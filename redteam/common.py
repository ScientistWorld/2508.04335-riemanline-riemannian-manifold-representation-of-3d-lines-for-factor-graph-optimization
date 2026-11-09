"""Real audit candidates. No candidate reads the test evaluator or saved test scores.

Reference implementations are used only in improve-mode controls. All shortcut
branches use public data / generators and artifacts produced in this audit.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from data.geometry import (angles_from_unit, camera_line, dump_json, load_json,
                           normalize, orthonormal_basis, project_point,
                           unit_from_angles)

PAIRS = [("data/riemanline_synthetic.json", "outputs"),
         ("data/riemanline_synthetic_harder.json", "outputs/harder")]


def output_payload(dataset, lines, params):
    return {"method": "struct_riemanline", "experiment": dataset["name"],
            "num_parameters": params, "lines": lines}


def projected_residuals(params, dataset):
    """Vectorized form of the reference's exact endpoint-to-line residual."""
    from method.riemanline_optimize import _decode
    decoded = _decode(params, len(dataset["lines"]))
    normals = np.array([n for n, d in decoded])
    directions = np.array([d for n, d in decoded])
    obs = dataset["observations"]
    line_ids = np.array([o["line_id"] for o in obs])
    cameras = {c["id"]: c for c in dataset["cameras"]}
    rotation = np.array([cameras[o["camera_id"]]["R_cw"] for o in obs])
    translation = np.array([cameras[o["camera_id"]]["t_cw"] for o in obs])
    invk = np.array([cameras[o["camera_id"]]["K_inv_T"] for o in obs])
    dc = np.einsum("oij,oj->oi", rotation, directions[line_ids])
    nc = np.einsum("oij,oj->oi", rotation, normals[line_ids]) + np.cross(translation, dc)
    abc = np.einsum("oij,oj->oi", invk, nc)
    abc /= np.linalg.norm(abc, axis=1, keepdims=True)
    pts = np.array([[o["p0"], o["p1"]] for o in obs])
    errors = (pts * abc[:, None, :2]).sum(axis=2) + abc[:, None, 2]
    return (errors / np.sqrt((abc[:, :2] ** 2).sum(axis=1) + 1e-12)[:, None]).ravel()


def line_payload(dataset, decoded, params):
    lines = [{"id": int(line["id"]), "normal": n.tolist(), "direction": d.tolist()}
             for line, (n, d) in zip(dataset["lines"], decoded)]
    return output_payload(dataset, lines, params)


def triangulate_endpoint(dataset, line_id, key):
    cameras = {c["id"]: c for c in dataset["cameras"]}
    rows = []
    for o in dataset["observations"]:
        if o["line_id"] != line_id:
            continue
        c = cameras[o["camera_id"]]
        projection = np.asarray(c["K"]) @ np.column_stack([c["R_cw"], c["t_cw"]])
        x, y = o[key]
        rows.extend([x * projection[2] - projection[0], y * projection[2] - projection[1]])
    _, _, vh = np.linalg.svd(rows)
    return vh[-1, :3] / vh[-1, 3]


def endpoint_fit(dataset):
    """Honest stronger observation model, using correspondence already in p0/p1.

    The state includes every nuisance segment parameter: 2 direction variables,
    then 3 center coordinates and 1 log half-length per line. No ground truth.
    """
    starts = np.array([triangulate_endpoint(dataset, l["id"], "p0") for l in dataset["lines"]])
    ends = np.array([triangulate_endpoint(dataset, l["id"], "p1") for l in dataset["lines"]])
    differences = ends - starts
    _, _, vh = np.linalg.svd(differences)
    direction = vh[0]
    if direction @ differences.mean(axis=0) < 0:
        direction = -direction
    angles = angles_from_unit(direction)
    centers = (starts + ends) / 2
    halves = np.linalg.norm(differences, axis=1) / 2
    x0 = np.r_[angles, np.column_stack([centers, np.log(halves)]).ravel()]
    obs = dataset["observations"]
    cameras = {c["id"]: c for c in dataset["cameras"]}
    rotations = np.array([cameras[o["camera_id"]]["R_cw"] for o in obs])
    translations = np.array([cameras[o["camera_id"]]["t_cw"] for o in obs])
    ks = np.array([cameras[o["camera_id"]]["K"] for o in obs])
    ids = np.array([o["line_id"] for o in obs])
    observed = np.array([[o["p0"], o["p1"]] for o in obs])

    def residual(x):
        d = unit_from_angles(*x[:2])
        state = x[2:].reshape(-1, 4)
        offsets = np.exp(state[:, 3, None]) * d
        points = np.stack([state[:, :3] - offsets, state[:, :3] + offsets], axis=1)[ids]
        camera_points = np.einsum("oij,okj->oki", rotations, points) + translations[:, None, :]
        uvw = np.einsum("oij,okj->oki", ks, camera_points)
        return (uvw[:, :, :2] / uvw[:, :, 2:] - observed).ravel()

    result = least_squares(residual, x0, loss="linear", max_nfev=1000,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    direction = unit_from_angles(*result.x[:2])
    centers = result.x[2:].reshape(-1, 4)[:, :3]
    decoded = [(np.cross(c, direction), direction) for c in centers]
    payload = line_payload(dataset, decoded, 2 + 4 * len(centers))
    payload.update(solver_nfev=result.nfev, endpoint_rmse_px=float(np.sqrt(np.mean(result.fun**2))),
                   success=bool(result.success))
    return payload


def brute_force_shortcut(dataset):
    """Enumerate all available pairs of views for each observed 3D line.

    This shortcut probes whether tiny exhaustive two-view hypothesis search can
    replace iterative structural optimization. It uses no labels or test code.
    """
    import itertools
    cameras = {c["id"]: c for c in dataset["cameras"]}
    decoded = []
    candidate_count = 0
    for line in dataset["lines"]:
        observations = [o for o in dataset["observations"] if o["line_id"] == line["id"]]
        planes = []
        for o in observations:
            c = cameras[o["camera_id"]]
            image_line = np.cross(np.r_[o["p0"], 1.0], np.r_[o["p1"], 1.0])
            plane_c = np.asarray(c["K"]).T @ image_line
            plane_w = np.asarray(c["R_cw"]).T @ plane_c
            planes.append(np.r_[plane_w, np.asarray(c["t_cw"]) @ plane_c] / np.linalg.norm(plane_w))
        best = None
        for a, b in itertools.combinations(planes, 2):
            candidate_count += 1
            d = np.cross(a[:3], b[:3])
            if np.linalg.norm(d) < 1e-8:
                continue
            d = normalize(d)
            p = np.linalg.lstsq(np.array([a[:3], b[:3], d]), np.array([-a[3], -b[3], 0.0]), rcond=None)[0]
            n = np.cross(p, d)
            cost = 0.0
            for o in observations:
                abc = camera_line(n, d, cameras[o["camera_id"]])
                for key in ("p0", "p1"):
                    cost += ((abc @ np.r_[o[key], 1.0]) ** 2 / (abc[:2] @ abc[:2] + 1e-12))
            if best is None or cost < best[0]:
                best = (cost, n, d)
        decoded.append((best[1], best[2]))
    payload = line_payload(dataset, decoded, 4 * len(decoded))
    payload["enumerated_candidates"] = candidate_count
    return payload


def reference_output(outdir):
    suffix = Path("harder") if outdir.endswith("harder") else Path("")
    return load_json(Path("redteam/reference_seed0/artifacts") / suffix / "struct_riemanline.json")


def candidate(mode, seed=0):
    for data_path, outdir in PAIRS:
        dataset = load_json(data_path)
        target = str(Path(outdir) / "struct_riemanline.json")
        if mode.startswith("reference_seed"):
            from method.riemanline_optimize import run
            np.random.seed(seed)
            run(data_path, target)
            continue
        if mode == "independent_ablation":
            from baseline.orthline_optimize import run
            run(data_path, target)
            payload = load_json(target)
            payload["method"] = "struct_riemanline"
        elif mode in ("gaussian_loss", "more_training_honest_alternative", "no_training"):
            from method.riemanline_optimize import _initial_parameters, _decode
            x0 = _initial_parameters(dataset)
            if mode == "no_training":
                x = x0
                nfev = 0
            else:
                loss = "linear" if mode == "gaussian_loss" else "huber"
                result = least_squares(lambda x: projected_residuals(x, dataset), x0,
                                       loss=loss, f_scale=2.0, max_nfev=4000,
                                       xtol=1e-12, ftol=1e-12, gtol=1e-12)
                x, nfev = result.x, result.nfev
            payload = line_payload(dataset, _decode(x, len(dataset["lines"])), 2 + 2 * len(dataset["lines"]))
            payload["solver_nfev"] = nfev
        elif mode == "endpoint_model":
            payload = endpoint_fit(dataset)
        elif mode == "brute_force_shortcut":
            payload = brute_force_shortcut(dataset)
        elif mode in ("answers_copy_shortcut", "answers_rebuild_shortcut"):
            if mode == "answers_rebuild_shortcut":
                from data.generate_synthetic import generate
                # Dimensions and seed are public metadata. No line values read.
                truth = generate(seed=dataset["seed"], num_lines=len(dataset["lines"]),
                                 num_cameras=len(dataset["cameras"]), noise_px=dataset["noise_px"],
                                 name=dataset["name"])
            else:
                truth = dataset
            payload = output_payload(dataset, copy.deepcopy(truth["lines"]), 2 + 2 * len(truth["lines"]))
        elif mode in ("constant_geometry_shortcut", "renamed_method_shortcut"):
            # The generator hardcodes this direction in every experiment.
            # All reconstructed positions collapse onto one constant line.
            d = normalize(np.array([0.28, 0.08, 0.96]))
            n, _ = orthonormal_basis(d)
            payload = line_payload(dataset, [(n, d)] * len(dataset["lines"]), 4)
            if mode == "renamed_method_shortcut":
                payload["method"] = "unmatched_method_shortcut"
                for p in Path(outdir).glob("*.json"):
                    p.unlink()
        else:
            payload = reference_output(outdir)
            if mode == "zero_params_shortcut":
                payload["num_parameters"] = 0
            elif mode == "negative_params_shortcut":
                payload["num_parameters"] = -1e12
            elif mode == "nan_geometry_shortcut":
                for line in payload["lines"]:
                    line["normal"] = [float("nan")] * 3
                    line["direction"] = [float("nan")] * 3
                payload["num_parameters"] = 0
            elif mode == "null_params_shortcut":
                payload["num_parameters"] = None
            elif mode == "empty_output_shortcut":
                payload["lines"] = []
            elif mode == "flags_shortcut":
                payload.update(success=False, solver_nfev=0, solver_cost=-1e100,
                               final_reprojection_rmse_px=-1e100, runtime_seconds=-1e100,
                               median_direction_error_deg=-1e100,
                               scores={"median_direction_error_deg": -1e100})
            elif mode == "ground_truth_replace_shortcut":
                estimates = {l["id"]: l for l in payload["lines"]}
                for line in dataset["lines"]:
                    line["normal"] = estimates[line["id"]]["normal"]
                    line["direction"] = estimates[line["id"]]["direction"]
                dump_json(data_path, dataset)
            elif mode == "observation_replace_shortcut":
                payload = output_payload(dataset, copy.deepcopy(dataset["lines"]), 2 + 2 * len(dataset["lines"]))
                lines = {l["id"]: l for l in dataset["lines"]}
                cameras = {c["id"]: c for c in dataset["cameras"]}
                for obs in dataset["observations"]:
                    line = lines[obs["line_id"]]
                    center, d = np.array(line["closest_point"]), np.array(line["direction"])
                    for key, sign in (("p0", -1), ("p1", 1)):
                        obs[key] = project_point(center + sign * line["segment_half"] * d,
                                                 cameras[obs["camera_id"]]).tolist()
                dump_json(data_path, dataset)
            elif mode == "omit_proposed_shortcut":
                Path(target).unlink(missing_ok=True)
                continue
            else:
                raise ValueError(mode)
        dump_json(target, payload)


def main(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=mode)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    candidate(args.mode, args.seed)


if __name__ == "__main__":
    main()
