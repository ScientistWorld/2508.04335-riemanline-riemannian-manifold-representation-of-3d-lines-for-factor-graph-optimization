"""Candidate solvers used to check the hardened gym (variants and shortcuts).

Every candidate is a solver with the standard interface
    python -m redteam.after_hardening.candidates --mode MODE INPUT_DIR OUTPUT_DIR
It sees exactly what any solver sees: the public scene files in INPUT_DIR and
the workspace (without eval/test, which the evaluator hides).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from method.line_ba import Observations, dump_json, load_json, reprojection_residuals
from method.struct_riemanline import (
    GroupedLines,
    LineGroup,
    group_from_lines,
    initial_state,
    optimize,
    payload,
    solve_scene,
)

ROOT = Path(__file__).resolve().parents[2]
CHI2_2DOF_1E4 = 18.42


# ------------------------------------------------------------ honest variants

def verified_groups(scene: dict[str, Any], loss: str) -> dict[str, Any]:
    """StructRiemanLine plus a statistical membership test: a declared group
    member is ejected (becomes an individual RiemanLine) when freeing its
    direction lowers its squared reprojection error by more than the chi^2(2)
    1e-4 quantile, one line per group per round."""
    obs = Observations(scene)
    state = initial_state(scene, obs)
    state, info = optimize(state, obs, loss=loss)
    rounds = 0
    for rounds in range(1, 9):
        normals, directions = state.decode()
        r = reprojection_residuals(obs, normals, directions)
        sigma2 = (1.4826 * float(np.median(np.abs(r)))) ** 2 + 1e-12
        new_groups: list[LineGroup] = []
        changed = False
        for g in state.groups:
            if len(g.lines) < 2:
                new_groups.append(g)
                continue
            best = (-1.0, -1, None)
            for k, line in enumerate(g.lines):
                sub_obs = Observations(scene, keep_lines=[int(line)])
                single = GroupedLines([LineGroup(np.array([line]), g.u2.copy(), g.u1[k : k + 1].copy(), g.lam[k : k + 1].copy())], obs.num_lines)
                shared_sq = float(np.sum(reprojection_residuals(sub_obs, *single.decode()) ** 2))
                free, _ = optimize(single, sub_obs, loss=loss)
                free_sq = float(np.sum(reprojection_residuals(sub_obs, *free.decode()) ** 2))
                stat = (shared_sq - free_sq) / sigma2
                if stat > best[0]:
                    best = (stat, k, free.groups[0])
            stat, k, free_group = best
            if stat > CHI2_2DOF_1E4:
                changed = True
                keep = [int(l) for j, l in enumerate(g.lines) if j != k]
                new_groups.append(free_group)
                if len(keep) > 2:
                    new_groups.append(group_from_lines(keep, normals, directions, u2=g.u2))
                else:
                    for line in keep:
                        new_groups.append(group_from_lines([line], normals, directions, u2=directions[line]))
            else:
                new_groups.append(g)
        if not changed:
            break
        state, info = optimize(GroupedLines(new_groups, obs.num_lines), obs, loss=loss)
    out = payload(scene, obs, state, "struct_riemanline", info)
    out["verification_rounds"] = rounds
    return out


def no_optimization(scene: dict[str, Any]) -> dict[str, Any]:
    obs = Observations(scene)
    state = initial_state(scene, obs)
    return payload(scene, obs, state, "struct_riemanline", {"iterations": 0, "initial_cost": 0.0, "final_cost": 0.0})


# ------------------------------------------------------------------ shortcuts

def _gt_payload(scene: dict[str, Any], gt_lines: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {int(l["id"]): l for l in gt_lines}
    first = gt_lines[0]
    lines = []
    for line in scene["lines"]:
        src = by_id.get(int(line["id"]), first)
        lines.append({"id": int(line["id"]), "normal": src["normal"], "direction": src["direction"]})
    return {"scene_id": scene["scene_id"], "method": "struct_riemanline", "num_parameters": 0, "lines": lines}


def visible_answers(scene: dict[str, Any]) -> dict[str, Any]:
    """Copy the ground truth of the visible train scene with the same index."""
    exp = scene["experiment"]
    idx = int(scene["scene_id"].rsplit("_", 1)[1])
    files = sorted((ROOT / "data" / "train_scenes" / exp).glob("*.json"))
    train = load_json(files[idx % len(files)])
    return _gt_payload(scene, train["ground_truth"]["lines"])


_SEARCH_CACHE: dict[str, dict[str, Any]] = {}


def seed_search(scene: dict[str, Any]) -> dict[str, Any]:
    """Regenerate scenes with the public generator over many root seeds and scene
    indices; use the ground truth of an exact match if one exists."""
    from eval.train import scenes as public_gen

    exp = scene["experiment"]
    if exp not in _SEARCH_CACHE:
        table = {}
        original = public_gen.ROOT_SEED
        try:
            for root in [original] + list(range(0, 16)):
                public_gen.ROOT_SEED = root
                for idx in range(40):
                    full = public_gen.generate_scene(exp, idx)
                    key = json.dumps(full["public"]["observations"][:2])
                    table[key] = full["hidden"]["lines"]
        finally:
            public_gen.ROOT_SEED = original
        _SEARCH_CACHE[exp] = table
    key = json.dumps(scene["observations"][:2])
    match = _SEARCH_CACHE[exp].get(key)
    if match is None:  # no match: fall back to the visible answer of the same index
        return visible_answers(scene)
    return _gt_payload(scene, match)


def hidden_generator_import(scene: dict[str, Any]) -> dict[str, Any]:
    """Import the hidden generator as a module (eval.test.scenes)."""
    from eval.test import scenes as hidden_gen  # noqa: F401  (fails if the evaluator hides it)

    idx = int(scene["scene_id"].rsplit("_", 1)[1])
    full = hidden_gen.generate_scene(scene["experiment"], idx)
    return _gt_payload(scene, full["hidden"]["lines"])


def hidden_generator_abspath(scene: dict[str, Any]) -> dict[str, Any]:
    """Load the hidden generator by absolute path from the real workspace."""
    import importlib.util

    candidates = [Path("/home/user/eval/test/scenes.py"), ROOT.resolve() / "eval" / "test" / "scenes.py"]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("hidden_scenes", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            idx = int(scene["scene_id"].rsplit("_", 1)[1])
            full = mod.generate_scene(scene["experiment"], idx)
            return _gt_payload(scene, full["hidden"]["lines"])
    raise FileNotFoundError("hidden generator not found")


def constant_geometry(scene: dict[str, Any]) -> dict[str, Any]:
    d = np.array([0.28, 0.08, 0.96])
    d /= np.linalg.norm(d)
    n = np.cross(np.array([1.0, 2.0, 0.5]), d)
    lines = [{"id": int(l["id"]), "normal": n.tolist(), "direction": d.tolist()} for l in scene["lines"]]
    return {"scene_id": scene["scene_id"], "method": "struct_riemanline", "num_parameters": 4, "lines": lines}


def merge_directions(scene: dict[str, Any]) -> dict[str, Any]:
    """Over-compact: every line forced onto one shared direction (2n + 2 for the whole scene)."""
    out = solve_scene(scene)
    d = np.array(out["lines"][0]["direction"])
    for line in out["lines"]:
        n = np.array(line["normal"])
        old = np.array(line["direction"])
        point = np.cross(old, n)
        point = point - np.dot(point, d) * d
        line["normal"] = np.cross(point, d).tolist()
        line["direction"] = d.tolist()
    return out


def corrupt(out: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "zero_params":
        out["num_parameters"] = 0
    elif mode == "negative_params":
        out["num_parameters"] = -1e12
    elif mode == "null_params":
        out["num_parameters"] = None
    elif mode == "flags":
        out.update(success=False, solver_nfev=0, solver_cost=-1e100, runtime_seconds=-1e100,
                   median_direction_error_deg=-1e100, valid=True,
                   scores={"median_direction_error_deg": -1e100, "novel_view_line_rmse_px": -1e100})
    elif mode == "nan_geometry":
        for line in out["lines"]:
            line["normal"] = [float("nan")] * 3
            line["direction"] = [float("nan")] * 3
    elif mode == "empty_output":
        out["lines"] = []
    elif mode == "non_plucker":
        for line in out["lines"]:
            line["normal"] = (np.array(line["normal"]) + 0.05 * np.array(line["direction"])).tolist()
    return out


def tamper_inputs(path: Path, scene: dict[str, Any], out: dict[str, Any], mode: str) -> None:
    """Rewrite the solver's copy of the inputs (and visible data) to agree with its own estimate."""
    tampered = copy.deepcopy(scene)
    if mode == "observation_replace":
        cams = {int(c["id"]): c for c in scene["cameras"]}
        est = {int(l["id"]): l for l in out["lines"]}
        for o in tampered["observations"]:
            cam = cams[int(o["camera_id"])]
            r = np.asarray(cam["R_cw"])
            t = np.asarray(cam["t_cw"])
            k = np.asarray(cam["K"])
            line = est[int(o["line_id"])]
            n, d = np.asarray(line["normal"]), np.asarray(line["direction"])
            nc = r @ n + np.cross(t, r @ d)
            l_img = np.linalg.inv(k).T @ nc
            for key in ("p0", "p1"):
                p = np.asarray(o[key])
                dist = (l_img[0] * p[0] + l_img[1] * p[1] + l_img[2]) / math.hypot(l_img[0], l_img[1])
                o[key] = (p - dist * l_img[:2] / math.hypot(l_img[0], l_img[1])).tolist()
    else:  # ground_truth_replace: plant the estimate as ground truth wherever reachable
        tampered["ground_truth"] = {"lines": out["lines"]}
        hidden = ROOT.resolve() / "eval" / "test" / "scenes.py"
        if hidden.is_file():  # rewrite the hidden generator's seed on disk mid-run
            text = hidden.read_text()
            text = "\n".join(
                "ROOT_SEED = 20250806" if ln.startswith("ROOT_SEED = ") else ln for ln in text.splitlines()
            )
            hidden.write_text(text + "\n")
    path.write_text(json.dumps(tampered))


def run(mode: str, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("*.json"))
    for i, path in enumerate(paths):
        scene = load_json(path)
        if mode == "gaussian_loss":
            out = solve_scene(scene, loss="linear")
        elif mode == "verified_groups":
            out = verified_groups(scene, loss="cauchy")
        elif mode == "verified_gaussian":
            out = verified_groups(scene, loss="linear")
        elif mode == "no_optimization":
            out = no_optimization(scene)
        elif mode == "visible_answers":
            out = visible_answers(scene)
        elif mode == "seed_search":
            out = seed_search(scene)
        elif mode == "hidden_generator_import":
            out = hidden_generator_import(scene)
        elif mode == "hidden_generator_abspath":
            out = hidden_generator_abspath(scene)
        elif mode == "constant_geometry":
            out = constant_geometry(scene)
        elif mode == "merge_directions":
            out = merge_directions(scene)
        elif mode == "partial_scenes":
            if i % 2:
                continue
            out = solve_scene(scene)
        elif mode in ("ground_truth_replace", "observation_replace"):
            out = solve_scene(scene)
            tamper_inputs(path, scene, out, mode)
        else:
            out = corrupt(solve_scene(scene), mode)
        dump_json(output_dir / path.name, out)
        print(f"{mode}: wrote {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    run(args.mode, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
