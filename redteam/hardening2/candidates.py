"""Candidate solvers for the second hardening pass (variants and shortcuts).

Every candidate is a solver with the standard interface
    python -m redteam.hardening2.candidates --mode MODE INPUT_DIR OUTPUT_DIR
and sees exactly what any solver sees: the public scene files in INPUT_DIR
(procedural line scenes and Open-Structure joint factor graphs, mixed) and
the workspace without eval/test.  Synthetic-scene behaviour of the earlier
candidates is reused from ``redteam/after_hardening/candidates.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from method.joint_ba import JointProblem, LineModel, Poses, frames_payload, robust, solve_joint
from method.line_ba import dump_json, load_json
from method.struct_riemanline import build_state, solve_joint_scene, solve_scene
from redteam.after_hardening import candidates as old

ROOT = Path(__file__).resolve().parents[2]
CAUCHY_PX = 3.0


def is_joint(scene: dict[str, Any]) -> bool:
    return scene.get("task") == "joint_ba"


def reference(scene: dict[str, Any]) -> dict[str, Any]:
    return solve_joint_scene(scene) if is_joint(scene) else solve_scene(scene)


# ------------------------------------------------------------ honest variants

def joint_no_optimization(scene: dict[str, Any]) -> dict[str, Any]:
    pb = JointProblem(scene)
    n, d = pb.initial_plucker()
    return {"frames": frames_payload(pb, Poses(pb.R_wc0, pb.t_wc0)),
            "lines": [{"id": pb.line_ids[i], "normal": n[i].tolist(), "direction": d[i].tolist()} for i in range(pb.num_lines)]}


def _retriangulate(pb: JointProblem, poses: Poses) -> tuple[np.ndarray, np.ndarray]:
    """Linear multi-view line triangulation from the segment measurements and the given poses."""
    normals, directions = pb.initial_plucker()
    r_wc = poses.R[pb.l_fr]
    t_wc = poses.t[pb.l_fr]
    hs = np.concatenate([pb.l_s, np.ones((len(pb.l_s), 1))], axis=1)
    he = np.concatenate([pb.l_e, np.ones((len(pb.l_e), 1))], axis=1)
    l_img = np.cross(hs, he)
    nw = np.einsum("kij,kj->ki", r_wc, l_img)  # R_wc l  (plane normal in the world)
    planes = np.concatenate([nw, -np.sum(nw * t_wc, axis=1, keepdims=True)], axis=1)
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True) + 1e-12
    for i in range(pb.num_lines):
        rows = planes[pb.l_ln == i]
        if len(rows) < 2:
            continue
        _, _, vt = np.linalg.svd(rows)
        x1, x2 = vt[-1], vt[-2]
        d = x1[3] * x2[:3] - x2[3] * x1[:3]
        n = np.cross(x1[:3], x2[:3])
        s = float(np.linalg.norm(d))
        if s < 1e-12:
            continue
        d, n = d / s, n / s
        normals[i] = n - np.dot(n, d) * d
        directions[i] = d
    return normals, directions


def _line_costs(pb: JointProblem, poses: Poses, normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    from method.joint_ba import line_residuals

    r = line_residuals(pb, poses, normals, directions).reshape(-1, 2)
    rho, _ = robust(np.sum(r * r, axis=1), "cauchy", CAUCHY_PX)
    return np.bincount(pb.l_ln, weights=rho, minlength=pb.num_lines)


def staged(scene: dict[str, Any], loss: str = "cauchy") -> dict[str, Any]:
    """StructRiemanLine with a better initial factor graph: (1) point-only
    bundle adjustment of the poses, (2) every line re-triangulated from its
    segment measurements with those poses (kept only where it explains the
    measurements better than the given initial line), (3) the reference
    StructRiemanLine joint optimisation from that graph."""
    pts_scene = dict(scene, lines=[], line_observations=[], parallel_groups=[])
    pb0 = JointProblem(pts_scene)
    empty = build_state([], *pb0.initial_plucker(), 0, use_groups=False)
    poses1, x1, _, _ = solve_joint(pb0, LineModel(empty, empty.column_lines()), loss=loss)

    pb = JointProblem(scene)
    n0, d0 = pb.initial_plucker()
    n1, d1 = _retriangulate(pb, poses1)
    c0 = _line_costs(pb, poses1, n0, d0)
    c1 = _line_costs(pb, poses1, n1, d1)
    use_new = c1 < c0
    normals = np.where(use_new[:, None], n1, n0)
    directions = np.where(use_new[:, None], d1, d0)
    lines = []
    for i, lid in enumerate(pb.line_ids):
        p = np.cross(directions[i], normals[i])  # closest point to the origin
        lines.append({"id": lid, "start": p.tolist(), "end": (p + directions[i]).tolist()})
    scene2 = dict(scene, frames=frames_payload(pb, poses1),
                  points=[{"id": pid, "xyz": x1[i].tolist()} for i, pid in enumerate(pb0.point_ids)],
                  lines=lines)
    out = solve_joint_scene(scene2, loss=loss)
    out["retriangulated_lines"] = int(np.sum(use_new))
    return out


# ------------------------------------------------------------------ shortcuts

def _read_traj(path: Path) -> dict[int, np.ndarray]:
    d = {}
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) >= 8 and not t[0].startswith("#"):
            d[int(float(t[0]))] = np.array(t[1:8], float)
    return d


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _sim3(src: np.ndarray, dst: np.ndarray):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(a.T @ b / len(src))
    dd = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        dd[2, 2] = -1
    r = vt.T @ dd @ u.T
    s = float(np.sum(sv * np.diag(dd))) / (float(np.sum(a * a)) / len(src))
    return s, r, md - s * r @ ms


_TRAJ_CACHE: dict[str, Any] = {}


def _match_sequence(scene: dict[str, Any], roots: list[Path]):
    """Find which Open-Structure sequence / keyframe phase a public instance came
    from, by aligning its initial poses with each sequence's initial (VO) poses."""
    r_cw = np.array([f["R_cw"] for f in scene["frames"]], float)
    t_cw = np.array([f["t_cw"] for f in scene["frames"]], float)
    centres = -np.einsum("nji,nj->ni", r_cw, t_cw)
    best = None
    for root in roots:
        if not root.is_dir():
            continue
        for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            key = str(seq_dir)
            if key not in _TRAJ_CACHE:
                venom = sorted(seq_dir.rglob("venom_sequence.txt"))
                gt = sorted(seq_dir.rglob("ground_truth.txt"))
                if not venom or not gt:
                    _TRAJ_CACHE[key] = None
                    continue
                vo = {}
                for line in venom[0].read_text().splitlines():
                    if line.startswith("Vertex:"):
                        t = line.split()
                        vo[int(t[1])] = np.array(t[2:9], float)
                _TRAJ_CACHE[key] = (vo, _read_traj(gt[0]))
            entry = _TRAJ_CACHE[key]
            if entry is None:
                continue
            vo, gt = entry
            frames_all = sorted(i for i in vo if i in gt)
            stride = max(1, math.ceil(len(frames_all) / 150))
            for offset in range(stride):
                kfs = frames_all[offset::stride]
                if len(kfs) != len(centres):
                    continue
                vo_c = np.array([vo[f][:3] for f in kfs])
                s, r, t = _sim3(centres, vo_c)
                res = float(np.sqrt(np.mean(np.sum((s * centres @ r.T + t - vo_c) ** 2, axis=1))))
                if best is None or res < best[0]:
                    best = (res, kfs, gt)
    if best is None or best[0] > 1e-3:
        return None
    return best[1], best[2]


def ground_truth_poses(scene: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    """Run the reference, then replace its poses by the matched sequence's
    ground-truth trajectory, expressed in the reference's own frame."""
    out = reference(scene)
    match = _match_sequence(scene, roots) if is_joint(scene) else None
    if match is None:
        out["gt_lookup"] = "no match"
        return out
    kfs, gt = match
    r_ref = np.array([f["R_cw"] for f in out["frames"]], float)
    t_ref = np.array([f["t_cw"] for f in out["frames"]], float)
    c_ref = -np.einsum("nji,nj->ni", r_ref, t_ref)
    c_gt = np.array([gt[f][:3] for f in kfs])
    s, r, t = _sim3(c_gt, c_ref)  # GT frame -> reference frame
    frames = []
    for i, f in enumerate(kfs):
        r_wc = r @ _quat_to_R(gt[f][3:7])
        c = s * r @ c_gt[i] + t
        frames.append({"id": out["frames"][i]["id"], "R_cw": r_wc.T.tolist(), "t_cw": (-r_wc.T @ c).tolist()})
    out["frames"] = frames
    out["gt_lookup"] = "matched"
    return out


def hidden_abspath(scene: dict[str, Any]) -> dict[str, Any]:
    if is_joint(scene):
        return ground_truth_poses(scene, [Path("/home/user/eval/test/downloads/open_structure"),
                                          ROOT.resolve() / "eval" / "test" / "downloads" / "open_structure"])
    return old.hidden_generator_abspath(scene)


def public_gt_lookup(scene: dict[str, Any]) -> dict[str, Any]:
    """Look the instance up among the public (train) Open-Structure sequences."""
    if is_joint(scene):
        return ground_truth_poses(scene, [ROOT / "data" / "downloads" / "open_structure"])
    return old.visible_answers(scene)


def scale_gauge(scene: dict[str, Any]) -> dict[str, Any]:
    """Reference output with the whole reconstruction shrunk by 2 about the first camera."""
    out = reference(scene)
    if not is_joint(scene):
        return out
    r0 = np.array(out["frames"][0]["R_cw"])
    c0 = -r0.T @ np.array(out["frames"][0]["t_cw"])
    sc = 0.5
    for f in out["frames"]:
        r = np.array(f["R_cw"])
        c = -r.T @ np.array(f["t_cw"])
        c2 = c0 + sc * (c - c0)
        f["t_cw"] = (-r @ c2).tolist()
    for line in out["lines"]:
        m, d = np.array(line["normal"]), np.array(line["direction"])
        # X -> c0 + sc (X - c0):  m' = sc m + (1 - sc) c0 x d
        line["normal"] = (sc * m + (1 - sc) * np.cross(c0, d)).tolist()
    return out


def constant_lines(scene: dict[str, Any]) -> dict[str, Any]:
    if not is_joint(scene):
        return old.constant_geometry(scene)
    out = joint_no_optimization(scene)
    d = np.array([0.28, 0.08, 0.96]) / np.linalg.norm([0.28, 0.08, 0.96])
    n = np.cross(np.array([1.0, 2.0, 0.5]), d)
    for line in out["lines"]:
        line["normal"], line["direction"] = n.tolist(), d.tolist()
    return out


def corrupt_joint(out: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "bad_rotation" and "frames" in out:
        for f in out["frames"]:
            f["R_cw"] = (1.01 * np.array(f["R_cw"])).tolist()
    elif mode == "missing_frames" and "frames" in out:
        out["frames"] = out["frames"][::2]
    elif mode == "nan_geometry":
        out = old.corrupt(out, mode)
        for f in out.get("frames", []):
            f["t_cw"] = [float("nan")] * 3
    return out


def observation_replace(path: Path, scene: dict[str, Any], out: dict[str, Any]) -> None:
    """Rewrite the solver's copy of the inputs so the measurements agree with its own estimate."""
    if not is_joint(scene):
        old.tamper_inputs(path, scene, out, "observation_replace")
        return
    tampered = copy.deepcopy(scene)
    intr = scene["intrinsics"]
    k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    frames = {int(f["id"]): (np.array(f["R_cw"]), np.array(f["t_cw"])) for f in out["frames"]}
    lines = {int(l["id"]): (np.array(l["normal"]), np.array(l["direction"])) for l in out["lines"]}
    for o in tampered["line_observations"]:
        r, t = frames[int(o[1])]
        m, d = lines[int(o[0])]
        l_img = np.linalg.inv(k).T @ (r @ m + np.cross(t, r @ d))
        nrm = math.hypot(l_img[0], l_img[1])
        for j in (2, 4):
            p = np.array(o[j:j + 2])
            dist = (l_img[0] * p[0] + l_img[1] * p[1] + l_img[2]) / nrm
            o[j:j + 2] = (p - dist * l_img[:2] / nrm).tolist()
    path.write_text(json.dumps(tampered))


def _guarded(fn, scene: dict[str, Any]) -> dict[str, Any]:
    """Honest variants fall back to the reference on an instance where they crash (logged)."""
    try:
        return fn(scene)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"WARNING: {fn.__name__} failed on {scene.get('scene_id')}: {exc!r}\n{traceback.format_exc()}", flush=True)
        out = reference(scene)
        out["fallback"] = "reference"
        return out


def run(mode: str, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("*.json"))
    for i, path in enumerate(paths):
        scene = load_json(path)
        joint = is_joint(scene)
        if mode == "no_optimization":
            out = joint_no_optimization(scene) if joint else old.no_optimization(scene)
        elif mode == "gaussian_loss":
            out = solve_joint_scene(scene, loss="linear") if joint else solve_scene(scene, loss="linear")
        elif mode == "verified_groups":
            out = reference(scene) if joint else old.verified_groups(scene, loss="cauchy")
        elif mode == "staged":
            out = _guarded(staged, scene) if joint else reference(scene)
        elif mode == "verified_staged":
            out = _guarded(staged, scene) if joint else old.verified_groups(scene, loss="cauchy")
        elif mode == "hidden_abspath":
            out = hidden_abspath(scene)
        elif mode == "public_gt_lookup":
            out = public_gt_lookup(scene)
        elif mode == "scale_gauge":
            out = scale_gauge(scene)
        elif mode == "constant_lines":
            out = constant_lines(scene)
        elif mode in ("bad_rotation", "missing_frames", "nan_geometry"):
            out = corrupt_joint(reference(scene), mode)
        elif mode == "observation_replace":
            out = reference(scene)
            observation_replace(path, scene, out)
        else:
            raise SystemExit(f"unknown mode {mode}")
        dump_json(output_dir / path.name, out)
        print(f"{mode}: wrote {path.name} ({'joint' if joint else 'lines'}) "
              f"{out.get('gt_lookup', '')}{out.get('retriangulated_lines', '')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    run(args.mode, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
