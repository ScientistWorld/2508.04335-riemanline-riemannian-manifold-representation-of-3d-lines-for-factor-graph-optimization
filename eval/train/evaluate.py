"""Train-split evaluator (same interface and scoring rules as the hidden test).

Generates the public-seed train scenes (``eval/train/scenes.py``), writes
only their public part to a scratch directory, runs every solver found in
``scripts/methods/*.sh`` on them (``bash <solver> INPUT_DIR OUTPUT_DIR``), and
recomputes every scored number from the raw line estimates.  Nothing a solver
writes besides its line estimates is read.  Solvers run from a symlinked view
of the workspace that does not contain ``eval/test``, each on a fresh copy of
the public inputs.

A solver run is *failed* when it exits non-zero, misses a scene or a
line, writes a non-finite or non-Plucker line, or does not solve the task
(median endpoint-to-line residual on the scene's own observations above
``GATE_MEDIAN_RESIDUAL_PX``).  A failed run receives the worst value of every
scored metric in every experiment, and so does every method row required by
``scoring/reference.json`` that has no solver on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.train import guard
from eval.train import real as real_lib
from eval.train import scenes as scene_lib

ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = ROOT / "scripts" / "methods"
REFERENCE = ROOT / "scoring" / "reference.json"

GATE_MEDIAN_RESIDUAL_PX = 10.0
PIXEL_CLIP = 100.0
SAME_DIRECTION_RAD = 1e-7
PLUCKER_TOL = 1e-6

METRICS = (
    "median_direction_error_deg",
    "p90_direction_error_deg",
    "median_normal_error_deg",
    "p90_normal_error_deg",
    "novel_view_line_rmse_px",
    "param_count_error_per_line",
)
WORST = {
    "median_direction_error_deg": 90.0,
    "p90_direction_error_deg": 90.0,
    "median_normal_error_deg": 90.0,
    "p90_normal_error_deg": 90.0,
    "novel_view_line_rmse_px": PIXEL_CLIP,
    "param_count_error_per_line": 2.0,
}


class InvalidOutput(Exception):
    pass


# ---------------------------------------------------------------- geometry

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two line vectors; degenerate input scores 90."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not (math.isfinite(na) and math.isfinite(nb)) or na < 1e-12 or nb < 1e-12:
        return 90.0
    c = abs(float(np.dot(a, b))) / (na * nb)
    if not math.isfinite(c):
        return 90.0
    return math.degrees(math.acos(min(1.0, max(0.0, c))))


def _image_lines(cams: list[dict], cam_ids: np.ndarray, normals: np.ndarray, directions: np.ndarray):
    by_id = {int(c["id"]): c for c in cams}
    r = np.array([by_id[int(i)]["R_cw"] for i in cam_ids], float)
    t = np.array([by_id[int(i)]["t_cw"] for i in cam_ids], float)
    kinv_t = np.array([np.linalg.inv(np.asarray(by_id[int(i)]["K"], float)).T for i in cam_ids])
    dc = np.einsum("mij,mj->mi", r, directions)
    nc = np.einsum("mij,mj->mi", r, normals) + np.cross(t, dc)
    return np.einsum("mij,mj->mi", kinv_t, nc)


def _point_line_px(lines: np.ndarray, pts: np.ndarray) -> np.ndarray:
    den = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    num = np.abs(lines[:, 0] * pts[:, 0] + lines[:, 1] * pts[:, 1] + lines[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(den > 1e-12, num / np.maximum(den, 1e-300), PIXEL_CLIP)
    dist = np.where(np.isfinite(dist), dist, PIXEL_CLIP)
    return np.minimum(dist, PIXEL_CLIP)


def _project_points(cam: dict, pts: np.ndarray):
    r = np.asarray(cam["R_cw"], float)
    t = np.asarray(cam["t_cw"], float)
    k = np.asarray(cam["K"], float)
    pc = pts @ r.T + t
    z = pc[:, 2]
    uv = (pc @ k.T)[:, :2] / np.where(np.abs(z) > 1e-12, z, 1e-12)[:, None]
    return uv, z


def _count_directions(directions: np.ndarray) -> int:
    n = len(directions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(np.cross(directions[i], directions[j])) < SAME_DIRECTION_RAD:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


# ---------------------------------------------------------------- scoring

def _read_lines(path: Path, expected_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise InvalidOutput(f"missing output {path.name}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise InvalidOutput(f"{path.name}: output JSON exceeds 16 MiB")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise InvalidOutput(f"{path.name}: unreadable JSON ({exc})") from exc
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list) or len(lines) != len(expected_ids):
        raise InvalidOutput(f"{path.name}: no 'lines' list")
    by_id: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for item in lines:
        try:
            lid = int(item["id"])
            n = np.asarray(item["normal"], dtype=float).reshape(3)
            d = np.asarray(item["direction"], dtype=float).reshape(3)
        except Exception as exc:  # noqa: BLE001
            raise InvalidOutput(f"{path.name}: malformed line entry ({exc})") from exc
        if lid in by_id:
            raise InvalidOutput(f"{path.name}: duplicate line id {lid}")
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(d))):
            raise InvalidOutput(f"{path.name}: non-finite line {lid}")
        dn = float(np.linalg.norm(d))
        if not math.isfinite(dn) or dn < 1e-9:
            raise InvalidOutput(f"{path.name}: zero direction for line {lid}")
        d = d / dn
        m = n / dn
        if not np.all(np.isfinite(m)) or float(np.linalg.norm(m)) > 1e6:
            raise InvalidOutput(f"{path.name}: line {lid} at infinity")
        if abs(float(np.dot(m, d))) > PLUCKER_TOL * max(1.0, float(np.linalg.norm(m))):
            raise InvalidOutput(f"{path.name}: line {lid} violates the Plucker constraint n.d = 0")
        by_id[lid] = (m, d)
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise InvalidOutput(f"{path.name}: line ids mismatch (missing {missing[:5]}, extra {extra[:5]})")
    normals = np.array([by_id[i][0] for i in expected_ids])
    directions = np.array([by_id[i][1] for i in expected_ids])
    return normals, directions


def _score_scene(scene: dict, output_path: Path) -> dict[str, Any]:
    public, hidden = scene["public"], scene["hidden"]
    ids = [int(line["id"]) for line in public["lines"]]
    normals, directions = _read_lines(output_path, ids)
    index = {lid: i for i, lid in enumerate(ids)}

    # Task gate on the scene's own observations (evaluator copy).
    obs = public["observations"]
    li = np.array([index[int(o["line_id"])] for o in obs])
    cam_ids = np.array([int(o["camera_id"]) for o in obs])
    img = _image_lines(public["cameras"], cam_ids, normals[li], directions[li])
    res = np.concatenate(
        [
            _point_line_px(img, np.array([o["p0"] for o in obs], float)),
            _point_line_px(img, np.array([o["p1"] for o in obs], float)),
        ]
    )
    input_median = float(np.median(res))
    if not input_median <= GATE_MEDIAN_RESIDUAL_PX:
        raise InvalidOutput(
            f"{public['scene_id']}: median residual {input_median:.2f}px on its own observations "
            f"exceeds {GATE_MEDIAN_RESIDUAL_PX}px (task not solved)"
        )

    gt = hidden["lines"]
    dir_err = []
    nrm_err = []
    for i, line in enumerate(gt):
        dir_err.append(_angle_deg(directions[i], np.asarray(line["direction"], float)))
        nrm_err.append(_angle_deg(normals[i], np.asarray(line["normal"], float)))

    # Novel views: noise-free projections of the true segments into cameras
    # the method never saw, scored against the projected estimated lines.
    sq = []
    for cam in hidden["heldout_cameras"]:
        for i, line in enumerate(gt):
            ends = np.array([line["start"], line["end"]], float)
            uv, z = _project_points(cam, ends)
            if np.any(z <= 0.2):
                continue
            if np.any(uv[:, 0] < -IMG_MARGIN_W) or np.any(uv[:, 0] > scene_lib.IMAGE_W + IMG_MARGIN_W):
                continue
            if np.any(uv[:, 1] < -IMG_MARGIN_H) or np.any(uv[:, 1] > scene_lib.IMAGE_H + IMG_MARGIN_H):
                continue
            l_img = _image_lines([cam], np.array([cam["id"]]), normals[i : i + 1], directions[i : i + 1])
            dist = _point_line_px(np.repeat(l_img, 2, axis=0), uv)
            sq.extend((dist**2).tolist())

    true_classes = len({int(line["true_class"]) for line in gt})
    found = _count_directions(directions)
    return {
        "scene_id": public["scene_id"],
        "num_lines": len(ids),
        "direction_errors_deg": [round(v, 6) for v in dir_err],
        "normal_errors_deg": [round(v, 6) for v in nrm_err],
        "novel_view_sq_sum": float(np.sum(sq)) if sq else 0.0,
        "novel_view_count": len(sq),
        "distinct_directions": int(found),
        "true_direction_classes": int(true_classes),
        "input_median_residual_px": input_median,
    }


IMG_MARGIN_W = scene_lib.IMAGE_W / 2.0
IMG_MARGIN_H = scene_lib.IMAGE_H / 2.0


def pooled_metrics(per_scene: list[dict[str, Any]]) -> dict[str, float]:
    dirs = np.concatenate([np.asarray(s["direction_errors_deg"], float) for s in per_scene])
    nrms = np.concatenate([np.asarray(s["normal_errors_deg"], float) for s in per_scene])
    sq_sum = sum(s["novel_view_sq_sum"] for s in per_scene)
    count = sum(s["novel_view_count"] for s in per_scene)
    lines = sum(s["num_lines"] for s in per_scene)
    param_err = sum(2 * abs(s["distinct_directions"] - s["true_direction_classes"]) for s in per_scene)
    return {
        "median_direction_error_deg": float(np.median(dirs)),
        "p90_direction_error_deg": float(np.percentile(dirs, 90)),
        "median_normal_error_deg": float(np.median(nrms)),
        "p90_normal_error_deg": float(np.percentile(nrms, 90)),
        "novel_view_line_rmse_px": float(math.sqrt(sq_sum / count)) if count else PIXEL_CLIP,
        "param_count_error_per_line": float(param_err / lines),
    }


# ---------------------------------------------------------------- running

def _fixed_assets_error() -> str:
    pins = json.loads(Path(__file__).with_name("baseline_hashes.json").read_text())
    for name, digest in pins.items():
        path = ROOT / name
        # A missing solver gets a complete failed row via _required_methods.
        if not path.exists() and name.startswith("scripts/methods/"):
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return f"evaluator-pinned comparison asset changed: {name}"
    return ""


def _required_methods() -> list[str]:
    try:
        ref = json.loads(REFERENCE.read_text())
    except Exception:  # noqa: BLE001
        return []
    names: set[str] = set()
    for exp in ref.get("experiments", {}).values():
        if isinstance(exp, dict):
            names.update(str(m) for m in exp.get("results", {}) if not str(m).startswith("_"))
    return sorted(names)


def _solver_workspace(scratch: Path) -> Path:
    """Symlinked view of the workspace without the hidden evaluator (eval/test)."""
    view = scratch / "solver_workspace"
    view.mkdir()
    for entry in sorted(ROOT.iterdir()):
        if entry.name in ("eval", "tmp", ".git"):
            continue
        (view / entry.name).symlink_to(entry)
    (view / "tmp").mkdir()
    eval_view = view / "eval"
    eval_view.mkdir()
    for entry in sorted((ROOT / "eval").iterdir()):
        if entry.name != "test":
            (eval_view / entry.name).symlink_to(entry)
    return view


def _run_solver(view: Path, script: Path, in_dir: Path, out_dir: Path, log_path: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{view}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(out_dir / ".tmp")
    (out_dir / ".tmp").mkdir(exist_ok=True)
    read_paths, write_paths = guard.solver_paths(ROOT, view, in_dir, out_dir)
    start = time.time()
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                ["bash", str(view / "scripts" / "methods" / script.name), str(in_dir), str(out_dir)],
                cwd=str(view),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: guard.restrict_filesystem(read_paths, write_paths),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"solver launch / answer isolation failed: {exc}"
    if proc.returncode != 0:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 600))
            tail = stream.read().decode(errors="replace")
        return False, f"exit code {proc.returncode}: {tail}"
    return True, f"ok in {time.time() - start:.1f}s"


def evaluate(scores_path: Path) -> None:
    synthetic = list(scene_lib.EXPERIMENTS)
    experiments = synthetic + list(real_lib.EXPERIMENTS)
    solvers = {p.stem: p for p in sorted(SOLVER_DIR.glob("*.sh"))} if SOLVER_DIR.is_dir() else {}
    methods = sorted(set(solvers) | set(_required_methods()))
    # Train inputs/outputs stay on disk for inspection (gitignored).
    scratch = ROOT / "outputs" / "train"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    status: dict[str, dict[str, Any]] = {m: {"failed": False, "reason": "", "runs": {}} for m in methods}
    per_scene: dict[str, dict[str, list]] = {m: {} for m in methods}
    try:
        all_scenes = {exp: scene_lib.all_scenes(exp) for exp in synthetic}
        for exp in real_lib.EXPERIMENTS:
            all_scenes[exp] = real_lib.all_instances(exp)
        view = _solver_workspace(scratch)
        # One solver call over the scenes of every experiment, interleaved under
        # opaque ids: a solver cannot tell which experiment a scene belongs to.
        in_dir = scratch / "inputs"
        in_dir.mkdir(parents=True)
        for exp in experiments:
            for scene in all_scenes[exp]:
                (in_dir / f"{scene['public']['scene_id']}.json").write_text(json.dumps(scene["public"]))
        fixed_error = _fixed_assets_error()
        for method in methods:
            if fixed_error:
                status[method].update(failed=True, reason=fixed_error)
                continue
            if method not in solvers:
                status[method].update(failed=True, reason="no solver script scripts/methods/%s.sh" % method)
                continue
            # Fresh copy of the public inputs for every solver, so no solver can
            # alter what the next one (or the scorer) sees.
            run_in = scratch / f"inputs_{method}"
            shutil.copytree(in_dir, run_in)
            out_dir = scratch / f"outputs_{method}"
            out_dir.mkdir(parents=True)
            ok, msg = _run_solver(view, solvers[method], run_in, out_dir, scratch / f"{method}.log")
            status[method]["runs"]["all"] = msg[:300]
            print(f"[train] {method}: {msg[:200]}", flush=True)
            if not ok:
                status[method].update(failed=True, reason=msg[:300])
                continue
            for exp in experiments:
                rows = []
                try:
                    for scene in all_scenes[exp]:
                        sid = scene["public"]["scene_id"]
                        if exp in real_lib.EXPERIMENTS:
                            rows.append(real_lib.score_instance(scene, out_dir / f"{sid}.json"))
                        else:
                            rows.append(_score_scene(scene, out_dir / f"{sid}.json"))
                except (InvalidOutput, real_lib.InvalidJointOutput) as exc:
                    status[method].update(failed=True, reason=f"{exp}: {exc}")
                    break
                except Exception as exc:  # noqa: BLE001
                    status[method].update(failed=True, reason=f"{exp}: scoring error {exc!r}")
                    break
                per_scene[method][exp] = rows
            if not status[method]["failed"]:
                for exp in experiments:
                    real = exp in real_lib.EXPERIMENTS
                    vals = (real_lib.pooled_metrics if real else pooled_metrics)(per_scene[method][exp])
                    worst = real_lib.WORST if real else WORST
                    if set(vals) != set(worst) or any(not math.isfinite(v) or v < 0 or v > worst[k] for k, v in vals.items()):
                        status[method].update(failed=True, reason=f"{exp}: impossible aggregate metric")
                        break
    finally:
        pass

    scores: dict[str, Any] = {
        "experiments": {},
        "_metadata": {
            "split": scene_lib.SPLIT,
            "scenes_per_experiment": scene_lib.NUM_SCENES,
            "solver_dir": "scripts/methods",
            "gate_median_residual_px": GATE_MEDIAN_RESIDUAL_PX,
            "worst_values": WORST,
            "joint_ba_instances": {e: len(all_scenes.get(e, [])) for e in real_lib.EXPERIMENTS},
            "joint_ba_gate_median_line_residual_px": real_lib.GATE_MEDIAN_LINE_RESIDUAL_PX,
            "joint_ba_worst_values": real_lib.WORST,
            "failed_methods": {m: s["reason"] for m, s in status.items() if s["failed"]},
        },
    }
    for exp in experiments:
        results = {}
        details = {}
        for method in methods:
            real = exp in real_lib.EXPERIMENTS
            if status[method]["failed"]:
                results[method] = dict(real_lib.WORST if real else WORST)
            else:
                results[method] = (real_lib.pooled_metrics if real else pooled_metrics)(per_scene[method][exp])
                details[method] = per_scene[method][exp]
        scores["experiments"][exp] = {
            "results": results,
            "_status": {m: {"failed": status[m]["failed"], "reason": status[m]["reason"]} for m in methods},
            "_per_scene": details,
        }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
    print(f"wrote {scores_path}")
    for exp in experiments:
        for method in methods:
            vals = scores["experiments"][exp]["results"][method]
            print(f"  {exp:28s} {method:22s} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()))


def export_scenes(out_dir: Path) -> None:
    """Write the full train scenes (public inputs + ground truth) for development."""
    for exp in scene_lib.EXPERIMENTS:
        target = out_dir / exp
        target.mkdir(parents=True, exist_ok=True)
        for scene in scene_lib.all_scenes(exp):
            sid = scene["public"]["scene_id"]
            payload = dict(scene["public"])
            payload["ground_truth"] = scene["hidden"]
            (target / f"{sid}.json").write_text(json.dumps(payload) + "\n")
    print(f"exported train scenes to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--export-scenes", default=None,
                        help="only write the full train scenes (with ground truth) to this directory")
    args = parser.parse_args()
    if args.export_scenes:
        export_scenes(Path(args.export_scenes))
        return
    evaluate(Path(args.scores))


if __name__ == "__main__":
    sys.exit(main())
