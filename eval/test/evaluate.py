"""Hidden-test evaluator.

Generates the hidden test scenes in memory (``eval/test/scenes.py``), writes
only their public part to a scratch directory, runs every solver found in
``scripts/methods/*.sh`` on them (``bash <solver> INPUT_DIR OUTPUT_DIR``), and
recomputes every scored number from the raw line estimates.  Nothing a solver
writes besides its line estimates is read.  Solvers run from a symlinked view
of the workspace that does not contain ``eval/test``, each on a fresh copy of
the public inputs.

A solver run is *failed* when it exits non-zero, times out, misses a scene or a
line, writes a non-finite or non-Plucker line, or does not solve the task
(median endpoint-to-line residual on the scene's own observations above
``GATE_MEDIAN_RESIDUAL_PX``).  A failed run receives the worst value of every
scored metric in every experiment, and so does every method row required by
``scoring/reference.json`` that has no solver on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.test import scenes as scene_lib

ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = ROOT / "scripts" / "methods"
REFERENCE = ROOT / "scoring" / "reference.json"

SOLVER_TIMEOUT_S = 3600
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
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise InvalidOutput(f"{path.name}: unreadable JSON ({exc})") from exc
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list):
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
        if dn < 1e-9:
            raise InvalidOutput(f"{path.name}: zero direction for line {lid}")
        d = d / dn
        m = n / dn
        if float(np.linalg.norm(m)) > 1e6:
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
    start = time.time()
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                ["bash", str(view / "scripts" / "methods" / script.name), str(in_dir), str(out_dir)],
                cwd=str(view),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=SOLVER_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {SOLVER_TIMEOUT_S}s"
    if proc.returncode != 0:
        tail = log_path.read_text()[-600:]
        return False, f"exit code {proc.returncode}: {tail}"
    return True, f"ok in {time.time() - start:.1f}s"


def evaluate(scores_path: Path) -> None:
    experiments = list(scene_lib.EXPERIMENTS)
    solvers = {p.stem: p for p in sorted(SOLVER_DIR.glob("*.sh"))} if SOLVER_DIR.is_dir() else {}
    methods = sorted(set(solvers) | set(_required_methods()))
    scratch_parent = ROOT / "tmp"
    try:
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="eval_test_", dir=str(scratch_parent)))
    except OSError:
        scratch = Path(tempfile.mkdtemp(prefix="eval_test_"))

    status: dict[str, dict[str, Any]] = {m: {"failed": False, "reason": "", "runs": {}} for m in methods}
    per_scene: dict[str, dict[str, list]] = {m: {} for m in methods}
    try:
        all_scenes = {exp: scene_lib.all_scenes(exp) for exp in experiments}
        view = _solver_workspace(scratch)
        for exp in experiments:
            in_dir = scratch / exp / "inputs"
            in_dir.mkdir(parents=True)
            for scene in all_scenes[exp]:
                (in_dir / f"{scene['public']['scene_id']}.json").write_text(json.dumps(scene["public"]))
        for method in methods:
            if method not in solvers:
                status[method].update(failed=True, reason="no solver script scripts/methods/%s.sh" % method)
                continue
            for exp in experiments:
                if status[method]["failed"]:
                    break
                # Fresh copy of the public inputs for every run, so no solver
                # can alter what the next one (or the scorer) sees.
                run_in = scratch / exp / f"inputs_{method}"
                shutil.copytree(scratch / exp / "inputs", run_in)
                out_dir = scratch / exp / f"outputs_{method}"
                out_dir.mkdir(parents=True)
                ok, msg = _run_solver(view, solvers[method], run_in, out_dir, scratch / f"{method}_{exp}.log")
                status[method]["runs"][exp] = msg[:300]
                print(f"[test] {method} / {exp}: {msg[:200]}", flush=True)
                if not ok:
                    status[method].update(failed=True, reason=f"{exp}: {msg[:300]}")
                    break
                rows = []
                try:
                    for scene in all_scenes[exp]:
                        sid = scene["public"]["scene_id"]
                        rows.append(_score_scene(scene, out_dir / f"{sid}.json"))
                except InvalidOutput as exc:
                    status[method].update(failed=True, reason=f"{exp}: {exc}")
                    break
                except Exception as exc:  # noqa: BLE001
                    status[method].update(failed=True, reason=f"{exp}: scoring error {exc!r}")
                    break
                per_scene[method][exp] = rows
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    scores: dict[str, Any] = {
        "experiments": {},
        "_metadata": {
            "split": scene_lib.SPLIT,
            "scenes_per_experiment": scene_lib.NUM_SCENES,
            "solver_dir": "scripts/methods",
            "gate_median_residual_px": GATE_MEDIAN_RESIDUAL_PX,
            "worst_values": WORST,
            "failed_methods": {m: s["reason"] for m, s in status.items() if s["failed"]},
        },
    }
    for exp in experiments:
        results = {}
        details = {}
        for method in methods:
            if status[method]["failed"]:
                results[method] = dict(WORST)
            else:
                results[method] = pooled_metrics(per_scene[method][exp])
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
            print(f"  {exp:28s} {method:22s} " + " ".join(f"{k}={vals[k]:.4f}" for k in METRICS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_test.json"))
    args = parser.parse_args()
    evaluate(Path(args.scores))


if __name__ == "__main__":
    sys.exit(main())
