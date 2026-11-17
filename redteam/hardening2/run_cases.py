"""Run every case of the second hardening pass through the real, no-argument hidden test.

Each case gets its own copy of the workspace (code, scripts, scoring; the
downloaded Open-Structure data is symlinked), changes only scripts/methods/,
then runs ``bash scripts/evaluate_test.sh`` there.  To save CPU the baseline
solver scripts are removed in every case (their rows then score as failed in
that case; the baseline numbers come from the reference run in
scoring/scores_test.json).  Full score files go to tmp/hardening2/<case>/; a
slim copy is kept in redteam/hardening2/cases/<case>/.  Finished cases are
skipped on resume.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "tmp" / "hardening2"
PRIVATE = ROOT / "eval" / "test" / "hardening_cases"

CANDIDATE = """#!/bin/bash
set -e
WORKSPACE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${{PYTHONPATH:-}}"
python -m redteam.hardening2.candidates --mode {mode} "$1" "$2"
"""

HONEST = ["reference_repeat", "no_optimization", "gaussian_loss", "verified_groups", "joint_verified", "joint_verified_strict"]
# Measured in the previous run on the dataset-association experiments only (worse than or tied
# with the reference there); kept runnable but not replayed:
EXTRA = ["staged", "verified_staged", "joint_warm", "joint_warm_verified"]
SHORTCUTS = ["hidden_abspath", "public_gt_lookup", "scale_gauge", "constant_lines", "bad_rotation",
             "missing_frames", "nan_geometry", "overflow_geometry", "zero_params", "negative_params", "flags", "observation_replace"]
CASES: dict[str, dict] = {m: {"replace": m} for m in HONEST + SHORTCUTS + EXTRA}
CASES["omit_proposed"] = {"remove": ["struct_riemanline"]}
BASELINES = ["orthline", "orthline_constr", "riemanline"]


def make_workspace(case: str, spec: dict) -> Path:
    ws = WORK / case / "workspace"
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "upstream", "cases", "*.pyc", "downloads", "outputs", "archive_before", "hardening_cases", "results_full.json")
    for name in ("baseline", "data", "eval", "method", "scripts", "scoring"):
        shutil.copytree(ROOT / name, ws / name, ignore=ignore, symlinks=True)
    (ws / "data" / "downloads").symlink_to(ROOT / "data" / "downloads")
    (ws / "eval" / "test" / "downloads").symlink_to(ROOT / "eval" / "test" / "downloads")
    (ws / "redteam").mkdir()
    shutil.copy(ROOT / "redteam" / "__init__.py", ws / "redteam" / "__init__.py")
    shutil.copytree(ROOT / "redteam" / "after_hardening", ws / "redteam" / "after_hardening", ignore=ignore)
    shutil.copytree(HERE, ws / "redteam" / "hardening2", ignore=ignore)
    methods = ws / "scripts" / "methods"
    for name in BASELINES:
        (methods / f"{name}.sh").unlink(missing_ok=True)
    if "replace" in spec:
        (methods / "struct_riemanline.sh").write_text(CANDIDATE.format(mode=spec["replace"]))
    for name in spec.get("remove", []):
        (methods / f"{name}.sh").unlink()
    for stale in ("scores.json", "scores_train.json", "scores_test.json"):
        (ws / "scoring" / stale).unlink(missing_ok=True)
    return ws


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def slim(scores: dict) -> dict:
    out = {"_metadata": scores.get("_metadata", {}), "experiments": {}}
    for exp, payload in scores.get("experiments", {}).items():
        per_scene = {}
        for method, rows in payload.get("_per_scene", {}).items():
            items = []
            for r in rows:
                item = {"scene_id": r["scene_id"],
                        "median_direction_error_deg": _median(r["direction_errors_deg"]),
                        "median_normal_error_deg": _median(r["normal_errors_deg"])}
                for key in ("distinct_directions", "true_direction_classes", "ate_rmse_cm", "rotation_rmse_deg",
                            "initial_ate_rmse_cm", "sim3_scale"):
                    if key in r:
                        item[key] = r[key]
                for key in ("input_median_residual_px", "line_gate_median_px"):
                    if key in r:
                        item[key] = round(r[key], 4)
                items.append(item)
            per_scene[method] = items
        out["experiments"][exp] = {"results": payload["results"], "_status": payload.get("_status", {}),
                                   "_note": "Detailed paired records are private to eval/test/."}
    return out


def run_case(case: str) -> None:
    case_dir = HERE / "cases" / case
    full = PRIVATE / case / "scores_test.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for folder in ("method", "baseline", "eval", "redteam/hardening2", "redteam/after_hardening"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "upstream" not in path.parts and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    digest.update((ROOT / "scoring/reference.json").read_bytes())
    code_hash = digest.hexdigest()
    old_status = json.loads((case_dir / "status.json").read_text()) if (case_dir / "status.json").is_file() else {}
    if old_status.get("code_hash") == code_hash and full.is_file():
        print(f"[skip] {case}", flush=True)
        return
    spec = CASES[case]
    ws = make_workspace(case, spec)
    case_dir.mkdir(parents=True, exist_ok=True)
    solver = ws / "scripts" / "methods" / "struct_riemanline.sh"
    (case_dir / "solvers.txt").write_text(
        "solver scripts on disk: " + ", ".join(sorted(p.name for p in (ws / "scripts" / "methods").glob("*.sh")))
        + "\n\nstruct_riemanline.sh:\n" + (solver.read_text() if solver.is_file() else "<removed>\n")
    )
    start = time.time()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(["bash", "scripts/evaluate_test.sh"], cwd=str(ws), capture_output=True, text=True, env=env)
    elapsed = time.time() - start
    (case_dir / "evaluate_test.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-5000:])
    scores_path = ws / "scoring" / "scores_test.json"
    if scores_path.is_file():
        scores = json.loads(scores_path.read_text())
        private_path = ws / "eval" / "test" / "results_full.json"
        shutil.copy(private_path if private_path.is_file() else scores_path, full)
        (case_dir / "scores_test.json").write_text(json.dumps(slim(scores), indent=1, sort_keys=True) + "\n")
    (case_dir / "status.json").write_text(json.dumps({"returncode": proc.returncode, "seconds": round(elapsed, 1), "code_hash": code_hash}) + "\n")
    shutil.rmtree(ws, ignore_errors=True)
    print(f"[done] {case}: rc={proc.returncode} in {elapsed:.0f}s", flush=True)


def main() -> None:
    cases = sys.argv[1:] or [c for c in CASES if c not in EXTRA]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
