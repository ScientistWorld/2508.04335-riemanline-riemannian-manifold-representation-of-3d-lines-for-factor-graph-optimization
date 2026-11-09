"""Run every hardening case through the real, no-argument hidden test.

Each case gets its own copy of the workspace (code, scripts, scoring, data),
modifies only scripts/methods/ (the solver set), then runs
``bash scripts/evaluate_test.sh`` there.  Full score files go to
tmp/hardening/<case>/ (for analysis); a slim copy is kept in
redteam/after_hardening/cases/<case>/.  Finished cases are skipped on resume.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "tmp" / "hardening"

CANDIDATE = """#!/bin/bash
set -e
WORKSPACE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${{PYTHONPATH:-}}"
python -m redteam.after_hardening.candidates --mode {mode} "$1" "$2"
"""

HONEST = ["gaussian_loss", "verified_groups", "verified_gaussian", "no_optimization"]
SHORTCUTS = [
    "visible_answers", "seed_search", "hidden_generator_import", "hidden_generator_abspath",
    "zero_params", "negative_params", "null_params", "flags", "ground_truth_replace",
    "observation_replace", "constant_geometry", "merge_directions", "nan_geometry",
    "empty_output", "partial_scenes", "non_plucker",
]
CASES: dict[str, dict] = {"reference": {}}
CASES.update({m: {"replace": m} for m in HONEST + SHORTCUTS})
CASES["renamed_method"] = {"remove": ["struct_riemanline"], "add_reference_as": "struct_riemanline_v2"}
CASES["omit_proposed"] = {"remove": ["struct_riemanline"]}


def make_workspace(case: str, spec: dict) -> Path:
    ws = WORK / case / "workspace"
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "upstream", "cases", "*.pyc")
    for name in ("baseline", "data", "eval", "method", "scripts", "scoring"):
        shutil.copytree(ROOT / name, ws / name, ignore=ignore, symlinks=True)
    (ws / "redteam").mkdir()
    shutil.copy(ROOT / "redteam" / "__init__.py", ws / "redteam" / "__init__.py")
    shutil.copytree(HERE, ws / "redteam" / "after_hardening", ignore=ignore)
    methods = ws / "scripts" / "methods"
    reference_script = (methods / "struct_riemanline.sh").read_text()
    if "replace" in spec:
        (methods / "struct_riemanline.sh").write_text(CANDIDATE.format(mode=spec["replace"]))
    for name in spec.get("remove", []):
        (methods / f"{name}.sh").unlink()
    if "add_reference_as" in spec:
        (methods / f"{spec['add_reference_as']}.sh").write_text(reference_script)
    for stale in ("scores.json", "scores_train.json", "scores_test.json"):
        (ws / "scoring" / stale).unlink(missing_ok=True)
    return ws


def slim(scores: dict) -> dict:
    out = {"_metadata": scores.get("_metadata", {}), "experiments": {}}
    for exp, payload in scores.get("experiments", {}).items():
        per_scene = {}
        for method, rows in payload.get("_per_scene", {}).items():
            per_scene[method] = [
                {
                    "scene_id": r["scene_id"],
                    "median_direction_error_deg": sorted(r["direction_errors_deg"])[len(r["direction_errors_deg"]) // 2],
                    "median_normal_error_deg": sorted(r["normal_errors_deg"])[len(r["normal_errors_deg"]) // 2],
                    "distinct_directions": r["distinct_directions"],
                    "true_direction_classes": r["true_direction_classes"],
                    "input_median_residual_px": round(r["input_median_residual_px"], 4),
                }
                for r in rows
            ]
        out["experiments"][exp] = {"results": payload["results"], "_status": payload.get("_status", {}),
                                   "_per_scene_summary": per_scene}
    return out


def run_case(case: str) -> None:
    case_dir = HERE / "cases" / case
    full = WORK / case / "scores_test.json"
    if (case_dir / "status.json").is_file() and full.is_file():
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
    proc = subprocess.run(["bash", "scripts/evaluate_test.sh"], cwd=str(ws), capture_output=True, text=True,
                          env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(ws), "OMP_NUM_THREADS": "1",
                               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    elapsed = time.time() - start
    (case_dir / "evaluate_test.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-5000:])
    scores_path = ws / "scoring" / "scores_test.json"
    if scores_path.is_file():
        scores = json.loads(scores_path.read_text())
        shutil.copy(scores_path, full)
        (case_dir / "scores_test.json").write_text(json.dumps(slim(scores), indent=1, sort_keys=True) + "\n")
    (case_dir / "status.json").write_text(json.dumps({"returncode": proc.returncode, "seconds": round(elapsed, 1)}) + "\n")
    shutil.rmtree(ws, ignore_errors=True)
    print(f"[done] {case}: rc={proc.returncode} in {elapsed:.0f}s", flush=True)


def main() -> None:
    cases = sys.argv[1:] or list(CASES)
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
