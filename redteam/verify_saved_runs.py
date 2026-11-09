"""Lightweight evidence checks; does not run any method or evaluator."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def main():
    statuses = {p.parent.name: read_json(p) for p in (ROOT / "redteam").glob("*/status.json")}
    assert len(statuses) == 22
    expected_failures = {"null_params_shortcut", "empty_output_shortcut"}
    assert {n for n, s in statuses.items() if s["test_exit"] != 0} == expected_failures
    hashes = statuses["reference_seed0"]["evaluator_sha256"]
    for name, status in statuses.items():
        folder = ROOT / "redteam" / name
        assert status["method_exit"] == 0
        assert status["train_exit"] == status["test_exit"]
        assert status["test_fresh_process"] and not status["test_called_with_arguments"]
        assert status["evaluator_sha256"] == hashes
        assert (folder / "scores_test.json").exists() == status["test_score_exists"]
        assert status["test_score_exists"] == (name not in expected_failures)
        assert (folder / "run.sh").exists() and (folder / "candidate.py").exists()
    for rel, digest in hashes.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel

    frozen = read_json(ROOT / "redteam/original/scoring/scores_test.json")
    summary = read_json(ROOT / "redteam/summary.json")
    for seed in range(3):
        scores = read_json(ROOT / f"redteam/reference_seed{seed}/scores_test.json")
        for exp in frozen["experiments"]:
            assert scores["experiments"][exp]["results"] == frozen["experiments"][exp]["results"]
    assert summary["seed_rank_spread"]["range"] == 0
    assert all(v["range"] == 0 for exp in summary["seed_spread"].values() for v in exp.values())
    cases = summary["cases"]
    for cohort in ("honest_scale", "audit_cohort", "pairwise"):
        assert cases["gaussian_loss"][cohort]["rank_delta"] > 0
        assert cases["no_training"][cohort]["rank_delta"] < 0
        assert cases["independent_ablation"][cohort]["rank_delta"] < 0
        for name in ("answers_copy_shortcut", "answers_rebuild_shortcut", "zero_params_shortcut",
                     "negative_params_shortcut", "ground_truth_replace_shortcut", "observation_replace_shortcut"):
            assert cases[name][cohort]["rank_delta"] > 0
        for name in ("omit_proposed_shortcut", "renamed_method_shortcut"):
            assert cases[name][cohort]["rank_delta"] == 0
            assert len(cases[name][cohort]["skipped_metrics"]) == 14
    assert cases["constant_geometry_shortcut"]["pairwise"]["rank_delta"] > 0
    assert cases["constant_geometry_shortcut"]["audit_cohort"]["rank_delta"] < 0

    # Inspect tracked scientific files against the actual iteration start, not
    # against the job-submission commit that temporarily changed run.sh.
    base = (ROOT / "redteam/original/commit.txt").read_text().strip()
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", base, "data", "method", "baseline",
         "scoring", "eval", "scripts", "environment"], cwd=ROOT, text=True).splitlines()
    for rel in tracked:
        expected = subprocess.check_output(["git", "show", f"{base}:{rel}"], cwd=ROOT)
        assert (ROOT / rel).read_bytes() == expected, rel

    python_files = list((ROOT / "redteam").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    shell_files = list((ROOT / "redteam").rglob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", (ROOT / "REDTEAM.md").read_text()):
        if not target.startswith(("http:", "https:")):
            assert (ROOT / target).exists(), (label, target)

    report = {"saved_runs": len(statuses), "successful_tests": 20,
              "expected_evaluation_failures": sorted(expected_failures),
              "evaluator_hashes_identical_and_current": True,
              "three_seed_results_equal_frozen_reference": True,
              "rank_direction_and_shortcut_claims_checked": True,
              "tracked_scientific_files_restored_to": base,
              "python_sources_parsed": len(python_files), "shell_sources_parsed": len(shell_files),
              "report_links_exist": True}
    (ROOT / "redteam/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
