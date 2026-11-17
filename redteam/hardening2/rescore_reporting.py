"""Complete calibration reporting without running a solver.

The default recomputes metrics from saved raw predictions and requires a job.
The lightweight --from-records mode adds only the zero-coefficient display
using direction counts already computed by the evaluator in job 1c13524f-b40.
It never changes a ranked number. Hidden detailed records stay in eval/test/.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISPLAY = "effective_line_parameters_per_line"


def complete_display_from_records() -> None:
    """Backfill the informational display from evaluator-owned primitives."""
    import copy
    reference = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    report = {"mode": "display_only_from_evaluator_records",
              "source_reference_job": "1c13524f-b40",
              "raw_prediction_rescore_run": False}
    for split in ("train", "test"):
        public_path = ROOT / "scoring" / f"scores_{split}.json"
        source_path = ROOT / "eval/test/results_full.json" if split == "test" else public_path
        scores = json.loads(source_path.read_text())
        previous_results = {e: copy.deepcopy(p["results"]) for e, p in scores["experiments"].items()}
        displays = {}
        for exp, cfg in reference.items():
            if DISPLAY not in cfg["metrics"]:
                continue
            assert cfg["metrics"][DISPLAY]["coefficient"] == 0.0
            payload = scores["experiments"][exp]
            assert set(payload["results"]) == set(cfg["results"])
            displays[exp] = {}
            for method, vals in payload["results"].items():
                assert not payload["_status"][method]["failed"], (split, exp, method)
                rows = payload["_per_scene"][method]
                assert rows
                for row in rows:
                    n = row["num_lines_scored"]
                    d = row["distinct_directions"]
                    assert type(n) is int and type(d) is int and 0 <= d <= n
                    assert n == len(row["direction_errors_deg"]) == len(row["normal_errors_deg"])
                n = sum(row["num_lines_scored"] for row in rows)
                assert n > 0
                value = 2.0 + 2.0 * sum(row["distinct_directions"] for row in rows) / n
                assert 2.0 < value <= 4.0
                vals[DISPLAY] = value
                displays[exp][method] = value
                if split == "test":
                    assert cfg["results"][method][DISPLAY] == value, (exp, method)
        # A zero-coefficient field cannot alter a normalized rank or interval.
        # Check exact preservation of every original ranked value, not a tolerance.
        for exp, payload in scores["experiments"].items():
            for method, vals in payload["results"].items():
                ranked = [m for m, mc in reference[exp]["metrics"].items() if abs(mc["coefficient"]) > 0]
                assert {m: vals[m] for m in ranked} == {m: previous_results[exp][method][m] for m in ranked}
                assert set(vals) == set(reference[exp]["metrics"])
        scores["_metadata"]["joint_ba_worst_values"][DISPLAY] = 4.0
        scores["_metadata"]["reporting_correction"] = (
            "Zero-coefficient dimensionality display derived from evaluator-owned direction/line counts "
            "saved by job 1c13524f-b40; every ranked value unchanged. No additional raw-prediction rescore."
        )
        source_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
        if split == "test":
            for payload in scores["experiments"].values():
                payload.pop("_per_scene", None)
            public_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
        report[split] = {"max_abs_change_to_ranked_values": 0.0, "display_primary": displays}
    (ROOT / "scoring/scores.json").write_bytes((ROOT / "scoring/scores_train.json").read_bytes())
    (ROOT / "redteam/hardening2/reporting_correction.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Completed both splits from evaluator-owned counts; every ranked value is unchanged.")


def rescore(split: str) -> dict:
    ev = importlib.import_module(f"eval.{split}.evaluate")
    real = ev.real_lib
    public_path = ROOT / "scoring" / f"scores_{split}.json"
    private_path = ROOT / "eval/test/results_full.json" if split == "test" else public_path
    scores = json.loads(private_path.read_text())
    fixed_error = ev._fixed_assets_error()
    if fixed_error:
        raise RuntimeError(fixed_error)
    instances = {exp: ev.scene_lib.all_scenes(exp) for exp in ev.scene_lib.EXPERIMENTS}
    instances.update({exp: real.all_instances(exp) for exp in real.EXPERIMENTS})
    methods = sorted(ev._required_methods())
    report = {"max_abs_change_to_ranked_values": 0.0, "display_primary": {}}
    for exp, scenes in instances.items():
        payload = scores["experiments"][exp]
        assert set(payload["results"]) == set(methods), (split, exp)
        joint = exp in real.EXPERIMENTS
        worst = real.WORST if joint else ev.WORST
        report["display_primary"][exp] = {}
        for method in methods:
            # These are evaluator-owned records of the original reference job.
            # Refuse to turn a previously failed process into a calibration run.
            assert not payload["_status"][method]["failed"], (split, exp, method)
            outputs = (ROOT / "eval/test/outputs" / method if split == "test"
                       else ROOT / "outputs/train" / f"outputs_{method}")
            scorer = real.score_instance if joint else ev._score_scene
            rows = [scorer(scene, outputs / f"{scene['public']['scene_id']}.json") for scene in scenes]
            vals = (real.pooled_metrics if joint else ev.pooled_metrics)(rows)
            assert set(vals) == set(worst), (split, exp, method)
            assert all(math.isfinite(v) and 0 <= v <= worst[m] for m, v in vals.items())
            for m, old in payload["results"][method].items():
                if m == DISPLAY:
                    continue
                diff = abs(old - vals[m])
                report["max_abs_change_to_ranked_values"] = max(report["max_abs_change_to_ranked_values"], diff)
                assert math.isclose(old, vals[m], rel_tol=1e-12, abs_tol=1e-12), (split, exp, method, m, old, vals[m])
            payload["results"][method] = vals
            payload["_per_scene"][method] = rows
            if joint:
                report["display_primary"][exp][method] = vals[DISPLAY]
            print(f"[rescore {split}] {exp} {method}: raw predictions verified", flush=True)
    scores["_metadata"]["joint_ba_worst_values"] = real.WORST
    scores["_metadata"]["reporting_correction"] = "Raw-prediction rescore of job 1c13524f-b40; added direct dimensionality display (coefficient zero)."
    private_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
    if split == "test":
        for payload in scores["experiments"].values():
            payload.pop("_per_scene", None)
        public_path.write_text(json.dumps(scores, indent=1, sort_keys=True) + "\n")
    return report


def main() -> None:
    report = {split: rescore(split) for split in ("train", "test")}
    (ROOT / "scoring/scores.json").write_bytes((ROOT / "scoring/scores_train.json").read_bytes())
    reference_path = ROOT / "scoring/reference.json"
    reference = json.loads(reference_path.read_text())
    test = json.loads((ROOT / "scoring/scores_test.json").read_text())
    for exp, cfg in reference["experiments"].items():
        for method, row in cfg["results"].items():
            for m, value in test["experiments"][exp]["results"][method].items():
                assert row[m] == value, (exp, method, m, row[m], value)
    # All previous ranks and intervals must remain identical, not just ordered.
    import subprocess
    summary_path = ROOT / "redteam/hardening2/summary.json"
    previous_summary = json.loads(summary_path.read_text())
    subprocess.run(["python", "redteam/hardening2/analyze.py"], cwd=ROOT, check=True)
    assert json.loads(summary_path.read_text()) == previous_summary, "Rank evidence changed during display-only correction"
    (ROOT / "redteam/hardening2/reporting_correction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-records", action="store_true",
                        help="Only add the unranked display from saved evaluator counts; no heavy computation.")
    args = parser.parse_args()
    if args.from_records:
        complete_display_from_records()
    else:
        main()
