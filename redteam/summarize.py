"""Apply the user-specified rank formula, with explicit audit-cohort scaling.

The production cohort/standard deviations are not present in this workspace.
Report both pairwise (reference + candidate) standardization and a fixed audit
cohort. Never pretend a cohort-dependent score is a universal final rank.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def proposed(scores, exp):
    return scores.get("experiments", {}).get(exp, {}).get("results", {}).get("struct_riemanline", {})


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def improvement(scores, reference, definition, cohort):
    total = 0.0
    skipped = []
    contributions = {}
    for exp, spec in definition["experiments"].items():
        parts = {}
        for metric, rule in spec["metrics"].items():
            a = proposed(scores, exp).get(metric)
            b = proposed(reference, exp).get(metric)
            if not (finite(a) and finite(b)):
                skipped.append(f"{exp}/{metric}")
                continue
            values = [proposed(x, exp).get(metric) for x in cohort]
            values = [float(x) for x in values if finite(x)]
            spread = statistics.pstdev(values) if values else 0.0
            if spread == 0.0:
                parts[metric] = 0.0
                continue
            direction = 1.0 if rule["higher_is_better"] else -1.0
            parts[metric] = spec["weight"] * abs(rule["coefficient"]) * direction * (a - b) / spread
            total += parts[metric]
        contributions[exp] = parts
    return {"rank_delta": total, "skipped_metrics": skipped, "contributions": contributions}


def main():
    definition = load(ROOT / "redteam/original/scoring/reference.json")
    frozen = load(ROOT / "redteam/original/scoring/scores_test.json")
    cases = {}
    for status_path in sorted((ROOT / "redteam").glob("*/status.json")):
        name = status_path.parent.name
        status = load(status_path)
        score_path = status_path.parent / "scores_test.json"
        if status["test_exit"] == 0 and score_path.exists():
            cases[name] = load(score_path)
    # Quality cohort excludes intentionally extreme shortcut outputs so that a
    # gigantic forged number does not decide the scaling for honest comparisons.
    honest_names = ["reference_seed0", "reference_seed1", "reference_seed2",
                    "gaussian_loss", "endpoint_model", "more_training_honest_alternative",
                    "no_training", "independent_ablation"]
    honest_cohort = [frozen] + [cases[n] for n in honest_names if n in cases]
    all_cohort = [frozen] + list(cases.values())
    summary = {"normalization": "population std per experiment/metric; proposed row; abs coefficients",
               "reference": "frozen original scoring/scores_test.json, not partial reference.json result values",
               "production_cohort_available": False,
               "honest_scale_definition": "honest cohort plus the candidate if not already present; candidate always contributes to its normalization",
               "nonfinite_policy": "NaN is skipped only for diagnostic calculations; production handling is unknown, so NaN success is conditional",
               "honest_cohort": ["frozen_reference"] + [n for n in honest_names if n in cases],
               "all_cohort": ["frozen_reference"] + list(cases), "cases": {}}
    for name, scores in cases.items():
        quality_cohort = honest_cohort if name in honest_names else honest_cohort + [scores]
        entry = {"pairwise": improvement(scores, frozen, definition, [frozen, scores]),
                 "honest_scale": improvement(scores, frozen, definition, quality_cohort),
                 "audit_cohort": improvement(scores, frozen, definition, all_cohort),
                 "proposed": {e: proposed(scores, e) for e in definition["experiments"]}}
        (ROOT / "redteam" / name / "rank.json").write_text(json.dumps(entry, indent=2, allow_nan=True) + "\n")
        summary["cases"][name] = entry
    summary["seed_spread"] = {}
    seed_ranks = [summary["cases"][f"reference_seed{i}"]["honest_scale"]["rank_delta"] for i in range(3)]
    summary["seed_rank_spread"] = {"seeds": [0, 1, 2], "values": seed_ranks,
                                     "range": max(seed_ranks) - min(seed_ranks)}
    for exp, spec in definition["experiments"].items():
        summary["seed_spread"][exp] = {}
        for metric in spec["metrics"]:
            values = [proposed(cases[f"reference_seed{i}"], exp)[metric] for i in range(3)]
            summary["seed_spread"][exp][metric] = {"min": min(values), "max": max(values),
                                                        "range": max(values) - min(values)}
    for name in ("null_params_shortcut", "empty_output_shortcut"):
        path = ROOT / "redteam" / name / "status.json"
        if path.exists():
            summary["cases"][name] = {"status": load(path), "rank_delta": None,
                                      "reason": "test fails; no per-metric worst values emitted"}
    (ROOT / "redteam/summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    for name, info in summary["cases"].items():
        if "pairwise" in info:
            print(name, "pairwise", round(info["pairwise"]["rank_delta"], 7),
                  "honest-scale", round(info["honest_scale"]["rank_delta"], 7),
                  "audit-cohort", round(info["audit_cohort"]["rank_delta"], 7))
        else:
            print(name, info["reason"])


if __name__ == "__main__":
    main()
