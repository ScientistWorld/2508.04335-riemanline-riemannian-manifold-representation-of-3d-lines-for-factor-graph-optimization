"""Summarise the hardening cases: rank deltas and paired bootstrap statistics.

Rank convention (as specified for this gym): for every experiment e and scored
metric m, improvement = (reference - candidate) for lower-is-better metrics,
divided by the population standard deviation of m across a cohort of
attempts, times |coefficient_m| and the experiment weight; summed.  The
cohort Q is the reference, the four honest variants and the three baseline
rows of the reference run, plus the candidate itself.  The proposed row
(struct_riemanline) is the compared row.

Paired statistics: every method sees the same hidden scenes, so differences
are bootstrapped over scenes (same resampled scene set for both methods).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from eval.test.evaluate import METRICS, WORST, pooled_metrics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "tmp" / "hardening"
PROPOSED = "struct_riemanline"
HONEST = ["gaussian_loss", "verified_groups", "verified_gaussian", "no_optimization"]
BASELINES = ["orthline", "orthline_constr", "riemanline"]
BOOT = 2000


def load(case: str) -> dict | None:
    path = WORK / case / "scores_test.json"
    return json.loads(path.read_text()) if path.is_file() else None


def row(scores: dict, exp: str, method: str = PROPOSED) -> dict:
    return scores["experiments"][exp]["results"].get(method, dict(WORST))


def rank_delta(cand: dict, ref: dict, cohort: list[dict], reference: dict) -> tuple[float, dict]:
    total = 0.0
    parts = {}
    for exp, spec in reference["experiments"].items():
        for m, mdef in spec["metrics"].items():
            vals = np.array([c[exp][m] for c in cohort], float)
            sd = float(vals.std())
            if sd <= 0 or not math.isfinite(sd):
                continue
            better = (cand[exp][m] - ref[exp][m]) if mdef["higher_is_better"] else (ref[exp][m] - cand[exp][m])
            contrib = spec["weight"] * abs(mdef["coefficient"]) * better / sd
            parts[f"{exp}/{m}"] = round(contrib, 6)
            total += contrib
    return total, parts


def per_scene(scores: dict, exp: str, method: str = PROPOSED):
    return scores["experiments"][exp]["_per_scene"].get(method)


def paired_bootstrap(a_rows, b_rows, reference: dict, exp: str, sds: dict, seed: int = 0):
    """Bootstrap pooled-metric differences (b - a) and weighted rank delta of b over a."""
    rng = np.random.default_rng(seed)
    n = len(a_rows)
    diffs = {m: [] for m in METRICS}
    ranks = []
    spec = reference["experiments"][exp]
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        pa = pooled_metrics([a_rows[i] for i in idx])
        pb = pooled_metrics([b_rows[i] for i in idx])
        r = 0.0
        for m in METRICS:
            diffs[m].append(pb[m] - pa[m])
            sd = sds.get(m, 0.0)
            if sd > 0:
                r += abs(spec["metrics"][m]["coefficient"]) * (pa[m] - pb[m]) / sd
        ranks.append(r)
    out = {}
    for m in METRICS:
        arr = np.array(diffs[m])
        out[m] = {"mean": float(arr.mean()), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
    arr = np.array(ranks)
    out["experiment_rank_delta"] = {"mean": float(arr.mean()),
                                    "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
    return out


def main() -> None:
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())
    exps = list(reference["experiments"])
    ref_scores = load("reference")
    if ref_scores is None:
        raise SystemExit("reference case missing")
    ref = {e: row(ref_scores, e) for e in exps}
    honest = {c: load(c) for c in HONEST}
    honest_rows = [{e: row(s, e) for e in exps} for s in honest.values() if s is not None]
    baseline_rows = [{e: row(ref_scores, e, b) for e in exps} for b in BASELINES]
    base_cohort = [ref] + honest_rows + baseline_rows

    summary = {"cohort": ["reference"] + [c for c, s in honest.items() if s is not None] + BASELINES,
               "cases": {}, "paired": {}}
    case_dirs = sorted(p.name for p in (HERE / "cases").iterdir() if p.is_dir())
    for case in case_dirs:
        scores = load(case)
        status = json.loads((HERE / "cases" / case / "status.json").read_text())
        if scores is None:
            summary["cases"][case] = {"returncode": status["returncode"], "scores": None}
            continue
        cand = {e: row(scores, e) for e in exps}
        cohort = base_cohort + ([cand] if case not in ["reference"] + HONEST else [])
        delta, parts = rank_delta(cand, ref, cohort, reference)
        failed = scores["_metadata"].get("failed_methods", {})
        summary["cases"][case] = {
            "returncode": status["returncode"],
            "seconds": status.get("seconds"),
            "proposed_failed": PROPOSED in failed,
            "failure_reason": failed.get(PROPOSED, "")[:300],
            "rank_delta_Q": round(delta, 6),
            "metrics": cand,
        }

    # Paired bootstrap: honest variants vs reference, and reference vs baselines.
    for exp in exps:
        sds = {m: float(np.array([c[exp][m] for c in base_cohort], float).std()) for m in METRICS}
        a_rows = per_scene(ref_scores, exp)
        for case, scores in honest.items():
            if scores is None or per_scene(scores, exp) is None:
                continue
            summary["paired"][f"{case}_vs_reference/{exp}"] = paired_bootstrap(a_rows, per_scene(scores, exp), reference, exp, sds)
        for b in BASELINES:
            b_rows = per_scene(ref_scores, exp, b)
            if b_rows is not None:
                summary["paired"][f"reference_vs_{b}/{exp}"] = paired_bootstrap(b_rows, a_rows, reference, exp, sds)

    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{'case':28s} {'rc':>3s} {'failed':>6s} {'rankQ':>10s}  " + " ".join(f"{e[:22]:>22s}" for e in exps))
    for case, info in summary["cases"].items():
        if info.get("metrics") is None:
            print(f"{case:28s} {info['returncode']:>3d}  no scores")
            continue
        cells = " ".join(
            f"{info['metrics'][e]['median_direction_error_deg']:7.3f}/{info['metrics'][e]['novel_view_line_rmse_px']:6.2f}/{info['metrics'][e]['param_count_error_per_line']:5.2f}"
            for e in exps
        )
        print(f"{case:28s} {info['returncode']:>3d} {str(info['proposed_failed']):>6s} {info['rank_delta_Q']:>10.4f}  {cells}")
    for key, val in summary["paired"].items():
        r = val["experiment_rank_delta"]
        print(f"paired {key:60s} rank {r['mean']:+.3f} CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"dmedDir {val['median_direction_error_deg']['mean']:+.4f} CI {np.round(val['median_direction_error_deg']['ci95'], 4).tolist()}")


if __name__ == "__main__":
    main()
