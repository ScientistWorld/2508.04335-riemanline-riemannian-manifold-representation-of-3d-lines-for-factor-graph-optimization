"""Analyse the second hardening pass (pure Python, no numpy needed).

Rank convention (as specified for this gym): for each experiment and scored
metric, improvement over the reference row is taken in the metric's better
direction, divided by the population standard deviation of that metric over a
cohort of attempts, multiplied by |coefficient| x experiment weight, and
summed.  Cohort Q = the reference, the three baselines and the honest
variants (plus the candidate itself when it is a shortcut).

Paired comparisons: every method sees the same hidden instances; a bootstrap
over instances (same resample for both methods) gives 95% CIs of the
per-experiment (unweighted) normalised delta.

    python redteam/hardening2/analyze.py            # writes redteam/hardening2/summary.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "eval" / "test" / "hardening_cases"
sys.path.insert(0, str(HERE))
from run_cases import CASES, HONEST, SHORTCUTS  # noqa: E402

REF = "struct_riemanline"
BASELINES = ["orthline", "orthline_constr", "riemanline"]
JOINT = "open_structure_joint_ba"
N_BOOT = 1000


def percentile(v: list[float], q: float) -> float:
    v = sorted(v)
    if not v:
        return float("nan")
    pos = (len(v) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def geomean(v: list[float], floor: float, ceil: float) -> float:
    v = [min(max(x, floor), ceil) for x in v]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else ceil


def pooled(exp: str, rows: list[dict]) -> dict[str, float]:
    dirs = [x for r in rows for x in r["direction_errors_deg"]]
    nrms = [x for r in rows for x in r["normal_errors_deg"]]
    if exp.startswith(JOINT):
        n = sum(r["num_lines_scored"] for r in rows)
        return {
            "ate_rmse_cm": geomean([r["ate_rmse_cm"] for r in rows], 0.01, 1000.0),
            "rotation_rmse_deg": geomean([r["rotation_rmse_deg"] for r in rows], 1e-3, 180.0),
            "median_direction_error_deg": percentile(dirs, 50),
            "p90_direction_error_deg": percentile(dirs, 90),
            "median_normal_error_deg": percentile(nrms, 50),
            "param_count_error_per_line": sum(2 * abs(r["distinct_directions"] - r["true_direction_classes"]) for r in rows) / max(1, n),
        }
    sq = sum(r["novel_view_sq_sum"] for r in rows)
    cnt = sum(r["novel_view_count"] for r in rows)
    lines = sum(r["num_lines"] for r in rows)
    return {
        "median_direction_error_deg": percentile(dirs, 50),
        "p90_direction_error_deg": percentile(dirs, 90),
        "median_normal_error_deg": percentile(nrms, 50),
        "p90_normal_error_deg": percentile(nrms, 90),
        "novel_view_line_rmse_px": math.sqrt(sq / cnt) if cnt else 100.0,
        "param_count_error_per_line": sum(2 * abs(r["distinct_directions"] - r["true_direction_classes"]) for r in rows) / lines,
    }


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    ref_cfg = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    base = load(ROOT / "eval" / "test" / "results_full.json")
    rows: dict[str, dict] = {}  # name -> {"results": {exp: metrics}, "per_scene": {exp: rows}, "failed": bool}
    for m in [REF] + BASELINES:
        rows[m] = {"results": {e: base["experiments"][e]["results"][m] for e in ref_cfg},
                   "per_scene": {e: base["experiments"][e]["_per_scene"].get(m) for e in ref_cfg},
                   "failed": bool(base["experiments"][next(iter(ref_cfg))]["_status"][m]["failed"])}
    status = {}
    for case in [c for c in CASES if c in HONEST or c in SHORTCUTS or c == "omit_proposed"]:
        sc = load(WORK / case / "scores_test.json")
        st = load(HERE / "cases" / case / "status.json")
        status[case] = st
        if sc is None:
            continue
        exp0 = next(iter(ref_cfg))
        failed = REF not in sc["experiments"][exp0]["results"] or bool(sc["experiments"][exp0]["_status"].get(REF, {}).get("failed"))
        rows[case] = {"results": {e: sc["experiments"][e]["results"].get(REF) for e in ref_cfg},
                      "per_scene": {e: sc["experiments"][e].get("_per_scene", {}).get(REF) for e in ref_cfg},
                      "failed": failed,
                      "reason": sc["experiments"][exp0]["_status"].get(REF, {}).get("reason", "")}

    # Every ranked attempt, including failed attempts, participates in scale.
    # Unreplayed historical EXTRA cases are not part of this final cohort.
    honest_cohort = [REF] + BASELINES + [c for c in HONEST if c in rows]

    def sds(cohort: list[str]) -> dict:
        out = {}
        for e, cfg in ref_cfg.items():
            for m in cfg["metrics"]:
                vals = [rows[c]["results"][e][m] for c in cohort if rows[c]["results"].get(e)]
                mu = sum(vals) / len(vals)
                out[(e, m)] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        return out

    def delta(res_c: dict, res_r: dict, sd: dict, exp: str | None = None, weighted: bool = True) -> float:
        tot = 0.0
        for e, cfg in ref_cfg.items():
            if exp is not None and e != exp:
                continue
            w = float(cfg.get("weight", 1.0)) if weighted else 1.0
            for m, mc in cfg["metrics"].items():
                s = sd[(e, m)]
                if s <= 0:
                    continue
                sign = 1.0 if mc.get("higher_is_better") else -1.0
                tot += w * abs(float(mc["coefficient"])) * sign * (res_c[e][m] - res_r[e][m]) / s
        return tot

    sd_q = sds(honest_cohort)
    summary: dict = {"cohort_honest": honest_cohort, "candidates": {}}
    before = load(HERE / "before_metrics.json") or {}
    rng = random.Random(0)
    for name in list(rows):
        if name == REF:
            continue
        is_shortcut = name in SHORTCUTS or name == "omit_proposed"
        sd = sds(honest_cohort + [name]) if is_shortcut else sd_q
        entry = {"failed": rows[name]["failed"], "reason": rows[name].get("reason", "")[:200],
                 "status": status.get(name),
                 "rank_delta_vs_reference": delta(rows[name]["results"], rows[REF]["results"], sd),
                 "per_experiment_delta": {e: delta(rows[name]["results"], rows[REF]["results"], sd, exp=e, weighted=False) for e in ref_cfg},
                 "results": rows[name]["results"]}
        if name in before:
            entry["before_rank_delta_same_scale"] = delta(before[name]["results"], before["_reference"]["results"], sd)
        if not rows[name]["failed"] and not is_shortcut:
            cis = {}
            total_boot = [0.0] * N_BOOT
            for e in ref_cfg:
                a, b = rows[name]["per_scene"][e], rows[REF]["per_scene"][e]
                if not a or not b or len(a) != len(b):
                    continue
                assert [v["scene_id"] for v in a] == [v["scene_id"] for v in b], "unpaired instances"
                n = len(a)
                vals = []
                for boot_i in range(N_BOOT):
                    idx = [rng.randrange(n) for _ in range(n)]
                    pa = pooled(e, [a[i] for i in idx])
                    pb = pooled(e, [b[i] for i in idx])
                    value = delta({e: pa}, {e: pb}, sd, exp=e, weighted=False)
                    vals.append(value)
                    total_boot[boot_i] += ref_cfg[e].get("weight", 1.0) * value
                cis[e] = [round(percentile(vals, 2.5), 3), round(percentile(vals, 97.5), 3)]
            blocks = {}
            for e in ref_cfg:
                if not e.startswith(JOINT):
                    continue
                a, b = rows[name]["per_scene"][e], rows[REF]["per_scene"][e]
                groups = {}
                for i, r in enumerate(a):
                    groups.setdefault(r["sequence"], []).append(i)
                keys = sorted(groups)
                vals = []
                for _ in range(N_BOOT):
                    idx = [i for _ in keys for i in groups[rng.choice(keys)]]
                    vals.append(delta({e: pooled(e, [a[i] for i in idx])},
                                      {e: pooled(e, [b[i] for i in idx])}, sd, exp=e, weighted=False))
                blocks[e] = [percentile(vals, 2.5), percentile(vals, 97.5)]
            entry["paired_sequence_block_ci95"] = blocks
            entry["per_experiment_delta_ci95"] = cis
            entry["rank_delta_ci95"] = [percentile(total_boot, 2.5), percentile(total_boot, 97.5)]
        summary["candidates"][name] = entry
    repeat = rows.get("reference_repeat")
    if repeat and not repeat["failed"]:
        summary["repeat_max_abs_metric_difference"] = max(abs(repeat["results"][e][m] - rows[REF]["results"][e][m])
            for e, cfg in ref_cfg.items() for m in cfg["metrics"])
    summary["reference_results"] = rows[REF]["results"]
    summary["before_reference_results"] = before.get("_reference", {}).get("results")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{'candidate':22s} {'rankQ':>8s}  " + "  ".join(f"{e[:22]:>22s}" for e in ref_cfg))
    for name, ent in summary["candidates"].items():
        cells = []
        for e in ref_cfg:
            d = ent["per_experiment_delta"][e]
            ci = ent.get("per_experiment_delta_ci95", {}).get(e)
            cells.append(f"{d:+7.3f}" + (f" [{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else " " * 15))
        print(f"{name:22s} {ent['rank_delta_vs_reference']:+8.3f}  " + "  ".join(cells)
              + ("  FAILED " + ent["reason"][:80] if ent["failed"] else ""))


if __name__ == "__main__":
    main()
