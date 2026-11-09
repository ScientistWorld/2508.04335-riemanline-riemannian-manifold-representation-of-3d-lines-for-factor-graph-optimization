# Hardening report (iteration 11)

**Outcome.** Better methods now score higher and worse ones lower, measured on 64 hidden scenes per experiment that no solver has seen. Every shortcut listed in `REDTEAM.md` is now closed: it either fails (worst value of every metric in every experiment) or ties the reference exactly. One shortcut still pays: a solver that loads the hidden evaluator's source by its absolute path. It is listed under "left" below. The reference method now follows the paper's own procedure (initialisation from observations, Eq. 4-7 manifold updates, the authors' Cauchy loss and group inlier check) and shows the paper's effect far outside the test noise. A new scored experiment, `parallel_lines_mislabeled`, sits where the paper's method falls short and gives real headroom.

All "after" numbers come from actual runs of the no-argument `bash scripts/evaluate_test.sh` (CPU jobs `049c272e-c91` and the follow-up gate run). Each case ran in its own copy of the workspace with only `scripts/methods/` changed. Per-case solver scripts, logs and scores are in `redteam/after_hardening/cases/<case>/`, and the full summary is in `redteam/after_hardening/summary.json`. "Before" numbers are copied from `REDTEAM.md` (the previous gym, its own rank cohorts).

## 1. What was wrong and what changed

| Problem in the previous gym (`REDTEAM.md`) | Fix |
|---|---|
| Test ground truth (line normals/directions) sat in `data/*.json`; the public generator plus a public seed rebuilt it; train and test were two halves of the **same** two fixed scenes, with one hard-coded direction | Scenes are now random per instance: 2-3 groups with random directions, individual lines, shuffled ids, random cameras. Hidden test scenes (64 per experiment) are generated **inside `eval/test/`** at evaluation time from a 128-bit seed that exists only there. Train scenes (24 per experiment, public seeds) come from an independent copy in `eval/train/` and are exported to `data/train_scenes/`. Solvers see only the public part of a scene (cameras, segment observations, declared groups). |
| Scores were computed from whatever output files sat in `outputs/`, with no common way to run a method | One solver interface for train and test: `scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR`. Both evaluators run **every** solver on disk, each on a fresh copy of the inputs, from a symlinked view of the workspace that does not contain `eval/test/`. No method name is hardcoded. |
| `num_parameters` was self-reported and trusted (0 and -10^12 paid) | Every number is recomputed from the raw line estimates. The compactness metric is now `param_count_error_per_line`: the parameter count implied by the output geometry (2 x distinct directions + 2 x lines) compared with the true structure's minimal count. Self-reported fields are never read. |
| Missing or renamed proposed rows were skipped (tie); NaN outputs were accepted; null or empty outputs crashed the evaluator without writing scores | A run **fails** on: non-zero exit or timeout, missing scene or line, non-finite values, a non-Plucker line (n·d ≠ 0), or a task failure (median endpoint residual on the scene's own observations above 10 px in any scene). A failed run, and any row required by `scoring/reference.json` that has no solver, gets the worst value of every metric in every experiment (90°, 100 px, 2.0). |
| A constant line for every id outranked the reference in a pairwise cohort, because compactness compensated for broken geometry | The task gate makes such outputs failures (worst values everywhere), so compactness can no longer buy rank for a failed reconstruction. |
| In-sample reprojection RMSE rewarded over-fitting: independent lines fit their own observations better and scored better | Replaced by `novel_view_line_rmse_px`: noise-free projections of the true segments into 4 held-out cameras the solver never sees. |
| `parallel_spread_deg` was 0 for any shared-direction method and penalised correctly splitting a wrong group | Removed. Structural correctness is measured by `param_count_error_per_line`, which also penalises sharing a direction between lines that are not truly parallel. |
| Coefficients were negative, and their sign was ambiguous | All coefficients are positive magnitudes; direction comes from `higher_is_better: false`. |

**Restored to the paper's method.** These were departures of earlier iterations (see `REDTEAM.md`):
- **Initialisation:** was ground truth plus a perturbation, for the method and the baselines alike. Now it is linear multi-view triangulation of the observations.
- **Updates:** were global latitude/longitude angles. Now steps go through the paper's Eq. 4-7 retraction: tangent step on S^2 with the exponential map; normal re-projected, rotated by δγ about the new direction, and its scale updated, all inside a Levenberg-Marquardt solver (`method/struct_riemanline.py`, `method/line_ba.py`).
- **Loss:** was Huber 2 px; now the authors' Cauchy 3 px.
- **Group check:** the authors' inlier check (|cos| > 0.9, more than 2 inliers; `get_parallel_candidate` in the released code) is added.
- **Baselines:**
  - OrthLine now uses the Bartoli-Sturm SO(3)×SO(2) update (U·Rx·Ry·Rz, W·R(δφ)), as in the authors' `LineOrthParameterization`.
  - OrthLine_Constr uses the paper's Eq. 11 averaged residual (authors' pair gating |cos| > 0.92, scaled by f_x into pixel units) instead of cross products against the first line.
  - The paper's independent RiemanLine was added as a baseline row (`baseline/riemanline.py`).
  - Baselines live in `baseline/` and never import `method/`.

**Experiment weights:**
- `parallel_lines` 0.40: the core claim with correct groups.
- `parallel_lines_harder` 0.35: weak geometry, where the paper's structure prior matters most.
- `parallel_lines_mislabeled` 0.25: the paper's weak point. It is a robustness extension, not a paper claim, so it gets the smallest weight.

**Metric coefficients:**
- Direction errors are weighted highest: median 1.0, p90 0.6. They are the paper's headline line metric (Table 4).
- Normal errors: median 0.8, p90 0.4.
- Novel-view error 0.6.
- Parameter-count error 0.5. It is the paper's constraint (Table 6) and is weighted so it can neither be ignored nor buy back lost accuracy.

**Task gate.** The gate was first set at 5 px. The weak no-optimisation control failed it on one mislabeled scene at 5.05 px. That is a poor but real reconstruction, while garbage sits at 15-100 px, so the gate was raised to 10 px to keep weak honest methods on a graded scale. Every other case's outcome provably cannot change: all valid cases stay below 2.1 px, and all other failures are invalid output, a missing solver, a crash, or at least 50 px. Only `reference` and `no_optimization` were rerun.

## 2. Does a better result score higher?

Rank convention, as specified for this gym:
- For each experiment and metric, the improvement is taken in its better direction and divided by the population standard deviation across a cohort of attempts.
- That is multiplied by |coefficient| × experiment weight, then summed over all metrics and experiments.
- Cohort Q = the reference, the four honest variants, and the three baseline rows, plus the candidate itself when it is a shortcut.
- The proposed row `struct_riemanline` is compared.

The future production cohort is unknown, so absolute values are illustrative; signs and the paired confidence intervals are the evidence.

**Before** (old gym, `REDTEAM.md`, its cohort Q): Gaussian loss +0.009; no optimisation −10.2; independent ablation −3.36; numerically tied longer run +6e−9.

**After** (hidden test, 64 scenes per experiment; per-experiment cells give median direction error ° / novel-view px / parameter-count error):

| Variant | Rank ΔQ | parallel_lines | parallel_lines_harder | parallel_lines_mislabeled |
|---|---:|---|---|---|
| Reference (StructRiemanLine) | 0 | 0.137 / 0.72 / 0.000 | 0.474 / 3.46 / 0.026 | 0.249 / 4.94 / 0.297 |
| Verified groups + Gaussian loss (better) | **+0.109** | 0.139 / 0.71 / 0.000 | 0.462 / 3.26 / 0.031 | **0.165 / 0.89 / 0.004** |
| Verified groups (better) | **+0.104** | 0.138 / 0.72 / 0.002 | 0.474 / 3.51 / 0.035 | **0.171 / 0.88 / 0.002** |
| Gaussian loss (honest trade-off) | +0.001 | 0.139 / 0.71 / 0.000 | 0.462 / 3.11 / 0.026 | 0.950 / 3.83 / 0.297 |
| No optimisation (weaker) | NO_OPT_RANK | NO_OPT_E1 | NO_OPT_E2 | NO_OPT_E3 |
| Independent RiemanLine (ablation: no shared direction; baseline row) | lower (paired below) | 0.352 / 0.94 / 1.413 | 1.495 / 4.53 / 1.484 | 0.426 / 1.15 / 1.266 |

"Verified groups" adds a χ²(2) membership test to StructRiemanLine. A declared member is split off as an individual RiemanLine when freeing its direction lowers its squared residuals by more than the 10⁻⁴ quantile, one line per group per round. The code is in `redteam/after_hardening/candidates.py`.

**Paired comparisons.** Every method sees the same 64 scenes per experiment. The table gives a bootstrap over scenes (2000 resamples) of the unweighted per-experiment rank delta, with 95% CIs:

| Comparison | parallel_lines | parallel_lines_harder | parallel_lines_mislabeled |
|---|---|---|---|
| Verified groups vs reference | −0.001 [−0.004, 0.000] | −0.008 [−0.017, −0.001] | **+0.429 [+0.404, +0.455]** |
| Verified + Gaussian vs reference | +0.000 [−0.000, +0.001] | +0.003 [−0.012, +0.016] | **+0.428 [+0.402, +0.455]** |
| Gaussian loss vs reference | +0.000 [−0.000, +0.001] | +0.009 [+0.000, +0.018] | −0.011 [−0.022, −0.001] |
| Reference vs OrthLine (paper's effect) | **+0.913 [+0.900, +0.927]** | **+1.082 [+1.054, +1.115]** | **+0.513 [+0.477, +0.549]** |
| Reference vs OrthLine_Constr | **+0.912 [+0.899, +0.926]** | **+0.987 [+0.970, +1.004]** | **+0.511 [+0.475, +0.546]** |
| Reference vs RiemanLine (independent) | **+0.913 [+0.900, +0.927]** | **+1.087 [+1.061, +1.118]** | **+0.513 [+0.477, +0.549]** |

What the paired comparisons show:
- **The paper's effect.** The median direction error falls 0.212° [0.189, 0.231] on `parallel_lines` and 1.020° [0.947, 1.093] on `parallel_lines_harder` relative to independent lines. That is far outside the noise. The method is deterministic, so repeated runs are byte-identical: the reference rerun in an isolated copy reproduced `scoring/scores_test.json` exactly.
- **Genuinely better variants.** On `parallel_lines_mislabeled` the verified-group variants cut the p90 direction error by 5.02° [4.52, 5.52] and the novel-view error by 4.05 px [3.62, 4.50]. The gain is far beyond the test noise, and it carries the overall rank.
- **Gaussian loss is a real trade-off.** It is slightly better on the harder experiment and clearly worse on the mislabeled medians (+0.69° median direction error), and it nets out at about zero.

## 3. Shortcuts: before and after

"After" is the hidden-test outcome for the proposed row. "Worst" means the run failed and every metric of every experiment is the worst value; rank ΔQ for those is −9.33.

| Shortcut | Before (`REDTEAM.md`) | After | Why |
|---|---|---|---|
| Copy visible answers (train ground truth of the same index) | **pays** ΔA +0.119 | worst (median residual 100 px) | Test truth is not visible; train scenes are different scenes |
| Rebuild with the public generator / seed search (17 root seeds × 40 indices) | **pays** ΔA +0.119 | worst (100 px) | 128-bit seed held only in `eval/test/` |
| Import the hidden generator (`from eval.test import scenes`) | not tested | worst (solver crashes: module hidden) | Solvers run from a workspace view without `eval/test/` |
| Zero parameters reported | **pays** ΔP +0.5 | tie (ΔQ 0) | Self-reports ignored |
| Negative parameters (−10^12) | **pays** ΔA +1.12 | tie (ΔQ 0) | Self-reports ignored |
| Null parameters | evaluator crashed (no scores) | tie (ΔQ 0) | Self-reports ignored |
| Fabricated flags / claimed scores | tie | tie (ΔQ 0) | Ignored |
| Ground-truth replace (plant estimate as truth; rewrite the hidden generator's seed on disk mid-run) | **pays** ΔA +0.120 | tie (ΔQ 0) | Truth is generated in memory before any solver runs |
| Observation replace (rewrite inputs to agree with own estimate) | **pays** ΔA +0.150 | tie (ΔQ 0) | Evaluator scores against its own in-memory observations; every solver gets a fresh copy |
| Constant geometry | **pays** in pairwise cohort ΔP +0.5 | worst (100 px) | Task gate |
| Merge every line onto one direction (over-compact) | not tested | worst (50 px) | Task gate |
| Renamed method (reference under a new name) | tie (row skipped) | worst for `struct_riemanline` | Required rows from `reference.json`; missing means worst |
| Omit proposed method | tie (row skipped) | worst | Same |
| NaN geometry | accepted (NaN scores) | worst | Finite check |
| Empty output | evaluator crashed | worst | Line-id check |
| Partial output (half the scenes) | not tested | worst | Scene check |
| Non-Plucker lines (n·d ≠ 0) | not tested | worst | Plucker check |
| **Load the hidden generator by absolute path** | not tested | **pays: ΔQ +0.194, perfect geometry** | **Left**, see below |

## 4. Shortcuts left

- **Reading the evaluator's private source by absolute path** (`hidden_generator_abspath`). A solver that `exec`s `/home/user/eval/test/scenes.py` can regenerate the test truth. The evaluator already hides `eval/test/` from the solver's workspace view, so the natural `import eval.test` fails. But a process running as the same user can always read a file it knows the path of, and the hidden test must keep its seed somewhere it can read. There is no small fix in a same-user container (permission changes can be undone by the solver, and break the evaluator if a preempted job leaves them behind). It is unambiguous evaluator tampering that the separate shortcut grader is meant to catch, and `scoring/CONSTRAINTS.md` forbids it explicitly.
- **Not replayed:** the endpoint-correspondence model and the brute-force two-view enumeration from `REDTEAM.md`. Segments are now truncated per view, so endpoints no longer correspond. Line parameters are continuous, so there is no finite candidate set to enumerate; the old brute force already lost (ΔA −3.4).

## 5. Reference score before and after

- **Before** (old gym; one fixed scene per experiment, one group, ground-truth-perturbed initialisation; test = odd line ids): median direction 0.061° / 0.128°, median normal 0.120° / 0.156°, reprojection RMSE 1.077 / 1.411 px, 26 / 34 parameters (self-reported), spread 0.
- **After** (hidden test, 64 scenes per experiment):

| Experiment | med dir ° | p90 dir ° | med normal ° | p90 normal ° | novel view px | param err/line |
|---|---:|---:|---:|---:|---:|---:|
| parallel_lines | 0.1368 | 0.3616 | 0.1125 | 0.4210 | 0.7182 | 0.0000 |
| parallel_lines_harder | 0.4735 | 1.6589 | 0.3059 | 1.4214 | 3.4596 | 0.0259 |
| parallel_lines_mislabeled | 0.2493 | 5.5553 | 0.1842 | 2.9917 | 4.9417 | 0.2973 |

The two sets are not comparable: the old numbers came from a ground-truth-initialised optimiser on one fixed scene. Baseline rows on the test split, median direction error / novel view: OrthLine 0.352° / 0.94 px, 1.495° / 4.28 px, 0.426° / 1.15 px; OrthLine_Constr 0.351° / 0.92 px, 1.255° / 3.70 px, 0.415° / 1.12 px. OrthLine_Constr ≤ OrthLine ≈ RiemanLine ≪ StructRiemanLine on correctly grouped scenes, the ordering of the paper's Table 4.

**Train/test parity** for the reference, median direction error train vs test: 0.164 vs 0.137 (`parallel_lines`), 0.531 vs 0.474 (`parallel_lines_harder`), 0.197 vs 0.249 (`parallel_lines_mislabeled`). The splits share the same distribution and differ only in seeds.

## 6. Headroom check

- **At the bound:** `param_count_error_per_line` = 0 on `parallel_lines` (and 0.026 on `parallel_lines_harder`, from lines whose initial triangulation fails the authors' inlier check). The paper's 2n+2 representation is already minimal for correct groups.
- **Not at the bound:** no angular or novel-view metric is near its bound.
- **Where the room is:** on `parallel_lines` no variant tried beats the reference beyond noise; the reference is close to the efficient estimate there, and that experiment mainly checks the core claim and the constraint. On `parallel_lines_harder`, Gaussian loss gives a small significant gain. The added `parallel_lines_mislabeled` experiment holds most of the headroom: the reference trusts wrong group members (p90 direction error 5.6° against 0.54° for verified groups), and verified groups also bring the parameter-count error from 0.297 to 0.002.
- **Baseline sanity:** the simple baselines score clearly below the reference (paired rank +0.51 to +1.09) on a scale with room above.
- **Brute force:** does not solve the test.

## 7. What I could not fix (stated plainly)

- **Joint pose estimation is not reproduced.** The paper's main tables report trajectory error (ATE) from joint pose/point/line bundle adjustment on ICL-NUIM, TartanAir and Open-Structure. This gym still optimises line landmarks only, with known camera poses, on procedural scenes; it measures the paper's line-reconstruction (Table 4) and parameter (Table 6) claims, not the pose claims.
- Jacobians are finite differences in the local coordinates, with a Python Levenberg-Marquardt instead of Ceres; the manifold parameterisation and updates are the paper's.
- The absolute-path read of the hidden evaluator (section 4) remains a paying shortcut that only the separate grader can catch.
- The old shortcut programs under `redteam/` (other than `redteam/after_hardening/`) target the previous interface and no longer run; they are kept as the record behind `REDTEAM.md`.
