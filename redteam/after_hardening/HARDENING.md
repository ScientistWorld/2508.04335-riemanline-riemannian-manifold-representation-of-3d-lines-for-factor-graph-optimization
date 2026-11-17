> Historical first-pass report. The current final pass supersedes its remaining gaps; see `/HARDENING.md`. Public per-instance details were subsequently removed to prevent feedback inversion and experiment-id lookup.

# Hardening report (iteration 11)

**Outcome.** Better methods now score higher and worse ones lower, measured on 64 hidden scenes per experiment that no solver has seen. Every shortcut listed in `REDTEAM.md` is now closed: it either fails (worst value of every metric in every experiment) or ties the reference exactly. One shortcut still pays: a solver that loads the hidden evaluator's source by its absolute path. It is listed under "left" below. The reference method now follows the paper's own procedure (initialisation from observations, Eq. 4-7 manifold updates, the authors' Cauchy loss and group inlier check) and shows the paper's effect far outside the test noise. A new scored experiment, `parallel_lines_mislabeled`, sits where the paper's method falls short and gives real headroom.

All "after" numbers come from actual runs of the no-argument `bash scripts/evaluate_test.sh` in the final configuration (CPU job `5f4a0c62-393`: reference stage plus all 22 cases). Each case ran in its own copy of the workspace with only `scripts/methods/` changed. Per-case solver scripts, logs and scores are in `redteam/after_hardening/cases/<case>/`, and the full summary is in `redteam/after_hardening/summary.json`. "Before" numbers are copied from `REDTEAM.md` (the previous gym, its own rank cohorts).

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
| Scene files and solver calls named their experiment (`"experiment": "parallel_lines_mislabeled"`, ids like `parallel_lines_mislabeled_007`, one call per experiment), so a solver could switch strategy on the label, e.g. ignore the declared groups only where they are known to be wrong | Opaque scene ids (hashes), no experiment field in public files, and **one solver call over all scenes of all experiments, interleaved**; the evaluator regroups by experiment internally. Scores were byte-for-byte unchanged by this fix (the scenes are the same), confirming determinism. |
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

**Task gate.** The gate was first set at 5 px. The weak no-optimisation control failed it on one mislabeled scene at 5.05 px. That is a poor but real reconstruction, while garbage sits at 15-100 px, so the gate was raised to 10 px to keep weak honest methods on a graded scale. All cases were then rerun in the final configuration.

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
| Verified groups + Gaussian loss (better) | **+1.017** | 0.139 / 0.71 / 0.000 | 0.462 / 3.26 / 0.031 | **0.165 / 0.89 / 0.004** |
| Verified groups (better) | **+0.926** | 0.138 / 0.72 / 0.002 | 0.474 / 3.51 / 0.035 | **0.171 / 0.88 / 0.002** |
| Gaussian loss (honest trade-off) | −0.938 | 0.139 / 0.71 / 0.000 | 0.462 / 3.11 / 0.026 | 0.950 / 3.83 / 0.297 |
| No optimisation (weaker control) | **−4.849** | 0.183 / 2.31 / 0.000 | 0.862 / 13.28 / 0.026 | 0.937 / 10.60 / 0.297 |
| Independent RiemanLine (ablation: no shared direction; baseline row) | lower (paired below) | 0.352 / 0.94 / 1.413 | 1.495 / 4.53 / 1.484 | 0.426 / 1.15 / 1.266 |

"Verified groups" adds a χ²(2) membership test to StructRiemanLine. A declared member is split off as an individual RiemanLine when freeing its direction lowers its squared residuals by more than the 10⁻⁴ quantile, one line per group per round. The code is in `redteam/after_hardening/candidates.py`.

**Paired comparisons.** Every method sees the same 64 scenes per experiment. The table gives a bootstrap over scenes (2000 resamples) of the unweighted per-experiment rank delta (cohort-Q standard deviations), with 95% CIs:

| Comparison | parallel_lines | parallel_lines_harder | parallel_lines_mislabeled |
|---|---|---|---|
| Verified groups vs reference | −0.014 [−0.084, +0.012] | −0.047 [−0.115, 0.000] | **+3.777 [+3.446, +4.121]** |
| Verified + Gaussian vs reference | +0.085 [−0.054, +0.238] | +0.071 [−0.106, +0.261] | **+3.796 [+3.447, +4.139]** |
| Gaussian loss vs reference | +0.085 [−0.054, +0.238] | +0.113 [−0.055, +0.286] | −4.044 [−4.656, −3.476] |
| No optimisation vs reference | **−4.402 [−5.179, −3.653]** | **−5.289 [−6.331, −4.330]** | **−5.013 [−5.482, −4.469]** |
| Reference vs OrthLine (paper's effect) | **+7.280 [+6.762, +7.835]** | **+7.643 [+7.041, +8.333]** | −0.853 [−1.245, −0.434] |
| Reference vs OrthLine_Constr | **+7.148 [+6.661, +7.646]** | **+5.225 [+4.923, +5.524]** | −0.937 [−1.323, −0.527] |
| Reference vs RiemanLine (independent) | **+7.280 [+6.762, +7.835]** | **+7.688 [+7.097, +8.362]** | −0.853 [−1.245, −0.434] |

What the paired comparisons show:
- **The paper's effect.** On the correctly grouped experiments, the median direction error falls 0.212° [0.189, 0.231] (`parallel_lines`) and 1.020° [0.947, 1.093] (`parallel_lines_harder`) relative to independent lines. That is far outside the noise. The method is deterministic: the reference rerun in an isolated copy reproduces `scoring/scores_test.json` byte for byte, and so did the rerun after the opaque-id change.
- **Genuinely better variants.** On `parallel_lines_mislabeled` the verified-group variants cut the p90 direction error by 5.02° [4.52, 5.52] and the novel-view error by 4.05 px [3.62, 4.50]. The gain is far beyond the test noise, and it carries the overall rank. On the clean experiments they are statistically tied with the reference.
- **Gaussian loss is a real trade-off.** Its gains on the clean experiments are not significant, and it is clearly worse on the mislabeled medians (+0.69° median direction error).
- **The weaker control loses everywhere.** No optimisation is significantly below the reference in every experiment.
- **Baselines beat the reference on `parallel_lines_mislabeled`.** Its trust in wrong group members gives it much heavier error tails, although its median errors and parameter-count error are better. That gap is the headroom of this experiment. It cannot be copied by switching strategy per experiment, because scenes no longer reveal their experiment; ignoring the groups everywhere costs ≈7 rank units on each clean experiment. Weighted over all experiments the reference stays far above every baseline.

## 3. Shortcuts: before and after

"After" is the hidden-test outcome for the proposed row. "Worst" means the run failed and every metric of every experiment is the worst value; rank ΔQ for those is −12.07.

| Shortcut | Before (`REDTEAM.md`) | After | Why |
|---|---|---|---|
| Copy visible answers (train ground truth of a train scene) | **pays** ΔA +0.119 | worst (median residual 100 px) | Test truth is not visible; train scenes are different scenes |
| Rebuild with the public generator / seed search (17 root seeds × 3 experiments × 40 indices) | **pays** ΔA +0.119 | worst (100 px) | 128-bit seed held only in `eval/test/` |
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
| **Load the hidden generator by absolute path** | not tested | **pays: ΔQ +4.29, perfect geometry** | **Left**, see below |

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
- **Baseline sanity:** the simple baselines score clearly below the reference on the correctly grouped experiments (paired rank +5.2 to +7.7), on a scale with room above. On the mislabeled experiment they are above it (−0.85), which is the room the paper's method leaves there.
- **Brute force:** does not solve the test.

## 7. What I could not fix (stated plainly)

- **Joint pose estimation is not reproduced.** The paper's main tables report trajectory error (ATE) from joint pose/point/line bundle adjustment on ICL-NUIM, TartanAir and Open-Structure. This gym still optimises line landmarks only, with known camera poses, on procedural scenes; it measures the paper's line-reconstruction (Table 4) and parameter (Table 6) claims, not the pose claims.
- Jacobians are finite differences in the local coordinates, with a Python Levenberg-Marquardt instead of Ceres; the manifold parameterisation and updates are the paper's.
- The absolute-path read of the hidden evaluator (section 4) remains a paying shortcut that only the separate grader can catch.
- The old shortcut programs under `redteam/` (other than `redteam/after_hardening/`) target the previous interface and no longer run; they are kept as the record behind `REDTEAM.md`.
