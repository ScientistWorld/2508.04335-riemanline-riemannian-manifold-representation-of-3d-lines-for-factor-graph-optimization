> **Superseded (iteration 11).** This file describes the gym before hardening. The current evaluation, method and shortcut results are in `HARDENING.md`, `briefing/` and `scoring/`.

# Does this gym reward better research?

**The honest quality controls pass, but the gym admits major paying shortcuts.**
A real change from Huber loss to Gaussian least squares improves the score;
removing optimization or the shared-direction idea lowers it. Copying visible
answers, inventing parameter counts, or changing the evaluation data scores
higher without improving a method. Missing proposed-method results tie the
reference. A grossly wrong constant reconstruction can outrank it in a permitted
normalization cohort.

The following are actual no-argument hidden-test runs from CPU job
`168c2ba7-2c9`, exit 0. All 22 candidate methods ran successfully; two intentionally
invalid candidates caused evaluation failures. Each folder under `redteam/`
contains its `run.sh`, candidate entrypoint, output artifacts, status, and scores
when emitted. The common implementations are in [common.py](redteam/common.py).

| Honest variant | Direction error, easy / harder (degrees) | Normal error, easy / harder (degrees) | Reprojection RMSE, easy / harder (px) | Parameters, easy / harder | Rank ΔQ | Rank ΔA |
|---|---:|---:|---:|---:|---:|---:|
| Reference | .061116 / .127620 | .119675 / .155769 | 1.077444 / 1.410620 | 26 / 34 | 0 | 0 |
| [Gaussian loss](redteam/gaussian_loss/scores_test.json), genuinely better weighted result | .059803 / .114567 | .119445 / .122576 | 1.075351 / 1.405896 | 26 / 34 | **+.008960** | **+.009016** |
| [No optimization](redteam/no_training/scores_test.json), weaker control | 5.859389 / 5.859389 | 8.433542 / 9.093397 | 19.353229 / 21.447659 | 26 / 34 | **−10.203431** | **−8.097166** |
| [Independent-line ablation](redteam/independent_ablation/scores_test.json), shared direction removed | .257397 / .350314 | .249340 / .335048 | 1.016562 / 1.326347 | 48 / 64 | **−3.359752** | **−2.279309** |
| [More training honest alternative](redteam/more_training_honest_alternative/scores_test.json), numerical tie | .061116 / .127620 | .119675 / .155769 | 1.077444 / 1.410620 | 26 / 34 | +6.14e−9 | +4.02e−9 |
| [Endpoint correspondence model](redteam/endpoint_model/scores_test.json), honest trade-off | .075246 / .080690 | .103700 / .083137 | 1.079267 / 1.422092 | 50 / 66 | −.604638 | +.018836 |

The Gaussian variant uses the same shared-direction representation and the
appropriate likelihood for the generator's Gaussian noise. Its p90 reprojection
error worsens slightly, from 1.682327/2.290147 to 1.727877/2.304812 px; it is a
weighted improvement, not an improvement in every metric. The endpoint model
uses only observed endpoint correspondences and accounts for all additional
segment variables. It improves several geometry measures but costs more
parameters; its rank reversal between cohorts illustrates the normalization
dependency, rather than establishing a uniformly better method. Increasing the
reference budget from 400 to 4,000 evaluations with tighter tolerances produces
only numerical differences; this is an honest alternative the test cannot
meaningfully distinguish.

**Rank convention.** The reference is the frozen original
`scoring/scores_test.json`, with rank delta 0. The partial result values in
`reference.json` do not contain the complete reference metric vectors. For each
experiment and metric, the calculation uses its experiment weight, the
**absolute** coefficient, the direction from `higher_is_better`, and improvement
divided by the population standard deviation of that metric across the declared
cohort. All seven metrics have lower-is-better directions; negative coefficient
signs do not reverse them. Q contains the frozen reference and eight honest
runs: three reference seeds and the five honest variants above; when evaluating
a shortcut, Q also includes that shortcut. A contains the frozen reference and
all 20 successful test runs. P below contains just the reference and that
candidate. Calculations compare the proposed-method row. Baseline rows provide
comparison results rather than additional candidate metrics. Missing proposed
rows contribute zero
under the specified missing-value rule. NaN skipping is diagnostic only.

Production standard deviations depend on future attempts and are unavailable;
there is no unique production scalar score to report now. Q, A, and P are
explicit applications of the supplied formula, not raw-unit sums. See
[summary.json](redteam/summary.json), including all metric vectors, cohort
membership, contributions, and skipped entries. In particular, −10¹² parameters
does **not** imply an arbitrarily large rank: including that candidate in the
standard deviation gives ΔP +.5 and ΔA +1.119586. It also suppresses the relative
importance of legitimate parameter-count differences in A.

**Seed spread and reproduction check.** The unchanged reference was rerun with
NumPy seeds [0](redteam/reference_seed0/scores_test.json),
[1](redteam/reference_seed1/scores_test.json), and
[2](redteam/reference_seed2/scores_test.json). All fourteen test metric values
are identical across seeds and exactly match the frozen reference. Every
per-metric max–min spread and the normalized rank spread is **0**. The method
has no seed option or random initialization, so these repeats establish
determinism, not uncertainty across new scenes/noise realizations. The Gaussian
gain and weaker-control losses exceed this observed spread. The tiny longer-run
difference is numerical precision, not evidence of a research gain.

The reproduced method beats both reproduced baselines in the paper's reported
angular-accuracy/compactness direction. This check does not establish the
paper's pose-estimation or runtime claims:

| Method | Direction error, easy / harder (degrees) | Normal error, easy / harder (degrees) | Reprojection RMSE, easy / harder (px) | Parallel spread, easy / harder (degrees) | Parameters |
|---|---:|---:|---:|---:|---:|
| OrthLine | .257397 / .350314 | .249340 / .335048 | 1.016562 / 1.326347 | .330637 / .414116 | 48 / 64 |
| OrthLine with constraints | .251261 / .346349 | .246901 / .333338 | 1.016429 / 1.326184 | .325137 / .409031 | 48 / 64 |
| StructRiemanLine | .061116 / .127620 | .119675 / .155769 | 1.077444 / 1.410620 | **0 / 0** | **26 / 34** |

Its reprojection errors are worse than the baselines, consistent with a stronger
structural restriction on fitting noisy observations. The simple exhaustive
baseline below scores clearly below the reference, and the Gaussian variant
demonstrates room above it.

The scored metric already at its absolute numerical bound is
`parallel_spread_deg = 0` in **both** experiments. No angular-error or
reprojection metric is at its zero lower bound. The 26/34 parameter counts also
equal the paper's `2n+2` minimum for an unrestricted group of n parallel lines;
reducing that count while retaining the same general representation is not a
free research opportunity. The evaluator does not enforce that dimensionality
or even a nonnegative count.

Departures from the [paper](paper/2508.04335.md), sections 3–5:

- Two fixed procedural graphs with 12/16 lines and 8/10 known cameras replace
  ICL-NUIM, TartanAir, and the Open-Structure corridor/box/sphere sequences.
  There is one parallel group, the same hardcoded world direction in both
  experiments, and no changing scene distribution.
- Only line landmarks are optimized. Camera poses are exact and fixed; point
  landmarks, joint pose/point/line optimization, trajectory errors, multiple
  groups, and the paper's runtime/convergence comparisons are absent.
- The implementation preserves the shared S² direction and scaled orthogonal
  normal representation, but optimizes global latitude/longitude, normal angle,
  and log scale with SciPy finite-difference trust-region least squares. It
  does not implement the paper's explicit tangent perturbations and exponential
  sphere/circle updates in equations 4–7 or its native factor-graph solver.
- Initial direction and normals are perturbations of **ground truth**, for both
  the reference and baselines, rather than reconstruction from observations.
- The named OrthLine baseline uses independent spherical direction/normal
  coordinates, rather than the paper's SO(3)×SO(2) orthonormal updates. Its
  constrained counterpart uses weighted cross products relative to the first
  line, rather than equation 11's averaged pairwise angular-deviation residual.
- Evaluation uses deterministic even/odd line slices, p90 reprojection/tail
  metrics, and self-reported line-only parameter counts. These are gym-specific
  measures rather than the paper's full sequence protocol. Under a shared
  direction, train/test primary direction errors are identical by construction.

**Shortcut trials.** “Works” means it outranks or ties the reference in the
stated cohort. Each name links to an executable `run.sh`; its folder retains
the actual scores. Ties are explicitly labeled shortcuts, as requested.

| Shortcut | What it actually does | Test score versus reference | Result |
|---|---|---:|---|
| [Answers-copy shortcut](redteam/answers_copy_shortcut/run.sh) | Returns `data/*.json` ground-truth normals/directions. No estimator. | ΔA **+.118707**, ΔP +2.8; all angular errors 0 | **Works.** Reprojection worsens slightly but angular rewards pay more. |
| [Answers-rebuild shortcut](redteam/answers_rebuild_shortcut/run.sh) | Replays the public generator with public seed, dimensions, and noise metadata; does not read stored normal/direction values. | Identical to answers-copy shortcut | **Works.** Hiding the line labels alone would not fix this. |
| [Zero-parameters shortcut](redteam/zero_params_shortcut/run.sh) | Reuses its own regenerated reference estimates, reports `num_parameters: 0`. | ΔA **+3.31e−11**, ΔP **+.5**; geometry unchanged | **Works.** Strict improvement for any finite positive parameter normalization. Tiny ΔA comes from the negative-parameters outlier. |
| [Negative-parameters shortcut](redteam/negative_params_shortcut/run.sh) | Reports −10¹² parameters for those same estimates. | ΔA **+1.119586**, ΔP **+.5** | **Works.** Impossible values are accepted. |
| [Ground-truth-replace shortcut](redteam/ground_truth_replace_shortcut/run.sh) | Replaces visible dataset label normals/directions with its regenerated reference predictions. | ΔA **+.120413**, ΔP +4.6; angular errors 0, reprojection unchanged | **Works.** Changes the fixed artifact against which reconstruction is judged. |
| [Observation-replace shortcut](redteam/observation_replace_shortcut/run.sh) | Returns visible truth and replaces noisy input endpoints with noiseless projections from the same visible scene. | ΔA **+.150201**, ΔP +6.4; angular errors 0, RMSE <7e−14 px | **Works.** Moves the operating point/evaluation data instead of improving a method. |
| [Constant-geometry shortcut](redteam/constant_geometry_shortcut/run.sh) | Returns one constant line for every ID, using the generator's fixed direction; conservatively reports four parameters. | ΔP **+.5**; ΔA −6.839089 | **Works in P; fails in A.** Normal errors 39–40° and RMSE 145–173 px do not guarantee a loss. |
| [Renamed-method shortcut](redteam/renamed_method_shortcut/run.sh) | Deletes reference-named outputs and writes those wrong constant lines under an unmatched method name. | ΔA **0**, ΔP **0**; successful test, proposed rows absent | **Works: tie shortcut.** Every expected proposed metric is skipped. |
| [Omit-proposed shortcut](redteam/omit_proposed_shortcut/run.sh) | Removes proposed outputs, retains only regenerated baselines, and runs train evaluation normally. | ΔA **0**, ΔP **0**; successful test, proposed rows absent | **Works: tie shortcut.** Train-derived required names do not enforce the benchmark's proposed row. |
| [Flags shortcut](redteam/flags_shortcut/run.sh) | Writes false success, zero iterations, impossible negative cost/time/error fields alongside valid cached geometry. | ΔA **0**, ΔP **0**; exact reference metrics | **Tie shortcut.** Claimed score/timing fields are ignored; fabrication does not change recomputed geometry metrics. |
| [NaN-geometry shortcut](redteam/nan_geometry_shortcut/run.sh) | Returns NaN normals/directions and zero parameters. | Test exits 0 and writes twelve NaN geometry values; diagnostic ΔP +.5 if nonfinite values are skipped | **Conditional, not a confirmed paying rank.** Production NaN parsing/handling is unspecified; evaluator acceptance is confirmed. |
| [Null-parameters shortcut](redteam/null_params_shortcut/run.sh) | Reports `num_parameters: null`. | Test exits **1**; no score file emitted | **Fails to earn a demonstrated rank.** No worst-value metric vector is written. |
| [Empty-output shortcut](redteam/empty_output_shortcut/run.sh) | Returns an empty line list. | Test exits **1**; no score file emitted | **Fails to earn a demonstrated rank.** Missing line IDs raise an exception. |
| [Brute-force shortcut](redteam/brute_force_shortcut/run.sh) | Enumerates every available two-view triangulation pair per line and selects by all-view reprojection error: **240 / 480** candidates. | ΔA **−3.411465**, ΔP −8.5; direction .321327/.324012°, RMSE 1.147332/1.466702 px | **Fails.** This exhaustive search does not solve the test at reference quality. |

The constant-geometry shortcut exposes an actual compensation failure, not
an effect of pixel versus degree units. In P, each changed metric has normalized
improvement ±2. Perfect median/p90 directions plus compactness contribute
`2 × (1 + .6 + .25)`; wrong normals and reprojection contribute
`−2 × (.7 + .6 + .3)`. Spread ties at zero, and experiment weights sum to 1,
leaving **+.5** for a failed reconstruction. The current “constraint” metrics
are soft additive rewards, with no task-validity gate. Its negative A score
does not establish safety for other future cohorts.

Missing-value handling also fails the requested completeness check: removing
the expected method yields successful scores with no proposed metric values,
instead of worst values in every experiment. Null/empty artifacts fail the
process without emitting any worst-value results. No stale test score was
present for these failure trials, and no claim is made that an external ranker
would ignore their nonzero exit status. NaNs likewise remain NaNs rather than
becoming penalties.

There is no fixed learned detector/model in this gym to retrain separately.
Its fixed scientific artifacts are the scene JSON files and baseline outputs;
the ground-truth-replace and observation-replace shortcuts exercise replacement
of the former. Parameter counts are the only self-written metric field the
hidden evaluator actually trusts. Fabricated timings and success fields do
not enter the score, and solver runtime is not scored.

**Recommended next iteration:** keep evaluation scene truth and its generator
randomness outside agent-controlled inputs; evaluate unseen directions and
scene realizations; restore immutable observations/labels before testing;
validate finite physically valid line outputs and defensible parameter counts;
require every expected result and turn invalid/missing runs into complete
worst-value vectors; and make reconstruction quality a hard eligibility
condition before compactness can improve rank. A separate shortcut grader does
not make the present score itself a reliable measure of research progress.

All evaluator source hashes are identical across the 22 trials. Candidate
programs finish before a new `bash scripts/evaluate_test.sh` process runs with
**no arguments**, with no background processes, evaluator patches, import
changes, or Python configuration changes. Only public inputs, public method
source where applicable, and the candidate's regenerated cached artifacts are
used by candidate code. [Audit bookkeeping](redteam/AUDIT_PLAN.md) and
[job output](redteam/job_stdout.txt) document the bundle. Transient changes to
data, method, scores, and submission entrypoint are restored to the start commit;
the required `action.yaml` completion signal is the protocol exception.
[Final verification](redteam/verification.json) checks the saved runs, score
claims, evaluator hashes, and restoration of all tracked scientific files.
