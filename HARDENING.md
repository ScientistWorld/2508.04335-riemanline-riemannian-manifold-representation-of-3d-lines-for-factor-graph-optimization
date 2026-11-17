# Hardening report

The final hidden test separates the tested scientific improvement from the reference and weaker controls, and rejects the previously paying hidden-answer reader. CPU job `1c13524f-b40` completed the final reference/baseline runs on both splits, better/weaker variants, affected shortcuts, paired analysis and validation. It resumed after a worker drain, retaining completed runs by source hash. No optimizer uses a wall-clock stopping condition. The remaining dataset train/test difficulty gap is stated below; this requirement is not fully met.

`REVIEW.md` requested evaluation against the paper datasets' ground truth and demonstrated headroom there. Both are now covered by joint pose/point/line experiments on ICL-NUIM and TartanAir, with an improvement whose sequence-block confidence interval excludes zero. Historical results remain in `REDTEAM.md` and `redteam/after_hardening/HARDENING.md`; they describe earlier setups and normalization cohorts.

## What changed

- Restored the paper's joint factor-graph procedure: poses and points are refined alongside shared S² directions and scaled S¹ line moments, using Eq. 4–7 manifold updates, point/line reprojection factors, Cauchy loss, the authors' group inlier check and at most 100 LM iterations. Point Schur elimination and local numerical line/pose Jacobians make the actual algorithm practical in Python. OrthLine, constrained OrthLine (Eq. 11) and independent RiemanLine baselines have separate implementations. This is not a learned surrogate.
- Added `open_structure_joint_ba` with dataset structural associations and `open_structure_joint_ba_clustered` with associations estimated from initial directions. Each has six disjoint sequences per split, four keyframe phases per sequence and at most 150 keyframes per instance. Each split contains four ICL-NUIM and two TartanAir sequences. Scoring uses dataset trajectories and available line-landmark truth. TartanAir's originally exact projections receive 1 px noise; ICL-NUIM retains its original measurements. Sim(3) alignment accounts for unobservable scale in the 2D-only graph.
- Closed absolute-path answer reads with unprivileged Landlock filesystem confinement installed before executing the solver. Public code and data remain readable; fresh inputs, outputs and temporary directories are writable. `eval/test/`, its source, seeds, downloaded answers and saved outputs are inaccessible by relative or absolute path. The evaluator retains access. Unsupported kernels fail closed (Landlock ABI at least 3 is required). Arbitrary estimation procedures remain expressible.
- Retained evaluator-owned observations and truth in memory, with a fresh public input copy per solver. Baseline source and launchers are pinned by evaluator manifests. Missing required rows/instances/ids, invalid rotations, non-Plucker lines, NaN and normalization overflow fail the entire run. JSON size is checked before parsing (16 MiB), and impossible aggregates are rejected. Every failure receives every metric's worst value in every experiment.
- Removed the three-hour solver cutoff. Iteration/convergence limits determine work; elapsed seconds are diagnostic only. References and controls run sequentially with one BLAS thread, with no other experiment launched alongside them. NumPy 2.4.6 and SciPy 1.17.1 are pinned to the verification container.
- Corrected a graph defect found in the first combined job, `75aa9655-ca8`: after keyframe/landmark pruning, some office1 poses had only 0–3 point tracks and escaped to distances up to 174 km. Both independent split implementations now iteratively remove cameras with fewer than six retained point tracks, retain the largest connected point co-visibility component, and repeat landmark pruning to stability. This uses public graph incidence only, never ground truth or error thresholds. All reference and baseline runs were repeated after this data change; their optimizer was unchanged.
- Removed public per-line test error vectors and per-scene summaries, which could constrain true lines or reveal experiment assignments. Detailed records, raw predictions and complete shortcut test outputs now live only under gitignored `eval/test/`. Public reports contain aggregates. Private saved predictions permit evaluator-only rescoring without another solve.
- Fixed paired analysis for both joint experiments, asserted matching instance ids, included failed attempts in normalization, and added sequence-block intervals so four phases are not treated as four independent sequences.

Every no-argument evaluation script discovers every solver on disk and writes its numbers under each experiment's `results` in the designated score file. Both splits use the same method interface. Canonical files contain only the reference and three baselines; audit candidates never replace those rows. `scripts/run.sh` again reproduces the reference suite end to end without running audit candidates.

## Ranking and weights

Weights are 0.20 / 0.15 / 0.15 for the three procedural experiments and 0.30 / 0.20 for dataset / clustered dataset evaluation. Half the weight now measures the paper's datasets. Correct structure receives more weight than the association-robustness extension. Procedural coefficients remain direction median 1, direction p90 0.6, moment median 0.8, moment p90 0.4, unseen-view error 0.6 and compactness 0.5. Joint coefficients are ATE 1, rotation 0.5, direction median 1 and p90 0.5, moment median 0.6 and compactness 0.5.

Rank adds experiment weight × absolute coefficient × improvement in the `higher_is_better` direction / population standard deviation across ranked attempts. Zero variance contributes zero. Only experiment `results` numbers are ranked. Failures participate with complete worst vectors: angles 90°, unseen-view error 100 px, compactness error 2, joint ATE 1000 cm and rotation 180°. Accepted aggregates cannot exceed these bounds, so a failed task cannot outrank a solved task through another metric.

Diagnostic cohort Q contains the reference, three baselines, reference repeat, no optimization, Gaussian loss, verified groups, joint verification and strict joint verification. Each shortcut is added to Q for its delta. Future production scales depend on future attempts; there is no universal scalar score. `redteam/hardening2/summary.json` records the cohort and aggregate vectors. Its deltas and intervals are evidence, not additional ranked fields.

Paper Tables 2, 3, 4 and 6 and earlier scalar summaries remain unchanged under `_paper_reported` in `scoring/reference.json`. Its calibration `results` rows are refreshed from the final held-out reference/baseline runs. No metric measures agreement with a paper value.

## Better and weaker variants, before and after

Pre-final metrics are frozen in `redteam/hardening2/before_metrics.json`; `before_observability.json` retains the first combined verification. These before/after deltas use the **same final Q metric standard deviations**. The graph correction changes some inputs, so these are setup comparisons, not paired changes across versions. Each final variant is compared with the final reference on identical inputs.

| Variant | Before final corrections, ΔQ | Final ΔQ | Final paired 95% interval |
|---|---:|---:|---|
| Reference | 0 | 0 | repeat metric difference exactly 0 |
| No optimization, weaker control | −8.316794 | −8.342824 | [−9.725927, −7.295539] |
| Gaussian loss | failed, −1092.435600 | −1.826735 | [−3.863049, −1.026624] |
| Verified groups | +0.558940 | +0.558940 | [+0.505694, +0.615606] |
| Joint verified groups | +0.541034 | +0.587852 | [+0.504673, +0.668984] |
| Strict joint verified groups | +0.558722 | **+0.594703** | **[+0.527369, +0.660945]** |

The large pre-correction Gaussian penalty is its complete failed vector divided by final scales from a cohort in which it is now valid. It is not a comparable raw error or a thousand-unit scientific change. Gaussian least squares now passes the gate but gives a worse weighted result. The original `REDTEAM.md` Gaussian gain (+0.008960) concerned two fixed scenes, not the current noisy, imperfect-association tasks.

Strict joint verification statistically tests group members, splits unsupported members and reruns joint refinement; it does not select stored reference outputs. Its conservative threshold is 73.68; the less conservative variant uses 18.42. These research controls impose no evaluator restriction. On the clustered dataset task:

| Scored quantity (lower is better) | Final reference | Strict better variant |
|---|---:|---:|
| ATE, cm | 1.771519 | 1.641705 |
| Rotation RMSE, ° | 0.405927 | 0.364228 |
| Median direction error, ° | 0.187782 | 0.136564 |
| Direction p90, ° | 1.450849 | 1.368522 |
| Median moment error, ° | 0.251263 | 0.245615 |
| Structural parameter error / line | 0.142617 | 0.069624 |

The unweighted normalized clustered-dataset improvement is +0.195; its 1,000-resample **sequence-block paired** 95% interval is **[+0.049904, +0.463743]**. Every resample includes all four phases of each selected sequence for both methods. This establishes headroom on paper data beyond repeated-sequence noise. The less conservative variant's block interval includes zero and is not claimed as independent dataset headroom evidence. Overall intervals in the preceding table use 1,000 paired instance resamples.

Final baseline deltas are OrthLine −5.055989 [−5.578922, −4.692893], constrained OrthLine −3.338633 [−3.484593, −3.183851], and independent RiemanLine −4.899763 [−5.321124, −4.603798]. On dataset associations their direction medians are 2.0227°, 0.9929° and 1.9044°, versus reference 0.1395°; compactness errors are 1.7369 versus 0.0123. The line-accuracy/compactness effect has the paper's direction and exceeds seed noise. ATE is mixed: constrained OrthLine has 1.6085 cm versus reference 1.6698 cm. Uniform trajectory dominance is not claimed.

## Major shortcuts

Historical before values use the cohorts stated in `REDTEAM.md` or the first hardening report. Final failure ΔQ is **−13.422123** in Q plus the shortcut. A metadata tie attached to valid geometry still requires the complete reconstruction.

| Shortcut | Before | Fix and final result |
|---|---|---|
| Absolute-path hidden-answer read | paid +4.29 in first hardening; pre-final attempt +0.119417 on final scales | Landlock denies the private subtree. Replayed reader exits with PermissionError; complete failure, −13.422123. |
| Copy visible answers or regenerate from public seeds | paid about +0.119 originally | Unseen instances, private seed/generator, disjoint dataset sequences. Earlier replay failed completely; unchanged scene generation was not replayed again. Final public-train-GT lookup gains 0 while falling back to a full solve. |
| Fabricate zero, negative or null parameter counts | zero paid +0.5 pairwise; negative +1.119586; null crashed evaluation | Structural degrees of freedom are recomputed from geometry. Final zero/negative attempts gain 0. Null was covered in the earlier pass and is ignored too. |
| Replace ground truth | paid +0.120 | In-memory evaluator truth and private answer files. Earlier replay gained 0; final reader replay checks the stronger file boundary. |
| Replace observations to fit output | paid +0.150 | Evaluator scores its own observations. Final replay gains 0; changing the solver's input copy changes no comparison. |
| Constant or over-compact geometry | constant paid +0.5 pairwise | Mandatory observed-line residual gate. Final constant replay fails at 100 px, −13.422123. Over-compact geometry was already rejected. |
| Omit or rename expected method | tied by leaving no values | Required rows receive complete worst vectors. Final omission fails, −13.422123. A real additional solver remains permitted and scored. |
| NaN, empty, partial, bad poses or non-Plucker outputs | NaN accepted; some invalid artifacts crashed without scores | Exact coverage and finite/manifold checks. Final NaN, overflow, missing-frame and bad-rotation attempts each get −13.422123. Empty/partial/non-Plucker cases were covered earlier. |
| Fake success, time, scores or verdicts | tied | Ignored. Final replay gains 0 with valid full reference geometry. |
| Replace baseline/comparison | comparisons previously mutable | Evaluator pins baseline source/launchers and owns operating points/truth. Hash mismatch fails; missing launchers get complete failed rows. Solver access to fixed source is read-only. |

Final `verify.py` checked every failed replay's entire vector in all five experiments, not just status or rank. There are no empty experiment rows or skipped penalties. Canonical `scores_test.json`, `scores_train.json` and `scores.json` were regenerated from the four reference/baseline runs alone.

## Reference before and after; headroom

The original fixed-scene reference had direction medians 0.061116° / 0.127620°, moment medians 0.119675° / 0.155769°, observed reprojection RMSE 1.077444 / 1.410620 px and self-reported 26 / 34 parameters. Those ground-truth-initialized results are not on the current scale. First hardening replaced them with unseen, partially observed scenes and restored the actual initialization/geometric procedure.

The procedural reference vectors are unchanged by this final pass:

| Experiment | Median direction ° | Direction p90 ° | Median moment ° | Moment p90 ° | Unseen-view px | Parameter error / line |
|---|---:|---:|---:|---:|---:|---:|
| `parallel_lines` | 0.136772 | 0.361558 | 0.112499 | 0.420989 | 0.718188 | 0 |
| `parallel_lines_harder` | 0.473505 | 1.658862 | 0.305875 | 1.421405 | 3.459577 | 0.025928 |
| `parallel_lines_mislabeled` | 0.249260 | 5.555281 | 0.184191 | 2.991728 | 4.941717 | 0.297330 |

| Dataset reference quantity | Dataset before | Dataset final | Clustered before | Clustered final |
|---|---:|---:|---:|---:|
| ATE, cm | 1.706285 | 1.669775 | 1.764921 | 1.771519 |
| Rotation, ° | 0.441339 | 0.371616 | 0.469400 | 0.405927 |
| Median direction, ° | 0.155067 | 0.139519 | 0.146707 | 0.187782 |
| Direction p90, ° | 1.335586 | 1.384506 | 1.424252 | 1.450849 |
| Median moment, ° | 0.260216 | 0.243609 | 0.250748 | 0.251263 |
| Parameter error / line | 0.012209 | 0.012263 | 0.141899 | 0.142617 |

The only exactly saturated reference metric is parameter error 0 in the clean procedural experiment. No angular, unseen-view, trajectory or rotation metric is at its lower bound. The added clustered dataset experiment measures unreliable associations in the paper's problem and has a demonstrated better variant, which reaches no dataset metric bound. Independent baselines remain clearly below the reference overall. Their wins on some mislabeled tail metrics are not our headroom evidence.

## Remaining limits and shortcuts left

- **Dataset train/test parity is not achieved.** Common preprocessing and the graph correction remove unidentifiable poses, but disjoint training sequences still have greater initial drift and difficult local minima. Reference ATE is 4.8401 / 5.7245 cm on train versus 1.6698 / 1.7715 cm on test; direction medians are 0.4862° / 0.9142° versus 0.1395° / 0.1878°. Office1 and TartanAir office dominate training errors. Pruning did not fix these residual errors. The disjoint sequences and actual optimizer are retained rather than hiding the gap or selecting sequences to match scores. This remains an unmet requirement. Procedural train/test direction medians are comparable: 0.1638/0.1368, 0.5310/0.4735 and 0.1972/0.2493 degrees.
- **Partial truth coverage.** The public carwelding archive has trajectory truth but no line-landmark truth. Its four instances contribute trajectory errors and undergo the observed-line gate; line accuracy and compactness pool the other five held-out sequences. No substitute truth is manufactured. The scientist-facing documents state this scope.
- **Unverified paper claims.** Reduced keyframe graphs, Python LM and local numerical Jacobians retain the paper's manifold/factor-graph method, but do not reproduce full-frame Ceres runtime or all reported ATE gains. Estimated associations and procedural mislabeled groups extend the paper's assumed associations. The milestone stays `secondary_claims`.
- **Legitimate ties.** Sim(3) gauge changes gain numerical noise only (2.2e−13); deterministic repeats tie exactly. Metadata lies attached to a valid reconstruction, rewriting an ignored input copy and an unsuccessful public-GT lookup followed by a full solve also tie. They provide no free reconstruction. A distinct honest method with identical measured quality is a headroom limitation, not a loophole.
- **Historical enumeration not replayed.** Exhaustive two-view pair selection already lost (−3.411465 in `REDTEAM.md`). Truncated segments remove endpoint correspondence; current joint graphs require continuous pose/point/line estimation rather than a small finite answer enumeration. General optimization of public observation residuals is allowed; there is no counted-query budget to evade. No paying short closed-form rule was found in the controls.
- **Residual attack scope.** Sophisticated kernel-boundary exploits and data memorization are left to the separate shortcut grader. Public original archives remain discoverable externally, although solving containers have no internet and hidden archives are inaccessible. No defenses against leftover processes or Python-environment tampering were added. Host Landlock support is required.

Both combined jobs passed `python validate.py`; the final also passed all evidence checks. Two heuristic warnings flag random instance construction in `real.py`; scoring reads submitted JSON through `_read_output` and `score_instance`, and replayed variants demonstrate that dependence. Final packaging changes only documentation, calibration rows and removal of audit candidates from the normal launcher. Scientific code/data are exactly those verified in job `1c13524f-b40`.
