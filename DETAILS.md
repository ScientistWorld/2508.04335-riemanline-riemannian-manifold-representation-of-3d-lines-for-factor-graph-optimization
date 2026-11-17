# Overview

- **Paper ID:** 2508.04335
- **Title:** RiemanLine: Riemannian Manifold Representation of 3D Lines for Factor Graph Optimization
- **Domain:** Computer Vision / SLAM
- **TL;DR:** RiemanLine represents parallel 3D line groups with a shared manifold direction and per-line local normals, improving structural consistency with fewer optimization variables.

## Short Summary

The paper addresses line landmark representation in visual factor graph optimization. Its method decouples line geometry into global direction and local normal components, then shares the global direction across known parallel-line groups. Experiments on ICL-NUIM, TartanAir, and synthetic scenes report better pose and line reconstruction accuracy, especially in structured environments, along with lower parameter complexity.

## Key Results (paper)

- TartanAir Hospital translation ATE drops from 7.29 cm for independent RiemanLine to 2.91 cm for StructRiemanLine.
- TartanAir Hospital median line direction/normal errors drop to 1.60 and 0.91 degrees.
- For 20 lines in two parallel groups, the paper reports 44 line parameters instead of 80 for independent orthonormal lines.

---

# Problem Definition

Recover accurate, compact 3D maps from calibrated multi-view observations when some line landmarks are believed to be parallel. Incorrect structural associations should not corrupt the geometry.

The procedural tasks provide known camera poses and noisy, partially visible 2D line segments. Segment endpoints do not correspond between views. The dataset tasks provide initial camera poses, point and line landmarks, pixel measurements, correspondences and declared parallel groups from ICL-NUIM and TartanAir factor graphs. Here camera poses also need refinement. Some tasks contain imperfect parallel associations.

For each input, output an infinite 3D line for every line id, represented by its direction and Plucker moment in the input world frame. For joint tasks also output every camera pose in that frame. A world point X maps to camera coordinates as R_cw X + t_cw. A good method reduces geometric error and trajectory error while representing the true line structure compactly, including on unseen sequences and imperfect associations.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

RiemanLine: a minimal Riemannian representation of 3D line landmarks that extends from individual lines to groups of parallel lines (StructRiemanLine).

## Approach

Each line is decoupled into a unit direction u2 on the sphere S^2 and a scaled normal lambda * u1, with u1 on the circle orthogonal to u2. A parallel group shares one u2 and keeps a per-line (u1, lambda): 2 + 2k degrees of freedom for k lines instead of 4k. Updates are on-manifold: the direction moves along the tangent plane of S^2 through the exponential map (Eq. 4-5), each normal is re-projected onto the plane orthogonal to the new direction and rotated by an angle about it, and its scale is updated (Eq. 7). Line reprojection factors (Eq. 10) measure the distance of both observed segment endpoints to the projected 3D line. The authors' optimiser uses a Cauchy loss (3 px) and keeps a declared group only for members that pass a coarse consensus check on their initial directions (|cos| > 0.9), with more than two inliers.

## Implementation in this gym

`method/struct_riemanline.py` (+ `method/line_ba.py`): linear multi-view triangulation as the initial factor graph, the authors' group consensus check, shared-direction initialization from the members' principal direction, and a Levenberg-Marquardt solver whose steps are applied through the Eq. 4-7 retraction (Jacobians in local coordinates by finite differences). `method/joint_ba.py` extends the same line states and factors to joint camera pose, point and line optimization on dataset factor graphs, using sparse line derivatives and point Schur elimination. Procedural cameras stay fixed. Dataset instances retain the initial graph estimates and prune landmarks observed by fewer than three selected cameras. Camera pruning requires at least six retained point tracks and keeps the largest connected point co-visibility component; camera and landmark pruning repeat to stability. This common preprocessing uses graph incidence without ground truth. At most 150 keyframes, four phases per sequence, replace the full-sequence scale. Joint ATE uses Sim(3) alignment because inputs have 2D measurements only. Full paper ATE gains and runtime are not claimed.

## Main Claims

Encoding parallelism inside the minimal representation improves line reconstruction (lower direction and normal errors, Table 4) and pose accuracy, while reducing the line state from 4n to 2n + 2 per group (Table 6), compared with independent orthonormal lines with or without explicit parallelism residuals (Eq. 11).

---

# Evaluation

Run `bash scripts/evaluate_train.sh` for visible development scores and `bash scripts/evaluate_test.sh` for the hidden test. Both take no arguments and evaluate every `scripts/methods/*.sh` on disk. Numbers are written under each experiment's `results` in `scoring/scores_train.json` and `scoring/scores_test.json`. Required methods without a solver receive complete failed rows.

A solver is called once per split as `bash scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR`, with all experiments mixed under opaque scene ids. It writes one JSON per input, named identically:

```
{"lines": [{"id": 1, "normal": [x,y,z], "direction": [x,y,z]}]}
```

Joint inputs (`task: "joint_ba"`) additionally require `"frames": [{"id": 1, "R_cw": [[...],[...],[...]], "t_cw": [x,y,z]}]`. All frame and line ids must be answered exactly once. Each output JSON is limited to 16 MiB. Direction vectors may have arbitrary nonzero scale if the moment has the same scale; moment and direction must be orthogonal. Other output fields are ignored.

Procedural inputs contain `cameras` (id, K, R_cw, t_cw), `observations` (line_id, camera_id, pixel endpoints p0, p1), `lines` (ids), `parallel_groups`, and `image_size`. Joint inputs contain `intrinsics` (fx, fy, cx, cy), `frames`, `points` (id, xyz), `lines` (id, start, end), `point_observations` ([point_id, frame_id, u, v]), `line_observations` ([line_id, frame_id, us, vs, ue, ve]), and `parallel_groups`. Initial landmark endpoints in joint inputs are estimates, not ground truth.

The three procedural experiments use 24 train and 64 test scenes each. The two joint experiments use six train sequences and six disjoint test sequences, four keyframe phases per sequence (24/24 instances per experiment), with at most 150 keyframes. ICL-NUIM retains its original measurement noise; TartanAir's exact projected measurements receive 1 px Gaussian noise. After landmark visibility pruning, cameras need at least six retained point tracks; only the largest connected point co-visibility component is retained, with pruning repeated to stability. This prevents unidentifiable camera poses from corrupting global trajectory alignment. All methods see identical instances. The clustered joint experiment derives imperfect associations from the initial estimates. Fixed seeds and deterministic reference optimization eliminate seed noise. Train and test use independent evaluator implementations and the same interface and preprocessing. Train procedural truth is exported to `data/train_scenes/`; train dataset truth is public in `data/downloads/`. Test data, seeds and generators remain in `eval/test/`. Filesystem confinement denies solver access there, including absolute paths; public code and model files remain readable. Inputs, output directories and temporary work directories are writable.

All metrics are lower-is-better and are recomputed from raw geometry against evaluator-owned observations and truth:

- Direction and moment errors: unsigned angular errors in degrees, median over lines; direction p90 is also scored in every experiment. Procedural experiments additionally score moment p90.
- `novel_view_line_rmse_px` (procedural only): RMS endpoint-to-line distance in unseen noise-free views, distances clipped at 100 px.
- `ate_rmse_cm` (joint only): camera-center RMSE after Sim(3) alignment to dataset truth, geometric mean over instances, clipped to [0.01, 1000] cm. Scale is unobservable from the input's 2D measurements.
- `rotation_rmse_deg` (joint only): orientation RMSE after the same alignment, geometric mean over instances, clipped to [0.001, 180] degrees.
- `param_count_error_per_line`: absolute difference between the geometric representation's effective and true minimal line parameter counts, divided by the number of scored lines. Each count is 2 times the number of distinct directions plus 2 times the number of lines. Estimated direction equality uses 1e-7 rad; dataset truth uses 0.01 degree to account for coordinate precision. This measures structural compactness; it does not count internal optimizer state or trust declared parameter counts.
- `effective_line_parameters_per_line` (joint only, display primary): effective geometric line parameters divided by the number of scored lines, or `2 + 2d/n` for `d` distinct estimated directions and `n` scored lines. The evaluator derives both counts from geometry. Its coefficient is zero because compactness already contributes through the preceding error metric; it adds no ranking term. This reports direct dimensionality rather than a residual against scene-specific true structure.

Dataset line metrics pool only landmarks with supplied ground truth. The public carwelding archive has no line-landmark truth, so its four instances contribute trajectory metrics and undergo the line reprojection gate, but do not contribute to line accuracy or compactness. Those quantities use the other five held-out sequences. The disjoint dataset splits have unequal observed difficulty despite the common preprocessing: reference ATE is 4.84 / 5.72 cm on train versus 1.67 / 1.77 cm on test (dataset / clustered associations). Treat train scores as development measurements, not a calibrated predictor of test error. The procedural split errors are comparable. Full numerical comparisons and remaining limitations are in `HARDENING.md`.

A nonzero exit, missing output/id, non-finite or impossible geometry, invalid rotation or Plucker constraint, or median endpoint reprojection residual above 10 px in any input fails the entire run. Every experiment then gets every metric's worst value: angular errors 90 degrees, novel-view error 100 px, parameter error 2; joint ATE 1000 cm and rotation 180 degrees. The joint dimensionality display receives 4 parameters per line. A failed task cannot gain rank through compactness. There is no scored runtime or solver wall-clock cutoff.

The ranking uses the sum, over experiments and metrics, of experiment weight times the absolute coefficient times improvement over the frozen reference in the metric's better direction, divided by that metric's population standard deviation across attempts being ranked (zero variance contributes zero). Failed runs participate with their complete worst-value vectors. Only `results` numbers are ranked; diagnostic status, runtime, bootstrap intervals and illustrative rank deltas are not inputs to ranking. `scoring/reference.json` defines directions, coefficients, weights and calibration rows. Paper numbers on a different scale are preserved under `_paper_reported`; agreement with a paper number is never a metric.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 00:00] - none
- Audited the inherited workspace and found only scaffolding, with no runnable method or evaluation.
- Read the paper and identified the core algorithm: Riemannian 3D line optimization with shared direction for parallel-line groups.
- Added a scaled synthetic factor-graph reproduction and prepared a CPU validation job.

### [2026-06-12 16:30] - none
- Reviewed failed CPU smoke job `c6264d3c-cf0`.
- Fixed Python import path handling in all runnable scripts.

### [2026-06-12 16:45] - core_claim
- CPU reproduction job `5562538f-736` completed successfully.
- Ran the synthetic multi-view parallel-line factor graph end to end: generated data, optimized the independent-line baseline, optimized the StructRiemanLine shared-direction method, and evaluated both outputs.
- StructRiemanLine improved median direction error from 0.257 deg to 0.061 deg, improved median normal error from 0.262 deg to 0.151 deg, reduced effective parameters from 48 to 26, and enforced zero parallel-direction spread with comparable reprojection RMSE.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:30] - core_claim
- Validated the scaled synthetic parallel-line factor graph reproduction with `python validate.py --compare`.
- The StructRiemanLine implementation reduces effective line parameters from 48 to 26, enforces zero parallel spread, and improves median direction and normal errors relative to the independent-line baseline while keeping reprojection RMSE in the same range.
- The evaluator is method-independent and reads the method output artifacts from `outputs/`.

### [2026-06-12 16:30] - core_claim
- Audited the inherited implementation after milestone reset. The method code implements the paper's shared-direction Riemannian line-group optimizer rather than a surrogate, and evaluation remains independent from `method/`.
- Added a larger, noisier synthetic setting for a pending `core_claim_plus` run.

### [2026-06-12 16:34] - core_claim_plus
- CPU job `afdb62b0-9b7` completed successfully and reran the full packaged pipeline.
- Added result coverage for `scaled_synthetic_parallel_lines_harder` with 16 lines, 10 cameras, and stronger pixel noise.
- StructRiemanLine retained the core claim on the harder setting: 34 effective parameters versus 64 for independent lines, zero parallel spread, lower median direction error, and lower median normal error with comparable reprojection RMSE.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

### [2026-06-12 17:11] - secondary_claims
- Patched the standard evaluator so `scoring/scores.json` covers both target experiments and all metrics, matching `TARGETS.md` and `reference.json`.
- Removed generated job and trajectory artifacts from the git index; a CPU validation job is being submitted to regenerate scores with the fixed contract.

### [2026-06-12 17:18] - secondary_claims
- CPU job `4b3ae4c3-066` completed the final validation run and regenerated `scores.json`, `scores_train.json`, and `scores_test.json` with both experiments and full metric coverage.
- Final verification passed with the milestone unchanged; the gym is ready at scaled synthetic scope.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

### [2026-06-12 17:11] - secondary_claims
- Patched the standard evaluator so `scoring/scores.json` covers both target experiments and all metrics, matching `TARGETS.md` and `reference.json`.
- Removed generated job and trajectory artifacts from the git index; a CPU validation job is being submitted to regenerate scores with the fixed contract.

### [2026-06-12 17:18] - secondary_claims
- CPU job `4b3ae4c3-066` completed the final validation run and regenerated `scores.json`, `scores_train.json`, and `scores_test.json` with both experiments and full metric coverage.
- Final verification passed with the milestone unchanged; the gym is ready at scaled synthetic scope.

### [2026-06-12 17:22] - secondary_claims
- Final verifier reran `python validate.py --compare`; scoring, import separation, train/test wiring, and shared-path checks passed.
- Tightened method-agnostic evaluation and scoring documentation while keeping the milestone unchanged.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

### [2026-06-12 17:11] - secondary_claims
- Patched the standard evaluator so `scoring/scores.json` covers both target experiments and all metrics, matching `TARGETS.md` and `reference.json`.
- Removed generated job and trajectory artifacts from the git index; a CPU validation job is being submitted to regenerate scores with the fixed contract.

### [2026-06-12 17:18] - secondary_claims
- CPU job `4b3ae4c3-066` completed the final validation run and regenerated `scores.json`, `scores_train.json`, and `scores_test.json` with both experiments and full metric coverage.
- Final verification passed with the milestone unchanged; the gym is ready at scaled synthetic scope.

### [2026-06-12 17:22] - secondary_claims
- Final verifier reran `python validate.py --compare`; scoring, import separation, train/test wiring, and shared-path checks passed.
- Tightened method-agnostic evaluation and scoring documentation while keeping the milestone unchanged.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 36m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

### [2026-06-12 17:11] - secondary_claims
- Patched the standard evaluator so `scoring/scores.json` covers both target experiments and all metrics, matching `TARGETS.md` and `reference.json`.
- Removed generated job and trajectory artifacts from the git index; a CPU validation job is being submitted to regenerate scores with the fixed contract.

### [2026-06-12 17:18] - secondary_claims
- CPU job `4b3ae4c3-066` completed the final validation run and regenerated `scores.json`, `scores_train.json`, and `scores_test.json` with both experiments and full metric coverage.
- Final verification passed with the milestone unchanged; the gym is ready at scaled synthetic scope.

### [2026-06-12 17:22] - secondary_claims
- Final verifier reran `python validate.py --compare`; scoring, import separation, train/test wiring, and shared-path checks passed.
- Tightened method-agnostic evaluation and scoring documentation while keeping the milestone unchanged.

### [2026-11-09] - secondary_claims
- Hardening: hidden-test scenes generated only inside `eval/test/`, one solver interface for train and test, recomputed metrics with worst-value failures, reference restored to the paper's procedure (triangulation init, Eq. 4-7 updates, Cauchy loss, authors' group inlier check), Eq. 11 / Bartoli-Sturm baselines, and a new `parallel_lines_mislabeled` headroom experiment.
- StructRiemanLine beats OrthLine, OrthLine_Constr and RiemanLine on the 64-scene hidden test of both correctly grouped experiments with paired bootstrap CIs far from zero; details in `HARDENING.md`.

</details>

### Iteration 11: gpt-6.1-sol (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 35m | **GPU**: 0.0h
- **Jobs**: 21 total (15 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 16:36] - method_runs
- Implemented the paper's line reprojection optimization with the RiemanLine shared-direction parameterization on a fixed synthetic parallel-line factor graph.
- Added the independent OrthLine baseline and evaluator that reads saved method outputs.

### [2026-06-12 16:36] - core_claim
- Reproduced the core compactness and structural-consistency claim on `scaled_synthetic_parallel_lines`.
- StructRiemanLine used 26 effective line parameters versus 48 for the independent-line baseline, with lower median direction and normal angular errors.

### [2026-06-12 16:36] - core_claim_plus
- Added `scaled_synthetic_parallel_lines_harder` with more lines, more cameras, and higher pixel noise.
- StructRiemanLine retained the compact `2n + 2` parameter count and lower line geometry errors than the independent baseline on the additional setting.

### [2026-06-12 16:52] - core_claim
- Continuing audit found the latest CPU packaging job failed before data generation with a Python import-path error.
- Hardened all shell entrypoints to derive the workspace root from their script location and added an `orthline_constr` explicit parallel-residual baseline for the next validation job.

### [2026-06-12 17:00] - core_claim_plus
- CPU job `fc48f35f-b33` completed the full two-setting reproduction in the container and passed `python validate.py --compare`.
- The additional harder setting again showed StructRiemanLine with lower line direction/normal errors and fewer parameters than independent-line baselines.

### [2026-06-12 17:00] - secondary_claims
- Added and scored the paper's explicit parallel-constraint baseline family as `orthline_constr`.
- StructRiemanLine preserved exact shared-direction consistency with 26/34 parameters, while explicit constrained OrthLine kept 48/64 independent-line parameters.

### [2026-06-12 17:08] - secondary_claims
- CPU job `556fa2ba-0ac` reran the full suite after the output metadata fix and passed `python validate.py --compare`.
- Confirmed all six output artifacts now report the correct experiment names for default and harder settings.

### [2026-06-12 17:11] - secondary_claims
- Patched the standard evaluator so `scoring/scores.json` covers both target experiments and all metrics, matching `TARGETS.md` and `reference.json`.
- Removed generated job and trajectory artifacts from the git index; a CPU validation job is being submitted to regenerate scores with the fixed contract.

### [2026-06-12 17:18] - secondary_claims
- CPU job `4b3ae4c3-066` completed the final validation run and regenerated `scores.json`, `scores_train.json`, and `scores_test.json` with both experiments and full metric coverage.
- Final verification passed with the milestone unchanged; the gym is ready at scaled synthetic scope.

### [2026-06-12 17:22] - secondary_claims
- Final verifier reran `python validate.py --compare`; scoring, import separation, train/test wiring, and shared-path checks passed.
- Tightened method-agnostic evaluation and scoring documentation while keeping the milestone unchanged.

### [2026-11-09] - secondary_claims
- Hardening: hidden-test scenes generated only inside `eval/test/`, one solver interface for train and test, recomputed metrics with worst-value failures, reference restored to the paper's procedure (triangulation init, Eq. 4-7 updates, Cauchy loss, authors' group inlier check), Eq. 11 / Bartoli-Sturm baselines, and a new `parallel_lines_mislabeled` headroom experiment.
- StructRiemanLine beats OrthLine, OrthLine_Constr and RiemanLine on the 64-scene hidden test of both correctly grouped experiments with paired bootstrap CIs far from zero; details in `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
