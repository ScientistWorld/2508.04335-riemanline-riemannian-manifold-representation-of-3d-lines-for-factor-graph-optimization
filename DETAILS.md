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

## Research Question

How should a visual mapping optimizer represent and refine 3D line landmarks, especially when many of them are known (or believed) to be parallel, so that the reconstructed line geometry is as accurate as possible while the map stays compact?

## Setting

Each scene is a small factor graph: a set of calibrated cameras with known poses, and noisy 2D line-segment detections of a set of 3D straight lines (several observations per line, from different cameras). Detected segments are noisy in pixel position and cover only part of the 3D segment, so endpoints do not correspond across views. Each scene also lists *declared parallel groups*: sets of line ids that an upstream module (e.g. vanishing-point clustering) believes to be parallel. Lines outside any group are individual lines. In some scenes these declarations contain mistakes.

The task is to output, for every line of every scene, an infinite 3D line in Plucker form: a direction vector and a normal (moment) vector, expressed in the world frame (the frame of the first camera).

## Why It Matters

Man-made environments are full of parallel line structures. A mapper that exploits them can reconstruct lines more accurately from the same images and store the map with fewer free parameters; a mapper that trusts wrong structure, or ignores structure, pays in accuracy.

## Success Criteria

A good method recovers line directions and positions more accurately (in 3D and when the lines are projected into views it never saw), represents the scene with no more and no fewer degrees of freedom than its true structure needs, and does so on unseen scenes, including scenes where the declared parallel groups are partly wrong.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

RiemanLine: a minimal Riemannian representation of 3D line landmarks that extends from individual lines to groups of parallel lines (StructRiemanLine).

## Approach

Each line is decoupled into a unit direction u2 on the sphere S^2 and a scaled normal lambda * u1, with u1 on the circle orthogonal to u2. A parallel group shares one u2 and keeps a per-line (u1, lambda): 2 + 2k degrees of freedom for k lines instead of 4k. Updates are on-manifold: the direction moves along the tangent plane of S^2 through the exponential map (Eq. 4-5), each normal is re-projected onto the plane orthogonal to the new direction and rotated by an angle about it, and its scale is updated (Eq. 7). Line reprojection factors (Eq. 10) measure the distance of both observed segment endpoints to the projected 3D line. The authors' optimiser uses a Cauchy loss (3 px) and keeps a declared group only for members that pass a coarse consensus check on their initial directions (|cos| > 0.9), with more than two inliers.

## Implementation in this gym

`method/struct_riemanline.py` (+ `method/line_ba.py`): linear multi-view triangulation as the initial factor graph, the authors' group consensus check, shared-direction initialization from the members' principal direction, and a Levenberg-Marquardt solver whose steps are applied through the Eq. 4-7 retraction (Jacobians in local coordinates by finite differences). Camera poses are known and fixed; the paper's joint pose/point optimisation and ATE evaluation are not reproduced.

## Main Claims

Encoding parallelism inside the minimal representation improves line reconstruction (lower direction and normal errors, Table 4) and pose accuracy, while reducing the line state from 4n to 2n + 2 per group (Table 6), compared with independent orthonormal lines with or without explicit parallelism residuals (Eq. 11).

---

# Evaluation

How success is measured for any method that reconstructs 3D lines from the scene files described in `briefing/problem.md`. This file describes the measurement only.

## Interface

A method is a solver script `scripts/methods/<name>.sh` called as

    bash scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR

It is called once per split with the scenes of all experiments mixed in one directory under opaque ids (a scene file does not say which experiment it belongs to). It must read every `INPUT_DIR/<scene_id>.json` and write `OUTPUT_DIR/<scene_id>.json` containing

    {"lines": [{"id": <int>, "normal": [x, y, z], "direction": [x, y, z]}, ...]}

with one entry per line id of the scene. Any other field is ignored. Scene files contain `scene_id`, `cameras` (`id`, `K`, `R_cw`, `t_cw`; a world point X maps to camera coordinates `R_cw X + t_cw`), `observations` (`line_id`, `camera_id`, segment endpoints `p0`, `p1` in pixels), `lines` (ids only), `parallel_groups` (declared groups of line ids) and `image_size`.

Every solver on disk is evaluated; the row named `struct_riemanline` is the reference row. A row listed in `scoring/reference.json` whose solver is missing is scored as a failed run.

## Splits

- **Train** (`scripts/evaluate_train.sh`, also `scripts/evaluate.sh`): 24 scenes per experiment generated by `eval/train/` with public seeds. Full scenes with ground truth are exported to `data/train_scenes/`; solver inputs and outputs are kept under `outputs/train/`.
- **Hidden test** (`scripts/evaluate_test.sh`, no arguments): 64 unseen scenes per experiment from the same distribution, generated inside `eval/test/` at evaluation time. Solvers run once on their public inputs; ground truth and held-out cameras never leave the evaluator.

## Validity

A run fails, and receives the worst value of every metric in every experiment, if the solver exits non-zero or times out, misses a scene or a line, writes a non-finite line, writes a line whose normal is not orthogonal to its direction (Plucker constraint), or does not solve the task: the median endpoint-to-line distance of the scene's own observations to the projected estimated lines exceeds 10 px in any scene (good reconstructions are near 1 px; constant or wrong-structure outputs are at 15-100 px). Worst values: 90 degrees for angles, 100 px for the novel-view error, 2.0 for the parameter-count error.

## Metrics (all lower is better; pooled over all lines of all scenes of an experiment)

- `median_direction_error_deg`, `p90_direction_error_deg`: unsigned angle between estimated and true line directions.
- `median_normal_error_deg`, `p90_normal_error_deg`: unsigned angle between estimated and true Plucker normal vectors (world frame at the first camera).
- `novel_view_line_rmse_px`: the true 3D segments are projected (noise-free) into held-out cameras that the method never sees; RMS distance (each distance clipped at 100 px) of those projected endpoints to the projection of the estimated line.
- `param_count_error_per_line`: the output's effective parameter count is 2 x (number of distinct estimated directions, exact equality up to 1e-7 rad) + 2 x (number of lines); the scene's minimal count is the same formula with the true distinct directions. The metric is |effective - minimal| summed over scenes, divided by the number of lines. Sharing directions that are truly parallel lowers it; sharing directions that are not parallel, or not sharing truly parallel ones, raises it.

All numbers are recomputed by the evaluator from the line estimates; self-reported counts, costs or flags are not read.

## Reference points

`scoring/scores_train.json` and `scoring/scores_test.json` hold the current reference row and three baseline rows on each split. `scoring/reference.json` holds the metric directions, coefficients and experiment weights, and the source paper's own reported numbers.

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


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
