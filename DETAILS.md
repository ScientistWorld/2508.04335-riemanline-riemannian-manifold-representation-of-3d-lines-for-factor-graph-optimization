# Overview

- **Paper ID:** 2508.04335
- **Title:** RiemanLine: Riemannian Manifold Representation of 3D Lines for Factor Graph Optimization
- **Domain:** Computer Vision / SLAM
- **TL;DR:** RiemanLine represents parallel 3D line groups with a shared manifold direction and per-line local normals, improving structural consistency with fewer optimization variables.

## Short Summary

The paper addresses line landmark representation in visual factor graph optimization. Its method decouples line geometry into global direction and local normal components, then shares the global direction across known parallel-line groups. Experiments on ICL-NUIM, TartanAir, and synthetic scenes report better pose and line reconstruction accuracy, especially in structured environments, along with lower parameter complexity.

## Key Results

- TartanAir Hospital translation ATE drops from 7.29 cm for independent RiemanLine to 2.91 cm for StructRiemanLine.
- TartanAir Hospital median line direction/normal errors drop to 1.60 and 0.91 degrees.
- For 20 lines in two parallel groups, the paper reports 44 line parameters instead of 80 for independent orthonormal lines.

---

# Problem Definition

## Research Question

How can a visual SLAM or structure-from-motion optimizer represent and refine 3D line landmarks, including groups of parallel lines, so that camera pose estimates and reconstructed line geometry improve under multi-view line observations?

## Why It Matters

Indoor and man-made environments contain many line structures. A good optimizer should use those geometric regularities to reduce drift, produce consistent maps, and avoid unnecessary optimization variables.

## Success Criteria

A successful system lowers line reconstruction errors and reprojection residuals while keeping the model compact and preserving known parallel relationships among line landmarks.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces RiemanLine, a minimal Riemannian representation for 3D line landmarks that extends naturally from independent lines to groups of parallel lines.

## Approach

Each 3D line is represented by a unit direction vector on the sphere and a scaled normal vector on the circle orthogonal to that direction. For a parallel-line group, all lines share one global unit direction while each line keeps its own local scaled normal component. Reprojection residuals are built from the distance between observed 2D segment endpoints and the projected 3D line, and the factor graph optimizes camera/landmark variables on their manifolds.

## Main Claims

The primary claim is that explicitly encoding parallel-line structure in a minimal manifold representation improves line-based factor graph optimization. Secondary claims are that the representation reduces dimensionality from `4n` independent line parameters to `2n + 2` for `n` parallel lines and improves convergence stability and solver efficiency.

---

# Evaluation

How success is measured for any method that outputs reconstructed 3D line landmarks. This file describes metrics and the evaluation procedure only.

## Metrics

The original benchmark reports Absolute Trajectory Error for pose estimation and angular direction/normal errors for line reconstruction. This scaled gym evaluates line reconstruction directly: median and p90 direction error in degrees, median normal error in degrees, line reprojection RMSE and p90 endpoint-to-line error in pixels, number of effective line parameters, and average pairwise direction spread inside a known parallel group. Lower is better for all metrics.

## Baselines and Targets

This gym includes saved reference outputs on the same synthetic factor graph. The target behavior is to keep line reprojection accurate, reduce line direction/normal error, reduce direction spread within the parallel group, and use fewer effective line parameters.

## Evaluation Protocol

A submitted method should write one JSON file per method variant under `outputs/` for `scaled_synthetic_parallel_lines` and under `outputs/harder/` for `scaled_synthetic_parallel_lines_harder`. Each file must include a `method` name, `num_parameters`, and a `lines` list containing `id`, `normal`, and `direction` for every scored 3D line.

Run `scripts/evaluate.sh` to score existing output JSON files. Run `scripts/reproduce.sh` or `scripts/run.sh` to regenerate the fixed-seed synthetic factor graphs, produce the current reference outputs, and call the evaluator. The standalone evaluator reads only saved output JSON files; it does not import or execute any method implementation.

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


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
