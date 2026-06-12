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

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The paper reports Absolute Trajectory Error for pose estimation and angular direction/normal errors for line reconstruction. This scaled reproduction evaluates line reconstruction directly: median direction error in degrees, median normal error in degrees, line reprojection RMSE in pixels, number of effective line parameters, and average pairwise direction spread inside a known parallel group. Lower is better for all metrics.

## Baselines and Targets

The paper compares point-line factor graph variants using independent line parameterizations and structure-aware variants. This gym includes an independent-line baseline and the proposed structure-aware method on the same synthetic factor graph. The target behavior is that the structure-aware method keeps line reprojection accurate, reduces line direction/normal error, reduces direction spread within the parallel group, and uses fewer parameters.

## Evaluation Protocol

Run `scripts/reproduce.sh` or `scripts/run.sh` to generate the fixed-seed synthetic factor graph, optimize both methods, and call `scripts/evaluate.sh`. The evaluator reads only saved output JSON files under `outputs/` and compares them with the ground-truth line geometry in `data/riemanline_synthetic.json`.

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


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
