# Progress

## Current state (iteration 11, hardening)

- The gym was rebuilt around a hidden test: procedural line-landmark scenes (2-3 parallel groups with random directions, individual lines, random cameras, truncated noisy segments) generated inside `eval/test/` (64 per experiment, secret 128-bit seed) and `eval/train/` (24 per experiment, public seed, exported to `data/train_scenes/`).
- Every solver `scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR` is run by both evaluators; all metrics are recomputed from raw line estimates; invalid or failed runs receive the worst value of every metric in every experiment.
- The reference implements the paper's StructRiemanLine: linear triangulation initialisation, the authors' parallel-group inlier check, shared S^2 direction plus scaled S^1 normals, Eq. 4-7 retraction inside Levenberg-Marquardt, Eq. 10 reprojection factors, Cauchy 3 px loss. Baselines: OrthLine (Bartoli-Sturm SO(3)xSO(2) updates), OrthLine_Constr (Eq. 11), independent RiemanLine.
- Three scored experiments: `parallel_lines` (0.40), `parallel_lines_harder` (0.35), `parallel_lines_mislabeled` (0.25, where the paper's method falls short).
- Shortcut and variant checks: see `HARDENING.md` and `redteam/after_hardening/`.

## Results (hidden test, 64 scenes per experiment; median direction error deg / novel-view px / parameter-count error per line)

| Method | parallel_lines | parallel_lines_harder | parallel_lines_mislabeled |
|---|---|---|---|
| `orthline` | 0.352 / 0.94 / 1.413 | 1.495 / 4.28 / 1.484 | 0.426 / 1.15 / 1.266 |
| `orthline_constr` | 0.351 / 0.92 / 1.413 | 1.255 / 3.70 / 1.484 | 0.415 / 1.12 / 1.266 |
| `riemanline` | 0.352 / 0.94 / 1.413 | 1.495 / 4.53 / 1.484 | 0.426 / 1.15 / 1.266 |
| `struct_riemanline` | **0.137 / 0.72 / 0.000** | **0.474 / 3.46 / 0.026** | 0.249 / 4.94 / 0.297 |

The shared-direction representation reproduces the paper's claim: lower direction and normal errors than every independent-line baseline (paired bootstrap far outside noise), with the minimal parameter count of the true structure. With wrong group declarations its error tail is worse than the baselines' (p90 direction 5.6 deg vs 1.0 deg), which is the headroom of the mislabeled experiment.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- The paper evaluates joint pose/point/line bundle adjustment on ICL-NUIM, TartanAir and Open-Structure and reports ATE. This gym optimises line landmarks only, with known camera poses, on procedural scenes; it reproduces the line-reconstruction (Table 4) and parameter-count (Table 6) claims, not the pose claims.
- Jacobians are finite differences in the manifold's local coordinates and the solver is a Python Levenberg-Marquardt rather than Ceres; the parameterisation and on-manifold updates are the paper's.
- The `parallel_lines_mislabeled` experiment (wrong group declarations) is an extension beyond the paper, added as the scored setting where the paper's method falls short.
