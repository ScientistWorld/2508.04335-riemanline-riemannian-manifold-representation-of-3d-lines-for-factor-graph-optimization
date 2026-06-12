# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim_plus`.
- Higher milestones would require reproducing the full point-line-pose benchmark suite from the original C++/Ceres stack on ICL-NUIM, TartanAir, or Open-Structure sequences.
