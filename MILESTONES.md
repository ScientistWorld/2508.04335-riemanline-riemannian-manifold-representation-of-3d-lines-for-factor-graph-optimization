# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-12 16:30] - core_claim
- Validated the scaled synthetic parallel-line factor graph reproduction with `python validate.py --compare`.
- The StructRiemanLine implementation reduces effective line parameters from 48 to 26, enforces zero parallel spread, and improves median direction and normal errors relative to the independent-line baseline while keeping reprojection RMSE in the same range.
- The evaluator is method-independent and reads the method output artifacts from `outputs/`.

## Stop Justification
- Completed at milestone `core_claim`.
- Higher milestones would require reproducing the full C++/Ceres point-line-pose bundle adjustment on ICL-NUIM, TartanAir, or Open-Structure sequences, which is beyond the packaged scaled benchmark currently available in this workspace.
