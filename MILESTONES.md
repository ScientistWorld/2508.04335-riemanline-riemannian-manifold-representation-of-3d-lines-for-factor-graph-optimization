# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-12 16:30] - core_claim
- Validated the scaled synthetic parallel-line factor graph reproduction with `python validate.py --compare`.
- The StructRiemanLine implementation reduces effective line parameters from 48 to 26, enforces zero parallel spread, and improves median direction and normal errors relative to the independent-line baseline while keeping reprojection RMSE in the same range.
- The evaluator is method-independent and reads the method output artifacts from `outputs/`.

### [2026-06-12 16:30] - core_claim
- Audited the inherited implementation after milestone reset. The method code implements the paper's shared-direction Riemannian line-group optimizer rather than a surrogate, and evaluation remains independent from `method/`.
- Added a larger, noisier synthetic setting for a pending `core_claim_plus` run.

## Stop Justification
- Not stopped: submitting the larger, noisier setting to test whether the core claim holds beyond the minimum experiment.
