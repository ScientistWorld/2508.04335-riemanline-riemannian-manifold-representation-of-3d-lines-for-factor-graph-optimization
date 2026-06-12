# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

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

## Stop Justification
- Completed at milestone `secondary_claims`.
