# Reproduction Milestones

**Current: core_claim**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim`.
