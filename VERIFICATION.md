> **Superseded (iteration 11).** This file describes the gym before hardening. The current evaluation, method and shortcut results are in `HARDENING.md`, `briefing/` and `scoring/`.

# Verification Report

## Initial Assessment
The workspace entered final verification at `secondary_claims`. The claim is justified for the scaled synthetic gym: the implemented method preserves the paper's core line-reprojection computation and compact parallel-line state, and prior CPU container jobs regenerated the two scored experiments successfully.

`python validate.py --compare` passes. The evaluator reads saved output JSON artifacts and does not import from `method/`. The score files cover both experiments listed in `reference.json`, include benefit metrics for 3D line reconstruction and reprojection fit, and include constraint metrics for effective parameter count and parallel-direction consistency.

## Issues Found
- **Minor:** `scoring/EXPERIMENTS.md` and `scoring/TARGETS.md` described one reference baseline mechanism more explicitly than needed for from-scratch users.
- **Minor:** The login-node Python environment in this verification shell does not have `numpy`, so `scripts/evaluate.sh` cannot be run directly outside the packaged environment. `environment/setup.sh` installs `numpy` and `scipy`, and prior container jobs confirm the scripts run in the intended environment.
- **Cosmetic / external:** `README.md` and `manifest.json` are currently special device nodes owned by the runtime environment, and `git restore` cannot unlink them. They are outside the reproduction code path; the scored gym files validate correctly.

## Fixes Applied
- Reworded `scoring/EXPERIMENTS.md` so it describes only what each experiment measures: geometric accuracy, reprojection fit, structural consistency, and compactness.
- Reworded `scoring/TARGETS.md` to focus on metrics, reference trade-offs, and fixed evaluation targets rather than baseline implementation details.
- Clarified `briefing/evaluation.md` with the reusable output JSON contract for future methods.

## Final Judgment
This workspace is ready to serve as a research gym at the reproduced `secondary_claims` milestone. Improve-mode agents can run the packaged scripts and compare against the saved method and baseline outputs. From-scratch agents get a method-agnostic problem statement, data generator, output contract, evaluator, and scoring rubric.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the gym reproduces the paper on fixed procedural line-only factor graphs rather than the full original point-line-pose SLAM benchmarks.
