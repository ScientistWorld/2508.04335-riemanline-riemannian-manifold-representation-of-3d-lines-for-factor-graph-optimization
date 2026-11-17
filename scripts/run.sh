#!/bin/bash
# Reproduce the reference end to end (CPU only).
#
# 1. Export the visible procedural train scenes (public inputs + ground truth)
#    to data/train_scenes/ for development.  (Open-Structure train instances
#    are built on the fly by eval/train/real.py from data/downloads/.)
# 2. Run every solver in scripts/methods/ (StructRiemanLine and the three
#    baselines) on the train instances  -> scoring/scores_train.json (= scores.json)
# 3. Run every solver on the hidden test instances built inside eval/test/
#                                       -> scoring/scores_test.json
# 4. Hardening checks: variants and shortcuts through the real hidden test
#    (redteam/hardening2/), then the analysis -> redteam/hardening2/summary.json
# 5. Validate the workspace.
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes

bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh

python redteam/hardening2/run_cases.py
python redteam/hardening2/analyze.py

python validate.py
