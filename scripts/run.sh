#!/bin/bash
# Reproduce the reference end to end (CPU only, a few minutes).
#
# 1. Export the visible train scenes (public inputs + ground truth) to
#    data/train_scenes/ for development.
# 2. Run every solver in scripts/methods/ (StructRiemanLine and the three
#    baselines) on the train scenes  -> scoring/scores_train.json (= scores.json)
# 3. Run every solver on the hidden test scenes generated inside eval/test/
#                                     -> scoring/scores_test.json
# 4. Validate the workspace.
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ ! -f tmp/hardening/REFERENCE_DONE_GATE10 ]; then
rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes

bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh

python validate.py
mkdir -p tmp/hardening && touch tmp/hardening/REFERENCE_DONE_GATE10
fi

# ---- Hardening checks (this job only).  All other cases are kept from the
# previous run: the gate change (5 -> 10 px) cannot alter them (valid cases
# never exceed 2.1 px; failed ones fail on invalid output or >= 50 px).
python -m redteam.after_hardening.run_cases reference no_optimization
python -m redteam.after_hardening.analyze
