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

# Jobs can be preempted and restarted: progress is kept in tmp/hardening and
# only discarded when the code that produced it changes.
STAMP=$(cat eval/test/*.py eval/train/*.py method/*.py baseline/*.py scripts/methods/*.sh \
        redteam/after_hardening/*.py scoring/reference.json | sha256sum | cut -d' ' -f1)
mkdir -p tmp/hardening
if [ "$(cat tmp/hardening/STAMP 2>/dev/null)" != "$STAMP" ]; then
  rm -rf tmp/hardening redteam/after_hardening/cases
  mkdir -p tmp/hardening
  echo "$STAMP" > tmp/hardening/STAMP
fi

if [ ! -f tmp/hardening/REFERENCE_DONE ]; then
rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes

bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh

python validate.py
touch tmp/hardening/REFERENCE_DONE
fi

# ---- Hardening checks (this job only): honest variants and shortcuts, each
# run through the real no-argument scripts/evaluate_test.sh in its own copy.
python -m redteam.after_hardening.run_cases
python -m redteam.after_hardening.analyze
