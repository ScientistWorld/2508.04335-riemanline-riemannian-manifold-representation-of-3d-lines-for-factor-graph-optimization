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
#
# Resumable (jobs can be preempted): steps 2-3 are skipped when their score
# file was produced by the same code (stamp in outputs/), and finished
# hardening cases are skipped.
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

code_hash() {
    find eval method baseline scripts/methods scoring/reference.json -type f \
        \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) \
        -not -path '*/__pycache__/*' -not -path '*/downloads/*' -not -path '*/outputs/*' -not -path 'method/upstream/*' \
        -not -path '*/hardening_cases/*' -not -path '*/archive_before/*' -not -name 'results_full.json' \
        | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16
}
HASH="$(code_hash)"
mkdir -p outputs
python -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'

rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes

if [ -f scoring/scores_train.json ] && [ "$(cat outputs/.stamp_train 2>/dev/null)" = "$HASH" ]; then
    echo "scores_train.json is current ($HASH); skipping train evaluation"
else
    bash scripts/evaluate_train.sh
    echo "$HASH" > outputs/.stamp_train
fi
cp scoring/scores_train.json scoring/scores.json
if [ -f scoring/scores_test.json ] && [ "$(cat outputs/.stamp_test 2>/dev/null)" = "$HASH" ]; then
    echo "scores_test.json is current ($HASH); skipping hidden-test evaluation"
else
    bash scripts/evaluate_test.sh
    echo "$HASH" > outputs/.stamp_test
fi

python redteam/hardening2/run_cases.py
python redteam/hardening2/analyze.py
python redteam/hardening2/verify.py

python validate.py
