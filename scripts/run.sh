#!/bin/bash
# TEMPORARY diagnostic job for the new Open-Structure joint-BA experiment.
# (The final run.sh reproduces the reference end to end; see tmp/run.sh.reference.)
set +e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
nproc
rm -f outputs/diag_joint.jsonl
python tmp/diag_joint.py no_optimization struct_verbose
rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes
time bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
time bash scripts/evaluate_test.sh
python tmp/diag_joint.py struct_iter300 struct_linear points_only
echo DIAG_DONE
