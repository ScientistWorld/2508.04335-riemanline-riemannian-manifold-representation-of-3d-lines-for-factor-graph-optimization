#!/bin/bash
# SMOKE JOB (temporary): checks the hardened pipeline on CPU and times it.
set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import numpy, scipy, sys; print(sys.version, numpy.__version__, scipy.__version__)"
nproc || true

echo "=== smoke"
python tmp/smoke.py 2 || echo "SMOKE FAILED"

echo "=== export train scenes"
rm -rf data/train_scenes
python -m eval.train.evaluate --export-scenes data/train_scenes
du -sh data/train_scenes

echo "=== evaluate_train"
time bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
echo "=== evaluate_test"
time bash scripts/evaluate_test.sh
python validate.py || echo "VALIDATE FAILED"

echo "=== harness case"
time python -m redteam.after_hardening.run_cases hidden_generator_import
cat redteam/after_hardening/cases/hidden_generator_import/status.json || true
python - <<'PY'
import json
s = json.load(open("tmp/hardening/hidden_generator_import/scores_test.json"))
print(json.dumps(s["_metadata"]["failed_methods"], indent=1)[:2000])
PY
