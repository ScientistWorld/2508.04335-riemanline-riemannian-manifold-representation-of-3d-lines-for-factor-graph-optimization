#!/bin/bash
# Reporting-only verification: rescore the final job's saved raw predictions.
# No optimizer, training or inference is run in this job.
set -euo pipefail
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m redteam.hardening2.rescore_reporting
python redteam/hardening2/verify.py
python validate.py --post-design
# Rescored artifacts cover all methods and experiments under the current code.
HASH="$(find eval method baseline scripts/methods scoring/reference.json -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) \
    -not -path '*/__pycache__/*' -not -path '*/downloads/*' -not -path '*/outputs/*' -not -path 'method/upstream/*' \
    -not -path '*/hardening_cases/*' -not -path '*/archive_before/*' -not -name 'results_full.json' \
    | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
printf '%s\n' "$HASH" > outputs/.stamp_train
printf '%s\n' "$HASH" > outputs/.stamp_test
