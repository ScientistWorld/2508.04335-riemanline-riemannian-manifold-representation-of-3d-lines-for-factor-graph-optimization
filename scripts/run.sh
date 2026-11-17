#!/bin/bash
# TEMPORARY: only the new/changed hardening cases (reference, baselines and
# evaluators are unchanged since the last full run; see tmp/run.sh.final).
set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python redteam/hardening2/run_cases.py joint_warm joint_verified joint_warm_verified public_gt_lookup
python redteam/hardening2/analyze.py
