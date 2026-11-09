#!/bin/bash
# Solver interface used by scripts/evaluate_{train,test}.sh:
#   bash scripts/methods/orthline_constr.sh INPUT_DIR OUTPUT_DIR
# reads every public scene INPUT_DIR/<scene>.json and writes OUTPUT_DIR/<scene>.json
# with {"lines": [{"id", "normal", "direction"}, ...]}.
# Baseline: orthonormal lines plus the explicit Eq. 11 parallelism residual.
set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
python -m baseline.orthline "$1" "$2" --constrained
