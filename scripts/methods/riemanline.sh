#!/bin/bash
# Solver interface used by scripts/evaluate_{train,test}.sh:
#   bash scripts/methods/riemanline.sh INPUT_DIR OUTPUT_DIR
# reads every public scene INPUT_DIR/<scene>.json and writes OUTPUT_DIR/<scene>.json
# with {"lines": [{"id", "normal", "direction"}, ...]}.
# Paper's independent-line RiemanLine (4 DoF per line, no shared directions).
set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
python -m method.struct_riemanline "$1" "$2" --no-groups
