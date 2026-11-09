#!/bin/bash
# Run one baseline solver on a directory of scenes.
#
# Usage:
#   bash scripts/baseline.sh <orthline|orthline_constr|riemanline> INPUT_DIR OUTPUT_DIR
#
# Baselines are scored together with every other solver by
# scripts/evaluate.sh / evaluate_train.sh / evaluate_test.sh.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
NAME="$1"
case "$NAME" in
  orthline|orthline_constr|riemanline) ;;
  *) echo "unknown baseline '$NAME' (orthline, orthline_constr, riemanline)" >&2; exit 2 ;;
esac
bash "scripts/methods/${NAME}.sh" "$2" "$3"
