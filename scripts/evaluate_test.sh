#!/bin/bash
# Hidden-test evaluation.  No arguments.
#
# Generates the hidden test scenes inside eval/test/ (their seeds and ground
# truth never leave that directory), runs every solver in scripts/methods/*.sh
# on the public part of those scenes, recomputes all metrics from the raw line
# estimates and writes scoring/scores_test.json.  Methods required by
# scoring/reference.json that are missing or fail get the worst value of every
# metric in every experiment.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
mkdir -p scoring

python -m eval.test.evaluate --scores scoring/scores_test.json
