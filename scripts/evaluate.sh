#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name>                             # evaluate with defaults
#   ./evaluate.sh <method_and_run_name> checkpoints/epoch_10.pt     # evaluate a specific checkpoint
#
# Called from run.sh after training, or independently.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

mkdir -p scoring

if [[ "$#" -gt 0 ]]; then
  DATA_PATH="${1:-data/riemanline_synthetic.json}"
  OUTPUT_DIR="${2:-outputs}"
  SCORES_PATH="${3:-scoring/scores.json}"
  python -m eval.evaluate_outputs \
    --pair "$DATA_PATH" "$OUTPUT_DIR" \
    --scores "$SCORES_PATH"
else
  python -m eval.evaluate_outputs \
    --pair data/riemanline_synthetic.json outputs \
    --pair data/riemanline_synthetic_harder.json outputs/harder \
    --scores scoring/scores.json
fi
