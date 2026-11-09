#!/bin/bash
# Run the paper's proposed method (StructRiemanLine) on a directory of scenes.
#
# Usage:
#   bash scripts/method.sh INPUT_DIR OUTPUT_DIR
#
# INPUT_DIR holds public scene JSON files (e.g. outputs/train/<experiment>/inputs
# after scripts/evaluate_train.sh, or data/train_scenes/<experiment>).
# Scoring is done by scripts/evaluate.sh, which runs this same solver through
# scripts/methods/struct_riemanline.sh.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
bash scripts/methods/struct_riemanline.sh "$1" "$2"
