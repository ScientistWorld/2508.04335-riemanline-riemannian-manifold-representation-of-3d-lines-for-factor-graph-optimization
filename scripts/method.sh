#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

DATA_PATH="${1:-data/riemanline_synthetic.json}"
OUTPUT_PATH="${2:-outputs/struct_riemanline.json}"

python method/riemanline_optimize.py --data "$DATA_PATH" --output "$OUTPUT_PATH"
bash scripts/evaluate.sh "$DATA_PATH" outputs scoring/scores.json
