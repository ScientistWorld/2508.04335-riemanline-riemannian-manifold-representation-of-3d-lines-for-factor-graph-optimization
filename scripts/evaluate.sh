#!/bin/bash
# Standard evaluation entry point (visible train scenes).
#
# Runs every solver in scripts/methods/*.sh on the public-seed train scenes
# through the same interface and scoring rules as the hidden test, and writes
# scoring/scores.json.  Uses only code in eval/; never imports method/.
#
# A solver is any executable shell script scripts/methods/<name>.sh taking
#   INPUT_DIR OUTPUT_DIR
# that reads every INPUT_DIR/<scene>.json and writes OUTPUT_DIR/<scene>.json
# containing {"lines": [{"id": int, "normal": [3], "direction": [3]}, ...]}.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"
mkdir -p scoring

python -m eval.train.evaluate --scores scoring/scores.json
