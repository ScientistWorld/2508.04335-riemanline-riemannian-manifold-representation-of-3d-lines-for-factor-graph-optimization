#!/bin/bash
# Reproduce the reference end to end (CPU only, a few minutes).
#
# Exports the visible train scenes, runs the paper's method and all baselines
# on the train scenes and on the hidden test scenes through the standard
# solver interface, writes scoring/scores{,_train,_test}.json and validates.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
bash scripts/run.sh
