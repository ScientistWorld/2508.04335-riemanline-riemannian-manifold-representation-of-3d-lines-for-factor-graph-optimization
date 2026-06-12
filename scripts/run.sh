#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

mkdir -p data outputs scoring tmp/scoring outputs/harder
rm -f outputs/*.json outputs/harder/*.json tmp/scoring/*.json

python -m data.generate_synthetic \
  --output data/riemanline_synthetic.json \
  --name scaled_synthetic_parallel_lines
bash scripts/baseline.sh data/riemanline_synthetic.json outputs/orthline.json
python -m baseline.orthline_constr_optimize \
  --data data/riemanline_synthetic.json \
  --output outputs/orthline_constr.json
bash scripts/method.sh data/riemanline_synthetic.json outputs/struct_riemanline.json

python -m data.generate_synthetic \
  --output data/riemanline_synthetic_harder.json \
  --seed 11 \
  --num-lines 16 \
  --num-cameras 10 \
  --noise-px 1.6 \
  --name scaled_synthetic_parallel_lines_harder
bash scripts/baseline.sh data/riemanline_synthetic_harder.json outputs/harder/orthline.json
python -m baseline.orthline_constr_optimize \
  --data data/riemanline_synthetic_harder.json \
  --output outputs/harder/orthline_constr.json
python -m method.riemanline_optimize --data data/riemanline_synthetic_harder.json --output outputs/harder/struct_riemanline.json
python -m eval.evaluate_outputs \
  --data data/riemanline_synthetic_harder.json \
  --outputs-dir outputs/harder \
  --scores tmp/scoring/harder_scores.json

python - <<'PY'
import json
from pathlib import Path

default_scores = json.loads(Path("scoring/scores.json").read_text())
harder_scores = json.loads(Path("tmp/scoring/harder_scores.json").read_text())
default_scores["experiments"].update(harder_scores["experiments"])
Path("scoring/scores.json").write_text(json.dumps(default_scores, indent=2, sort_keys=True) + "\n")
PY

python validate.py --compare
