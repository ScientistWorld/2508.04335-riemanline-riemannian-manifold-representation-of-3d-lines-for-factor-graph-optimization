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

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p data outputs scoring

python -m data.generate_synthetic --output data/riemanline_synthetic.json
bash scripts/baseline.sh data/riemanline_synthetic.json outputs/orthline.json
bash scripts/method.sh data/riemanline_synthetic.json outputs/struct_riemanline.json
python validate.py --compare
