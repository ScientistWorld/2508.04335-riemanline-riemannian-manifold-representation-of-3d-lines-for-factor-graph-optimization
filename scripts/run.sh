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

mkdir -p data outputs scoring outputs/harder
rm -f outputs/*.json outputs/harder/*.json scoring/scores_train.json scoring/scores_test.json

python -m data.generate_synthetic \
  --output data/riemanline_synthetic.json \
  --name scaled_synthetic_parallel_lines
bash scripts/baseline.sh data/riemanline_synthetic.json outputs/orthline.json
python -m baseline.orthline_constr_optimize \
  --data data/riemanline_synthetic.json \
  --output outputs/orthline_constr.json
python -m method.riemanline_optimize --data data/riemanline_synthetic.json --output outputs/struct_riemanline.json

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

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python - <<'PY'
import json
from pathlib import Path

reference = json.loads(Path("scoring/reference.json").read_text())
train = json.loads(Path("scoring/scores_train.json").read_text())
test = json.loads(Path("scoring/scores_test.json").read_text())

expected_experiments = set(reference["experiments"])
for label, scores in (("train", train), ("test", test)):
    experiments = {
        name: payload
        for name, payload in scores["experiments"].items()
        if not name.startswith("_")
    }
    if set(experiments) != expected_experiments:
        raise SystemExit(
            f"{label} experiments {sorted(experiments)} do not match "
            f"reference {sorted(expected_experiments)}"
        )
    for exp_name, ref_exp in reference["experiments"].items():
        metrics = set(ref_exp["metrics"])
        methods = set(ref_exp["results"])
        results = experiments[exp_name]["results"]
        if set(results) != methods:
            raise SystemExit(
                f"{label} {exp_name} methods {sorted(results)} do not match "
                f"reference {sorted(methods)}"
            )
        for method, values in results.items():
            if set(values) != metrics:
                raise SystemExit(
                    f"{label} {exp_name}/{method} metrics {sorted(values)} "
                    f"do not match reference {sorted(metrics)}"
                )

for exp_name in expected_experiments:
    train_methods = set(train["experiments"][exp_name]["results"])
    test_methods = set(test["experiments"][exp_name]["results"])
    if train_methods != test_methods:
        raise SystemExit(f"{exp_name} train/test method mismatch")

print("split score files match reference experiments, methods, and metrics")
PY

python validate.py --reference-only
