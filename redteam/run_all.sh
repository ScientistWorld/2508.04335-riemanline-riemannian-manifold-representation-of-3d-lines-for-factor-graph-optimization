#!/bin/bash
set -euo pipefail
cd /home/user
trap 'python -m redteam.control restore' EXIT
mkdir -p redteam/control_artifacts/harder
# Regenerate baselines from visible source rather than taking hidden scores or
# inherited output artifacts as candidate input. Cache only to avoid rerunning.
python -m baseline.orthline_optimize --data data/riemanline_synthetic.json --output redteam/control_artifacts/orthline.json
python -m baseline.orthline_constr_optimize --data data/riemanline_synthetic.json --output redteam/control_artifacts/orthline_constr.json
python -m baseline.orthline_optimize --data data/riemanline_synthetic_harder.json --output redteam/control_artifacts/harder/orthline.json
python -m baseline.orthline_constr_optimize --data data/riemanline_synthetic_harder.json --output redteam/control_artifacts/harder/orthline_constr.json
for case_name in \
  reference_seed0 reference_seed1 reference_seed2 \
  gaussian_loss endpoint_model more_training_honest_alternative \
  no_training independent_ablation brute_force_shortcut \
  answers_copy_shortcut answers_rebuild_shortcut \
  zero_params_shortcut negative_params_shortcut flags_shortcut \
  ground_truth_replace_shortcut observation_replace_shortcut \
  constant_geometry_shortcut renamed_method_shortcut omit_proposed_shortcut \
  nan_geometry_shortcut null_params_shortcut empty_output_shortcut; do
  bash "redteam/$case_name/run.sh"
done
python -m redteam.summarize
