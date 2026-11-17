# Evaluation Targets

All quantities are lower-is-better. `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh` write every target under every experiment's `results`. Improve over the frozen reference in `scores_test.json`; paper numbers are context, not matching targets.

- All five experiments: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg` measure line accuracy; `param_count_error_per_line` measures the compactness constraint. Display primary: median direction error for procedural tasks, direct effective line parameters per line for joint tasks. Ranking uses every metric with its stated coefficient; the display primary does not restrict ranking.
- `parallel_lines`, `parallel_lines_harder`, `parallel_lines_mislabeled`: also `p90_normal_error_deg` and `novel_view_line_rmse_px`.
- `open_structure_joint_ba`, `open_structure_joint_ba_clustered`: also `ate_rmse_cm` and `rotation_rmse_deg` against dataset truth.

Joint experiments additionally display `effective_line_parameters_per_line`, computed from output directions. It has coefficient zero: compactness is already ranked through `param_count_error_per_line`, which penalizes both over- and under-sharing. Direct dimensionality is a readable summary, not an additional optimization target.

Validity is mandatory: every scene, frame and line answered, finite valid geometry, proper rotations, and median observed line residual at most 10 px per instance. Any failure gives worst values in all scored metrics of all experiments. Compare against the fixed `orthline`, `orthline_constr`, and `riemanline` rows as well as the reference. Accuracy gains must be assessed together with compactness and trajectory behavior, not by optimizing only one side.
