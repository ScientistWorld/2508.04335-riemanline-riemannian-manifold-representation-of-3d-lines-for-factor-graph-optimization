# Evaluation Targets

All targets are measured by `bash scripts/evaluate_test.sh` (hidden test, no arguments) on the row `struct_riemanline`; `bash scripts/evaluate.sh` gives the same measurements on the visible train scenes. Every metric is lower-is-better.

## Primary: line reconstruction accuracy (all three experiments)
- **Metrics**: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg`, `p90_normal_error_deg`, `novel_view_line_rmse_px`.
- **Goal**: lower errors than the current reference row in `scoring/scores_test.json`, on `parallel_lines`, `parallel_lines_harder` and `parallel_lines_mislabeled`.
- **Run**: `bash scripts/evaluate_test.sh`

## Constraint: compact, structurally correct representation (all three experiments)
- **Metric**: `param_count_error_per_line` (0 means the output uses exactly the degrees of freedom of the true structure: one shared direction per truly parallel set).
- **Goal**: do not trade it away for accuracy. The reference is at 0 when the declared groups are correct; on `parallel_lines_mislabeled` it is above 0 because some declared members are not parallel.
- `scoring/reference.json` lists the paper's own value (Table 6 expressed per line: 0 for the shared-direction representation, 1.8 for independent 4n representations).

## Constraint: validity
- Every scene and line answered, finite Plucker lines (normal orthogonal to direction), median residual on the scene's own observations at most 10 px in every scene. Violations score the worst value of every metric in every experiment.

## Comparison rows
- `orthline`, `orthline_constr`, `riemanline`: independent-line baselines (with and without explicit parallel residuals) scored on the same scenes.
