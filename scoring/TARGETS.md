# Evaluation Targets

## Primary: scaled_synthetic_parallel_lines
Tests whether a submitted representation preserves the paper's compactness claim for one known group of 12 parallel 3D lines.
- **Metric**: `effective_params` (lower is better; paper formula gives `4n = 48` for independent lines and `2n + 2 = 26` for StructRiemanLine)
- **Run**: `bash scripts/reproduce.sh`

## Constraint: scaled_synthetic_parallel_lines
Tests that compactness does not come from ignoring the reconstruction task or breaking the known parallel structure.
- **Metrics**: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg`, `reprojection_rmse_px`, `p90_reprojection_error_px`, and `parallel_spread_deg` (lower is better)
- **Baselines**: `orthline` uses independent line directions; `orthline_constr` uses independent line directions plus explicit parallelism residuals.
- **Run**: `bash scripts/reproduce.sh`

## Additional Setting: scaled_synthetic_parallel_lines_harder
Tests the same goals on a larger, noisier fixed-seed factor graph with 16 parallel lines and 10 cameras.
- **Metric**: `effective_params` (lower is better; `4n = 64` for independent lines and `2n + 2 = 34` for StructRiemanLine)
- **Constraint metrics**: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg`, `reprojection_rmse_px`, `p90_reprojection_error_px`, and `parallel_spread_deg` (lower is better)
- **Run**: `bash scripts/reproduce.sh`
