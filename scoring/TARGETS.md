# Evaluation Targets

## Primary: scaled_synthetic_parallel_lines
Tests whether a submitted method can represent one known group of 12 parallel 3D lines compactly while still reconstructing the 3D geometry.
- **Metric**: `effective_params` (lower is better; the current reproduced method uses 26 effective line parameters versus 48 for the least compact reference output)
- **Run**: `bash scripts/reproduce.sh`

## Constraint: scaled_synthetic_parallel_lines
Tests that compactness does not come from ignoring the reconstruction task or breaking the known parallel structure.
- **Metrics**: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg`, `reprojection_rmse_px`, `p90_reprojection_error_px`, and `parallel_spread_deg` (lower is better)
- **Reference outputs**: `orthline`, `orthline_constr`, and `struct_riemanline` provide current comparison points with different accuracy, consistency, and parameter-count trade-offs.
- **Run**: `bash scripts/reproduce.sh`

## Additional Setting: scaled_synthetic_parallel_lines_harder
Tests the same goals on a larger, noisier fixed-seed factor graph with 16 parallel lines and 10 cameras.
- **Metric**: `effective_params` (lower is better; the current reproduced method uses 34 effective line parameters versus 64 for the least compact reference output)
- **Constraint metrics**: `median_direction_error_deg`, `p90_direction_error_deg`, `median_normal_error_deg`, `reprojection_rmse_px`, `p90_reprojection_error_px`, and `parallel_spread_deg` (lower is better)
- **Run**: `bash scripts/reproduce.sh`
