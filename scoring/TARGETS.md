# Evaluation Targets

## Primary: scaled_synthetic_parallel_lines
Tests line reconstruction on a fixed-seed multi-view factor graph containing one known group of parallel 3D lines.
- **Metric**: `median_direction_error_deg` (lower is better)
- **Run**: `bash scripts/reproduce.sh`

## Constraint: scaled_synthetic_parallel_lines
Tests that improvements preserve the compact representation and the known parallel structure instead of treating every line as unconstrained.
- **Metrics**: `effective_params` and `parallel_spread_deg` (lower is better)
- **Run**: `bash scripts/reproduce.sh`
