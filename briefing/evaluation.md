# Evaluation

How success is measured for any method that outputs reconstructed 3D line landmarks. This file describes metrics and the evaluation procedure only.

## Metrics

The original benchmark reports Absolute Trajectory Error for pose estimation and angular direction/normal errors for line reconstruction. This scaled gym evaluates line reconstruction directly: median and p90 direction error in degrees, median normal error in degrees, line reprojection RMSE and p90 endpoint-to-line error in pixels, number of effective line parameters, and average pairwise direction spread inside a known parallel group. Lower is better for all metrics.

## Baselines and Targets

This gym includes saved reference outputs on the same synthetic factor graph. The target behavior is to keep line reprojection accurate, reduce line direction/normal error, reduce direction spread within the parallel group, and use fewer effective line parameters.

## Evaluation Protocol

A submitted method should write one JSON file per method variant under `outputs/` for `scaled_synthetic_parallel_lines` and under `outputs/harder/` for `scaled_synthetic_parallel_lines_harder`. Each file must include a `method` name, `num_parameters`, and a `lines` list containing `id`, `normal`, and `direction` for every scored 3D line.

Run `scripts/evaluate.sh` to score existing output JSON files. Run `scripts/reproduce.sh` or `scripts/run.sh` to regenerate the fixed-seed synthetic factor graphs, produce the current reference outputs, and call the evaluator. The standalone evaluator reads only saved output JSON files; it does not import or execute any method implementation.
