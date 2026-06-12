# Evaluation

How success is measured for any method that outputs reconstructed 3D line landmarks. This file describes metrics and the evaluation procedure only.

## Metrics

The original benchmark reports Absolute Trajectory Error for pose estimation and angular direction/normal errors for line reconstruction. This scaled gym evaluates line reconstruction directly: median and p90 direction error in degrees, median normal error in degrees, line reprojection RMSE and p90 endpoint-to-line error in pixels, number of effective line parameters, and average pairwise direction spread inside a known parallel group. Lower is better for all metrics.

## Baselines and Targets

This gym includes saved outputs from multiple geometric line optimizers on the same synthetic factor graph. The target behavior is to keep line reprojection accurate, reduce line direction/normal error, reduce direction spread within the parallel group, and use fewer effective line parameters.

## Evaluation Protocol

Run `scripts/reproduce.sh` or `scripts/run.sh` to generate the fixed-seed synthetic factor graphs, produce optimizer output JSON files, and call `scripts/evaluate.sh`. The standalone evaluator reads only saved output JSON files under `outputs/` and `outputs/harder/`; it does not import or execute any method implementation.
