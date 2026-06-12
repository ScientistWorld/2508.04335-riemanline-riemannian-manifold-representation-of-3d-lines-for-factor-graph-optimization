# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The paper reports Absolute Trajectory Error for pose estimation and angular direction/normal errors for line reconstruction. This scaled reproduction evaluates line reconstruction directly: median direction error in degrees, median normal error in degrees, line reprojection RMSE in pixels, number of effective line parameters, and average pairwise direction spread inside a known parallel group. Lower is better for all metrics.

## Baselines and Targets

The paper compares point-line factor graph variants using independent line parameterizations and structure-aware variants. This gym includes an independent-line baseline and the proposed structure-aware method on the same synthetic factor graph. The target behavior is that the structure-aware method keeps line reprojection accurate, reduces line direction/normal error, reduces direction spread within the parallel group, and uses fewer parameters.

## Evaluation Protocol

Run `scripts/reproduce.sh` or `scripts/run.sh` to generate the fixed-seed synthetic factor graph, optimize both methods, and call `scripts/evaluate.sh`. The evaluator reads only saved output JSON files under `outputs/` and compares them with the ground-truth line geometry in `data/riemanline_synthetic.json`.
