# Train/Test Evaluation Design Notes

## Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The test script invokes `eval/test/`, writes `scoring/scores_test.json`, and passes a train-score path only to require the same method names when available.
- Reviewed `eval/train/evaluate.py` and `eval/test/evaluate.py`. Both slices read saved method output JSON files from disk and do not rerun or import the paper method.
- Checked for train/test cross-imports and method-name defaults. No cross-imports or hardcoded method names were found.
- Confirmed there were no `scoring/scores_train.json` or `scoring/scores_test.json` stubs.
- Ran shell syntax checks for both split scripts and reference-schema validation. No experiments or evaluators were run.

## Changed

- Added `p90_direction_error_deg` to both train and test evaluators. This makes the split harder to game by methods that keep median direction error low while sacrificing a tail of scored lines.
- Added `p90_reprojection_error_px` to both train and test evaluators. This catches methods that fit enough endpoints for the RMSE/median-style metrics while leaving large image-space outliers.
- Updated `scoring/reference.json` so every nonzero metric has `higher_is_better`, `coefficient`, and `description`.
- Reduced the coefficient on `effective_params` and increased the harder experiment weight from `0.35` to `0.45`. Compactness remains part of the paper claim, but the score now puts more pressure on maintaining geometric quality under the larger/noisier setting.

## Runtime Concerns For Next Iteration

- The new p90 metrics are source-defined but have not been numerically exercised in this iteration. The runner should verify both `scores_train.json` and `scores_test.json` include `p90_direction_error_deg` and `p90_reprojection_error_px` for every method and experiment.
- Watch for saturation. The synthetic line geometry is small, so a method that copies ground-truth lines from the visible data would still score perfectly; the benchmark relies on the held-out test slice being hidden from future agents.
- `primary_metric` remains `effective_params` because existing reference results only contain parameter counts. The metric coefficients now counterbalance this, but a later cleanup could add reference-side geometric values and switch the primary metric to a reconstruction-quality measure.
