# Progress

## What Works

- The paper has been analyzed and the upstream code release has been cloned under `method/upstream` for reference.
- A self-contained scaled reproduction has been implemented: fixed-seed synthetic multi-view line observations, an independent-line baseline, the StructRiemanLine shared-direction optimizer, and an independent evaluator.
- `scripts/run.sh` completed on CPU in job `5562538f-736`, including data generation, both optimizers, evaluation, and `python validate.py --compare`.
- CPU job `afdb62b0-9b7` completed the expanded two-setting evaluation and passed `python validate.py --compare`.

## Results

The reproduced core-claim experiment is `scaled_synthetic_parallel_lines`.

| Method | Median direction error (deg) | Median normal error (deg) | Reprojection RMSE (px) | Effective params | Parallel spread (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `orthline` baseline | 0.257 | 0.262 | 0.985 | 48 | 0.320 |
| `struct_riemanline` | 0.061 | 0.151 | 1.047 | 26 | 0.000 |

This supports the core claim in the scaled environment: the shared-direction Riemannian representation gives more accurate and consistent parallel-line geometry with fewer landmark parameters, while keeping reprojection fit in the same range as the independent-line baseline.

The added `scaled_synthetic_parallel_lines_harder` experiment uses 16 lines, 10 cameras, and higher pixel noise.

| Method | Median direction error (deg) | Median normal error (deg) | Reprojection RMSE (px) | Effective params | Parallel spread (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `orthline` baseline | 0.386 | 0.377 | 1.289 | 64 | 0.544 |
| `struct_riemanline` | 0.128 | 0.199 | 1.417 | 34 | 0.000 |

This supports `core_claim_plus`: the shared-direction representation preserves the claim on an additional noisier setting, improving geometry and structural consistency while using fewer parameters.

## Remaining

- Higher milestones would require reproducing more of the original benchmark suite, including point-line-pose bundle adjustment on the paper's ICL-NUIM, TartanAir, or Open-Structure sequences.
- The reusable gym is ready for the current scaled benchmark: future agents can run `scripts/reproduce.sh` or improve the method and evaluate through `scripts/evaluate.sh`.

## Issues

- The inherited workspace was essentially empty despite previous files being present.
- The paper's public repository depends on a C++/Ceres/Pangolin GUI stack and does not include the full benchmark datasets in the clone, so this reproduction uses a scaled procedural factor graph.
- First CPU smoke job failed before running optimization because Python executed scripts by path and did not put `/home/user` on `sys.path`; scripts now set `PYTHONPATH` and use `python -m`.
- The successful CPU job showed a slightly higher reprojection RMSE for StructRiemanLine than the independent baseline, which is expected for the stronger shared-direction constraint; the geometry accuracy, parallel consistency, and parameter-count metrics improved.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- The paper evaluates on ICL-NUIM, TartanAir, and Open-Structure simulation sequences. This workspace currently uses a fixed-seed procedural synthetic parallel-line factor graph because the public clone did not include the full benchmark data and the original stack is a GUI-oriented C++ SLAM simulator.
- The reproduction optimizes line landmarks with known camera poses rather than the full point-line-pose bundle adjustment. The retained core computation is the line reprojection factor and the Riemannian shared-direction parameterization for parallel lines.
