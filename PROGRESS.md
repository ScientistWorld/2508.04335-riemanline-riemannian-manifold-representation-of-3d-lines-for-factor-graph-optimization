# Progress

## What Works

- The paper has been analyzed and the upstream code release has been cloned under `method/upstream` for reference.
- A self-contained scaled reproduction has been implemented: fixed-seed synthetic multi-view line observations, an independent-line baseline, the StructRiemanLine shared-direction optimizer, and an independent evaluator.

## Results

- No completed job results yet in this continuation. The next submitted CPU test job will run `scripts/run.sh` and write `scoring/scores.json`.

## Remaining

- Run the CPU reproduction job and inspect `scoring/scores.json`.
- If the structure-aware method does not improve direction/normal accuracy and parameter count relative to the independent-line baseline, tune the synthetic benchmark or optimizer initialization while preserving the paper's computation.

## Issues

- The inherited workspace was essentially empty despite previous files being present.
- The paper's public repository depends on a C++/Ceres/Pangolin GUI stack and does not include the full benchmark datasets in the clone, so this reproduction uses a scaled procedural factor graph.
- First CPU smoke job failed before running optimization because Python executed scripts by path and did not put `/home/user` on `sys.path`; scripts now set `PYTHONPATH` and use `python -m`.

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
