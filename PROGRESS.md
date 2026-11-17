# Progress

Post-design validation follow-up: the protected validator mistook a scaled held-out compactness residual for a paper transcription and rejected its ratio to the harder train split. The joint display primary now reports direct effective parameters per line, with coefficient zero to retain the existing ranked compactness constraint and every scientific weight. A CPU job will rescore the saved raw predictions, verify unchanged ranks/intervals, and run the stricter `--post-design` validation. The documented dataset difficulty gap remains.

Final job `1c13524f-b40` completed with exit code 0. It reran the reference and three baselines on both splits after the graph-observability correction, then the better/weaker variants and affected shortcuts through the hidden test. All evidence checks and validation passed. It resumed hashed completed work after a worker drain.

Five experiments cover procedural line reconstruction and joint pose/point/line refinement on disjoint ICL-NUIM and TartanAir sequences. The reference retains the paper's S²/scaled S¹ representation, Eq. 4–7 updates, reprojection factors, Cauchy loss, group inlier check and joint optimization. Dataset graphs use at most 150 keyframes, with common visibility/connectivity filtering. Package versions are pinned to the verification container.

The strict better variant gains +0.594703 in the declared normalized cohort, with paired 95% CI [+0.527369, +0.660945]. On clustered dataset graphs its sequence-block paired interval is [+0.049904, +0.463743]. Reference repetition changes no metric. Baselines and weaker controls score lower. The paying absolute-path answer reader now fails with complete worst vectors, as do invalid geometry and missing outputs. Metadata/input-copy tampering gains nothing. Canonical score files contain the reference and its three baselines only.

`HARDENING.md` records before/after scores, shortcut outcomes, weights, headroom and remaining limits. Dataset train/test difficulty remains unequal despite common preprocessing; carwelding has trajectory truth but no line-landmark truth. Full-sequence ATE gains/runtime claims remain unverified. The milestone remains `secondary_claims`.

Reference calibration is refreshed from final raw-output scores without changing paper-reported tables. Scientist-facing documents describe the final interface, measurements, graph selection and truth coverage. `scripts/run.sh` again reproduces the reference suite end to end; audit candidates remain separate.
