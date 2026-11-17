# Progress

The final hardening pass is ready for combined verification. Five experiments cover procedural line reconstruction and joint pose/point/line refinement on disjoint ICL-NUIM and TartanAir sequences. Train and test use the same interface, noise and preprocessing; six dataset sequences each balance four ICL-NUIM and two TartanAir sequences.

The reference retains the paper's shared S² direction/scaled S¹ moment representation, Eq. 4–7 updates, reprojection factors, Cauchy loss and authors' group inlier check. Joint optimization refines camera poses and points alongside lines. Independent OrthLine, constrained OrthLine and RiemanLine baselines remain separate from method code.

Evaluator hardening confines solver file access away from hidden answers, pins comparisons, validates raw geometry including overflow, and assigns complete worst-value vectors to failures. No solver wall-clock cutoff or runtime score remains. Reference runs and controls are sequential and deterministic. The forthcoming job regenerates canonical reference/baseline scores and runs paired variant/shortcut checks; `HARDENING.md` records the changes and remaining limits.

Scaled settings: at most 150 keyframes per dataset instance, four phases per sequence, original ICL noise and 1 px added TartanAir noise, Sim(3) alignment for the 2D-only graph. Python LM and local numerical line/pose derivatives replace full-sequence Ceres execution while retaining its geometric procedure. Full paper ATE and runtime improvements are not claimed; current evidence is strong for line accuracy and compactness with mixed trajectory behavior.
