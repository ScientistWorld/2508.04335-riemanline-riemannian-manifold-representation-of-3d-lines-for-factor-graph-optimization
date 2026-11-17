# Experiment Overview

Every experiment measures accuracy together with structural compactness. Invalid or failed runs receive complete worst-value rows in every experiment. All methods see the same inputs. These descriptions concern the measurement, not a prescribed solution.

| Experiment | Weight | What it establishes |
|---|---:|---|
| `parallel_lines` | 0.20 | Accuracy and compactness with correct associations, eight known cameras, 1 px noise and partial/truncated detections. |
| `parallel_lines_harder` | 0.15 | The same quantities under weak triangulation: six cameras on a 110-degree arc, 1.5 px noise, shorter segments and lower visibility. |
| `parallel_lines_mislabeled` | 0.15 | Robustness to incorrect parallel declarations, including near-parallel members and gross association errors. This is an extension beyond the paper's assumed associations. |
| `open_structure_joint_ba` | 0.30 | Trajectory and line accuracy against dataset ground truth on ICL-NUIM and TartanAir, together with compactness, using dataset structural associations. |
| `open_structure_joint_ba_clustered` | 0.20 | The same dataset quantities with imperfect associations derived from initial map estimates. Tests an upstream association weakness in the paper's setting. |

The procedural experiments use 24 train / 64 hidden test scenes each. Joint experiments use 24 train / 24 hidden test instances each: four keyframe phases of six/six disjoint sequences, up to 150 keyframes per instance. Dataset test sequences are livingroom2, livingroom3, office2, office3, hospital and carwelding. TartanAir's originally exact measurements receive 1 px noise; ICL-NUIM retains its original noise. After landmark visibility pruning, cameras need at least six retained point tracks; only the largest connected point co-visibility component is retained, with pruning repeated to stability. This prevents unidentifiable camera poses from corrupting global trajectory alignment. Dataset metrics use Sim(3) trajectory alignment. They are scaled evaluations, not direct replications of the paper's full-sequence tables.

Procedural metrics are median and p90 direction/moment angles, error in unseen views, and structural parameter-count error. Dataset metrics are ATE, rotation RMSE, median and p90 direction error, median moment error, and structural parameter-count error. Every quantity has a corresponding entry in `scores_train.json` and `scores_test.json`. See `briefing/evaluation.md` for definitions and failure rules.

Line metrics use only landmarks for which the archive supplies ground truth. Carwelding supplies trajectory truth but no line-landmark truth: it contributes trajectory scores and is subject to the observed-line validity gate; the other five test sequences contribute line metrics. The dataset train split remains harder than test: reference ATE is 4.84 / 5.72 cm on train and 1.67 / 1.77 cm on test for dataset / clustered associations. Identical interfaces and preprocessing do not remove this sequence difficulty gap. This limitation and the paired headroom evidence are recorded in `HARDENING.md`.
