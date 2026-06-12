# Experiment Overview

## scaled_synthetic_parallel_lines
Tests whether a line-based optimizer can recover a group of parallel 3D line landmarks from noisy multi-view 2D segment observations. The experiment measures geometric accuracy, image-space fit, structural consistency within the parallel group, and parameter compactness.

## scaled_synthetic_parallel_lines_harder
Repeats the same evaluation with more line landmarks, more camera views, and stronger pixel noise. This checks whether the compactness and structural-consistency benefits persist beyond the minimum scored setting.

Each experiment includes reference outputs with different parameter counts and structural consistency, so improvements must be judged across accuracy, image-space fit, and compactness rather than any single metric alone.
