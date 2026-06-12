# Research Direction

## Research Problem
Improve 3D line landmark optimization in visual factor graphs, especially when the scene contains known groups of parallel lines. The goal is better geometric reconstruction and image reprojection consistency without inflating the optimization state.

## Core Contribution
The paper's contribution is a compact manifold representation for line landmarks and line groups. Future work should stay focused on geometric line representations, manifold updates, factor design, initialization, or robust optimization for this same line-based factor graph setting.

## Approach Scope
Allowed changes include alternative minimal line parameterizations, different manifold update rules, robust losses, initialization strategies, or structural factors, provided the result is still a geometric factor-graph method for 3D line landmarks.

## Out of Bounds
- Replacing the problem with neural image-to-geometry prediction
- Adding extra cameras, observations, or labels to the fixed scored dataset
- Optimizing only 2D image lines without reconstructing 3D line geometry
- Reporting scores from a method that does not expose line normals and directions for evaluation
