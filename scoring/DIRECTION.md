# Research Direction

## Research Problem
Improve 3D line landmark reconstruction in multi-view line factor graphs that contain groups of parallel lines: more accurate line geometry (also when seen from new viewpoints) with a representation whose degrees of freedom match the true structure, robust to errors in the declared parallel groups.

## Core Contribution
The source paper contributes a minimal manifold representation for line landmarks that shares one direction across a parallel group and optimizes it inside a reprojection factor graph. Work in this gym should stay a geometric reconstruction method for this setting: line parameterizations, manifold updates, factors and robust losses, initialization, structure verification or estimation, and solver design.

## Approach Scope
Allowed: any geometric estimation procedure that reconstructs 3D lines from the given observations, including learned components trained only on the train split; deciding which declared groups or members to trust; alternative minimal representations; better initialization and optimization.

## Out of Bounds
- Reading, importing or regenerating the hidden test scenes (`eval/test/`), or using ground truth at solve time
- Editing the evaluator, `scoring/reference.json` or the baseline solvers
- Outputs that are not real 3D line estimates (constant, copied or partially missing lines)
