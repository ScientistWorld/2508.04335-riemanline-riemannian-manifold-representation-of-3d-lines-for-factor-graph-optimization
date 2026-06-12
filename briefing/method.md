# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces RiemanLine, a minimal Riemannian representation for 3D line landmarks that extends naturally from independent lines to groups of parallel lines.

## Approach

Each 3D line is represented by a unit direction vector on the sphere and a scaled normal vector on the circle orthogonal to that direction. For a parallel-line group, all lines share one global unit direction while each line keeps its own local scaled normal component. Reprojection residuals are built from the distance between observed 2D segment endpoints and the projected 3D line, and the factor graph optimizes camera/landmark variables on their manifolds.

## Main Claims

The primary claim is that explicitly encoding parallel-line structure in a minimal manifold representation improves line-based factor graph optimization. Secondary claims are that the representation reduces dimensionality from `4n` independent line parameters to `2n + 2` for `n` parallel lines and improves convergence stability and solver efficiency.
