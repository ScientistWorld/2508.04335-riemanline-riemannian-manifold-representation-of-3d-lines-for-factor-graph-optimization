# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

RiemanLine: a minimal Riemannian representation of 3D line landmarks that extends from individual lines to groups of parallel lines (StructRiemanLine).

## Approach

Each line is decoupled into a unit direction u2 on the sphere S^2 and a scaled normal lambda * u1, with u1 on the circle orthogonal to u2. A parallel group shares one u2 and keeps a per-line (u1, lambda): 2 + 2k degrees of freedom for k lines instead of 4k. Updates are on-manifold: the direction moves along the tangent plane of S^2 through the exponential map (Eq. 4-5), each normal is re-projected onto the plane orthogonal to the new direction and rotated by an angle about it, and its scale is updated (Eq. 7). Line reprojection factors (Eq. 10) measure the distance of both observed segment endpoints to the projected 3D line. The authors' optimiser uses a Cauchy loss (3 px) and keeps a declared group only for members that pass a coarse consensus check on their initial directions (|cos| > 0.9), with more than two inliers.

## Implementation in this gym

`method/struct_riemanline.py` (+ `method/line_ba.py`): linear multi-view triangulation as the initial factor graph, the authors' group consensus check, shared-direction initialization from the members' principal direction, and a Levenberg-Marquardt solver whose steps are applied through the Eq. 4-7 retraction (Jacobians in local coordinates by finite differences). Camera poses are known and fixed; the paper's joint pose/point optimisation and ATE evaluation are not reproduced.

## Main Claims

Encoding parallelism inside the minimal representation improves line reconstruction (lower direction and normal errors, Table 4) and pose accuracy, while reducing the line state from 4n to 2n + 2 per group (Table 6), compared with independent orthonormal lines with or without explicit parallelism residuals (Eq. 11).
