# Problem Definition

Recover accurate, compact 3D maps from calibrated multi-view observations when some line landmarks are believed to be parallel. Incorrect structural associations should not corrupt the geometry.

The procedural tasks provide known camera poses and noisy, partially visible 2D line segments. Segment endpoints do not correspond between views. The dataset tasks provide initial camera poses, point and line landmarks, pixel measurements, correspondences and declared parallel groups from ICL-NUIM and TartanAir factor graphs. Here camera poses also need refinement. Some tasks contain imperfect parallel associations.

For each input, output an infinite 3D line for every line id, represented by its direction and Plucker moment in the input world frame. For joint tasks also output every camera pose in that frame. A world point X maps to camera coordinates as R_cw X + t_cw. A good method reduces geometric error and trajectory error while representing the true line structure compactly, including on unseen sequences and imperfect associations.
