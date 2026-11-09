# Problem Definition

## Research Question

How should a visual mapping optimizer represent and refine 3D line landmarks, especially when many of them are known (or believed) to be parallel, so that the reconstructed line geometry is as accurate as possible while the map stays compact?

## Setting

Each scene is a small factor graph: a set of calibrated cameras with known poses, and noisy 2D line-segment detections of a set of 3D straight lines (several observations per line, from different cameras). Detected segments are noisy in pixel position and cover only part of the 3D segment, so endpoints do not correspond across views. Each scene also lists *declared parallel groups*: sets of line ids that an upstream module (e.g. vanishing-point clustering) believes to be parallel. Lines outside any group are individual lines. In some scenes these declarations contain mistakes.

The task is to output, for every line of every scene, an infinite 3D line in Plucker form: a direction vector and a normal (moment) vector, expressed in the world frame (the frame of the first camera).

## Why It Matters

Man-made environments are full of parallel line structures. A mapper that exploits them can reconstruct lines more accurately from the same images and store the map with fewer free parameters; a mapper that trusts wrong structure, or ignores structure, pays in accuracy.

## Success Criteria

A good method recovers line directions and positions more accurately (in 3D and when the lines are projected into views it never saw), represents the scene with no more and no fewer degrees of freedom than its true structure needs, and does so on unseen scenes, including scenes where the declared parallel groups are partly wrong.
