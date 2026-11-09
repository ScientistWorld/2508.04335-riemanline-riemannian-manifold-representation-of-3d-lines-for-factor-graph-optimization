# Experiment Overview

All three experiments are scored on 64 hidden scenes each (train: 24 visible scenes each, same distribution). Every method sees the same scenes.

## parallel_lines (weight 0.40)
Establishes the basic claim: when the declared parallel groups are correct, can a line optimizer reconstruct the lines more accurately than treating each line independently, while representing the scene with exactly the degrees of freedom its structure needs? Eight cameras around the scene, 1 px detection noise, partial visibility and truncated segments.

## parallel_lines_harder (weight 0.35)
Tests whether the benefit holds when the geometry is weak: a narrow 110-degree camera arc with six cameras, 1.5 px noise, 70% visibility, shorter segments and more lines per scene. Individually, many lines are poorly constrained here, so this experiment separates methods by how well they use structure to compensate.

## parallel_lines_mislabeled (weight 0.25)
Tests robustness to wrong structure: each declared group contains one or two lines that are actually 3-10 degrees off parallel, and half of the scenes contain one grossly wrong member. A method that trusts the declarations blindly pays in accuracy and in the parameter-count error; a method that ignores the declarations pays in accuracy on the correct members. This is where the current reference leaves the most room for improvement.

In every experiment, accuracy (3D direction and normal errors, median and tail, and the error in unseen views) is measured together with the parameter-count error, so a method cannot improve one by giving up the other unnoticed. Runs that fail the task or produce invalid output receive the worst value of every metric.
