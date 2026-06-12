# Problem Definition

## Research Question

How can a visual SLAM or structure-from-motion optimizer represent and refine 3D line landmarks, including groups of parallel lines, so that camera pose estimates and reconstructed line geometry improve under multi-view line observations?

## Why It Matters

Indoor and man-made environments contain many line structures. A good optimizer should use those geometric regularities to reduce drift, produce consistent maps, and avoid unnecessary optimization variables.

## Success Criteria

A successful system lowers line reconstruction errors and reprojection residuals while keeping the model compact and preserving known parallel relationships among line landmarks.
