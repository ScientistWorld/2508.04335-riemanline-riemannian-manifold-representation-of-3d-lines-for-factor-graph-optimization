# Overview

- **Paper ID:** 2508.04335
- **Title:** RiemanLine: Riemannian Manifold Representation of 3D Lines for Factor Graph Optimization
- **Domain:** Computer Vision / SLAM
- **TL;DR:** RiemanLine represents parallel 3D line groups with a shared manifold direction and per-line local normals, improving structural consistency with fewer optimization variables.

## Short Summary

The paper addresses line landmark representation in visual factor graph optimization. Its method decouples line geometry into global direction and local normal components, then shares the global direction across known parallel-line groups. Experiments on ICL-NUIM, TartanAir, and synthetic scenes report better pose and line reconstruction accuracy, especially in structured environments, along with lower parameter complexity.

## Key Results (paper)

- TartanAir Hospital translation ATE drops from 7.29 cm for independent RiemanLine to 2.91 cm for StructRiemanLine.
- TartanAir Hospital median line direction/normal errors drop to 1.60 and 0.91 degrees.
- For 20 lines in two parallel groups, the paper reports 44 line parameters instead of 80 for independent orthonormal lines.
