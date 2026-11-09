# Scientist Constraints

## Data
Use only what a scene file provides (cameras, segment observations, declared parallel groups) when solving a scene. The train scenes in `data/train_scenes/` and the train generator `eval/train/scenes.py` may be used freely for development, tuning and learning. A solver must not read, import or regenerate anything from `eval/test/`, and must not use ground truth at solve time.

## Interface
A method is a solver script `scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR` that writes one JSON per scene with `{"lines": [{"id", "normal", "direction"}]}`. The scored row for your method is `struct_riemanline`: replace `scripts/methods/struct_riemanline.sh` (or the code it calls) with your method. Keep the baseline solvers unchanged.

## Evaluation
`eval/` must stay unchanged and must not import from `method/`. The hidden test (`scripts/evaluate_test.sh`, no arguments) regenerates its scenes, runs every solver once and recomputes all metrics; nothing a solver writes besides its line estimates is read.

## Compliance Checklist
- [ ] Solvers use only the scene file they are given (no ground truth, no access to `eval/test/`)
- [ ] `eval/` and `scoring/reference.json` are unchanged
- [ ] `scripts/methods/struct_riemanline.sh` runs your method and writes every scene and line
- [ ] Output lines are finite Plucker lines in the world frame of the first camera
