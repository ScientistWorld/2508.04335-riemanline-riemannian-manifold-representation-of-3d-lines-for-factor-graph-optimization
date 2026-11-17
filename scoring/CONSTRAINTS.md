# Scientist Constraints

Use the public input instances and train split for development. Do not read, import or regenerate hidden inputs, answers or seeds from `eval/test/`, or use dataset ground truth to solve test instances. Network access is absent during execution. The hidden evaluator confines solver file access to public assets, its fresh input copy, outputs and temporary work directory.

Supply any estimation procedure through `scripts/methods/<name>.sh INPUT_DIR OUTPUT_DIR`; replace `struct_riemanline.sh` to submit under the reference row, or add a solver to compare it. Reconstruct every line and, for joint tasks, every camera pose. See `briefing/evaluation.md` for raw JSON formats. Learned components trained on the train split are allowed. You may change initialization, optimization, representations, losses or association handling; you are not limited to reference settings.

Keep `eval/`, `scoring/reference.json`, and baseline source/launchers unchanged. Baseline hashes are pinned by the evaluator. Self-reported counts, scores, validity flags and runtime have no effect. Output geometry must satisfy the task and validity gates before any improvement can count. Sim(3) gauge changes are equivalent reconstructions, not scientific gains.
