# Audit execution

Start commit: recorded in `original/commit.txt`. Pre-existing working-tree
differences are recorded in `original/status.txt`; they belong to the runtime
and are not changed by this audit.

`run_all.sh` regenerates the two baselines, runs the unchanged reference three
times with NumPy seeds 0, 1, 2, and then runs all variant/shortcut `run.sh` files.
Every candidate finishes before `bash scripts/evaluate_test.sh` starts with no
arguments in a separate process. There are no background processes or evaluator
environment changes. Test code and its imported geometry utilities stay intact.
Each run saves evaluator hashes, exit codes, train/test scores when emitted,
and its actual output artifacts. Failed evaluations start without stale scores.

The common candidate implementation reads only scientist-visible data, source,
and its own previously regenerated reference artifacts. `control.py` uses the
administrative original-file snapshots only to reset and restore experiments;
`summarize.py` reads the frozen score/reference files only after all trials.
These administrative files do not supply answers to any candidate.

Honest checks: Gaussian rather than Huber loss for Gaussian observation noise;
an endpoint-correspondence geometric model that accounts for all its additional
parameters; a longer, tighter optimization budget; no optimization; independent
line ablation. The brute force shortcut enumerates every pair of available views
for each line, selecting the best reprojection hypothesis. This is a continuous
optimization problem, so there is no literal finite space of all possible lines.

Shortcut trials cover visible/rebuilt answers, false/negative parameter counts,
ignored flags and claimed scores, substituted ground truth/observations, constant
reconstructions, omitted/renamed method rows, NaN outputs, null parameters, and
empty reconstructions. No shortcut edits evaluator code, its process, imports,
or Python configuration.

The rank is a weighted sum of direction-aware improvements using **absolute**
coefficient magnitudes, divided by per-metric standard deviations. Production
standard deviations depend on future attempts and are unavailable here.
`summary.json` therefore reports a declared honest audit cohort (adding the
tested candidate when necessary), a full audit cohort, and
reference-versus-candidate pairwise normalization. Every candidate contributes
to its own normalization. A rank delta of 0
is the reference. Raw metric vectors are retained for any future normalization.
NaN handling is explicitly conditional because the supplied rank description
only promises skipping missing values, not handling non-standard JSON numbers.

The archived job stdout contains preliminary scalar calculations. After the job,
the honest-cohort normalization was corrected to include the tested shortcut in
its own denominator, and the calculator was made standard-library-only. The
final `summary.json` and per-case `rank.json` files supersede those preliminary
stdout scalars; all method artifacts and evaluator-produced metric files are
unchanged. This recalculation requires no training or evaluation reruns.

All transient changes outside `redteam/` are restored after each case and at
job exit. The submission-only `scripts/run.sh` is restored before final delivery.
