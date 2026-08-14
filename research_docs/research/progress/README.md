# Detailed Progress Reports

This directory is the permanent, human-readable research history.

`research/STATE.md` stays compact and describes only the current research state. `research/EXPERIMENTS.jsonl` stays machine-readable. This directory contains one detailed Markdown report for every baseline and candidate experiment.

## Naming

Use monotonically increasing IDs:

- `PROGRESS_0000_BASELINE.md`
- `PROGRESS_0001_<short-candidate-name>.md`
- `PROGRESS_0002_<short-candidate-name>.md`

Use lowercase ASCII words separated by underscores in the candidate suffix. Never overwrite or renumber an older report.

## Lifecycle

Create the report as soon as the candidate is selected, then update the same report through the experiment:

1. `PLANNED` — candidate, parent, reason, intended training/evaluation plan;
2. `RUNNING` — exact code state, command, hardware/software snapshot, training settings;
3. progress events — intermediate checkpoints, cheap LIBERO progress checks, continue/extend/stop/select decisions;
4. broader evaluation — candidate screen, confirmation, diagnostics, canonical evaluation as applicable;
5. final status — `PROMOTE`, `REJECT`, `BRANCH`, `RETEST`, or `CRASH`.

A completed experiment is not fully recorded until its report is finalized.

## Detail standard

The report should let a future researcher understand and reproduce the run without relying on conversation history. Record concrete values and artifact paths rather than vague summaries. Keep huge raw logs and package listings on disk and reference them from the report instead of pasting thousands of lines.
