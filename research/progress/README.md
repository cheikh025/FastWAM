# Detailed Progress Reports

This directory is the permanent human-readable history for the **new RoboTwin multi-embodiment project**.

`research/STATE.md` stays compact. `research/EXPERIMENTS.jsonl` stays machine-readable. This directory contains one detailed Markdown report for parent setup and every new candidate.

Do not import/renumber the previous LIBERO project's reports.

## Naming

Use monotonically increasing IDs:

- `PROGRESS_0000_PARENT_BASELINE.md`
- `PROGRESS_0001_<short-candidate-name>.md`
- `PROGRESS_0002_<short-candidate-name>.md`

Use lowercase ASCII words separated by underscores in candidate suffixes. Never overwrite or renumber an older report.

## Lifecycle

Create the report as soon as a candidate is selected, then update the same report through:

1. `PLANNED` — parent, candidate, reason, intended training/evaluation plan;
2. `RUNNING` — exact code/data/embodiment state, command, hardware/software;
3. progress events — intermediate checkpoints, RoboTwin progress checks, LIBERO retention sentinels, continue/extend/stop/select decisions;
4. broader evaluation — screen, confirmation, diagnostics, canonical evaluation as applicable;
5. final status — `PROMOTE`, `REJECT`, `BRANCH`, `RETEST`, or `CRASH`.

`PROGRESS_0000_PARENT_BASELINE.md` is special: it records inherited exp0019 canonical LIBERO evidence separately from fresh-machine setup/sentinel evidence and RoboTwin evaluator/reference validation. It must not pretend the inherited canonical numbers were rerun by this project.

## Detail standard

A future researcher should be able to understand/reproduce the run without conversation history. Record concrete values and artifact paths. Keep huge raw logs/package listings on disk and reference them instead of pasting thousands of lines.
