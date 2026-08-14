# FastWAM AutoResearch Agent Kit

A repo-local autonomous research setup for one concrete goal:

> Start from the released FastWAM LIBERO checkpoint, improve LIBERO-90 to at least 90% success, and keep LIBERO-Spatial, Object, Goal, and Long at or above 90% each.

## Structure

`CLAUDE.md` contains the always-on Claude Code project rules and objective.

`AUTORESEARCH.md` contains the continuous experiment loop.

The six skills separate the main research phases:

1. `setup-fastwam-research` — understand this checkout/machine, run one-time smoke tests, establish the baseline and adaptive evaluation plan;
2. `choose-fastwam-experiment` — decide what is worth trying next;
3. `run-fastwam-training` — execute and control a planned candidate, including intermediate checkpoints;
4. `evaluate-fastwam-on-libero` — collect adaptive LIBERO evidence for progress control, candidate screening, confirmation, diagnosis, and promotion;
5. `review-fastwam-experiment` — make the final candidate-level promote/reject/branch/retest decision;
6. `investigate-fastwam-problem` — deeper diagnosis when ordinary iteration stalls.

Every baseline and candidate also receives a permanent detailed Markdown report under `research/progress/`. `STATE.md` remains the short current-state view; the reports preserve full reproducibility details, including hardware/software, exact commands/configuration, training timeline, intermediate checkpoints, progress checks, metrics, task-level evidence, artifacts, interpretation, and decision.

The run/evaluation commands are intentionally not hard-coded into the skills. `setup-fastwam-research` inspects the actual FastWAM repository and fills `research/RUNBOOK.md` with verified commands and paths.

## Install

Unzip/copy this directory into the root of the FastWAM checkout so that `CLAUDE.md`, `AUTORESEARCH.md`, `.claude/`, and `research/` sit at repository root.

Make the official released checkpoint available locally. The default checkpoint name is:

`libero_uncond_2cam224.pt`

Then start the coding agent in the repository and instruct it to read `AUTORESEARCH.md` and begin.

## Public FastWAM context

The Fast-WAM project page reports the Fast-WAM model at:

- LIBERO-Spatial: 98.2%
- LIBERO-Object: 100.0%
- LIBERO-Goal: 97.0%
- LIBERO-Long: 95.2%
- Average: 97.6%

The research loop still measures the released checkpoint locally before any modification and uses that measured baseline for comparisons.

## Experiment philosophy

The loop borrows useful ideas from AutoResearch: baseline first, fixed evaluator semantics, Git-backed experiment state, concise result extraction, preservation of failed/crashed experiments, evidence-based keep/revert decisions, simplicity for near-ties, and continuous autonomous iteration.

It does not force a single-file edit, a single scalar objective, a fixed tiny training budget, or one-variable-at-a-time experimentation.

An experiment is a coherent candidate and may contain multiple related modifications.

## Adaptive evaluation

Evaluation is used as a research control signal, not only after training finishes.

A cheap progress check may evaluate an intermediate checkpoint to decide whether to:

- continue training;
- extend a promising run;
- stop a bad/regressive run early;
- select an earlier checkpoint;
- train more and recheck;
- diagnose unexpected behavior.

After or between training phases, cheap candidate screening and broader confirmation decide whether full evaluation is worth the cost. Only the fixed canonical five-suite evaluation can promote a checkpoint.

The exact cheap task set, rollout count, and progress-check timing are chosen by the coding agent from measured baseline variance, current failure patterns, evaluation cost, and the candidate's risk. They are not hard-coded here.

Setup smoke tests are separate: evaluation/training/checkpoint-reload smoke tests run during initial environment setup only, or after a material infrastructure change. They are not repeated before ordinary experiments.

## Checkpoint persistence

Important checkpoints are backed up to the Hugging Face model repository `cheikh025/ASR`. Authentication is read from the `HF_TOKEN` environment variable. Local checkpoints remain on disk after upload; remote backup does not trigger local deletion. If disk space later becomes constrained, the agent verifies the remote artifact before removing any local checkpoint and preserves the active parent/continuation checkpoint whenever practical.

## Research records

The package uses three levels of state:

- `research/STATE.md` — short current working state;
- `research/EXPERIMENTS.jsonl` — structured experiment ledger;
- `research/progress/PROGRESS_XXXX_*.md` — detailed immutable report for each baseline/experiment.

The detailed report is created when an experiment is planned and updated during training, progress checks, evaluation, and review. This avoids relying on conversational memory to reconstruct the run afterward.
