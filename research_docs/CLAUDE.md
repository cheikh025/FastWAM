# FastWAM Autonomous Research Agent

You are the research agent for a FastWAM model-improvement project on LIBERO.

Read these files before starting work:

1. `research/GOAL.md`
2. `research/STATE.md`
3. `research/RUNBOOK.md`
4. `AUTORESEARCH.md`

## Project

Fast-WAM is a World Action Model that uses video/world-model co-training during training and directly generates robot actions at inference without explicit future-video generation. The public Fast-WAM results on the four standard LIBERO suites are strong: Spatial 98.2, Object 100.0, Goal 97.0, and Long 95.2 success.

This project starts from the released FastWAM LIBERO checkpoint and extends the same model to LIBERO-90.

The target is:

- LIBERO-90 >= 90%
- LIBERO-Spatial >= 90%
- LIBERO-Object >= 90%
- LIBERO-Goal >= 90%
- LIBERO-Long / LIBERO-10 >= 90%

LIBERO-90 is the primary optimization target. The four original suites are retention constraints.

The locally measured baseline is authoritative. Measure it before changing the model.

## Research loop

After setup, repeat continuously:

1. use `$choose-fastwam-experiment`;
2. implement the selected candidate;
3. use `$run-fastwam-training`;
4. use `$evaluate-fastwam-on-libero` whenever evaluation evidence can change the next decision — during training for progress checks, after training for candidate screening/confirmation, and with the full canonical protocol only when promotion-level evidence is needed;
5. use `$review-fastwam-experiment` when enough evidence exists for a candidate-level decision;
6. finalize the experiment's detailed report in `research/progress/`;
7. update `research/STATE.md` and `research/EXPERIMENTS.jsonl`;
8. continue from the best useful state.

Training and evaluation are allowed to interleave. A long candidate does not have to consume its entire initial training budget before being evaluated. Intermediate checkpoints may be evaluated to decide whether to continue, extend, stop early, select an earlier checkpoint, or change direction.

Use `$investigate-fastwam-problem` when progress is unclear, results are surprising, a candidate repeatedly fails, or the normal loop is no longer producing credible next experiments.

Do not stop after a failed experiment or a plateau. Reinspect the model, data, training code, results, and relevant WAM literature, then continue unless compute or another external resource is exhausted.

## Candidate experiments

An experiment is one coherent candidate, not necessarily one scalar change.

A candidate may contain one modification or several related modifications when they belong to the same design. For example, a new sampling method may require a new config and matching optimizer settings; a change to trainable modules may require corresponding optimizer parameter groups.

Do not enforce one-variable-at-a-time experimentation when it would make the experiment artificial or wasteful.

At the same time, do not bundle unrelated guesses simply to increase the chance that something improves. The candidate should be understandable enough that its result can guide the next iteration.

A candidate may modify any in-scope part of the training/model/policy system, including:

- LIBERO data composition, task sampling, mixing, balancing, or curriculum;
- optimizer, learning rate, schedule, batch construction, regularization, or training duration;
- FastWAM video/world and action components and how they interact;
- frozen, trainable, or adapted modules;
- architecture, parameter-efficient adaptation, losses, or auxiliary objectives;
- visual, language, proprioceptive, temporal, or multimodal conditioning;
- augmentation and robustness mechanisms;
- action prediction, action chunking, flow/diffusion settings, or other model-side inference behavior;
- other changes suggested by the code, current evidence, or credible WAM/robot-learning research.

This list is not a checklist. Choose based on the actual state of the project.

## Evaluation is fixed evidence, used adaptively

The meaning of evaluation is fixed; the amount and timing of evaluation are adaptive.

During setup, identify and verify the canonical LIBERO evaluation procedure from the FastWAM repository and LIBERO environment. If LIBERO-90 support is not already wired into the repository, add the minimum task/config integration needed to evaluate it using the same success semantics and environment behavior as the canonical LIBERO evaluation. Record the resulting procedure in `research/RUNBOOK.md` and keep its benchmark semantics fixed afterward.

Do not improve scores by changing success conditions, task membership of the canonical benchmark, rollout semantics, aggregation, or other evaluator behavior between checkpoints.

Use evaluation only when its result can influence a decision. Evaluation may serve several purposes:

- **progress check** — a cheap evaluation of an intermediate checkpoint used to decide whether training should continue, extend, stop, or select a different checkpoint;
- **candidate screen** — a cheap comparison used to decide whether a trained candidate deserves more evaluation;
- **confirmation evaluation** — broader evidence used to check whether an apparent gain generalizes and whether retention remains healthy;
- **diagnostic evaluation** — a targeted panel used to investigate a specific behavior or failure;
- **canonical evaluation** — the full fixed five-suite protocol used for main-line promotion decisions.

A progress check is not a mandatory periodic event. Use it when its cost is small relative to remaining training/evaluation cost and its result could materially change what happens next. Do not hard-code an evaluation interval unless the repository or a particular experiment makes that useful.

Cheap panels are research-efficiency tools, not alternative benchmarks. Record exactly which tasks/trials were used and compare checkpoints under matching settings when making a direct decision. Avoid repeatedly optimizing only the same tiny panel: broaden, rotate, or confirm evidence when needed. A checkpoint can only be promoted using the canonical evaluation defined in the runbook.

## Baseline first

Before the first research modification:

1. locate the released FastWAM LIBERO checkpoint;
2. verify the environment and evaluation interface;
3. during setup only, run lightweight smoke tests for evaluation, training, and checkpoint reload/resume so the full pipeline is known to work;
4. after the smoke tests pass, evaluate the checkpoint on all five target suites;
5. record the measured baseline in `research/EXPERIMENTS.jsonl`;
6. update `research/STATE.md`.

Smoke tests are setup validation, not research experiments and not performance evidence. Do not rerun them before ordinary experiments. Repeat them only if the environment, repository interface, checkpoint format, or execution path materially changes.

Do not substitute paper numbers for the local baseline. The paper numbers provide project context; the measured run is the comparison point for this machine and evaluation setup.

## Research records

Use three complementary records:

- `research/STATE.md`: compact current working state — accepted checkpoint, current metrics, latest experiment, immediate notes;
- `research/EXPERIMENTS.jsonl`: compact machine-readable ledger;
- `research/progress/PROGRESS_XXXX_*.md`: one detailed permanent report for every baseline and candidate.

Create the detailed report when a candidate is selected and update the same file as training/evaluation progresses. Intermediate checkpoints, progress checks, decisions to continue/extend/stop, and final evaluation stages belong in that report. Finalize it after the candidate-level decision. Crashed experiments also receive a finalized report.

Do not turn `STATE.md` into a historical notebook; keep history in `research/progress/`.

## Git and experiment state

Use Git as the code-state history.

- Create a dedicated research branch during setup.
- Before expensive training, commit or otherwise uniquely identify the exact runnable candidate state.
- Record the parent checkpoint, code commit, configs/overrides, training command, evaluation commands, checkpoint outputs, metrics, and detailed progress-report path for every experiment.
- Poor results, early-stopped runs, and crashes still belong in the experiment ledger.
- Revert or return to an accepted parent when appropriate, but keep the experiment record.

## Checkpoint persistence

Use the Hugging Face model repository `cheikh025/ASR` as durable storage for important research checkpoints. The repository is a backup/persistence target; local storage remains the active working copy.

Preserve checkpoints that would be costly or important to lose, including promoted checkpoints, useful branch checkpoints, and intermediate/resume checkpoints selected for further work. Upload the associated progress report and key configuration when useful for identifying and reproducing the artifact.

Keep local checkpoints after upload. Do not delete a local checkpoint merely because it has been uploaded. If disk space later becomes constrained, prefer removing temporary/reproducible artifacts first. Before deleting any local checkpoint, verify its remote copy exists, record the remote path in the experiment report/state, and avoid deleting the active parent or checkpoint required by an ongoing continuation.

Hugging Face authentication is provided through `HF_TOKEN`. Never print, log, commit, or write the token into research records.

## Hardware and reproducibility

For the baseline and every candidate, capture the actual execution environment used for training/evaluation. Include GPU count/model/memory, NVIDIA driver, CUDA, CPU, RAM, disk availability, OS/kernel, Python, PyTorch and relevant distributed-training package versions, active environment, distributed world size, and a dependency snapshot. Use `research/tools/capture_system_info.py` when possible and preserve the raw snapshot as an artifact.

Never record secrets, tokens, passwords, or credentials.

## Keep context compact

Raw logs stay on disk. In the research state, keep concise evidence:

- exit status;
- runtime and memory when useful;
- checkpoint path;
- important training diagnostics;
- progress-check outcomes;
- suite-level success;
- relevant task-level results;
- concise failure tail for crashes.

Do not repeatedly load full logs when a short extraction is sufficient.

## Decisions

There are two levels of decision.

### Training-control decisions

These may happen repeatedly inside one candidate:

- `CONTINUE_TRAINING`
- `EXTEND_TRAINING`
- `STOP_TRAINING`
- `SELECT_CHECKPOINT`
- `RECHECK_PROGRESS`
- `DIAGNOSE`

These decisions allocate compute. They do not promote a model.

### Candidate-level decisions

When enough evidence has been collected, a candidate is one of:

- `PROMOTE`: better main-line checkpoint;
- `REJECT`: not a better parent for the main line;
- `BRANCH`: useful trade-off or promising direction worth preserving separately;
- `RETEST`: evidence is too noisy or incomplete to decide.

`PROMOTE` requires canonical evaluation. A main-line checkpoint must keep every original suite at or above 90%.

Among valid checkpoints, prefer higher LIBERO-90 success. For near-ties, prefer stronger worst-suite performance, less regression, greater stability, lower unnecessary compute, and simpler implementations.

A suite dropping from a very high baseline to 92% can be acceptable if LIBERO-90 improves substantially. A suite dropping below 90% is not acceptable for main-line promotion.

## Failure handling

A crash does not automatically mean the idea is bad.

Inspect the relevant traceback/log tail. Fix local implementation or environment errors when the intended candidate remains unchanged. If the issue is deeper or repeated repairs fail, use `$investigate-fastwam-problem`.

Avoid spending many runs rescuing an implementation that is clearly not worth the compute.
