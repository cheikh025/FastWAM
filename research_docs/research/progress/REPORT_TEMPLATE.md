# PROGRESS_XXXX — <candidate name>

- **Experiment ID:**
- **Status:** `PLANNED | RUNNING | EVALUATING | PROMOTE | REJECT | BRANCH | RETEST | CRASH`
- **Created:**
- **Updated:**
- **Parent experiment:**
- **Parent checkpoint:**
- **Selected candidate checkpoint:**
- **Git branch:**
- **Git commit:**

## 1. Result at a glance

Summarize the candidate, the strongest evidence collected, how training was controlled, and the final decision. Keep this short enough to scan later.

## 2. Research state before experiment

Record the accepted five-suite metrics and the specific weakness/opportunity that motivated this candidate. Include task-level evidence when it materially influenced the choice.

## 3. Candidate design

### Modifications

List every intentional modification. A candidate may contain several related changes.

1. ...

### Why this candidate

Explain why this candidate was worth the compute based on current evidence, code, or relevant research.

### What to watch

State the result patterns that matter most: LIBERO-90 gain, particular tasks, retention risk, stability, etc.

### Initial compute plan

- Initial training budget:
- Checkpoint/save plan:
- When a progress check might be useful:
- Expected training/evaluation cost:

This is a plan, not a promise to consume the whole budget.

## 4. Exact code and configuration state

- Git commit:
- Git branch:
- Working tree clean/dirty before launch:
- Files changed:
- Diff summary / `git diff --stat`:
- Training config(s):
- Config overrides:
- Dataset config(s):
- Sampler/mixing configuration:
- Model/trainable-module configuration:
- Optimizer / LR / scheduler:
- Batch size / gradient accumulation / effective batch:
- Initial training steps / epochs / budget:
- Checkpoint/save cadence:
- Random seed(s), if controlled:
- Resume source:

Record exact values. Do not merely say "same as before" for settings important to reproduction; repeat them or point to an immutable parent report/config file.

## 5. Hardware and software environment

### GPU

- GPU count:
- GPU model(s):
- Memory per GPU:
- NVIDIA driver:
- CUDA runtime/toolkit:
- `CUDA_VISIBLE_DEVICES` / world size:

### CPU / RAM / storage

- CPU model:
- Logical CPU count:
- System RAM:
- Relevant disk total/free before run:

### Software

- OS / kernel:
- Python executable and version:
- PyTorch version:
- PyTorch CUDA version:
- cuDNN version, if available:
- DeepSpeed version, if used:
- Accelerate version, if used:
- FastWAM repository commit:
- Environment/venv/conda identifier:
- Dependency snapshot path:

Use `research/tools/capture_system_info.py` when possible, then summarize its output here and preserve the raw snapshot path. Do not record secrets, access tokens, or credentials.

### Setup validation (baseline report only)

For `PROGRESS_0000_BASELINE.md`, record the one-time setup smoke tests:

- evaluation smoke test command/result/artifacts;
- training smoke test command/result/artifacts;
- checkpoint reload/resume smoke test command/result/artifacts.

For ordinary experiment reports: `not applicable — setup already validated` unless infrastructure materially changed.

## 6. Training execution and control timeline

- Exact launch command:
- Start time:
- End time:
- Wall-clock runtime:
- Exit code/status:
- Number of GPUs / distributed world size:
- Steps completed:
- Throughput / step time, if available:
- Peak GPU memory, if available:
- Important training diagnostics / losses:
- Training log path:
- System snapshot path:
- Dependency snapshot path:

### Intermediate checkpoints and progress decisions

Add one row for every checkpoint that materially affected compute allocation. Add rows as the run progresses.

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| | | `progress_check` | | `CONTINUE_TRAINING / EXTEND_TRAINING / STOP_TRAINING / SELECT_CHECKPOINT / RECHECK_PROGRESS / DIAGNOSE` | |

If no intermediate progress evaluation was useful, state why the run was short/cheap enough to train directly to completion.

### Why training ended

Record whether the run ended by planned completion, evidence-based early stop, extension completion, selected intermediate checkpoint, manual resource limit, or crash.

### Training anomalies

Record stalls, OOMs, NaNs, restarts, data-loader problems, unexpected warnings, or other events that affect interpretation. For crashes, include a concise relevant traceback/log tail and preserve the full log path.

## 7. Evaluation events

Evaluation is adaptive. Add an event for every evaluation that influenced a decision. Do not force every type to appear.

### Event <N> — `<progress_check | candidate_screen | confirmation | diagnostic | canonical>`

- Checkpoint / training step:
- Decision this evaluation was meant to inform:
- Exact task subset / suite coverage:
- Why this panel was used:
- Parent/reference checkpoint and matching result:
- Candidate result:
- Trials per task:
- Exact command/config:
- Raw results path:
- Runtime:
- Validity checks:
- Decision enabled by this evidence:
- Reason:

Repeat this subsection as needed.

### Canonical promotion evaluation

Fill only if canonical evaluation ran.

- Exact evaluation command/config:
- Trial count per task:
- Task membership:
- Seed / initial-state procedure:
- Episode horizon:
- Action/replan settings:
- Raw results path:
- Evaluation runtime:

| Suite | Parent | Candidate | Delta | Constraint |
|---|---:|---:|---:|---|
| LIBERO-90 | | | | target >=90% |
| LIBERO-Spatial | | | | >=90% |
| LIBERO-Object | | | | >=90% |
| LIBERO-Goal | | | | >=90% |
| LIBERO-Long / LIBERO-10 | | | | >=90% |

### Task-level evidence

Record important per-task results, failure clusters, or regressions from any evaluation event. If tables are large, preserve them as artifacts and summarize the important patterns here.

### Evaluation validity

Confirm requested tasks completed, checkpoint identity was correct, direct comparisons used matching settings, duplicates/missing tasks were handled, and aggregation matched `research/RUNBOOK.md`. Note any uncertainty.

## 8. Comparison and interpretation

Explain what changed relative to the parent/accepted checkpoint and across the training trajectory. Did more training help, plateau, or hurt? Did an earlier checkpoint outperform a later one? For multi-modification candidates, do not attribute causality to individual components unless evidence isolates them. Distinguish measured evidence from interpretation.

## 9. Decision

- **Decision:** `PROMOTE | REJECT | BRANCH | RETEST | CRASH`
- **Retention gate passed:** yes/no/not evaluated
- **Reason:**
- **Checkpoint/branch to preserve:**
- **Next main-line parent:**

## 10. What this changes for the next experiment

Record the most useful lessons, remaining weaknesses, and one or more promising next directions. This is not a requirement to choose the next candidate immediately.

## 11. Artifacts

List exact paths for:

- training log;
- intermediate checkpoint(s);
- selected checkpoint;
- Hugging Face remote checkpoint path and upload verification, if persisted;
- config(s);
- raw progress-evaluation outputs;
- raw candidate/confirmation/canonical evaluation outputs;
- parsed task/suite metrics;
- videos/rollouts used for diagnosis, if any;
- hardware/software snapshot;
- dependency snapshot;
- diagnostic scripts/results created specifically for this experiment.

## 12. Reproducibility checklist

- [ ] exact candidate commit recorded
- [ ] parent checkpoint recorded
- [ ] intentional modifications listed
- [ ] initial training plan recorded
- [ ] training configuration and command recorded
- [ ] hardware/software environment recorded
- [ ] intermediate checkpoints and progress decisions recorded when used
- [ ] logs and checkpoint paths recorded
- [ ] remote checkpoint path recorded and verified if an HF backup was created
- [ ] every evaluation event has purpose/settings/raw results recorded
- [ ] five-suite metrics recorded when canonical evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision and reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
