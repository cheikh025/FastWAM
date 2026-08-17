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

Summarize the candidate, strongest RoboTwin evidence, LIBERO retention evidence, training-control decisions, and final candidate-level decision.

## 2. Research state before experiment

### Accepted RoboTwin state

- canonical metric(s):
- important weak tasks/difficulties:

### Accepted LIBERO retention state

| Suite | Success |
|---|---:|
| LIBERO-90 | |
| LIBERO-Spatial | |
| LIBERO-Object | |
| LIBERO-Goal | |
| LIBERO-Long / LIBERO-10 | |

Record the specific weakness/opportunity motivating the candidate. Include task-level evidence when material.

For `PROGRESS_0000_PARENT_BASELINE.md`, distinguish inherited exp0019 canonical metrics from fresh-machine validation.

## 3. Candidate design

### Modifications

List every intentional modification. A candidate may contain several related changes.

1. ...

### Why this candidate

Explain why the candidate is worth compute based on current evidence, code, data, or relevant literature.

### Multi-embodiment representation/configuration

Record when relevant:

- shared action dimension:
- LIBERO valid channels/mask:
- RoboTwin valid channels/mask:
- normalization/statistics behavior:
- checkpoint projection expansion/initialization:
- embodiment/control conditioning:
- camera/observation handling:
- inference slicing/decoding:

### Data and learning strategy

- LIBERO datasets/tasks:
- RoboTwin datasets/tasks:
- sampling/mixing ratios:
- per-task/per-dataset weights:
- replay/rehearsal strategy:
- loss weights:
- retention/distillation/regularization:
- trainable/frozen modules:

### What to watch

State the patterns that determine next action: RoboTwin gain, weak RoboTwin tasks/difficulties, LIBERO forgetting risk, stability, loss/mask correctness, etc.

### Initial compute plan

- initial training budget:
- checkpoint/save plan:
- when a progress check might be useful:
- expected training/evaluation cost:

This is a plan, not a promise to consume the whole budget.

## 4. Exact code and configuration state

- Git commit:
- Git branch:
- parent code commit:
- working tree clean/dirty before launch:
- files changed:
- diff summary / `git diff --stat`:
- training config(s):
- config overrides:
- LIBERO dataset config(s):
- RoboTwin dataset config(s):
- sampler/mixing configuration:
- action/state normalization configuration:
- action validity-mask configuration:
- model/trainable-module configuration:
- optimizer / LR / scheduler:
- batch size / gradient accumulation / effective batch:
- initial training steps / epochs / budget:
- checkpoint/save cadence:
- random seed(s), if controlled:
- resume source and resume type:

Record exact values. Do not merely say "same as before" for reproduction-critical settings.

## 5. Hardware and software environment

### GPU

- GPU count:
- GPU model(s):
- memory per GPU:
- NVIDIA driver:
- CUDA runtime/toolkit:
- `CUDA_VISIBLE_DEVICES` / world size:

### CPU / RAM / storage

- CPU model:
- logical CPU count:
- system RAM:
- relevant disk total/free before run:

### Software

- OS / kernel:
- Python executable/version:
- PyTorch version:
- PyTorch CUDA version:
- cuDNN version, if available:
- DeepSpeed version, if used:
- Accelerate version, if used:
- FastWAM repository commit:
- RoboTwin repository/revision:
- LIBERO repository/revision:
- environment identifier:
- dependency snapshot path:

Use `research/tools/capture_system_info.py` when possible and preserve raw snapshots. Never record secrets.

### Setup validation (`PROGRESS_0000_PARENT_BASELINE.md` only)

Record:

- exact parent checkpoint identity/download verification;
- parent commit/branch provenance verification;
- creation/activation of `autoresearch/robotwin-multiembodiment-v1` before tracked changes;
- Qwen-VLA + relevant literature review completion;
- verified LIBERO/RoboTwin action/state/normalization semantics;
- LIBERO evaluation smoke/sentinel command/result/artifacts;
- RoboTwin evaluator/environment smoke/reference command/result/artifacts;
- training smoke command/result/artifacts;
- checkpoint reload/resume smoke command/result/artifacts.

For ordinary experiment reports: `not applicable — setup already validated` unless infrastructure materially changed.

## 6. Training execution and control timeline

- exact launch command:
- start time:
- end time:
- wall-clock runtime:
- exit code/status:
- number of GPUs/world size:
- steps completed:
- throughput/step time:
- peak GPU memory:
- important losses/diagnostics:
- RoboTwin vs LIBERO batch/loss diagnostics if available:
- training log path:
- system snapshot path:
- dependency snapshot path:

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Runtime so far | Eval purpose | RoboTwin evidence | LIBERO retention evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|---|
| | | `progress_check` | | | `CONTINUE_TRAINING / EXTEND_TRAINING / STOP_TRAINING / SELECT_CHECKPOINT / RECHECK_PROGRESS / DIAGNOSE` | |

If no intermediate evaluation was useful, state why.

### Why training ended

Record planned completion, evidence-based early stop, extension completion, selected earlier checkpoint, resource limit, or crash.

### Training anomalies

Record OOMs, NaNs, stalls, restarts, data-loader problems, mask/action-dimension issues, simulator/data errors, unexpected warnings, or other interpretation-relevant events.

## 7. Evaluation events

Add an event for every evaluation that influenced a decision. Do not force every type to appear.

### Event <N> — `<progress_check | candidate_screen | confirmation | diagnostic | canonical>`

- benchmark: `robotwin | libero | joint`
- checkpoint / training step:
- decision this evaluation was meant to inform:
- exact task/suite/difficulty coverage:
- why this panel was used:
- parent/reference checkpoint and matching result:
- candidate result:
- trials/episodes per task:
- exact command/config:
- raw results path:
- runtime:
- validity checks:
- decision enabled by this evidence:
- reason:

Repeat as needed.

### Canonical RoboTwin promotion evaluation

Fill only if canonical RoboTwin evaluation ran.

- exact command/config:
- task set:
- difficulty/domain-randomization split(s):
- episodes/trials per task:
- seed/scene/randomization procedure:
- instruction mode:
- episode horizon:
- action/replan settings:
- raw results path:
- runtime:

| RoboTwin metric | Parent | Candidate | Delta |
|---|---:|---:|---:|
| <canonical metric 1> | | | |
| <canonical metric 2 if applicable> | | | |

### Canonical LIBERO retention evaluation

Fill only if full canonical LIBERO evaluation ran.

- exact command/config:
- trial count per task:
- task membership:
- seed/initial-state procedure:
- episode horizon:
- action/replan settings:
- raw results path:
- runtime:

| Suite | Parent | Candidate | Delta | Constraint |
|---|---:|---:|---:|---|
| LIBERO-90 | | | | >=90% |
| LIBERO-Spatial | | | | >=90% |
| LIBERO-Object | | | | >=90% |
| LIBERO-Goal | | | | >=90% |
| LIBERO-Long / LIBERO-10 | | | | >=90% |

### Task-level evidence

Record important RoboTwin/LIBERO per-task failure clusters or regressions. Preserve large tables as artifacts and summarize patterns here.

### Evaluation validity

Confirm requested tasks completed, checkpoint identity was correct, comparison settings matched, duplicates/missing tasks were handled, and aggregation matched `research/RUNBOOK.md`.

## 8. Comparison and interpretation

Explain what changed relative to the accepted checkpoint across RoboTwin and LIBERO. Did more training help, plateau, or hurt? Was forgetting localized or broad? Did an earlier checkpoint offer a better trade-off? For multi-modification candidates, do not over-attribute causality unless isolated.

## 9. Decision

- **Decision:** `PROMOTE | REJECT | BRANCH | RETEST | CRASH`
- **Canonical RoboTwin evidence available:** yes/no
- **All five LIBERO >=90% canonical:** yes/no/not evaluated
- **Reason:**
- **Checkpoint/branch to preserve:**
- **Next main-line parent:**

## 10. What this changes for the next experiment

Record reusable lessons, remaining RoboTwin weaknesses, LIBERO retention risks, and promising next directions.

## 11. Artifacts

List exact paths for:

- training log;
- intermediate checkpoints;
- selected checkpoint;
- HF remote checkpoint path + verification if persisted;
- configs;
- action/mask/normalization implementation tests;
- raw RoboTwin evaluations;
- raw LIBERO evaluations;
- parsed task/suite metrics;
- videos/rollouts used for diagnosis;
- hardware/software snapshot;
- dependency snapshot;
- diagnostic scripts/results.

## 12. Reproducibility checklist

- [ ] exact candidate commit and RoboTwin branch recorded
- [ ] parent checkpoint recorded
- [ ] intentional modifications listed
- [ ] multi-embodiment representation/mask/normalization details recorded when relevant
- [ ] dataset mixture and retention strategy recorded
- [ ] initial training plan recorded
- [ ] training configuration/command recorded
- [ ] hardware/software environment recorded
- [ ] intermediate checkpoints/progress decisions recorded when used
- [ ] logs/checkpoint paths recorded
- [ ] remote checkpoint path verified if HF backup created
- [ ] every evaluation event has benchmark/purpose/settings/raw results
- [ ] canonical RoboTwin metrics recorded when canonical evaluation ran
- [ ] five LIBERO suite metrics recorded when canonical retention evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision/reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
