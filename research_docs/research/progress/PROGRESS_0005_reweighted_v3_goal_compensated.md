# PROGRESS_0005 — reweighted_v3_goal_compensated

- **Experiment ID:** 0005_reweighted_v3_goal_compensated
- **Status:** `RETEST`
- **Created:** 2026-08-09
- **Updated:** 2026-08-09
- **Parent experiment:** 0004_reweighted_v2_long_boost
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- **Selected candidate checkpoint:** TBD (pending training)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `eb0a53bcd18a96ca46147c8595639ad5e29d20c8`

## 1. Result at a glance

Compensating Goal's oversampling weight (5x->7x, restoring its proportional mix share to ~25.9%, above the ~24.3% level that worked in 0003) **did not restore Goal's performance** — it stayed at 50% panel avg, essentially unchanged from 0004's 45% and far below 0003's 70%, directly contradicting the "Goal's regression was caused by proportional-share dilution" hypothesis that motivated this experiment. Worse, **LIBERO-Object regressed from a perfect 100% (in both 0003 and 0004) to 85%**, despite its own weight never being touched across any of these three experiments. Task 73 (book-in-caddy) swung wildly across the three experiments (40%->20%->100%) with no design change plausibly explaining that trajectory. **Decision: `RETEST`**, not a clean `REJECT` — the evidence pattern (an unweighted suite regressing, a compensated suite not recovering, a task swinging over the full 0-100% range) looks more consistent with **single-run training variance dominating the true effect size** than with a reliable, interpretable signal about this specific data-mix change. Recommend pausing further single-shot weight-tuning iterations and reconsidering evaluation/comparison methodology (larger trial counts and/or repeated-seed training) before continuing.

## 2. Research state before experiment

0004 (rejected) revealed that the training mix behaves as a zero-sum resource allocation: pushing Long's oversampling from 5x to 10x improved Long (panel avg 20%->35%) but caused Goal to regress sharply (70%->45%) even though Goal's own absolute oversampling weight was unchanged — because the total mix grew (8917->10857 episodes), diluting Goal's *proportional* share from ~24.3% (0003, where it worked well) to ~19.9% (0004, where it regressed). Spatial also regressed somewhat (80%->65%), though more mildly and concentrated on a single task, possibly partly run-to-run noise.

## 3. Candidate design

### Modifications

1. New data config `configs/data/libero_2cam_plus90_reweighted_v3.yaml`: **Goal's oversampling raised from 5x to 7x** (433 -> 3031 episodes) to compensate for the larger total mix and restore its proportional share to ~25.9% — above the ~24.3% level that worked in 0003. **Long stays at 10x** (388 -> 3880 episodes, unchanged from 0004, since it showed real dose-response improvement there). Spatial and Object remain at 1x (their regressions in 0004 looked more consistent with single-run noise than a systematic share-sensitivity effect, given Spatial's own share barely moved between 0003/0004 — this experiment does not change them, providing a further data point on whether they regress again in the same direction).
2. New task config `configs/task/libero_uncond_2cam224_plus90_reweighted_v3_3e-5.yaml`: identical to 0003/0004's task config (lr=3e-5, batch_size=4, grad_accum=4, max_steps=4000, save_every=500) — only the data mix changes.
3. Training to be launched with `resume=<released checkpoint>.pt` (fresh branch), and `checkpoints/state` symlinked to the second, ephemeral filesystem (per the disk-management fix validated in 0004) to prevent any repeat of 0003's disk-full crash.

### Why this candidate

This is the most direct test of the hypothesis formed from 0004's zero-sum finding: that Goal's regression was caused by its proportional-share reduction (not some other confound), and that compensating for it directly (rather than reducing Long's weight back down) should let both Long's improvement and Goal's retention coexist. If Goal recovers to ~0003-or-better levels while Long holds its 0004-level gains, this validates the "compensate everyone whose share would otherwise shrink" principle for future mix design. If Goal still doesn't fully recover despite now having a larger absolute and proportional share than in 0003, that would suggest Goal's issue involves something beyond simple proportional dilution (e.g. genuine training-run variance, or an interaction effect specific to how goal and the now-much-larger long interact during training).

### What to watch

- **Primary**: Goal's 4 sentinel tasks — does the panel average recover toward or above 0003's 70%, especially task 5 (100%->20% in 0004, the most severe single-task regression)?
- **Secondary**: does Long hold its 0004-level improvement (~35% avg) now that its weight is unchanged from that experiment?
- **Tertiary**: Spatial — if it regresses again in the same pattern as 0004 (task 8 specifically), that would suggest a real, not noise, sensitivity for Spatial too, worth addressing in a future candidate.
- **Trade-off**: LIBERO-90's mix share drops further (33.4%, down from 36.1% in 0004 and 44.0% in 0003) as the total mix grows again — check tasks 9/24/46 hold their gains; task 73 and 19 have already shown some volatility across 0003/0004 and are worth watching for further movement.

### Initial compute plan

- Initial training budget: `max_steps=4000` (consistent with 0001/0003/0004 for comparability).
- Checkpoint/save plan: `save_every=500`, state checkpoints on the second filesystem via symlink (validated fix from 0004).
- When a progress check might be useful: same approach as 0003/0004 — evaluate only after training reaches a clean stop, using the same 21-task widened panel for direct three/four-way comparison against 0003 and 0004.
- Expected training/evaluation cost: ~4000 steps at ~3s/step (slightly slower given the largest dataset yet, 11723 episodes) ≈ ~3.3-3.5 hours; eval panel ~35 min.

## 4. Exact code and configuration state

- Git commit: `eb0a53bcd18a96ca46147c8595639ad5e29d20c8`
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: clean (committed at `eb0a53b`).
- Files changed:
  - `configs/data/libero_2cam_plus90_reweighted_v3.yaml` (new)
  - `configs/task/libero_uncond_2cam224_plus90_reweighted_v3_3e-5.yaml` (new)
- Training config(s): `task=libero_uncond_2cam224_plus90_reweighted_v3_3e-5`
- Config overrides: `resume=<released checkpoint>`
- Dataset config(s): `data=libero_2cam_plus90_reweighted_v3` (spatial 1x=434, object 1x=457, goal 7x=3031, long 10x=3880, libero_90 1x=3921; total 11723 episodes)
- Sampler/mixing configuration: oversampling via repeated `dataset_dirs` entries, unchanged mechanism, goal's repeat count raised 5->7
- Model/trainable-module configuration: unchanged
- Optimizer / LR / scheduler: unchanged (AdamW, cosine, lr=3e-5)
- Batch size / gradient accumulation / effective batch: unchanged (4 / 4 / 64 global)
- Initial training steps / epochs / budget: `max_steps=4000`
- Checkpoint/save cadence: `save_every=500`; state saves redirected to the second filesystem via symlink
- Random seed(s): default (`seed: 42`) — same seed as all prior experiments, so this is not a repeated-seed noise check, just a mix-composition comparison
- Resume source: `resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` (fresh branch)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5.

### Setup validation (baseline report only)

Not applicable — setup already validated.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_v3_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    output_dir=./runs/reweighted_v3_libero90_finetune/exp0005 \
    wandb.name=exp0005_reweighted_v3_goal_compensated
  ```
- Start time: 2026-08-09 06:39:12 UTC
- End time: 2026-08-09 10:09 UTC (`max_steps reached step=4000`)
- Wall-clock runtime: ~3h30min
- Exit code/status: 0 (clean, no crash — disk-management fix held again, root filesystem cycled normally between checkpoint saves and auto-cleanups)
- Steps completed: 4000/4000
- Training log path: `runs/reweighted_v3_libero90_finetune/exp0005/train.log`
- Disk management: confirmed working correctly a second time (root filesystem oscillated 258-338GB free as expected across 8 save/cleanup cycles; `/workspace` ended at 343GB free / 67% used, no crash). Also fixed a minor bug in the disk-monitor script itself this run: the previous version's self-termination check (`pgrep -f "scripts/train.py"`) matched its own command line (which contained that literal string) and never returned false; this run's monitor checked the actual training PID directly and correctly self-terminated when training completed.

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_004000 (final) | ~3h30min | `candidate_screen` | 21-task panel vs exp0003/exp0004: Goal did not recover (50% vs 0003's 70%), Object regressed unexpectedly (100%->85%), task 73 swung to 100% | `RETEST` — evidence too noisy to trust as a clean signal about this design change | n/a — training already complete |

### Why training ended

Planned completion — reached `max_steps=4000` cleanly.

### Training anomalies

None. Training itself was healthy (matching 0003/0004's pattern of clean loss curves and no divergence) — the surprising result is in the *evaluation* outcome, not a training-process problem.

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/reweighted_v3_libero90_finetune/exp0005/checkpoints/weights/step_004000.pt` (final)
- Decision this evaluation was meant to inform: whether compensating Goal's weight (to counteract the proportional-dilution effect identified in exp0004) restores its retention while keeping Long's gains.
- Exact task subset / suite coverage: identical 21-task widened panel used in exp0003 and exp0004, for direct three-way comparison.
- Trials per task: 5
- Exact command/config: same pattern as 0003/0004, `CKPT=.../exp0005/checkpoints/weights/step_004000.pt`, `OUTPUT_DIR=./evaluate_results/exp0005_cheap_panel`
- Raw results path: `evaluate_results/exp0005_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- Runtime: ~35 min
- Validity checks: 21/21 tasks present, 5/5 episodes each, `failed_tasks.txt` empty, correct checkpoint and this run's own `dataset_stats.json`.

| Suite | Task | Baseline | exp0003 (5x/5x) | exp0004 (5x/10x) | exp0005 (7x/10x) |
|---|---:|---:|---:|---:|---:|
| Spatial | 0 | 98% | 100% | 100% | 100% |
| Spatial | 3 | 96% | 100% | 100% | 100% |
| Spatial | 5 | 94% | 20% | 60% | 40% |
| Spatial | 8 | 96% | 100% | 0% | 0% |
| Object | 0 | 98% | 100% | 100% | **80%** |
| Object | 3 | 100% | 100% | 100% | **60%** |
| Object | 5 | 96% | 100% | 100% | 100% |
| Object | 8 | 100% | 100% | 100% | 100% |
| Goal | 0 | 100% | 0% | 0% | 0% |
| Goal | 3 | 88% | 80% | 60% | 60% |
| Goal | 5 | 100% | 100% | 20% | 40% |
| Goal | 8 | 100% | 100% | 100% | 100% |
| Long | 0 | 94% | 60% | 80% | 80% |
| Long | 3 | 96% | 0% | 20% | 0% |
| Long | 5 | 100% | 20% | 40% | 60% |
| Long | 8 | 92% | 0% | 0% | 0% |
| LIBERO-90 | 24 | 100% | 100% | 100% | 100% |
| LIBERO-90 | 19 | 96% | 40% | 20% | 20% |
| LIBERO-90 | 46 | 70% | 100% | 100% | **40%** |
| LIBERO-90 | 9 | 26% | 60% | 60% | **20%** |
| LIBERO-90 | 73 | 0% | 40% | 20% | **100%** |

- Decision enabled by this evidence: `RETEST` — the evidence does not cleanly support or refute the Goal-compensation hypothesis. Goal did not recover despite a proportional share exceeding 0003's working level, which should have shown clear recovery under the dilution-only theory; simultaneously, Object (completely untouched across all three experiments) regressed, which the dilution theory cannot explain at all. This pattern is more consistent with substantial single-run training variance than a reliable design effect.
- Reason: three consecutive experiments (0003, 0004, 0005) at 5 eval trials/task and single training seeds each have produced a result pattern too volatile to interpret confidently task-by-task. Continuing to tune mix weights one experiment at a time under these conditions risks chasing noise.

### Task-level evidence

**This is the clearest evidence yet that noise, not just design changes, is a major contributor to the observed differences between experiments 0003-0005.** Concrete red flags:
- Object regressed 100%->85% despite zero design change to its own weight or the training recipe across three experiments.
- Goal did not respond to a compensating weight increase that, under the leading hypothesis from exp0004, should have restored it.
- LIBERO-90 task 73 moved 0%(baseline)->40%(0003)->20%(0004)->100%(0005) — a full-range swing with no obvious mechanism tied to the actual design changes (which only touched goal/long weights, not anything specific to book-in-caddy tasks).
- LIBERO-90 tasks 9 and 46, both strong and stable across 0003/0004 (60%/100% both times), dropped sharply in 0005 (20%/40%) despite libero_90's own absolute weight being unchanged across all three experiments.

### Evaluation validity

Confirmed technically valid: 21/21 tasks, 5/5 episodes each, zero failures, correct checkpoint, matching dataset_stats.json. The *technical* execution is not in question — the concern is statistical: n=5 trials/task and n=1 training run/candidate is not enough to reliably separate a true effect from noise at the magnitude of differences being observed here.

## 8. Comparison and interpretation

- **LIBERO-90 change**: volatile across 0003/0004/0005 in ways not obviously tied to the (goal/long-only) design changes — task 73's 0-100% swing and tasks 9/46's unexplained drop both suggest meaningful run-to-run variance affecting even the untouched libero_90 portion of the mix.
- **Original-suite change**: Object's unexpected regression is the single most important data point in this report — it is direct evidence that *something* varies substantially between training runs even when composition for that suite is literally unchanged, undermining confident attribution of any of the goal/spatial/long movements purely to the mix-weight changes being tested.
- **Task-level pattern**: no longer a clean "dose-response" story as it appeared after just 0003->0004. With 0005 added, the pattern looks noisy enough that the earlier "zero-sum dilution" narrative from 0004 should be held more tentatively — it may still be a real, smaller effect, but it is not cleanly demonstrated by this data.
- **Training progression**: training remained numerically healthy in all three experiments; the volatility is in outcomes, not training stability.
- **Retention satisfied**: no, but the more important finding here is methodological, not a retention verdict for this specific checkpoint.

## 9. Decision

- **Decision:** `RETEST`
- **Retention gate passed:** no (Goal 50%, Long 35%, Spatial 60% all fail; Object's 85% also technically fails the floor despite being the "healthy" suite in prior experiments), but this is not the primary conclusion of this report.
- **Reason:** the evidence quality across 0003-0005 is insufficient to confidently attribute suite-level movements to the specific weight changes tested, given an untouched suite (Object) regressed and a compensated suite (Goal) failed to recover as predicted. Recommend addressing evaluation/comparison methodology (see Section 10) before running another single-shot weight-tuning iteration.
- **Checkpoint/branch to preserve:** not preserved as a main-line branch, but uploaded to `cheikh025/ASR` per standing policy (see Section 11).
- **Next main-line parent:** unchanged — `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (0000_baseline remains accepted).

## 10. What this changes for the next experiment

- **Methodological pivot needed before further weight-tuning iterations.** Three single-shot experiments (0003, 0004, 0005) at n=5 eval trials/task and n=1 training seed/candidate have produced a result pattern that cannot be confidently attributed to the design changes being tested, given an untouched suite regressed and a compensated suite failed to respond as predicted. Options to improve signal quality going forward:
  - Increase eval trial count for the cheap panel (e.g. 10-15 instead of 5) to reduce per-task measurement noise, accepting the added cost.
  - Consider a repeated-seed check (same recipe, different training seed) to directly measure how much of the observed 0003->0004->0005 variation is training-run noise vs design-change effect, before trusting any further single-run comparison.
  - Consider whether broader confirmation-level evidence (more tasks per suite, not just the 4-task sentinel sample) would give a more stable picture than continuing to read fine-grained task-level deltas from a 4-task/suite panel.
- **Do not treat exp0004's "zero-sum dilution" narrative as confirmed** — it may still be partially real, but exp0005 does not cleanly support it, and a more careful test (ideally with reduced noise) is needed before building further candidate designs on top of that specific theory.
- **Object's regression is worth understanding on its own** — investigate whether there's a source of run-to-run variance in the training/data pipeline beyond what's expected from ordinary stochastic gradient descent (e.g. dataloader shuffling interacting with the larger dataset size, checkpoint-loading nondeterminism, or something specific to this experiment) before assuming it's pure noise.

## 11. Artifacts

- training log: `runs/reweighted_v3_libero90_finetune/exp0005/train.log`
- intermediate checkpoint(s): `runs/reweighted_v3_libero90_finetune/exp0005/checkpoints/weights/step_{500,...,3500}.pt` retained locally
- selected checkpoint: `runs/reweighted_v3_libero90_finetune/exp0005/checkpoints/weights/step_004000.pt`
- Hugging Face remote checkpoint path: `cheikh025/ASR/rejected/0005_reweighted_v3_goal_compensated/step_004000.pt` — uploaded and verified 2026-08-09
- config(s): `configs/data/libero_2cam_plus90_reweighted_v3.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_v3_3e-5.yaml` (commit `eb0a53b`)
- raw progress-evaluation outputs: none (no real-time progress check)
- raw candidate/confirmation/canonical evaluation outputs: `evaluate_results/exp0005_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- parsed task/suite metrics: table in Section 7
- videos/rollouts: per-episode MP4s under `evaluate_results/exp0005_cheap_panel/<suite>/videos/`
- hardware/software snapshot: unchanged from baseline/0001
- dependency snapshot: unchanged from baseline/0001
- diagnostic scripts/results: none beyond the panel above; the fixed disk-monitor script (checking actual PID rather than a self-matching grep pattern) is a reusable improvement for future runs

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded
- [x] intermediate checkpoints and progress decisions recorded when used
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — canonical not run)
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated

## 11. Artifacts

- training log: TBD
- intermediate checkpoint(s): TBD
- selected checkpoint: TBD
- Hugging Face remote checkpoint path: TBD (standing policy — upload regardless of outcome)
- config(s): `configs/data/libero_2cam_plus90_reweighted_v3.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_v3_3e-5.yaml`
- raw progress-evaluation outputs: TBD
- raw candidate/confirmation/canonical evaluation outputs: TBD
- parsed task/suite metrics: TBD
- videos/rollouts: TBD
- hardware/software snapshot: unchanged from baseline
- dependency snapshot: unchanged from baseline
- diagnostic scripts/results: none yet

## 12. Reproducibility checklist

- [ ] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded (config side; exact launch command to be recorded by `$run-fastwam-training`)
- [ ] hardware/software environment recorded (unchanged, will confirm no drift at launch)
- [ ] intermediate checkpoints and progress decisions recorded when used
- [ ] logs and checkpoint paths recorded
- [ ] remote checkpoint path recorded and verified if an HF backup was created
- [ ] every evaluation event has purpose/settings/raw results recorded
- [ ] five-suite metrics recorded when canonical evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision and reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
