# PROGRESS_0007 — extended_reweighted_v1

- **Experiment ID:** 0007_extended_reweighted_v1
- **Status:** `RUNNING`
- **Created:** 2026-08-09
- **Updated:** 2026-08-09
- **Parent experiment:** 0006_exp0003_reproducibility_check
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/state/step_004000` (full accelerate/deepspeed state, global_step=4000)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `eb0a53bcd18a96ca46147c8595639ad5e29d20c8` (same as exp0003/0006 — no config changes, only `max_steps` override and `resume=` pointed at a state dir)

## 1. Result at a glance

Extends exp0006's training (goal 5x / long 5x reweighted mix, same recipe as exp0003) from step 4000 to step 8000 by resuming its full optimizer/scheduler state, to test whether LIBERO-Goal/Long/90 continue improving with more training time on this specific recipe — a question raised but not directly tested in the 0003-0006 series (exp0002 tested "more training helps" only for the original unweighted 4-suite+90 mix, not the reweighted one).

## 2. Research state before experiment

exp0006 (exact rerun of exp0003's recipe, different seed-realization due to distributed non-determinism) just completed with cheap-panel (21-task, 5 trials/task) suite results: spatial 70%, object 95%, goal 55%, long(10) 30%, libero_90 48%, overall 59.6% — versus exp0003's original run on the identical recipe: spatial 80%, object 100%, goal 70%, long 20%, libero_90 68%, overall 67.6%. Comparing per-task, **11 of 21 tasks differ between the two identical-seed reruns**, with several suite-level swings of 10-20pp and task-level swings up to 40pp (see `research/NOTES.md` and `PROGRESS_0006` for the full comparison table) — establishing that training-side run-to-run noise is large for this pipeline (no explicit determinism flags are set; multi-GPU NCCL/cuDNN algorithm selection and dataloader ordering are the likely source). Only `libero_goal` task 0 (0/5 in both runs) reproduced exactly, reinforcing that it is a genuine systematic failure (see NOTES.md diagnostic on this task — likely instruction/scene interference between the "middle drawer" instruction and LIBERO-90's heavily-oversampled "top/bottom drawer" tasks).

Neither exp0006 run individually clears the 90% retention floor on goal/long/90 — this experiment tests whether more training on the same recipe moves those suites closer, independent of the reweighting question the earlier series was trying to answer.

## 3. Candidate design

### Modifications

1. No recipe change. Resume exp0006's full training state (`checkpoints/state/step_004000`, optimizer + scheduler + step counter) and continue to `max_steps=8000` (4000 additional steps) in a fresh output directory (`runs/reweighted_libero90_finetune/exp0007_extended/`).

### Why this candidate

exp0001's own diagnostics showed suite recovery from the LIBERO-90 stability-gap collapse is non-uniform and can be late (Object recovered specifically between step 2000-4000, not gradually from the start). The reweighted recipe (exp0003-family) has never been trained past 4000 steps. Given goal/long/90 are still well below the 90% retention floor at step 4000 on this recipe, extending training is the cheapest way to test whether they are still on an improving trajectory (worth more budget) or have plateaued (worth abandoning this recipe's step budget and trying a different design lever).

### What to watch

A progress-check panel at an intermediate step (e.g. step 6000) compared against exp0006's own step-4000 result — but given exp0006 just demonstrated a large run-to-run noise floor, any observed single-panel delta must be interpreted cautiously: treat per-task deltas under ~20-30pp and suite-level deltas under ~10-15pp as within the demonstrated noise band, not as confident signal. Look for a directional pattern across multiple suites/tasks rather than a single task's movement.

### Initial compute plan

- Initial training budget: extend to `max_steps=8000` (4000 more steps, matching the original run's cost, ~3.5h more).
- Checkpoint/save plan: `save_every=500`, full state on the ephemeral scratch disk via symlink (validated fix), weights-only `.pt` in `runs/.../exp0007_extended/checkpoints/weights/`.
- Progress check: plan to run the same 21-task cheap panel around step 6000 if training is healthy, to decide continue/stop before committing the full remaining budget.
- Expected cost: ~3.5h training + ~15-20min per progress-check eval.

## 4. Exact code and configuration state

- Git commit: `eb0a53bcd18a96ca46147c8595639ad5e29d20c8` (unchanged)
- Git branch: `autoresearch/libero90-v1`
- Working tree: clean, no file changes (pure resume + max_steps override)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5` (goal 5x, long 5x mix; lr=3e-5, batch_size=4, grad_accum=4)
- Config overrides: `max_steps=8000`, `resume=/home/claudeuser/fastwam_state_scratch/exp0006_rerun/state/step_004000`, `output_dir=./runs/reweighted_libero90_finetune/exp0007_extended`
- Random seed: `seed=42` (unchanged; irrelevant to reproducibility now that non-determinism is confirmed regardless of seed)
- Resume source: exp0006's full accelerate/deepspeed state at step 4000 (optimizer + scheduler + step counter restored; NOTE — the LR scheduler is reconstructed for `remaining_steps = max_steps - global_step` at resume time, i.e. a fresh cosine decay over the new 4000-step extension window, not a continuation of the original 4000-step cosine curve; same mechanism previously used successfully in exp0002)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=/home/claudeuser/fastwam_state_scratch/exp0006_rerun/state/step_004000 \
    max_steps=8000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0007_extended \
    wandb.name=exp0007_extended_from_exp0006
  ```
- Start time: 2026-08-09 ~21:56 UTC
- End time: 2026-08-10 01:26 UTC (`max_steps reached step=8000`, clean exit, no crash, no OOM, no disk-full warning — extension runtime ~3h30min, consistent with the original 4000-step run's pace)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0007_extended/checkpoints/weights/step_008000.pt` (12,041,735,545 bytes ≈ 11.2GB)
- Final training diagnostics at step 8000: `val_loss=0.1851 infer_psnr=25.2663 infer_ssim=0.8420 action_l2=0.0069 action_l1=0.0564` (action_l2/l1 continued decreasing from step-4000 levels, video metrics roughly stable — consistent with continued useful learning, not divergence)
- Disk management: symlinked-state + auto-cleanup monitor ran the full duration with no warnings.

### Intermediate checkpoints and progress decisions

Planned progress check around step 6000 was reconsidered as training approached that point: an eval panel cannot run concurrently with training (GPU memory fully saturated under ZeRO-1, confirmed earlier this project), so a step-6000 progress check would require pausing/interrupting the run. With training progressing smoothly (no crashes, no anomalies, loss decreasing normally) and only ~2000 steps (~1.5-2h) remaining to the planned step-8000 completion, the cost of interrupting outweighs the value of an early progress check at this late stage — decision: `CONTINUE_TRAINING` to planned completion (step 8000), run the cheap panel once at final completion instead of mid-flight.

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/reweighted_libero90_finetune/exp0007_extended/checkpoints/weights/step_008000.pt`
- Decision this evaluation was meant to inform: does more training on the reweighted recipe (beyond step 4000) improve goal/long/90 retention
- Exact task subset: same 21-task widened panel (`libero_90:24,19,46,9,73`; `libero_spatial:0,3,5,8`; `libero_object:0,3,5,8`; `libero_goal:0,3,5,8`; `libero_10:0,3,5,8`)
- Trials per task: 5
- Reference checkpoint: exp0006's step_004000.pt (the parent of this extension)
- Raw results path: `evaluate_results/exp0007_cheap_panel/`

**Suite-level result — CATASTROPHIC COLLAPSE, not improvement:**

| Suite | exp0006 (step 4000) | exp0007 (step 8000) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 70.00% | 5.00% | **-65pp** |
| LIBERO-Object | 95.00% | 10.00% | **-85pp** |
| LIBERO-Goal | 55.00% | 15.00% | **-40pp** |
| LIBERO-Long/10 | 30.00% | 0.00% | **-30pp** |
| LIBERO-90 (5-task) | 48.00% | 24.00% | -24pp |
| **Overall** | **59.60%** | **10.80%** | **-49pp** |

This is far beyond the training-run noise floor quantified in `PROGRESS_0006` (max ~20pp/suite, ~40pp/task) — this is total model breakdown, not run-to-run variance.

## 8. Comparison and interpretation

**Root cause identified: a learning-rate scheduler bug in the resume-from-full-state-directory + `max_steps` override pattern.** Extracting `lr=` from the training log across the full extension shows a smooth, monotonic climb from `lr=3.01e-07` at step 4010 up to the full original peak `lr=3.00e-05` by step 7710-7860, staying at peak for the rest of training and only barely beginning to tick down (`2.99e-05`) in the final ~150 steps:

```
step=4010  lr=3.01e-07   (near-zero, effectively a fresh warmup start)
step=5010  lr=5.18e-06
step=6010  lr=1.65e-05
step=7010  lr=2.69e-05
step=7710  lr=3.00e-05   (= the ORIGINAL peak LR from the first 4000 steps)
step=7960  lr=2.99e-05   (still ~peak, only just starting to decay)
```

`trainer.py`'s `__init__` always builds a **fresh** `SequentialLR` (warmup + cosine) sized for the new `max_steps` (8000) regardless of the checkpoint's resumed `global_step` (4000). When `accelerator.load_state()` restores the old scheduler's internal step counter into this differently-shaped new scheduler object, the practical effect is that nearly the entire 4000-step extension is spent **re-warming the learning rate from near-zero back up to the original peak (3e-5)** — exactly the opposite of continuing the intended decay. The model, which had already converged to a good low-LR optimum at step 4000, was then trained for ~3700 steps at a rapidly-increasing and ultimately peak-level learning rate, catastrophically disrupting the converged weights. This fully explains the -49pp overall collapse, dwarfing anything attributable to "more training on this recipe doesn't help."

**This candidate's result says nothing about whether more training helps the reweighted recipe** — it is purely an artifact of the broken LR continuation. The `resume=<state_dir>` + `max_steps=<new_total>` pattern (also used by exp0002 to extend exp0001) should not be trusted until fixed; exp0002's "extending training helped" conclusion should be treated with new skepticism and revisited if it becomes decision-relevant again.

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** no — catastrophic failure on every suite, but for a root-caused infrastructure reason, not a property of the design.
- **Reason:** Candidate outcome is invalidated by a training-infrastructure bug (LR re-warmup on resume-from-state + max_steps extension), not a real signal about the reweighted recipe's response to additional training. exp0006's step_004000.pt remains the best-evidenced checkpoint in this lineage.
- **Checkpoint/branch to preserve:** uploaded to HF for the record per standing policy; not usable as a working checkpoint (catastrophically degraded).
- **Next main-line parent:** unchanged — still `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (baseline, PROMOTE); exp0006's step_004000.pt remains the best reweighted-recipe candidate, not yet promoted (fails retention gate as-is).

## 10. What this changes for the next experiment

1. **Critical infra fix needed**: the `resume=<state_dir>` + `max_steps=<new_total>` extension pattern is broken (see NOTES.md for the full writeup) — either fix `_build_scheduler`/`_resume_or_load_checkpoint` to correctly continue a decayed schedule (e.g., construct the scheduler for `remaining_steps = max_steps - global_step` with `warmup_steps=0` when resuming past a nonzero `global_step`), or avoid this resume pattern entirely.
2. **Immediately launched a corrected continuation as experiment 0008**: instead of resuming exp0006's full state directory, resume from its **weights-only checkpoint** (`step_004000.pt`, the same clean mechanism already used successfully by every prior experiment 0001/0003/0004/0005/0006) with a fresh, properly-shaped schedule (`lr=1e-5`, `max_steps=2000`, new task config `libero_uncond_2cam224_plus90_reweighted_continued_1e-5.yaml`) — this cleanly tests "does further training from this checkpoint help" without the scheduler bug, at a lower LR appropriate for continuing an already-adapted checkpoint rather than re-running a full first-stage fine-tune.
3. Do not reuse the resume-from-state-directory extension pattern for any future experiment until the underlying bug is fixed and verified.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0007_extended/train.log` (full copy at `/tmp/.../scratchpad/train_exp0007.log`)
- final weights checkpoint: `runs/reweighted_libero90_finetune/exp0007_extended/checkpoints/weights/step_008000.pt` (11.2GB, catastrophically degraded — preserved for the record only)
- HF backup: `cheikh025/ASR:rejected/0007_extended_reweighted_v1/step_008000.pt` (uploaded and existence-verified 2026-08-10, per standing policy)
- config(s): unchanged (`configs/data/libero_2cam_plus90_reweighted.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml`)
- eval results: `evaluate_results/exp0007_cheap_panel/`
- root-cause evidence: LR-vs-step trace extracted from `train_exp0007.log`, reproduced in Section 8 above

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded (unchanged from baseline, no drift)
- [x] logs and checkpoint paths recorded
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] final decision and reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated (next step)
- [ ] `research/STATE.md` updated (next step)
