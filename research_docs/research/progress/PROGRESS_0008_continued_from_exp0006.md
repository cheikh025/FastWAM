# PROGRESS_0008 — continued_from_exp0006

- **Experiment ID:** 0008_continued_from_exp0006
- **Status:** `RUNNING`
- **Created:** 2026-08-10
- **Updated:** 2026-08-10
- **Parent experiment:** 0006_exp0003_reproducibility_check
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt` (weights-only)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `71fcf4a` (adds new task config `libero_uncond_2cam224_plus90_reweighted_continued_1e-5.yaml`)

## 1. Result at a glance

Retests "does more training help the reweighted (goal5x/long5x) recipe" after experiment 0007's attempt was invalidated by a training-infra bug (LR re-warmup on resume-from-state-directory + `max_steps` override — see `research/NOTES.md`). This time, resumes from exp0006's **weights-only** checkpoint (the same clean mechanism used successfully by every other experiment in this project) with a fresh, correctly-shaped schedule at a lower LR (`1e-5` vs the original `3e-5`), appropriate for further-adapting an already-fine-tuned checkpoint rather than repeating a first-stage fine-tune.

## 2. Research state before experiment

exp0006's step_004000.pt (goal5x/long5x reweighted recipe) is the best-evidenced checkpoint in this lineage: cheap-panel suite results spatial 70%, object 95%, goal 55%, long 30%, libero_90 48% (this specific run; exp0003's original run on the identical recipe scored somewhat higher — spatial 80/object 100/goal 70/long 20/libero_90 68 — with the difference attributed to demonstrated training-run noise, see `PROGRESS_0006`). None of these clear the 90% retention floor on goal/long/90. exp0007 attempted to test whether more training closes this gap but was invalidated by a scheduler bug that caused catastrophic collapse (overall 59.6%->10.8%) unrelated to the recipe itself.

## 3. Candidate design

### Modifications

1. New task config `configs/task/libero_uncond_2cam224_plus90_reweighted_continued_1e-5.yaml`: identical to the reweighted recipe (`data: libero_2cam_plus90_reweighted`, same batch/grad-accum/weight-decay) except `learning_rate: 1e-5` (down from `3e-5`) and `max_steps: 2000` (a moderate additional budget, smaller than exp0007's ill-fated 4000-step extension, to limit compute risk while the "does more training help" question is retested).
2. `resume=<step_004000.pt>` (weights-only file, not a state directory) — deliberately avoids the buggy directory-resume path. This means optimizer momentum/variance and the original schedule position are NOT continued; this is a fresh second-stage fine-tune starting from the exp0006 weights, not a mathematically pure continuation.

### Why this candidate

Directly retests the training-duration question exp0007 was meant to answer, using a mechanism free of the newly-discovered scheduler bug. A lower LR than the original 3e-5 is standard practice for continuing to fine-tune an already-adapted checkpoint (avoids re-disrupting learned structure the way a fresh 3e-5 warmup would).

### What to watch

Same 21-task cheap panel comparison against exp0006's step-4000 result. Given exp0006 already established a real training-run noise floor (~20pp/suite, ~40pp/task), only interpret this result confidently if it moves well beyond that band, and in a consistent direction across multiple suites/tasks rather than a single outlier.

### Initial compute plan

- Initial training budget: `max_steps=2000` (~1.7h at the established ~0.33 step/s throughput).
- Checkpoint/save plan: `save_every=500`, state on ephemeral scratch disk via symlink (validated fix).
- Progress check: consider a cheap panel at step 1000 if training looks healthy and GPU can be freed briefly; otherwise evaluate once at completion (2000 steps is short enough that a single evaluation at the end is reasonable).
- Expected cost: ~1.7h training + ~2min eval.

## 4. Exact code and configuration state

- Git commit: `71fcf4a`
- Git branch: `autoresearch/libero90-v1`
- Working tree: clean after commit (new task config file only; data config unchanged)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_continued_1e-5`
- Config overrides: none beyond the new task config's own values
- Dataset config: `data=libero_2cam_plus90_reweighted` (unchanged — goal 5x, long 5x)
- Optimizer / LR / scheduler: `lr_scheduler_type=cosine`, `learning_rate=1e-5`, fresh warmup+cosine for `max_steps=2000`
- Batch size / gradient accumulation: `batch_size=4`, `gradient_accumulation_steps=4` (unchanged)
- Resume source: `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt` (weights-only)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_continued_1e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt \
    output_dir=./runs/reweighted_libero90_finetune/exp0008_continued \
    wandb.name=exp0008_continued_from_exp0006
  ```
- Start time: 2026-08-10 ~01:50 UTC
- End time: 2026-08-10 03:37 UTC (`max_steps reached step=2000`, clean exit, no crash, no OOM — runtime ~1h47min)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0008_continued/checkpoints/weights/step_002000.pt` (12,041,735,545 bytes ≈ 11.2GB)
- Final training diagnostics at step 2000: `val_loss=0.2415 infer_psnr=25.1012 infer_ssim=0.8189 action_l2=0.0133 action_l1=0.0756 lr=1.00e-07`
- **LR schedule sanity check (critical after exp0007's bug)**: LR correctly decayed to near-zero (`1.00e-07`) by the final step, confirming the weights-only-resume workaround produces a correctly-shaped schedule (no re-warmup repeat).

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/reweighted_libero90_finetune/exp0008_continued/checkpoints/weights/step_002000.pt`
- Reference checkpoint: exp0006's step_004000.pt (the parent)
- Raw results path: `evaluate_results/exp0008_cheap_panel/`

**Suite-level result — CATASTROPHIC COLLAPSE, worse than exp0007:**

| Suite | exp0006 (step 4000) | exp0008 (step 2000, lr=1e-5) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 70.00% | 0.00% | **-70pp** |
| LIBERO-Object | 95.00% | 5.00% | **-90pp** |
| LIBERO-Goal | 55.00% | 10.00% | **-45pp** |
| LIBERO-Long/10 | 30.00% | 0.00% | **-30pp** |
| LIBERO-90 (5-task) | 48.00% | 0.00% | **-48pp** |
| **Overall** | **59.60%** | **3.00%** | **-56.6pp** |

## 8. Comparison and interpretation

**This result rules out the LR-schedule bug as the (sole) explanation for exp0007's collapse.** exp0008 used a correctly-shaped, properly-decaying schedule (verified: `lr` fell to `1.00e-07` by the final step, no re-warmup) and still collapsed just as badly as exp0007 (in fact worse: 3.0% vs 10.8% overall). Internal training-time metrics throughout the run looked unremarkable, not alarming (`val_loss` ranged 0.22-0.46 across the run, final `val_loss=0.2415`, `infer_psnr=25.1`, `infer_ssim=0.82`, `action_l2=0.0133` — all in a similar range to exp0007's own final metrics, which also looked "fine" despite that checkpoint's own catastrophic collapse). This disconnect between reasonable-looking internal training/validation metrics and near-total LIBERO rollout failure, replicated across **two independent attempts** with different resume mechanisms and different LR schedules, points to something more fundamental going wrong specifically when **continuing training from one of this project's own previously-saved checkpoints** (as opposed to the originally-released checkpoint, which every prior successful experiment — 0001, 0003, 0004, 0005, 0006 — resumed from without issue).

**Diagnostic result (fully resolved)**: a targeted 1-step reload/resave/re-eval test (reload exp0006's checkpoint, run exactly 1 training step at `lr≈1e-7`, i.e. essentially zero real optimization, then evaluate on 4 tasks that scored 100% in exp0006's own eval) also produced **total failure (0/5 on all 4 tasks)**. Root cause fully identified via a full 1649-tensor numerical diff: `mixtures.action.action_encoder.{weight,bias}` and `mixtures.action.head.weight` are the only tensors that change drastically (relative diff ~1.3, vs <0.9 for everything else) between the parent checkpoint and the 1-step-resaved checkpoint — exactly the `random_kept_prefixes=['action_encoder.', 'head.']` submodules that training's model construction randomly initializes before `resume=` is applied. Cross-checked against exp0006's own (successful) training log: `action_l2` starts at 0.087 (step 200) and decreases to 0.015 by step 4000 — a from-scratch learning curve, not evidence of a correctly-loaded pretrained head. **Conclusion: `resume=` never actually restores `action_encoder`/`head` for any checkpoint, released or our own — every successful run was silently relearning that small layer from scratch over its own full training budget.** Short/gentle continuations (this experiment's 2000 steps at lr=1e-5, exp0007's schedule-wasted budget, this 1-step diagnostic) simply didn't have enough effective high-LR training time to redo that relearning. Full writeup: `research/NOTES.md` "ROOT CAUSE FOUND".

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** no — catastrophic failure on every suite.
- **Reason:** Result is invalidated pending root-cause diagnosis; not attributable to the `lr=1e-5` design choice itself given the disconnect between healthy training metrics and catastrophic rollout failure, replicated independently of exp0007's LR-schedule bug.
- **Checkpoint/branch to preserve:** uploaded to HF for the record per standing policy; not usable as a working checkpoint.
- **Next main-line parent:** unchanged — still `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (baseline, PROMOTE); exp0006's step_004000.pt remains the best reweighted-recipe candidate (not promotable as-is, retention gate not met).

## 10. What this changes for the next experiment

1. **Confirmed via diagnostic: do not resume training from any of this project's own saved checkpoints** — only `resume=checkpoints/fastwam_release/libero_uncond_2cam224.pt` is currently safe. This is now a hard constraint on candidate design until the underlying bug is fixed (see `research/NOTES.md`).
2. Given both extension attempts failed for infra reasons, exp0006's original step_004000.pt remains the best-evidenced state in this lineage; exp0007/exp0008 are not informative about "does more training help the reweighted recipe."
3. Immediately launched **experiment 0009**: a fresh, single, continuous 8000-step run of the reweighted recipe starting from the originally-released checkpoint (not resuming any intermediate checkpoint), to cleanly retest "does more training help" without the broken resume path — at the cost of repeating the first 4000 steps' compute.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0008_continued/train.log` (live copy at `/tmp/.../scratchpad/train_exp0008.log`)
- final weights checkpoint: `runs/reweighted_libero90_finetune/exp0008_continued/checkpoints/weights/step_002000.pt` (11.2GB, catastrophically degraded — preserved for the record only)
- HF backup: `cheikh025/ASR:rejected/0008_continued_from_exp0006/step_002000.pt` (uploaded and existence-verified 2026-08-10, per standing policy)
- config(s): `configs/data/libero_2cam_plus90_reweighted.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_continued_1e-5.yaml`
- eval results: `evaluate_results/exp0008_cheap_panel/`

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
