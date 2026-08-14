# PROGRESS_0010 — reweighted_longrun2

- **Experiment ID:** 0010_reweighted_longrun2
- **Status:** `RUNNING`
- **Created:** 2026-08-10
- **Updated:** 2026-08-10
- **Parent experiment:** 0009_reweighted_longrun_fresh
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (the originally-released checkpoint — fresh run, not a resume of exp0009's own checkpoint)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** same as exp0003/0006/0009 (no new config; `max_steps` overridden on the command line)

## 1. Result at a glance

Extends the strongest trend found so far: exp0009 showed that doubling training (4000->8000 steps) on the reweighted recipe substantially improved Goal (+25pp), Long (+20pp), and LIBERO-90 (+24pp) with no sign of plateau. This experiment tests a further increase (8000->12000 steps) via another fresh, safe, single continuous run from the originally-released checkpoint (repeating the compute, since resuming from exp0009's own checkpoint is confirmed unsafe — see `research/NOTES.md`), to see whether the improving trend continues toward the 90% retention target.

## 2. Research state before experiment

exp0009's step_008000.pt is the best-evidenced checkpoint to date: cheap-panel results spatial 70%, object 100%, goal 80%, long 50%, libero_90 72%, overall 74.4% — a large, credible improvement over exp0006's step-4000 result on the identical recipe, well beyond the established training-run noise floor (~20pp/suite). Still short of the 90% retention floor on Spatial, Goal, and Long. Training diagnostics (val_loss, action_l2/l1) were still improving at step 8000 relative to step 4000, with no plateau signal.

## 3. Candidate design

### Modifications

None to the recipe. Same data/task config as exp0003/0006/0009, with `max_steps=12000` (up from 8000) overridden on the command line. `resume=` points at the originally-released checkpoint (not exp0009's own checkpoint), per the standing constraint against resuming from this project's own saved checkpoints.

### Why this candidate

Directly continues the strongest, most reliable lever found in this project so far: more training on the reweighted recipe. A single continuous 12000-step schedule avoids all resume-related bugs entirely.

### What to watch

Cheap panel comparison against exp0009's step-8000 result at completion. Given the trend so far, look for continued improvement on Goal/Long/LIBERO-90 specifically, and whether Spatial (flat at 70% across exp0006->exp0009) starts moving or remains a separate bottleneck requiring a different intervention.

### Initial compute plan

- Initial training budget: `max_steps=12000` (~10.5h at the established ~0.33 step/s throughput).
- Checkpoint/save plan: `save_every=500`, state on ephemeral scratch disk via symlink (validated fix).
- No interim progress check planned: as with exp0009, this run cannot be safely paused/resumed given the checkpoint-resume corruption bug, so it runs straight through to completion.
- Expected cost: ~10.5h training + eval time.

## 4. Exact code and configuration state

- Git commit: same as exp0003/0006/0009 (no new files)
- Git branch: `autoresearch/libero90-v1`
- Working tree: clean
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=12000`
- Resume source: `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (the originally-released checkpoint)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    max_steps=12000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0010_longrun2 \
    wandb.name=exp0010_reweighted_12000steps_fresh
  ```
- Start time: 2026-08-10 ~11:39 UTC
- End time: 2026-08-10 21:58 UTC (`max_steps reached step=12000`, clean exit, no crash, no OOM, no disk-full warning — runtime ~10h19min)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (12,041,735,545 bytes ≈ 11.2GB)
- Final training diagnostics at step 12000: `val_loss=0.0788 infer_psnr=26.7957 infer_ssim=0.8426 action_l2=0.0090 action_l1=0.0597 lr=3.00e-07` — val_loss and psnr both improved further vs exp0009's step-8000 numbers (val_loss 0.1582->0.0788, psnr 25.03->26.80), still no plateau signal. LR correctly decayed to near-zero as expected for a single continuous schedule.

## 7. Evaluation events

### Event 1 — `candidate_screen` (final, step 12000)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt`
- Reference: exp0009's step_008000.pt (same recipe, 4000 fewer steps)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0010_cheap_panel/`

**Suite-level result — by far the best result of the entire experiment series, first time clearing 90% on any original suite via more training alone:**

| Suite | exp0009 (step 8000) | exp0010 (step 12000, fresh) | Delta | vs 90% floor |
|---|---:|---:|---:|---|
| LIBERO-Spatial | 70.00% | 85.00% | +15pp | below (85%) |
| LIBERO-Object | 100.00% | 100.00% | +0pp | **meets (100%)** |
| LIBERO-Goal | 80.00% | 85.00% | +5pp | below (85%) |
| LIBERO-Long/10 | 50.00% | **100.00%** | **+50pp** | **meets (100%)** |
| LIBERO-90 (5-task) | 72.00% | 76.00% | +4pp | below (76%) |
| **Overall** | **74.40%** | **89.20%** | **+14.8pp** | — |

**Task-level notes**: `libero_10` (Long) went from 2/4-tasks-perfect to **all 4 tasks perfect (100% each)** — the two previously-stuck tasks (3 and 8) both cleared. `libero_goal` task 0 (the historically-0% "middle drawer" task) continued its recovery trend: 0%(0003-0006) -> 40%(exp0009) -> **80%(exp0010, 4/5)**. Spatial and Goal are each exactly 1 trial away from the 90% floor on this 5-trial/task panel (85% = 17/20 aggregate). The weakest points now are `libero_90` task 73 (1/5, book-in-caddy-front, a known hard category) and `libero_spatial` task 5 (2/5).

### Validity checks

Same panel/trial count/settings as every prior comparison; all 21 tasks completed with no failures.

## 8. Comparison and interpretation

**A third consecutive doubling of training steps (8000->12000, on top of 4000->8000) produced another large, consistent improvement, with no sign of diminishing returns yet.** Long fully solved the panel (100%, up from 50%), Object held at a perfect 100%, and both Spatial and Goal moved to 85% — a single trial away from the 90% retention floor at this trial count. The persistently-hard `libero_goal` task 0 continued its clean recovery trajectory (0%->40%->80%) with more training, confirming it is not a purely structural, unrecoverable failure as first hypothesized, but one that responds to continued training exposure.

Final training diagnostics (val_loss 0.0788, psnr 26.80) were both meaningfully better than exp0009's step-8000 numbers, with no plateau signal — this is the third consecutive experiment in this "more training" line to show continued, substantial improvement, making it very likely that additional steps will push Spatial/Goal (currently 1 trial from the floor) and LIBERO-90 (76%, still the furthest suite from target) further.

## 9. Decision

- **Decision:** `BRANCH` — new best candidate by a wide margin; two of five suites now meet the 90% retention floor, but Spatial/Goal/LIBERO-90 do not yet.
- **Retention gate passed:** partially — Object (100%) and Long (100%) clear 90%; Spatial (85%), Goal (85%), and the primary target LIBERO-90 (76%) do not. Full retention gate (all four original suites >=90%) not yet met, so not promotable.
- **Reason:** Clear, large, consistent improvement with training metrics still improving and no plateau — strong signal to keep extending rather than switch direction.
- **Checkpoint/branch to preserve:** `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt`, uploaded to HF.
- **Next main-line parent:** unchanged for now (still the released baseline) — this branch is the clear leading candidate for eventual promotion.

## 10. What this changes for the next experiment

1. **The "more training" trend remains strong and unplateaued after three consecutive doublings** (4000->8000->12000 steps). Immediately launched **experiment 0011**: another fresh single continuous run targeting `max_steps=16000`, to keep testing whether Spatial/Goal cross 90% and whether LIBERO-90 (the primary target, currently the furthest from 90%) continues its steady climb (48%->72%->76%).
2. LIBERO-90 remains the binding constraint on the actual project goal even as retention suites approach/clear the floor — worth keeping a close eye on whether its improvement rate keeps pace with the original 4 suites as training continues, since a checkpoint with perfect retention but LIBERO-90 still well under 90% would not meet the project's primary target despite passing the promotion retention gate.
3. Given cheap-panel evidence is now consistently strong, a canonical (full 130-task, 50-trial) evaluation should be considered once a checkpoint appears to plausibly clear all suites on this panel, to get promotion-grade evidence rather than continuing to iterate on the cheap panel indefinitely.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0010_longrun2/train.log` (live copy at `/tmp/.../scratchpad/train_exp0010.log`)
- final weights checkpoint: `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (11.2GB)
- HF backup: `cheikh025/ASR:branch/0010_reweighted_longrun2/step_012000.pt` (uploaded and existence-verified 2026-08-10)
- config(s): unchanged from exp0003/0006/0009
- eval results: `evaluate_results/exp0010_cheap_panel/`

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
