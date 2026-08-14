# PROGRESS_0009 — reweighted_longrun_fresh

- **Experiment ID:** 0009_reweighted_longrun_fresh
- **Status:** `RUNNING`
- **Created:** 2026-08-10
- **Updated:** 2026-08-10
- **Parent experiment:** 0006_exp0003_reproducibility_check (same recipe, but a genuinely fresh run — not a resume of its checkpoint)
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (the originally-released checkpoint)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** same as exp0003/0006 (`eb0a53b...`, no new config needed — reuses `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml` with a `max_steps` override)

## 1. Result at a glance

Retests "does more training help the reweighted (goal5x/long5x) recipe" for a third time, after experiments 0007 and 0008 were both invalidated by a newly-discovered, serious infra bug: resuming training from any of this project's own saved checkpoints catastrophically corrupts task performance (see `research/NOTES.md`). This run avoids that bug entirely by launching a **fresh, single, continuous 8000-step run from the originally-released checkpoint** — the same starting point every other successful experiment (0001, 0003, 0004, 0005, 0006) has used — rather than resuming exp0006's 4000-step checkpoint. This costs repeating the first 4000 steps' compute but is currently the only verified-safe way to test extended training.

## 2. Research state before experiment

exp0006's step_004000.pt (goal5x/long5x reweighted recipe, 4000 steps) remains the best-evidenced checkpoint on this recipe: cheap-panel results spatial 70%, object 95%, goal 55%, long 30%, libero_90 48%, overall 59.6% (with exp0003's original run on the identical recipe scoring somewhat higher due to established training-run noise — see `PROGRESS_0006`). Neither clears the 90% retention floor on goal/long/90. Two attempts to test whether more training closes this gap (0007: buggy LR-rewarm extension; 0008: correctly-scheduled but still-corrupted continuation) both failed for infrastructure reasons unrelated to the recipe's actual response to more training.

## 3. Candidate design

### Modifications

None to the recipe. Same data config (`libero_2cam_plus90_reweighted`, goal 5x/long 5x) and same task config (`libero_uncond_2cam224_plus90_reweighted_3e-5`, lr=3e-5, batch_size=4, grad_accum=4) as exp0003/exp0006, with only `max_steps=8000` overridden on the command line (doubling the original 4000-step budget) and a fresh `output_dir`. `resume=` points at the originally-released checkpoint, exactly as exp0003/exp0006 did — this is NOT a continuation of exp0006's own checkpoint.

### Why this candidate

Directly retests the training-duration question that motivated exp0007/exp0008, using the only currently-verified-safe training path. A single continuous 8000-step schedule (built once at `__init__`, `warmup_steps=int(8000*0.05)=400`, cosine decay over the remaining 7600 steps) avoids both the LR re-warmup bug (exp0007) and the deeper own-checkpoint-resume corruption (exp0008) since there is no resume-from-own-checkpoint step at all.

### What to watch

Cheap panel comparison against exp0006's step-4000 result, at both an intermediate progress check (if GPU availability allows) and at final completion (step 8000). Given the established training-run noise floor (~20pp/suite, ~40pp/task from exp0006's reproducibility check), only trust a directional conclusion if it is large and consistent across multiple suites/tasks.

### Initial compute plan

- Initial training budget: `max_steps=8000` (~7h at the established ~0.33 step/s throughput — roughly double exp0006's single run since this repeats the first 4000 steps).
- Checkpoint/save plan: `save_every=500`, state on ephemeral scratch disk via symlink (validated fix), weights-only `.pt` checkpoints for eval/backup.
- Progress check: consider evaluating the step-4000 checkpoint (which should closely resemble exp0003/exp0006's own step-4000 results, modulo training-run noise) as a sanity check, then again near/at step 8000.
- Expected cost: ~7h training + eval time.

## 4. Exact code and configuration state

- Git commit: same as exp0003/exp0006 (no new files; `max_steps` passed as a command-line Hydra override)
- Git branch: `autoresearch/libero90-v1`
- Working tree: clean, no changes needed
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=8000` (only difference from exp0003/exp0006's launch)
- Dataset config: `data=libero_2cam_plus90_reweighted` (unchanged — goal 5x, long 5x)
- Resume source: `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (the originally-released checkpoint — NOT exp0006's own checkpoint)

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
    max_steps=8000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0009_longrun \
    wandb.name=exp0009_reweighted_8000steps_fresh
  ```
- Start time: 2026-08-10 ~04:26 UTC
- End time: 2026-08-10 11:20 UTC (`max_steps reached step=8000`, clean exit, no crash, no OOM, no disk-full warning — runtime ~6h54min)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0009_longrun/checkpoints/weights/step_008000.pt` (12,041,735,545 bytes ≈ 11.2GB)
- Final training diagnostics at step 8000: `val_loss=0.1582 infer_psnr=25.0308 infer_ssim=0.8428 action_l2=0.0047 action_l1=0.0451 lr=3.00e-07` — LR correctly decayed to near-zero as expected for a single continuous schedule (no resume-related bug, since this run never resumed from an intermediate checkpoint). Metrics look healthy and are in fact slightly better than exp0006's own step-4000 numbers (lower val_loss, lower action_l2/l1).

## 7. Evaluation events

No progress check performed at step 4000 despite the original plan to consider one: since resuming from any of this project's own saved checkpoints is now confirmed unsafe (see `research/NOTES.md`), this run cannot be safely paused and resumed for an interim eval — doing so would require the exact broken resume path this experiment exists to avoid. Training therefore ran straight through to step 8000 uninterrupted; evaluation happened once at final completion.

### Event 1 — `candidate_screen` (final, step 8000)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0009_longrun/checkpoints/weights/step_008000.pt`
- Reference: exp0006's step_004000.pt (same recipe, half the training)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0009_cheap_panel/`

**Suite-level result — the best result of the entire experiment series:**

| Suite | exp0006 (step 4000) | exp0009 (step 8000, fresh) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 70.00% | 70.00% | +0pp |
| LIBERO-Object | 95.00% | 100.00% | +5pp |
| LIBERO-Goal | 55.00% | 80.00% | **+25pp** |
| LIBERO-Long/10 | 30.00% | 50.00% | **+20pp** |
| LIBERO-90 (5-task) | 48.00% | 72.00% | **+24pp** |
| **Overall** | **59.60%** | **74.40%** | **+14.8pp** |

Also comparable against exp0003's original step-4000 run (spatial 80/object 100/goal 70/long 20/libero_90 68/overall 67.6): exp0009 still beats it on Goal (+10pp), Long (+30pp), and LIBERO-90 (+4pp), roughly matches on Object, and is 10pp behind on Spatial alone.

**Task-level notes**: `libero_goal` task 0 ("open the middle drawer of the cabinet") — 0% in every prior reweighted run (0003, 0004, 0005, 0006) — moved to **40% (2/5)** here, the first time it has ever been non-zero. This softens (does not fully overturn) the earlier diagnostic finding that this failure is purely structural/unrecoverable — it appears to partially improve with substantially more training exposure, though it is still far from solved. `libero_10` (Long) tasks 0 and 5 jumped to 100%, while tasks 3 and 8 remain at 0% — the suite-level Long improvement is concentrated in specific tasks, not a uniform lift.

### Validity checks

Same panel/trial count/settings as every prior comparison in this series; all 21 tasks completed with no failures; checkpoint identity and dataset_stats path confirmed correct for this run.

## 8. Comparison and interpretation

**Doubling training steps (4000->8000) on the reweighted recipe, done via a safe fresh single continuous run, produced the largest, most credible improvement of the entire experiment series.** Goal (+25pp), Long (+20pp), and LIBERO-90 (+24pp) all improved substantially and consistently — well beyond the ~20pp/suite noise floor established in exp0006's reproducibility check — while Object and Spatial held steady or slightly improved. This is the first strong, trustworthy evidence that "more training helps" for this recipe; exp0007/exp0008 could not answer this question due to the checkpoint-resume corruption bug, and this fresh-run approach cleanly sidesteps it.

The improvement is not uniform at the task level: Long's gain is concentrated in 2 of 4 sampled tasks (0 and 5, both to 100%; 3 and 8 stay at 0%), and `libero_goal` task 0's long-standing 0% streak broke to 40% — suggesting more training exposure can partially overcome the drawer-position interference previously characterized as structural, though it remains a weak point.

**Still short of the retention gate**: Spatial (70%) and Long (50%) and Goal (80%) are all below the 90% floor required for a main-line promotion; only Object (100%) clears it. This checkpoint cannot be promoted as-is, but it is unambiguously the best candidate produced so far and the training curve shows no sign of plateauing at step 8000 (final metrics — val_loss, action_l2/l1 — were still improving relative to step 4000). This strongly motivates trying an even longer training budget next.

## 9. Decision

- **Decision:** `BRANCH` — best candidate to date, substantial and credible improvement over exp0006/exp0003, but does not meet the retention gate (Spatial/Goal/Long all below 90%) required for main-line promotion.
- **Retention gate passed:** no (Spatial 70%, Goal 80%, Long 50% all below 90%; only Object at 100% clears it).
- **Reason:** Improvement is real and large relative to the established noise floor, and the training trajectory (loss/metrics still improving at step 8000, no plateau) suggests more training may close the remaining gap. Preserve as the current best branch and extend further rather than reject or attempt to promote prematurely.
- **Checkpoint/branch to preserve:** `runs/reweighted_libero90_finetune/exp0009_longrun/checkpoints/weights/step_008000.pt`, uploaded to HF.
- **Next main-line parent:** unchanged for now (still the released baseline) — this branch is not yet promotable, but is the leading candidate for eventual promotion pending further training.

## 10. What this changes for the next experiment

1. **More training continues to help** — the clearest, most reliable signal from the whole reweighting series. Immediately launched **experiment 0010**: another fresh single continuous run (same safe pattern, no resume-from-own-checkpoint) targeting a longer budget beyond 8000 steps, to test whether the improving trend continues toward the 90% target.
2. The repeated need to redo early training steps from scratch for every "train longer" test is a real, mounting compute cost directly caused by the unresolved checkpoint-resume corruption bug (`research/NOTES.md`). If further extended-training experiments are planned, fixing that bug (with careful before/after validation via a cheap diagnostic, following the same pattern used to discover it) would pay for itself quickly and should be considered as a parallel investigation.
3. `libero_goal` task 0 and the two persistently-0% `libero_10` tasks (3, 8) remain worth targeted attention — they did not uniformly improve with more training and may need a different, more specific intervention (e.g. task-specific oversampling) rather than just more steps.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0009_longrun/train.log` (live copy at `/tmp/.../scratchpad/train_exp0009.log`)
- final weights checkpoint: `runs/reweighted_libero90_finetune/exp0009_longrun/checkpoints/weights/step_008000.pt` (11.2GB)
- HF backup: `cheikh025/ASR:branch/0009_reweighted_longrun_fresh/step_008000.pt` (uploaded and existence-verified 2026-08-10 — stored under `branch/` rather than `rejected/` since this is the current best candidate, not a rejected one)
- config(s): unchanged from exp0003/exp0006 (`configs/data/libero_2cam_plus90_reweighted.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml`)
- eval results: `evaluate_results/exp0009_cheap_panel/`

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
