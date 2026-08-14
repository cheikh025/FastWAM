# PROGRESS_0013 — continued_from_exp0010

- **Experiment ID:** 0013_continued_from_exp0010
- **Status:** `PROMOTE`
- **Created:** 2026-08-10
- **Updated:** 2026-08-11
- **Parent experiment:** 0010_reweighted_longrun2
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (weights-only)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

Relaunches exp0012's goal (continue training from exp0010's best checkpoint, 89.2% overall on the cheap panel) now that both resume mechanisms are validated. **Important caveat, flagged directly by the user during exp0012's own review**: this run uses *weights-only* resume, not *directory/full-state* resume — a real distinction, not just a matter of which fix applies. Weights-only resume is the correct tool for "start a new phase from these weights" (fresh optimizer/schedule is fine); directory resume is the correct tool for "keep this exact run going longer" (same optimizer momentum, continuous LR decay). The semantically correct choice for "just train exp0010 more" would have been directory resume, but **exp0010's own full-state checkpoint no longer exists** — it was deleted during a disk cleanup earlier in the session, before the directory-resume fix even existed. Weights-only resume from exp0010's saved weights is therefore the best available option, not the ideal one: this is a fresh second-stage run built on exp0010's trained state, not a byte-for-byte continuation of exp0010's exact training trajectory.

**Going forward, this run's own full-state checkpoints will be preserved** (only the single newest one is kept locally at any time, via the standard disk-management pruning, but never deleted down to zero) so that if further extension is wanted later, a proper directory-resume continuation is available this time.

## 2. Research state before experiment

exp0010's step_012000.pt is the best-evidenced checkpoint to date: cheap-panel results Object 100%, Long 100%, Spatial 85%, Goal 85%, LIBERO-90 76%, overall 89.2% — Object and Long clear the 90% retention floor, Spatial/Goal are one trial away, LIBERO-90 (the primary target) remains furthest out. Both resume mechanisms are now confirmed fixed: weights-only (commit `99826cc`, validated via 1-step diagnostic and exp0012's own early training behavior) and directory/full-state (commit `b2b49d0`, validated via a separate diagnostic using exp0012's partial state checkpoint).

## 3. Candidate design

### Modifications

None to the recipe. Same data/task config as exp0003/0006/0009/0010/0012 (`libero_uncond_2cam224_plus90_reweighted_3e-5`, goal5x/long5x reweighted mix, lr=3e-5). `resume=` points at exp0010's weights-only checkpoint. `max_steps=8000` (a fresh schedule sized for 8000 steps from this starting point).

### Why this candidate

Continues the strongest, most reliable lever in this project (more training on the reweighted recipe) using the now-fully-validated resume infrastructure, at much lower compute cost than a fresh restart from the released checkpoint would require.

### What to watch

- Cheap panel vs exp0010's step-12000 result at completion (or at an intermediate save point if a progress check becomes worthwhile).
- Whether Spatial/Goal cross 90% (currently 85% each) and whether LIBERO-90 continues climbing (48%->72%->76%->?).
- Early training health (val_loss/action_l2 trajectory) as an ongoing sanity check that the resume fix continues to hold.

### Initial compute plan

- Initial training budget: `max_steps=8000` (~7h at the established throughput).
- Checkpoint/save plan: `save_every=500`, only the single newest full-state checkpoint kept locally (not deleted to zero, to preserve directory-resume extension option for the future).
- No interim progress check planned by default (GPU contention with eval, as in prior long runs), but reconsider if this run's own full-state checkpoints make a brief pause-and-resume cheap enough to be worth it now that directory resume is validated.

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=8000`
- Resume source: `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (weights-only — see Section 1 caveat)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`).

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt \
    max_steps=8000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0013_continued_fixed \
    wandb.name=exp0013_continued_from_exp0010_validated
  ```
- Start time: 2026-08-10 ~23:14 UTC
- End time: 2026-08-11 06:09 UTC (`max_steps reached step=8000`, clean exit, no crash, no OOM, no disk-full warning — runtime ~6h55min)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/weights/step_008000.pt` (12,041,735,545 bytes ≈ 11.2GB)
- HF backup: `cheikh025/ASR:branch/0013_continued_from_exp0010/step_008000.pt` (uploaded and existence-verified 2026-08-11)
- Final training diagnostics at step 8000: `val_loss=0.1420 infer_psnr=26.1285 infer_ssim=0.8565 action_l2=0.0033 action_l1=0.0358` — action_l2 notably lower than exp0010's own final value (0.0090), suggesting continued improvement.
- Final full-state checkpoint (`step_008000`) preserved on the scratch disk (not pruned to zero), per the disk-management discipline discussed with the user, to keep a true directory-resume extension option available for this run specifically.

## 7. Evaluation events

### Event 1 — `candidate_screen` (cheap panel)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/weights/step_008000.pt`
- Reference: exp0010's step_012000.pt
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0013_cheap_panel/`

**Suite-level result — FIRST TIME ALL FOUR ORIGINAL SUITES CLEAR THE 90% RETENTION FLOOR:**

| Suite | exp0010 (step 12000) | exp0013 (step 8000, this run) | Delta | vs 90% floor |
|---|---:|---:|---:|---|
| LIBERO-Spatial | 85.00% | 95.00% | +10pp | **meets (95%)** |
| LIBERO-Object | 100.00% | 90.00% | -10pp | **meets (90%, exactly at floor)** |
| LIBERO-Goal | 85.00% | 95.00% | +10pp | **meets (95%)** |
| LIBERO-Long/10 | 100.00% | 100.00% | +0pp | **meets (100%)** |
| LIBERO-90 (5-task) | 76.00% | 80.00% | +4pp | below (80%) |
| **Overall** | **89.20%** | **92.00%** | **+2.8pp** | — |

**Task-level notes**: `libero_goal` task 0 (the historically-hardest "middle drawer" task) reached **5/5 (100%)** — a complete recovery: 0%→40%→80%→100% across the extension series (exp0006→0009→0010→0013). Object task 5 dipped to 3/5 (60%, the one notable regression, still within the run-to-run noise band established in exp0006's reproducibility check). LIBERO-90 continues its steady climb (48%→72%→76%→80%) but remains the binding constraint on the project's primary target.

### Event 2 — `confirmation` (full 130-task coverage, 15 trials/task)

Launched immediately given the strength of the cheap-panel result, per the project's own adaptive-evaluation guidance ("before spending a full canonical run, confirm a promising cheap-panel signal generalizes"). Full 130-task coverage (all 90 LIBERO-90 + all 40 original-suite tasks), 15 trials/task (vs. 50 canonical).

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=15`, fresh `output_dir=evaluate_results/exp0013_confirmation`
- Runtime: ~2h07min (2026-08-11 06:25 -> 08:32 UTC), all 130 tasks completed, 0 failures.

**Result — retention gate confirmed with real margin, LIBERO-90 tantalizingly close:**

| Suite | Confirmation (15 trials/task) | vs 90% floor |
|---|---:|---|
| LIBERO-Spatial | 96.00% | **meets, comfortable margin** |
| LIBERO-Object | 96.67% | **meets, comfortable margin** |
| LIBERO-Goal | 98.67% | **meets, comfortable margin** |
| LIBERO-Long/10 | 96.67% | **meets, comfortable margin** |
| LIBERO-90 (all 90 tasks) | **88.59%** | just below (1.41pp short) |
| **Overall** | **95.32%** | — |

This is a much stronger and more robust confirmation than the cheap panel suggested: all four original suites are comfortably above 90% (96-98.67%, not just barely at the floor), and LIBERO-90 across its full 90-task set is close enough (88.59%) that only canonical-grade (50-trial) evidence can settle whether it clears 90%.

### Event 3 — `canonical` (full 130-task coverage, 50 trials/task) — promotion-grade evidence

Launched immediately given how close the confirmation result was — this is exactly the situation the project's promotion rule anticipates ("PROMOTE requires canonical evaluation").

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=50`, fresh `output_dir=evaluate_results/exp0013_canonical`
- Start time: 2026-08-11 ~08:32 UTC
- End time: 2026-08-11 ~15:56 UTC (~7h24min, all 130 tasks completed, 0 failures)
- Raw results: `evaluate_results/exp0013_canonical/`

**Result — canonical, promotion-grade evidence:**

| Suite | Canonical (50 trials/task) | vs 90% floor |
|---|---:|---|
| LIBERO-Spatial | 94.40% | **meets** |
| LIBERO-Object | 98.00% | **meets** |
| LIBERO-Goal | 97.40% | **meets** |
| LIBERO-Long/10 | 93.00% | **meets** |
| LIBERO-90 (all 90 tasks) | **88.49%** | below (1.51pp short) |
| **Overall** | **94.26%** | — |

All four original suites clear the 90% retention floor with real margin (93.00–98.00%). LIBERO-90 lands at 88.49% — consistent with the confirmation run's 88.59% (a 0.10pp difference, well within noise for 15-vs-50 trials/task), so this is a stable, well-evidenced result, not a fluke of either evaluation. It does not clear the 90% target for LIBERO-90 specifically, but it is a massive improvement over the current accepted main-line checkpoint (the unmodified release, LIBERO-90 = 15.40%) while comfortably keeping every original suite above 90%.

## 8. Comparison and interpretation

This is by far the strongest result in the project so far, and it held up under increasingly rigorous evidence at every stage. Three consecutive doublings of training steps on the reweighted recipe (exp0009: 8000, exp0010: 12000, and now this continuation to a further 8000 steps built on exp0010's checkpoint) produced a checkpoint that:

- Cleared the 90% retention floor on all four original suites on the 21-task cheap panel (Spatial 95%, Object 90%, Goal 95%, Long 100%).
- **Confirmed this with real margin** on full 130-task/15-trial coverage: Spatial 96.00%, Object 96.67%, Goal 98.67%, Long 96.67% — not narrowly scraping the floor, comfortably above it.
- Landed LIBERO-90 (the project's primary target) at 88.59% on the full 90-task confirmation set — a huge jump from the 5-task cheap-panel sample (80%), and only 1.41pp below the 90% target with 15 trials/task.

This confirms the "more training helps" trend was real and generalizes far beyond the small cheap panel that originally revealed it — the panel's LIBERO-90 signal (48%→72%→76%→80% across exp0006→0009→0010→0013) understated the true improvement once measured across all 90 tasks. The canonical (50-trial) result (88.49%) landed within 0.1pp of the confirmation (15-trial) result (88.59%), so this checkpoint's LIBERO-90 level is now precisely and reliably measured, not a noisy estimate.

## 9. Decision

- **Decision:** `PROMOTE`. exp0013's checkpoint (`step_008000.pt`) becomes the new main-line accepted checkpoint, replacing the unmodified released checkpoint that has been the accepted baseline since project setup.
- **Retention gate passed:** yes, with real margin, at canonical (50-trial) precision: LIBERO-Spatial 94.40%, LIBERO-Object 98.00%, LIBERO-Goal 97.40%, LIBERO-Long/10 93.00% — every original suite clears 90%.
- **Primary target (LIBERO-90):** 88.49% at canonical precision — a huge improvement over the current main-line's 15.40% (+73.09pp), and consistent (within 0.1pp) between confirmation and canonical evaluation, but **still 1.51pp below the project's ultimate 90% target for LIBERO-90.** Per the project's promotion rule, `PROMOTE` requires only that every original suite clear the 90% retention floor and that the candidate be the best available main-line parent — both conditions are clearly met here (LIBERO-90 improvement of +73pp is unambiguously material, not marginal). It does **not** by itself satisfy the overall project goal (all five suites ≥90%), so research continues immediately from this new accepted checkpoint to close the remaining LIBERO-90 gap.

## 10. What this changes for the next experiment

1. **New main-line accepted checkpoint**: `runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/weights/step_008000.pt`. `research/STATE.md` updated accordingly. HF backup uploaded to `cheikh025/ASR:promoted/0013_continued_from_exp0010/step_008000.pt` (existence-verified) — the prior `branch/` copy remains as well, no need to remove it.
2. **Immediate next step**: continue training from this exact checkpoint via directory/full-state resume (validated, commit `b2b49d0`) — this run's own full-state checkpoint (`step_008000`, ~80GB) was deliberately preserved at `/home/claudeuser/fastwam_state_scratch/exp0013_continued_fixed/state/step_008000/` specifically to make this a true, seamless continuation (preserved optimizer momentum, continuous LR decay) rather than another fresh weights-only restart. The trend across exp0009→0010→0013 has shown no plateau (LIBERO-90: 48%→72%→76%→88.49%), so more training remains the best-evidenced lever for closing the remaining ~1.5pp gap.
3. `libero_goal` task 0's full recovery (0%→100%) confirms the earlier hypothesis that its difficulty was training-exposure-related, not a structural limitation — worth noting as a general lesson: apparent "hard-capped" task failures in this project may simply need more training rather than a targeted architectural fix.
4. The cheap 5-task LIBERO-90 sentinel panel meaningfully understated the true LIBERO-90 improvement (80% sampled vs. 88.49-88.59% on full 90-task coverage) — worth keeping in mind when interpreting future cheap-panel LIBERO-90 numbers as a lower bound rather than a precise estimate.
5. Watch per-suite headroom while extending further: LIBERO-Long/10 (93.00%) and LIBERO-Spatial (94.40%) are the closest to the 90% floor among the four original suites — if LIBERO-90 gains come at the cost of retention, these are the suites most likely to cross back below 90% first.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0013_continued_fixed/train.log`
- config(s): unchanged from exp0003/0006/0009/0010/0012
- fix commits: `99826cc`, `b2b49d0`
- eval results: `evaluate_results/exp0013_cheap_panel/`, `evaluate_results/exp0013_confirmation/`, `evaluate_results/exp0013_canonical/`
- full-state checkpoint (preserved for continuation): `/home/claudeuser/fastwam_state_scratch/exp0013_continued_fixed/state/step_008000/` (~80GB, ephemeral container filesystem — not backed up, disposable if a further continuation is not pursued)
- HF backup (promoted): `cheikh025/ASR:promoted/0013_continued_from_exp0010/{step_008000.pt, dataset_stats.json, config.yaml, PROGRESS_0013_continued_from_exp0010.md}` (existence-verified)
- HF backup (branch, superseded by promoted): `cheikh025/ASR:branch/0013_continued_from_exp0010/step_008000.pt`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded (unchanged from baseline except fix commits `99826cc`, `b2b49d0`)
- [x] logs and checkpoint paths recorded
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
