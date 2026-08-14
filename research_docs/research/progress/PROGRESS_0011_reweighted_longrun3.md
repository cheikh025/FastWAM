# PROGRESS_0011 — reweighted_longrun3

- **Experiment ID:** 0011_reweighted_longrun3
- **Status:** `STOPPED_INTENTIONALLY`
- **Created:** 2026-08-10
- **Updated:** 2026-08-10
- **Parent experiment:** 0010_reweighted_longrun2
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (fresh run, not a resume of exp0010's own checkpoint)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** same as exp0003/0006/0009/0010

## 1. Result at a glance

Continues the strongest, most consistent trend in the project: three consecutive doublings of training steps (4000->8000->12000) on the reweighted recipe each produced large, consistent improvements with no plateau signal. exp0010 (12000 steps) reached 89.2% overall on the cheap panel — Object and Long both at 100%, Spatial and Goal at 85% (one trial from the 90% retention floor), LIBERO-90 (the primary target) at 76%. This experiment extends to `max_steps=16000` via another fresh, safe, single continuous run to test whether Spatial/Goal cross 90% and whether LIBERO-90 keeps climbing.

## 2. Research state before experiment

exp0010's step_012000.pt is the best-evidenced checkpoint to date and the first to clear the 90% floor on any original suite through more training alone (Object 100%, Long 100%). Spatial (85%) and Goal (85%) are each one trial away from the floor on the 5-trial/task cheap panel. LIBERO-90 (76%) remains the furthest from the 90% target and the binding constraint on the project's primary goal. Training diagnostics (val_loss, infer_psnr) were still improving at step 12000 with no plateau.

## 3. Candidate design

### Modifications

None to the recipe. Same data/task config as exp0003/0006/0009/0010, with `max_steps=16000` (up from 12000) overridden on the command line. `resume=` points at the originally-released checkpoint per the standing constraint.

### Why this candidate

Directly continues the strongest lever in this project. Given no plateau after three doublings, a further extension is the highest-expected-value use of the next block of compute.

### What to watch

Cheap panel vs exp0010's step-12000 result. Specifically: does Spatial and/or Goal cross 90%; does LIBERO-90 continue its climb (48%->72%->76%->?); does `libero_goal` task 0 continue recovering (0%->40%->80%->?).

### Initial compute plan

- Initial training budget: `max_steps=16000` (~13.7h at the established ~0.33 step/s throughput).
- Checkpoint/save plan: `save_every=500`, state on ephemeral scratch disk via symlink.
- No interim progress check (same reasoning as exp0009/0010 — cannot safely pause/resume given the resume-corruption constraint).
- Expected cost: ~13.7h training + eval time. This is now a substantial single-run cost; if the improvement trend shows clear diminishing returns at this checkpoint, consider whether a canonical evaluation of exp0010 or exp0011's intermediate progress is a better use of compute than continuing to extend further.

## 4. Exact code and configuration state

- Git commit: same as exp0003/0006/0009/0010 (no new files)
- Config overrides: `max_steps=16000`
- Resume source: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`

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
    max_steps=16000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0011_longrun3 \
    wandb.name=exp0011_reweighted_16000steps_fresh
  ```
- Start time: 2026-08-10 ~22:13 UTC
- **Stopped intentionally at ~step 250-500 (2026-08-10 ~22:25 UTC), per explicit user instruction**, on the reasoning that another ~13.7h full-restart run to test "more steps" is wasteful compute now that the exact root cause of the resume-corruption bug is known (see `PROGRESS_0008` / `research/NOTES.md` "ROOT CAUSE FOUND" — `action_encoder`/`head` never receive the resumed checkpoint's values). Redirected effort to actually fixing `resume=` so future extensions can cheaply continue from exp0010's checkpoint instead of repeating all prior steps from scratch.

## 7. Evaluation events

Not applicable — stopped before any checkpoint worth evaluating was produced.

## 8. Comparison and interpretation

Not applicable.

## 9. Decision

- **Decision:** `STOPPED_INTENTIONALLY` (not a candidate-level REJECT/PROMOTE/BRANCH — training-control decision to redirect compute toward fixing the underlying resume bug rather than brute-force-repeating full training runs).
- **Retention gate passed:** not applicable — no meaningful checkpoint produced.

## 10. What this changes for the next experiment

Redirected to a targeted fix for the `resume=` checkpoint-corruption bug (see `research/NOTES.md`). If the fix is validated (via the same cheap 1-step-diagnostic pattern used to find the bug), the next experiment should resume directly from exp0010's step_012000.pt checkpoint (weights or, ideally, full state for optimizer continuity) and extend training incrementally, rather than repeating a full fresh run each time. If the fix cannot be validated quickly/safely, fall back to a fresh long run (the exp0011 pattern) as the known-safe default.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0011_longrun3/train.log` (pending; live copy at `/tmp/.../scratchpad/train_exp0011.log`)
- config(s): unchanged from exp0003/0006/0009/0010

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [ ] hardware/software environment recorded (unchanged, to confirm no drift)
- [ ] logs and checkpoint paths recorded (pending)
- [ ] every evaluation event has purpose/settings/raw results recorded (pending)
- [ ] final decision and reasoning recorded (pending)
- [ ] `research/EXPERIMENTS.jsonl` updated (pending)
- [ ] `research/STATE.md` updated (pending)
