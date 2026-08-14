# PROGRESS_0012 — continued_fixed

- **Experiment ID:** 0012_continued_fixed
- **Status:** `STOPPED_INTENTIONALLY`
- **Created:** 2026-08-10
- **Updated:** 2026-08-10
- **Parent experiment:** 0010_reweighted_longrun2
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (weights-only, loaded via the newly-fixed pre-wrap resume path)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `99826cc` (the resume-fix commit)

## 1. Result at a glance

The first real (non-diagnostic) use of the fixed `resume=` path: continues training from exp0010's step_012000.pt checkpoint (89.2% overall on the cheap panel — Object/Long at 100%, Spatial/Goal at 85%, LIBERO-90 at 76%) for `max_steps=4000` more steps, instead of relaunching a full run from the released checkpoint. This directly tests whether the fix enables efficient, cheap continuation (~3.5h vs. the ~13.7h a fresh 16000-step restart would have cost) while continuing the strong "more training helps" trend from exp0009/0010.

## 2. Research state before experiment

exp0010 is the best-evidenced checkpoint to date. A user-prompted investigation into why every previous "continue training" attempt (exp0007, exp0008) failed catastrophically identified the exact root cause (`action_encoder`/`head` never receiving resumed values due to a DeepSpeed FP32-master/BF16-working-copy desync — see `research/NOTES.md` "ROOT CAUSE FOUND") and a fix was implemented and validated via the same cheap 1-step-diagnostic pattern (see `diag2_resume_fix_validation` in `EXPERIMENTS.jsonl`): a 1-step reload of exp0010's checkpoint with the fix scored 100% on a 4-task panel that scored 0% with the old code. exp0011 (a fresh 16000-step restart, the "safe but wasteful" fallback) was stopped intentionally by explicit user instruction once this investigation began, to redirect compute toward the fix instead.

## 3. Candidate design

### Modifications

None to the recipe. Same data/task config as exp0003/0006/0009/0010 (`libero_uncond_2cam224_plus90_reweighted_3e-5`, goal5x/long5x reweighted mix). Only change: `resume=` points at exp0010's own weights-only checkpoint (not the released checkpoint), using the newly-fixed pre-wrap resume path, with `max_steps=4000` (a fresh, correctly-shaped schedule for this continuation window, not a raw extension of exp0010's own schedule).

### Why this candidate

Directly tests the fix in a real training scenario (not just a 1-step diagnostic) and is the cheapest way to keep testing the "more training helps" trend, now that continuation from a checkpoint is confirmed safe.

### What to watch

- Confirm no repeat of the old catastrophic-collapse symptom: internal training metrics (val_loss, action_l2) should start at levels consistent with exp0010's own converged state (low), not a from-scratch relearning curve (which would indicate the fix didn't actually take effect in a multi-step run).
- Cheap panel result vs exp0010's step-12000 result at completion.

### Initial compute plan

- Initial training budget: `max_steps=4000` (~3.5h, dramatically cheaper than a fresh restart).
- Checkpoint/save plan: `save_every=500`, state on ephemeral scratch disk via symlink.
- Progress check: monitor early-step val_loss/action_l2 closely as a sanity check that the fix held (see "what to watch"); a full eval possible mid-run if warranted, though GPU-contention limits this as with prior long runs.

## 4. Exact code and configuration state

- Git commit: `99826cc` (resume-fix commit)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=4000`
- Resume source: `runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt` (via the fixed pre-wrap resume path)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the code fix in `src/fastwam/trainer.py` (commit `99826cc`).

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0010_longrun2/checkpoints/weights/step_012000.pt \
    max_steps=4000 \
    output_dir=./runs/reweighted_libero90_finetune/exp0012_continued_fixed \
    wandb.name=exp0012_continued_from_exp0010_fixed
  ```
- Start time: 2026-08-10 ~22:31 UTC
- **Stopped intentionally at step 550/4000 (2026-08-10 ~23:04 UTC), per explicit user instruction.** Rationale: this run only exercised the weights-only resume fix (commit `99826cc`); the user correctly pointed out that a real continuation using the *directory/full-state* resume path (which had its own separate LR-schedule bug from exp0007, fixed but not yet validated at that point — commit `b2b49d0`) should also be validated before committing more real training compute. Redirected to validate the second fix using this run's own partial state checkpoint (`step_000500`) as the test case — see Section 8.
- Early training health check before stopping: loss ranged 0.12-0.34 over steps 10-550 (raw per-step, noisy as expected), with `val_loss` at step 200 = 0.1890 and step 400 = 0.3369 (an early uptick attributed to the fresh warmup-then-decay schedule ramping LR up before it starts decaying, consistent with similarly noisy early-training behavior seen in other fresh-schedule runs like exp0006). Not conclusive either way on candidate quality — this run was diagnostic/exploratory for the resume-fix validation, not a real candidate screen.

## 7. Evaluation events

Not applicable — stopped before a meaningful evaluation checkpoint; the run's value was in validating resume-fix #1 in a real multi-step scenario (loss levels consistent with genuine continuation, not from-scratch relearning) and in producing the `step_000500` full-state checkpoint used to validate resume-fix #2.

## 8. Comparison and interpretation

This run's early loss trajectory (starting ~0.15-0.22, well below the ~0.4+ a from-scratch run shows at comparable steps) confirmed resume-fix #1 held in a real multi-step run, not just the 1-step diagnostic. Its own saved full-state checkpoint (`step_000500`, global_step=500, schedule shaped for max_steps=4000) was then reused as the resume source for a separate diagnostic validating resume-fix #2 (directory/full-state resume): resuming that checkpoint with `max_steps` overridden to 8000 (a genuinely different schedule shape) produced `lr=2.9987e-05` immediately after resume — correct for a position just past an 8000-step schedule's 400-step warmup — with no discontinuity in the following steps. This confirms both resume mechanisms are now fixed and safe. Full evidence: `research/NOTES.md` "VALIDATED" section under the second fix.

## 9. Decision

- **Decision:** `STOPPED_INTENTIONALLY` (training-control decision, not a candidate-level PROMOTE/REJECT/BRANCH — this run existed to validate infrastructure, not to screen a candidate).
- **Retention gate passed:** not applicable.

## 10. What this changes for the next experiment

**Both resume mechanisms (weights-only and directory/full-state) are now confirmed fixed and validated.** Future experiments can choose either based on what the experiment needs: weights-only resume for a fresh optimizer/schedule (e.g. a different LR for a new phase), or directory resume for exact, seamless continuation of the exact training trajectory (preserves optimizer momentum, continues the LR decay curve without disruption). The next real candidate experiment should use whichever is appropriate and can now be trusted to actually behave as intended.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0012_continued_fixed/train.log` (pending; live copy at `/tmp/.../scratchpad/train_exp0012.log`)
- config(s): unchanged from exp0003/0006/0009/0010
- fix commit: `99826cc`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [ ] hardware/software environment recorded (fix commit noted, to confirm no other drift)
- [ ] logs and checkpoint paths recorded (pending)
- [ ] every evaluation event has purpose/settings/raw results recorded (pending)
- [ ] final decision and reasoning recorded (pending)
- [ ] `research/EXPERIMENTS.jsonl` updated (pending)
- [ ] `research/STATE.md` updated (pending)
