# PROGRESS_0016 — lr_warm_restart_from_exp0014

- **Experiment ID:** 0016_lr_warm_restart_from_exp0014
- **Status:** `REJECT`
- **Created:** 2026-08-12
- **Updated:** 2026-08-12
- **Parent experiment:** 0014_continued_from_exp0013_dirresume (`PROMOTE`, current accepted main-line checkpoint)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt` (weights-only resume)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

exp0015 (directory-resume extension of exp0014, +8000 steps) plateaued: confirmation-level LIBERO-90 moved only 91.09%->91.78% (within noise), with Goal/Long dipping slightly. This candidate tests a different lever for the same underlying goal (push LIBERO-90/Long toward the project's >=95% target): a **weights-only resume from exp0014** (not directory-resume), giving a **fresh full cosine LR cycle back up to peak 3e-5** with its own warmup, rather than another directory-resume extension whose LR only partially recovers based on its position in an extended schedule. This directly tests whether the exp0015 plateau was a shallow-schedule artifact (each directory-resume extension reaching a progressively lower peak LR: exp0014 resumed at ~1.29e-5, exp0015 at ~9.8e-6) rather than a genuine data/capacity limit.

## 2. Research state before experiment

exp0014 remains the accepted main-line checkpoint: LIBERO-Spatial 97.20%, LIBERO-Object 100.00%, LIBERO-Goal 96.60%, LIBERO-Long/10 93.40%, LIBERO-90 91.09% (canonical). exp0015 (directory-resume, +8000 steps) showed no material further LIBERO-90 gain (91.78% at confirmation-level, within noise) and was rejected. Investigated the data mix as an alternative lever: LIBERO-90 already contributes ~44% of the training mix by raw episode count, so naive further suite-level oversampling is not clearly justified and risks diluting Goal/Long (see `PROGRESS_0015` Section 10 for full reasoning); no task-level oversampling mechanism currently exists in the data pipeline. Given this, the lowest-risk, best-justified next lever is testing the LR-schedule hypothesis before investing in data-engineering work.

**LR pattern across the "more training" lineage** (peak LR reached at each resume point):
- exp0009: fresh run from released checkpoint, full cosine cycle to peak 3e-5 (worked — 48%->72% LIBERO-90 panel).
- exp0010: fresh run (not a resume), full cosine cycle to peak 3e-5 over 12000 steps (worked — 72%->76%).
- exp0013: **weights-only resume** from exp0010, fresh full cosine cycle to peak 3e-5 (worked strongly — 76%->88.49% canonical).
- exp0014: **directory-resume** from exp0013, extended schedule 8000->14000, LR resumed at ~1.29e-5 (partial recovery) (worked strongly — 88.49%->91.09% canonical).
- exp0015: **directory-resume** from exp0014, extended schedule 14000->22000, LR resumed at ~9.8e-6 (weaker recovery) (plateaued — 91.09%->91.78%, within noise).

The two directory-resume extensions reached progressively lower peak LRs and showed progressively weaker (then no) improvement, while the two weights-only resumes (which both reach the full peak 3e-5 via a fresh warmup) both produced strong gains. This is a plausible, testable explanation for the plateau.

## 3. Candidate design

### Modifications

1. `resume=./runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt` (weights-only resume — fresh optimizer state and fresh LR schedule, not a continuation of exp0014/0015's exact trajectory).
2. `learning_rate=3e-5` (same peak as every prior successful weights-only-resume round in this lineage — a genuine fresh warm-restart, not a partially-decayed continuation).
3. `max_steps=8000` (matches the scale of exp0009's and exp0013's successful weights-only-resume budgets).
4. No other recipe changes (same reweighted goal5x/long5x mix, same `task=libero_uncond_2cam224_plus90_reweighted_3e-5` base config).

### Why this candidate

Tests the LR-schedule-plateau hypothesis directly, using only the already-validated weights-only resume mechanism (no new code, no new data engineering, low implementation risk). If this produces a further gain comparable to exp0013/0014's, it confirms directory-resume's partial LR recovery was the limiting factor in exp0015 and establishes "periodic full LR warm-restarts" as the right pattern for further extensions. If this also plateaus, it rules out the LR-schedule explanation and points more strongly toward a genuine data/capacity limit, motivating the task-level data-engineering investigation flagged in `PROGRESS_0015`.

### What to watch

- Whether LIBERO-90 shows a material gain again (comparable in scale to exp0013's +12pp or exp0014's +3pp canonical gains), not just noise-level movement.
- Whether Goal/Long (the two suites that dipped slightly in exp0015) recover or continue to soften — worth watching given this is again training on the identical data mix.
- Whether the fresh optimizer state (discarding exp0014/0015's accumulated momentum) causes any early instability, given this model is already well-converged (unlike exp0009/0013's earlier, less-converged starting points).

### Initial compute plan

- Initial training budget: `max_steps=8000` (~7h at established throughput, matching exp0009/exp0013's scale).
- Checkpoint/save plan: `save_every=500`, only the newest full-state checkpoint kept locally (disk-monitor pattern).
- No interim progress check planned by default (GPU contention with eval), consistent with prior rounds — but reconsider if early training diagnostics look anomalous.

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `learning_rate=3e-5 max_steps=8000`
- Resume source: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt` (weights-only)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`). Venv: `/workspace/venv-fastwam`.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  source /workspace/venv-fastwam/bin/activate
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt \
    learning_rate=3e-5 \
    max_steps=8000 \
    save_every=500 \
    output_dir=./runs/reweighted_libero90_finetune/exp0016_lr_restart \
    wandb.name=exp0016_lr_warm_restart_from_exp0014
  ```
- Start time: 2026-08-12 ~12:46 UTC (immediately after exp0015's records were finalized and GPUs freed — no idle time)
- End time: 2026-08-12 ~19:41 UTC (`max_steps reached step=8000`, clean exit, no crash, no OOM — runtime ~6h55min)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0016_lr_restart/checkpoints/weights/step_008000.pt` (12,041,735,545 bytes)
- Final training diagnostics at step 8000: `val_loss=0.1348 infer_psnr=26.6521 infer_ssim=0.8632 action_l2=0.0025 action_l1=0.0288`
- Training health check: confirmed genuine fresh warm-restart — LR started at ~8.23e-7 right after resume (step 10) and ramped up through a fresh warmup, unlike the directory-resume runs which jump straight to a mid-schedule value. Loss/val_loss/action_l2 stayed in normal healthy ranges throughout (action_l2 0.003-0.04 band, no "relearning from scratch" signature).
- Final full-state checkpoint (`step_008000`) preserved on the scratch disk.

### Why training ended

Planned completion — reached `max_steps=8000` cleanly with no early-stop trigger. No interim progress check performed (GPU contention with eval).

## 7. Evaluation events

### Event 1 — `candidate_screen` (cheap panel)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0016_lr_restart/checkpoints/weights/step_008000.pt`
- Reference: exp0014's panel (Spatial 100%, Object 100%, Goal 100%, Long 100%, LIBERO-90 92%, overall 98.40%) and exp0015's panel (Spatial 90%, Object 100%, Goal 100%, Long 95%, LIBERO-90 92%, overall 95.40%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0016_cheap_panel/`

**Result — inconclusive, identical overall score to the rejected exp0015:**

| Suite | exp0014 (panel) | exp0015 (panel) | exp0016 (panel) |
|---|---:|---:|---:|
| LIBERO-Spatial | 100.00% | 90.00% | 90.00% |
| LIBERO-Object | 100.00% | 100.00% | 100.00% |
| LIBERO-Goal | 100.00% | 100.00% | 95.00% |
| LIBERO-Long/10 | 100.00% | 95.00% | 100.00% |
| LIBERO-90 (5-task) | 92.00% | 92.00% | 92.00% |
| **Overall** | **98.40%** | **95.40%** | **95.40%** |

The 5-task/25-trial LIBERO-90 sentinel has now returned exactly 92.00% (23/25) for three consecutive checkpoints (exp0014, 0015, 0016) — this small panel appears to have saturated as a discriminating signal at this checkpoint quality level and cannot distinguish this candidate from the already-rejected exp0015. Confirmation evaluation (full 130-task coverage, 15 trials/task) launched immediately to get a real reading.

### Event 2 — `confirmation` (full 130-task coverage, 15 trials/task)

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=15`, fresh `output_dir=evaluate_results/exp0016_confirmation`
- Runtime: ~2h05min (2026-08-12 19:57 -> ~22:02 UTC), all 130 tasks completed, 0 failures. Verified: every suite at exactly 15 trials/task.

**Result — four original suites hit their best-ever scores, but LIBERO-90 (the worst-case suite) went slightly the wrong direction:**

| Suite | exp0014 (canonical) | exp0015 (confirmation) | exp0016 (confirmation) |
|---|---:|---:|---:|
| LIBERO-Spatial | 97.20% | 97.33% | **98.67%** |
| LIBERO-Object | 100.00% | 100.00% | **100.00%** |
| LIBERO-Goal | 96.60% | 96.00% | **100.00%** |
| LIBERO-Long/10 | 93.40% | 92.67% | **97.33%** |
| **LIBERO-90** | 91.09% | 91.78% | **90.59%** (lowest of the three) |
| Overall (suite avg) | 95.66% | 95.56% | **97.32%** (highest) |

All three original-suite dips seen in exp0015 fully reversed and then some — Spatial/Goal/Long all now clear the project's updated 95% target for the first time. But LIBERO-90 itself, the metric the active goal explicitly centers on maximizing as the worst-case suite, came in *below* both exp0014 and exp0015, not above. The aggregate suite-average is the best of the whole lineage (97.32%), but under the goal's explicit "maximize worst-case" framing this is a step backward, not forward, since the worst-case suite (still LIBERO-90) got slightly worse.

## 8. Comparison and interpretation

The LR-warm-restart hypothesis from Section 1 is **not supported** by this result: a genuine fresh full-peak-LR (3e-5) warm restart did not produce a material LIBERO-90 gain over exp0014/0015's partial-LR-recovery directory-resume extensions — if anything, LIBERO-90 was slightly worse (90.59% vs 91.09%/91.78%). Instead, the fresh warm-restart's extra gradient signal appears to have gone disproportionately toward the four original suites (which are simpler, more homogeneous, and already well-represented), pushing them to new highs, rather than toward LIBERO-90's much larger and more diverse 90-task distribution. This is useful negative evidence: three different generic-training variants in a row (exp0014's directory-resume, exp0015's further directory-resume, exp0016's fresh warm-restart) have now failed to produce a *material, reliable* further LIBERO-90 gain (91.09% -> 91.78% -> 90.59%, no clear upward trend, all within noise of each other) even though two of them clearly still help the other suites. This points strongly away from any further generic-training variant and toward a genuinely LIBERO-90-targeted intervention.

## 9. Decision

- **Decision:** `REJECT`. Despite the highest suite-average in the project's history, exp0016 does not improve the worst-case suite (LIBERO-90) over the current accepted checkpoint — per the project's goal (maximize worst-case, not average), this is not a better main-line parent.
- **Retention gate passed:** yes, with the strongest margin yet on all four original suites (90.59% LIBERO-90 is still above the original 90% floor, and every other suite clears the new 95% target).
- **exp0014 remains the accepted main-line checkpoint** (LIBERO-90 91.09% canonical is still the best worst-case-suite result in the project).
- exp0016's checkpoint is kept locally (not HF-uploaded) — its main value is as evidence, not as a candidate parent, though it is worth noting as the strongest-ever result on the four original suites if a future candidate needs a "safe" starting point for suite retention.

## 10. What this changes for the next experiment

**Three consecutive generic-training variants (exp0014's extension, exp0015's further extension, exp0016's warm-restart) have failed to produce a further material LIBERO-90 gain.** This is now strong, convergent evidence that the "more/different generic training on the unchanged data mix" lever has been exhausted for this suite specifically, even though it continues to help the four original suites. The next candidate should pursue the task-level data-engineering direction flagged in `PROGRESS_0015`: build a filtered, oversample-able subset of LIBERO-90's own weakest task categories (task 51's data gap, and the book-in-caddy cluster: tasks 73/75/81/82/83) so the data mix can target LIBERO-90's actual weak points directly, rather than relying on suite-level or global training-budget levers that have now visibly plateaued.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0016_lr_restart/train.log` (live copy: `/tmp/.../scratchpad/train_exp0016.log`)
- full-state checkpoint (preserved): `/home/claudeuser/fastwam_state_scratch/exp0016_lr_restart/state/step_008000/`
- config(s): unchanged from exp0003/0006/0009/0010/0013/0014/0015 except `learning_rate` override
- fix commits: `99826cc`, `b2b49d0`
- eval results: `evaluate_results/exp0016_cheap_panel/`, `evaluate_results/exp0016_confirmation/`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created (n/a — not uploaded, see Section 9)
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — confirmation-level evidence was sufficient for REJECT)
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
