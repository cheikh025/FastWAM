# PROGRESS_0014 — continued_from_exp0013_dirresume

- **Experiment ID:** 0014_continued_from_exp0013_dirresume
- **Status:** `PROMOTE`
- **Created:** 2026-08-11
- **Updated:** 2026-08-12
- **Parent experiment:** 0013_continued_from_exp0010 (`PROMOTE`)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/state/step_008000` (full accelerate/deepspeed state, directory resume)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

Direct continuation of the newly-promoted exp0013 checkpoint via **directory/full-state resume** (not weights-only) — the first real production use of the validated directory-resume fix (commit `b2b49d0`). exp0013's own 8000-step schedule had fully decayed to `lr≈3.0e-7` by its end; extending `max_steps` from 8000 to 14000 and resuming from the full-state directory rebuilds the scheduler for the new 14000-step shape and fast-forwards it to `global_step=8000`, which correctly resumes at a substantially higher LR (~1.6e-5, the mathematically correct position 8000/14000 through a 14000-step cosine schedule) rather than continuing at the near-zero LR exp0013's own (too-short) schedule ended on. This preserves optimizer momentum/variance state exactly, unlike exp0013's own weights-only resume from exp0010.

## 2. Research state before experiment

exp0013 was just promoted (canonical): LIBERO-Spatial 94.40%, LIBERO-Object 98.00%, LIBERO-Goal 97.40%, LIBERO-Long/10 93.00% (all clear the 90% retention floor with real margin), **LIBERO-90 88.49%** — 1.51pp short of the project's 90% target, the sole remaining gap. The exp0009→0010→0013 trend (LIBERO-90: 48%→72%→76%→88.49%) has shown no plateau across three consecutive extensions, making more training the best-evidenced next lever. exp0013's own full-state checkpoint was deliberately preserved (not pruned) specifically to enable this true continuation.

## 3. Candidate design

### Modifications

1. `resume=./runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/state/step_008000` (directory/full-state resume of exp0013, not weights-only).
2. `max_steps=14000` (extends exp0013's own 8000-step schedule by 6000 steps — a similar relative extension to exp0009→0010's +4000/1.5x).
3. No other recipe changes (same reweighted goal5x/long5x mix, same `task=libero_uncond_2cam224_plus90_reweighted_3e-5`).

### Why this candidate

Directory resume is the semantically correct tool for "keep this exact run going longer" (preserves optimizer momentum, continues from the exact trained state) and is now fully validated (commit `b2b49d0`, diagnostic-tested). The trend has shown consistent, large, non-plateauing gains from more training on this exact recipe across three prior extensions; the remaining LIBERO-90 gap (1.51pp) is small enough that a moderate further extension is plausible to close it, especially with optimizer state preserved this time (a genuine advantage over exp0013's own weights-only continuation from exp0010).

### What to watch

- Whether LIBERO-90 continues climbing past 88.49% toward/past 90%.
- Whether any of the four original suites (especially LIBERO-Long/10 at 93.00% and LIBERO-Spatial at 94.40%, the closest to the 90% floor) regress as LIBERO-90 improves further.
- Early training health (val_loss/action_l2, LR trajectory) to confirm the resync fix continues to behave correctly in a real multi-thousand-step run (not just the short diagnostic).

### Initial compute plan

- Initial training budget: `max_steps=14000` (6000 new steps from the resumed `global_step=8000`, ~5-6h at established throughput).
- Checkpoint/save plan: `save_every=500`, only the newest full-state checkpoint kept locally at any time (disk-monitor pattern), preserved (not pruned to zero) at run end for a possible further continuation.
- Progress check: plan a cheap-panel progress check around step ~11000-12000 if GPU contention allows, to decide whether to extend further, stop early, or proceed to confirmation/canonical evaluation at the full budget.
  - **Decision at step ~10260 (2026-08-11 ~16:08 UTC): skip the interim progress check.** All 4 GPUs are saturated by training (eval cannot run concurrently under ZeRO-1, per the established operational lesson in `research/RUNBOOK.md`/`STATE.md`), remaining budget is small (~3700 steps, ~3h), and the trend across three prior extensions has shown no plateau or instability requiring early intervention. Cheaper to let training reach `max_steps=14000` and evaluate the final checkpoint directly (`CONTINUE_TRAINING`).

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=14000`
- Resume source: `runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/state/step_008000` (full accelerate/deepspeed state — directory resume)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`).

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  source /workspace/venv-fastwam/bin/activate
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0013_continued_fixed/checkpoints/state/step_008000 \
    max_steps=14000 \
    save_every=500 \
    output_dir=./runs/reweighted_libero90_finetune/exp0014_continued_dirresume \
    wandb.name=exp0014_continued_from_exp0013_dirresume
  ```
- Start time: 2026-08-11 ~14:13 UTC (immediately after exp0013's canonical evaluation completed and GPUs freed — no idle time)
- End time: 2026-08-11 ~19:19 UTC (`max_steps reached step=14000`, clean exit, no crash, no OOM, no disk-full warning — runtime ~5h6min for the 6000 new steps)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt` (10,525,605,888 bytes; loads cleanly, 1649 tensors, bf16 — verified via direct load, same tensor count/dtype as exp0013's checkpoint; the byte-size difference from exp0013's `.pt` is non-tensor serialization overhead, not missing content)
- Final training diagnostics at step 14000: `val_loss=0.0694 infer_psnr=28.3011 infer_ssim=0.8797 action_l2=0.0059 action_l1=0.0370`, `lr=3.00e-07` (schedule fully decayed, as expected at the end of the 14000-step schedule)
- Note on incidental infra issue: the first launch attempt failed immediately (`accelerate: command not found`) because the default `/venv/main` venv doesn't have this project's dependencies — this project uses a dedicated venv at `/workspace/venv-fastwam`. Relaunched correctly with that venv activated; no impact on the run itself (failed attempt produced no checkpoints).
- Training health check (real-run validation of the directory-resume fix): no discontinuity or "relearning from scratch" signature at the resume boundary. `action_l2` stayed in the same low 0.003–0.025 band throughout (e.g. step 8200: 0.0091, step 10600: 0.0172, step 13400: 0.0106) rather than jumping back to the ~0.087 "randomly-initialized action head" level seen in the historically broken resumes — confirms the fix correctly preserved the trained action head and optimizer state in a real multi-thousand-step run, not just the short diagnostic.
- Final full-state checkpoint (`step_014000`) preserved on the scratch disk (older `step_013500` pruned), keeping a true directory-resume extension option available if needed.

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_010260 | ~2h | `progress_check` (considered) | GPUs fully saturated by training (eval cannot run concurrently under ZeRO-1); ~3700 steps / ~3h remaining; no plateau/instability signal in loss curve | `CONTINUE_TRAINING` (interim check skipped as not cost-effective) | none — proceed to `max_steps=14000` |

### Why training ended

Planned completion — reached `max_steps=14000` cleanly with no early-stop trigger.

### Training anomalies

None during the successful run. (One unrelated launch-command error before the run started, see above — no training compute was spent on the failed attempt.)

## 7. Evaluation events

### Event 1 — `candidate_screen` (cheap panel)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt`
- Reference: exp0013's step_008000.pt panel result (Spatial 95%, Object 90%, Goal 95%, Long 100%, LIBERO-90 80%, overall 92.0%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0014_cheap_panel_v2/` (note: first launch attempt, `exp0014_cheap_panel/`, hit a stuck-tmux infra bug — see `research/NOTES.md` — and was discarded with zero real results produced; re-launched cleanly as `_v2`)

**Result — strong improvement across every suite, including a first-time-ever >90% LIBERO-90 panel score:**

| Suite | exp0013 (panel) | exp0014 (panel) | Delta | vs 90% floor |
|---|---:|---:|---:|---|
| LIBERO-Spatial | 95.00% | **100.00%** | +5pp | **meets** |
| LIBERO-Object | 90.00% | **100.00%** | +10pp | **meets** |
| LIBERO-Goal | 95.00% | **100.00%** | +5pp | **meets** |
| LIBERO-Long/10 | 100.00% | **100.00%** | +0pp | **meets** |
| LIBERO-90 (5-task) | 80.00% | **92.00%** | +12pp | **meets (first time)** |
| **Overall** | **92.00%** | **98.40%** | **+6.4pp** | — |

Task-level: `libero_90` task 19 ("put the moka pot on the stove") and task 73 ("book in front compartment of caddy") were the only misses (4/5 each, 80%); all other panel tasks across all five suites hit 5/5 (100%).

Given the strength of this result (first-ever >90% LIBERO-90 panel score, all four original suites perfect on the panel), a confirmation evaluation (full 130-task coverage, 15 trials/task) was launched immediately to check whether this generalizes, following the same adaptive-evaluation pattern used for exp0013.

### Event 2 — `confirmation` (full 130-task coverage, 15 trials/task)

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=15`, fresh `output_dir=evaluate_results/exp0014_confirmation`
- Runtime: ~1h (2026-08-11 20:22 -> ~21:22 UTC), all 130 tasks completed, 0 failures.

**Result — LIBERO-90 clears the 90% target for the first time in the project:**

| Suite | Confirmation (15 trials/task) | vs 90% floor |
|---|---:|---|
| LIBERO-Spatial | 97.33% | **meets** |
| LIBERO-Object | 100.00% | **meets** |
| LIBERO-Goal | 97.33% | **meets** |
| LIBERO-Long/10 | 93.33% | **meets** |
| LIBERO-90 (all 90 tasks) | **91.19%** | **meets — first time ever** |
| **Overall** | **95.84%** | — |

Given this crosses the 90% target with real evidence (all 90 tasks, 15 trials each = 1350 LIBERO-90 episodes), canonical (50-trial) evaluation was launched immediately as the promotion-grade confirmation, following the same evidence ladder used for exp0013.

### Event 3 — `canonical` (full 130-task coverage, 50 trials/task) — promotion-grade evidence

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=50`, fresh `output_dir=evaluate_results/exp0014_canonical`
- Start time: 2026-08-11 ~21:23 UTC
- End time: 2026-08-12 ~03:14 UTC (~5h51min, all 130 tasks completed, 0 failures)
- Raw results: `evaluate_results/exp0014_canonical/`; verified via direct per-task JSON aggregation (every task at exactly 50/50 trials, no partial data)

**Result — canonical, promotion-grade evidence — ALL FIVE SUITES CLEAR 90% FOR THE FIRST TIME IN THE PROJECT:**

| Suite | Canonical (50 trials/task) | vs 90% floor | vs exp0013 canonical |
|---|---:|---:|---:|
| LIBERO-Spatial | 97.20% | **meets** | +2.80pp |
| LIBERO-Object | 100.00% | **meets** | +2.00pp |
| LIBERO-Goal | 96.60% | **meets** | -0.80pp |
| LIBERO-Long/10 | 93.40% | **meets** | +0.40pp |
| LIBERO-90 (all 90 tasks) | **91.09%** | **meets — first time ever** | +2.60pp |
| **Overall** (suite average) | **95.66%** | — | +1.40pp |

LIBERO-90 landed at 91.09%, within 0.10pp of the confirmation run's 91.19% — the same tight confirmation-to-canonical consistency seen with exp0013 (88.59% -> 88.49%, also 0.10pp), so this is a precise, reliable measurement, not a lucky sample. Every original suite improved or stayed flat relative to exp0013's canonical result except Goal, which dipped slightly (97.40% -> 96.60%) but remains comfortably clear of the 90% floor.

## 8. Comparison and interpretation

The cheap panel showed a clean, large improvement over exp0013 on every axis at once (not a trade-off), and both the confirmation and canonical evaluations validated this at full precision: LIBERO-90 91.09-91.19%, above the project's original 90% target for the first time, while every original suite stayed comfortably clear of the retention floor (93.40-100.00%). This is a stronger and more complete result than exp0013's own canonical (88.49%, 1.51pp short) — the additional 6000 steps of directory-resume continuation (preserving optimizer momentum, unlike exp0013's own weights-only continuation from exp0010) closed the remaining gap and then some. The "more training helps" trend (exp0009 -> 0010 -> 0013 -> 0014) has still shown no plateau across four consecutive extensions.

## 9. Decision

- **Decision:** `PROMOTE`. exp0014's checkpoint (`step_014000.pt`) becomes the new main-line accepted checkpoint, replacing exp0013.
- **Retention gate passed:** yes, with strong margin at canonical precision: LIBERO-Spatial 97.20%, LIBERO-Object 100.00%, LIBERO-Goal 96.60%, LIBERO-Long/10 93.40%.
- **Primary target (LIBERO-90):** 91.09% at canonical precision — **the first checkpoint in the project to clear the 90% target on every one of the five suites simultaneously**, satisfying the project's original goal. Consistent (within 0.1pp) between confirmation and canonical evaluation.
- **Note on the project's evolved goal**: the active research goal was updated (2026-08-12) to target >=95% on all five suites, maximizing the worst-case suite. Under that higher bar, LIBERO-90 (91.09%) and LIBERO-Long/10 (93.40%) are now the two weakest suites and the natural focus for the next candidate; the other three (Spatial 97.20%, Object 100.00%, Goal 96.60%) already clear 95%.

## 10. What this changes for the next experiment

1. **New main-line accepted checkpoint**: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/weights/step_014000.pt`. `research/STATE.md` updated accordingly.
2. **Original project goal met** (all five suites >=90%) — this is the first checkpoint to do so. The active goal has since moved to >=95% on all five suites, with explicit priority on the current worst-case suite.
3. **Worst-case suite is now LIBERO-90 (91.09%)**, with LIBERO-Long/10 (93.40%) second-weakest — both are the natural targets for the next candidate under the updated goal, while Spatial/Object/Goal (96.60-100.00%) already clear the new 95% bar and mainly need to be preserved, not necessarily improved further.
4. **"More training" continues to be the best-evidenced lever with no plateau across four consecutive extensions** (exp0009->0010->0013->0014). The full-state checkpoint for this run should be preserved to enable another directory-resume continuation as the next candidate, consistent with the same validated mechanism.
5. **Verified per-task LIBERO-90 weak points (canonical, 50 trials/task)** — the 15 weakest tasks:
   ```
   38.00%  task 51  pick up the butter and put it in the basket        <- ZERO training demos (known data gap, see STATE.md)
   44.00%  task 32  put the ketchup in the top drawer of the cabinet
   46.00%  task 81  pick up the book and place it in the front compartment of the caddy
   54.00%  task  6  open the bottom drawer of the cabinet
   56.00%  task 89  pick up the book on the right and place it under the cabinet shelf
   64.00%  task 21  turn on the stove and put the frying pan on it
   66.00%  task 27  put the wine bottle on the wine rack
   66.00%  task 63  stack the left bowl on the right bowl and place them in the tray
   66.00%  task 73  pick up the book and place it in the front compartment of the caddy
   66.00%  task 75  pick up the book and place it in the right compartment of the caddy
   74.00%  task 33  close the microwave
   74.00%  task 82  pick up the book and place it in the left compartment of the caddy
   76.00%  task 83  pick up the book and place it in the right compartment of the caddy
   78.00%  task 23  close the bottom drawer of the cabinet and open the top drawer
   78.00%  task 30  put the black bowl on the plate
   ```
   Two clear patterns: (a) **task 51 is a genuine data gap** (zero training demonstrations, documented in `research/STATE.md`'s data-coverage note) — likely needs a targeted data fix (e.g. sourcing/synthesizing demos) rather than more generic training, since there is nothing to learn from; (b) **the "book in caddy compartment" category remains the weakest cluster overall** (5 of the 15 weakest tasks: 81, 73, 75, 82, 83, all in the 46-76% range) — consistent with this category's known history of being slow to improve. A future candidate could target this cluster specifically (e.g. oversampling book/caddy demonstrations within the LIBERO-90 data) if generic continued training alone plateaus on it.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/train.log` (live copy: `/tmp/.../scratchpad/train_exp0014.log`)
- full-state checkpoint (preserved): `/home/claudeuser/fastwam_state_scratch/exp0014_continued_dirresume/state/step_014000/`
- config(s): unchanged from exp0003/0006/0009/0010/0013
- fix commits: `99826cc`, `b2b49d0`
- eval results: `evaluate_results/exp0014_cheap_panel_v2/`, `evaluate_results/exp0014_confirmation/`, `evaluate_results/exp0014_canonical/`
- HF backup (promoted): `cheikh025/ASR:promoted/0014_continued_from_exp0013_dirresume/{step_014000.pt,dataset_stats.json,config.yaml,PROGRESS_0014_continued_from_exp0013_dirresume.md}`

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
- [x] five-suite metrics recorded when canonical evaluation ran
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
