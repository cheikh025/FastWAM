# PROGRESS_0018 — libero90_weak_task_oversampling_v2

- **Experiment ID:** 0018_libero90_weak_task_oversampling_v2
- **Status:** `PROMOTE`
- **Created:** 2026-08-13
- **Updated:** 2026-08-14
- **Parent experiment:** 0017_libero90_weak_task_oversampling (`PROMOTE`, current accepted main-line checkpoint)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0017_weak_task_oversample/checkpoints/weights/step_008000.pt` (weights-only resume)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

exp0017 (4x oversampling of 7 weak LIBERO-90 tasks) produced a new project-best LIBERO-90 (92.60% canonical) with all four original suites clearing the updated 95% goal, but task-level analysis showed the gain came mostly from 3 of the 7 targeted tasks (bottom-drawer, ketchup-drawer, book-under-shelf), with the primary "book in caddy" cluster showing only mixed/marginal movement, and 4 new weak points emerging that weren't targeted (task23, 64, 65, 66). This candidate broadens the weak-task subset to 11 tasks (the original 7 plus the 4 newly-emerged ones), keeping the same validated mechanism (weights-only resume, 4x oversampling, 8000 steps) — testing whether wider coverage of the current weakest-task set produces a further gain.

## 2. Research state before experiment

exp0017 is the accepted main-line checkpoint: LIBERO-Spatial 96.80%, LIBERO-Object 99.60%, LIBERO-Goal 97.20%, LIBERO-Long/10 96.00% (all clear the 95% goal), **LIBERO-90 92.60%** (canonical) — the best result in the project, still 2.40pp short of the 95% goal for LIBERO-90 specifically. Per-task analysis of the weakest 15 LIBERO-90 tasks (canonical) identified: task51 (32%, confirmed zero-demo data gap, excluded from this candidate), task81 (36%, book-in-caddy, still weak), task23 (50%, newly emerged, not previously targeted), task64/65/66 (60-78%, newly emerged), task21/63/75 (66%, mixed), task73/6/32/83 (70-80%, improved substantially from exp0014 but not fully solved).

## 3. Candidate design

### Modifications

1. **Broadened filtered subset dataset** `data/libero_mujoco3.3.2/libero_90_weak_subset_v2_lerobot/` (654 episodes, 11 target tasks): the original 7 from exp0017 (book-in-caddy x4 variants, ketchup-drawer, bottom-drawer-open, book-under-shelf) plus 4 newly-emerged weak tasks from exp0017's canonical eval — "close the bottom drawer of the cabinet and open the top drawer" (34 episodes), "stack the right bowl on the left bowl and place them in the tray" (37 episodes), "put the red mug on the left plate" (34 episodes), "put the red mug on the right plate" (45 episodes). Built with the same corrected script (rewrites parquet `episode_index`/`index`/`task_index` columns, symlinks video files) validated for exp0017. Verified via a direct `BaseLerobotDataset` smoke test (0 errors, 53 random samples) before launching.
2. **New data config** `configs/data/libero_2cam_plus90_reweighted_weaktasks_v2.yaml`: identical to exp0017's recipe except the weak-subset directory is the broadened v2 (654 episodes, listed 4x = 2616 episodes, ~19% of the resulting ~13,600-episode mix).
3. **New task config** `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v2_3e-5.yaml`.
4. `resume=./runs/reweighted_libero90_finetune/exp0017_weak_task_oversample/checkpoints/weights/step_008000.pt` (weights-only resume from the new best checkpoint).
5. `learning_rate=3e-5`, `max_steps=8000` — identical scale to exp0017 for a clean comparison.
6. **Cache pre-warmed via a real 4-rank `accelerate launch` dry run before the actual training launch** (lesson from exp0017's extensive debugging saga — see `research/NOTES.md`), and full process/GPU cleanup verified before the real launch.

### Why this candidate

Directly extends the one validated real lever (task-level data engineering) with the most direct, low-risk next step: broaden coverage to the current weakest-task set rather than assume the same 7 tasks are still the right target. Uses only already-validated mechanisms (weights-only resume, filtered-subset oversampling) — no new infrastructure risk.

### What to watch

- Whether the 4 newly-added tasks (23, 64, 65, 66) show a clear improvement.
- Whether the book-in-caddy cluster (still muted in exp0017) responds any differently with more of the total data-mix "budget" going toward LIBERO-90-weak-task episodes.
- Whether LIBERO-90 aggregate improves beyond exp0017's 92.60%.
- Whether the four original suites hold their newly-achieved 95%+ margins (2616 weak-subset episodes is a somewhat larger share of the mix than exp0017's 2016, slightly more dilution risk).
- Task51: expected to remain unchanged/noisy (not included, still a data gap).

### Initial compute plan

- Initial training budget: `max_steps=8000` (~7h, matching exp0017's scale for a clean comparison).
- Checkpoint/save plan: `save_every=500`, prune intermediate weights aggressively given disk pressure observed during exp0017 (was as low as ~113GB free at one point).
- No interim progress check planned by default (GPU contention with eval), consistent with prior rounds.

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes) + new, uncommitted config/data files and new data directory (`data/libero_mujoco3.3.2/libero_90_weak_subset_v2_lerobot/`).
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_weaktasks_v2_3e-5`
- Config overrides: `learning_rate=3e-5 max_steps=8000`
- Resume source: `runs/reweighted_libero90_finetune/exp0017_weak_task_oversample/checkpoints/weights/step_008000.pt` (weights-only)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`). Venv: `/workspace/venv-fastwam`.

## 6. Training execution and control timeline

- Exact launch command: see below (launched 2026-08-13, into a verified-clean environment with a pre-warmed cache from the start)
- Start time: 2026-08-13 (immediately after exp0017's records were finalized and GPUs freed — no idle time)
- Status: running

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_002500 | ~1h47min | `progress_check` | Discussed with user whether `max_steps=8000` was necessary for this small an incremental data-mix change (weak-subset grew 504->654 episodes, ~87% of the recipe unchanged from exp0017). Loss diagnostics (steps 0-2500) reviewed: train loss noisy 0.10-0.25, val_loss noisy 0.12-0.34, action_l2 healthy 0.003-0.04 — no clear trend either way, ordinary noise, not decisive alone. | `STOP_TRAINING` (early) + `SELECT_CHECKPOINT` (step_002500) for a screen, before deciding whether to extend | Stopped at step 2500/8000 (~31%) rather than continuing to the original 8000-step budget; screening step_002500.pt via cheap panel before deciding whether more training is warranted (possibly via a properly-shaped shorter-schedule restart rather than continuing this run, given the LR-schedule-shape caveat: this checkpoint's schedule was built for 8000 steps, so it isn't a "converged, fully-decayed" checkpoint the way exp0017's final one was) |

### Why training ended

Stopped early (`STOP_TRAINING`, user-directed efficiency check) at step 2500/8000 rather than planned completion — see progress-decision row above. **Operational note**: stopping required both a `SIGTERM` to the wrapper script *and* an explicit `pkill` sweep for `scripts/train.py`/`accelerate launch` — the wrapper exiting did not kill the underlying distributed worker processes (all 4 ranks + dataloader workers stayed alive, GPUs stayed fully loaded) until they were killed directly. Consistent with the process-hygiene lesson from exp0017 (see `research/NOTES.md`) — always verify full cleanup (`ps aux` + `nvidia-smi`), not just that the top-level wrapper PID is gone.

### Full completion (after directory-resume continuation)

- Resumed via `resume=./runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/state/step_002500` (directory/full-state resume), `max_steps=8000` unchanged.
- Resync confirmed correct: `Re-synchronized LR scheduler to global_step=2500 (shape: total_train_steps=8000, warmup_steps=400); lr=2.4748e-05` — exactly matching the LR at the point training was stopped (2.47e-05), confirming a genuine continuation, not a fresh restart.
- End time: 2026-08-13 ~21:23 UTC (`max_steps reached step=8000`, clean exit, no crash — this time all distributed worker processes exited fully on their own, unlike the earlier `SIGTERM`-to-wrapper-only stop which left the underlying `accelerate`/`torchrun` processes running until explicitly killed — see the process-hygiene note in Section 6 above).
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt` (12,041,735,545 bytes)
- Final training diagnostics at step 8000: `val_loss=0.1771 infer_psnr=28.6849 infer_ssim=0.8690 action_l2=0.0050 action_l1=0.0344` — healthy, normal range throughout the continuation (steps 2500-8000: val_loss noisy 0.09-0.27, action_l2 healthy 0.003-0.02, no anomalies).

## 7. Evaluation events

### Event 1 — `progress_check` (cheap panel on step_002500)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_002500.pt` (early stop, ~31% through the planned 8000-step budget, LR not yet fully decayed — still ~2.47e-5 at stop time)
- Reference: exp0017's final canonical result (LIBERO-90 92.60%, Spatial 96.80%, Object 99.60%, Goal 97.20%, Long 96.00%) and exp0017's own cheap panel (Spatial 85%, Object 100%, Goal 100%, Long 100%, LIBERO-90 92%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0018_progress_step2500/` (first launch attempt hit the tmux-death infra bug again, discarded, relaunched cleanly)

**Result — clear regression relative to exp0017, confirming the mid-schedule checkpoint is genuinely under-converged, not just "good enough sooner":**

| Suite | exp0017 (panel) | exp0018 step_002500 (panel) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 85.00% | 95.00% | +10pp |
| LIBERO-Object | 100.00% | 100.00% | 0pp |
| LIBERO-Goal | 100.00% | **70.00%** | **-30pp** |
| LIBERO-Long/10 | 100.00% | **75.00%** | **-25pp** |
| LIBERO-90 (5-task) | 92.00% | 80.00% | -12pp |
| **Overall** | **95.40%** | **84.00%** | **-11.4pp** |

Goal task0 ("open the middle drawer of the cabinet" — the historically-hardest task in the project, only recovered to 100% after extensive training in earlier experiments) and Long task8 ("put both moka pots on the stove") both dropped to **0%** at this intermediate checkpoint. This is decisive evidence against the "8000 steps might be more than necessary" hypothesis at the *early-stop* end: a checkpoint pulled mid-schedule is not simply "a smaller version of the final result" — it's genuinely regressed on some suites relative to its own starting point (exp0017), consistent with the well-established pattern in this project (see `research/STATE.md`'s retention chart) where original-suite performance dips mid-training before recovering as the LR fully decays.

**Decision: `CONTINUE_TRAINING`.** Resumed via directory/full-state resume from `checkpoints/state/step_002500` (optimizer momentum preserved) back toward the original `max_steps=8000` target — this answers the efficiency question with real evidence (early-stopping at ~31% produces a worse checkpoint, not just a "good enough, cheaper" one) while avoiding wasted compute by resuming rather than restarting from scratch. The efficiency lesson for *future* similar candidates is still valid (see Section 1) — it should be applied by choosing a shorter `max_steps` *from the start* of a new launch, not by truncating a run already in progress.

### Event 2 — `candidate_screen` (cheap panel, step 8000, full schedule completed)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt`
- Reference: exp0017's panel (Spatial 85%, Object 100%, Goal 100%, Long 100%, LIBERO-90 92%, overall 95.40%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0018_cheap_panel/`

**Result — best-ever LIBERO-90 panel score (first-ever perfect 100%):**

| Suite | exp0017 (panel) | exp0018 (panel) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 85.00% | 90.00% | +5pp |
| LIBERO-Object | 100.00% | 100.00% | 0pp |
| LIBERO-Goal | 100.00% | 95.00% | -5pp |
| LIBERO-Long/10 | 100.00% | 100.00% | 0pp |
| LIBERO-90 (5-task) | 92.00% | **100.00%** | **+8pp (first-ever perfect)** |
| **Overall** | **95.40%** | **97.00%** | **+1.6pp** |

Task73 ("book in front compartment of caddy" — historically one of the weakest, part of the muted-response caddy cluster in exp0017) scored a perfect 5/5 this time. Given the strength of this result, confirmation evaluation (full 130-task coverage, 15 trials/task) launched immediately.

### Event 3 — `confirmation` (full 130-task coverage, 15 trials/task)

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=15`, fresh `output_dir=evaluate_results/exp0018_confirmation`
- Runtime: ~1h50min (2026-08-13 21:38 -> ~23:28 UTC), all 130 tasks completed, 0 failures.

**Result — best confirmation-level result in the project's history, LIBERO-90 within 0.26pp of the 95% goal:**

| Suite | exp0017 confirmation | exp0017 canonical | exp0018 confirmation |
|---|---:|---:|---:|
| LIBERO-Spatial | 94.00% | 96.80% | 94.67% |
| LIBERO-Object | 100.00% | 99.60% | 99.33% |
| LIBERO-Goal | 100.00% | 97.20% | **100.00%** |
| LIBERO-Long/10 | 96.67% | 96.00% | 95.33% |
| **LIBERO-90** | 91.63% | 92.60% | **94.74%** |
| Overall | 96.46% | 96.44% | **96.81%** |

LIBERO-90's confirmation-level running score held remarkably steady throughout the evaluation (97.8% at 21/90 tasks, 95.7% at 51/90, 95.3% at 61-77/90, settling at 94.74% final) — a much more stable trajectory than earlier candidates, suggesting a genuinely more consistent improvement rather than a few lucky tasks dominating a small sample. Canonical (50-trial) evaluation launched immediately given how close this is to the 95% goal.

## 8. Comparison and interpretation

Training-control note: the mid-run progress-check + directory-resume continuation (see Sections 6-7, Event 1) was itself informative — it demonstrated concretely that early-stopping this recipe at ~31% through its schedule produces a genuinely worse checkpoint (Goal 70%, Long 75% on the panel) than either its starting point (exp0017) or its own eventual completion (Goal 95%, Long 100%), confirming the full schedule was needed for this candidate specifically, separate from the general efficiency question about *future* candidates' budget sizing.

The final result is encouraging: LIBERO-90's cheap-panel score reached its first-ever perfect 100%, including a full recovery on the previously-muted-response task73. Confirmation evaluation will show whether this holds at full 90-task coverage.

### Event 4 — `canonical` (full 130-task coverage, 50 trials/task) — promotion-grade evidence

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=50`, fresh `output_dir=evaluate_results/exp0018_canonical`
- Start time: 2026-08-13 ~23:29 UTC
- End time: 2026-08-14 ~04:57 UTC (~5h28min, all 130 tasks completed, 0 failures). Verified: every suite at exactly 50/50 trials, no partial data.

**Result — LIBERO-90 clears the 95% goal for the first time ever; 4 of 5 suites now clear 95% simultaneously:**

| Suite | Canonical (50 trials/task) | vs 90% floor | vs 95% goal | vs exp0017 canonical |
|---|---:|---|---|---:|
| LIBERO-Spatial | 94.80% | meets | just below (-0.20pp) | -2.00pp |
| LIBERO-Object | 99.60% | meets | **meets** | 0.00pp |
| LIBERO-Goal | 97.40% | meets | **meets** | +0.20pp |
| LIBERO-Long/10 | 95.20% | meets | **meets** | -0.80pp |
| **LIBERO-90** | **95.49%** | meets | **meets — first time ever** | **+2.89pp** |
| **Overall** | **96.50%** | — | — | +0.06pp |

**LIBERO-90, the project's primary target throughout, clears the 95% goal for the first time.** Only LIBERO-Spatial falls short of the 95% goal, and only by 0.20pp (94.80%, down from exp0017's 96.80%) — the closest the project has ever come to satisfying the full updated goal on all five suites simultaneously.

**Task-level**: every one of the 11 targeted weak tasks improved over exp0017's canonical values, several substantially — task81 (36%->82%, +46pp), task23 (50%->88%, +38pp), task64 (60%->84%, +24pp), task73 (70%->86%, +16pp), task6 (80%->92%, +12pp), task51 (32%->58%, +26pp, notable even without training signal — plausibly some indirect transfer from broader improvement, though still the single weakest task in the project). Only task75 (68%, book-in-right-compartment) remains a clear laggard within the targeted set. Two non-targeted tasks (task21 "stove+pan" 60%, task27 "wine rack" 70%) are now among the weakest, alongside task17/63 (bowl-stacking variants, 82-84%, related to the targeted task64 but not identical).

## 9. Decision

- **Decision:** `PROMOTE`. exp0018's checkpoint (`step_008000.pt`) becomes the new main-line accepted checkpoint, replacing exp0017.
- **Retention gate passed:** yes, all four original suites clear 90% with real margin (94.80-99.60%).
- **Primary target (LIBERO-90):** 95.49% at canonical precision — **the first checkpoint in the project to clear the project's updated 95% goal on LIBERO-90**, a +2.89pp gain over exp0017. Consistent with confirmation (94.74%, +0.75pp — canonical came in slightly higher, within normal noise).
- **Overall goal status**: 4 of 5 suites (LIBERO-90, Object, Goal, Long) now clear 95% simultaneously for the first time. Only LIBERO-Spatial remains, at 94.80% — 0.20pp short.

## 10. What this changes for the next experiment

1. **New main-line accepted checkpoint**: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt`. `research/STATE.md` updated accordingly.
2. **Broadening the targeted weak-task subset was clearly effective** — every one of the 11 targeted tasks improved, several substantially. This validates iterating the same approach further: the next candidate could add the newly-emerged weak tasks (task21 "stove+pan", task27 "wine rack", task17/63 bowl-stacking variants) to a v3 subset, following the exact same validated mechanism.
3. **LIBERO-Spatial is now the sole worst-case suite** (94.80%, just 0.20pp short of 95%) — a genuinely different situation from every prior round, where LIBERO-90 was always the clear bottleneck. Worth checking Spatial's own per-task breakdown for a quick, targeted fix (e.g. task_spatial_4 scored 76.00% canonical, the weakest Spatial task) rather than assuming it needs the same LIBERO-90-focused intervention.
4. **Task51's data gap remains real** but showed unexpected movement (32%->58%) despite having no direct training signal — worth a note of caution against over-interpreting single-task movements at this trial count, though it's not concerning since it's still comfortably the weakest task and the explanation (zero exact-matching demos) still holds.
5. Given how close the full goal now is (4/5 suites clearing 95%, the 5th only 0.20pp short), the next candidate has a realistic chance of completing the project's active goal outright.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/train.log`
- new data directory: `data/libero_mujoco3.3.2/libero_90_weak_subset_v2_lerobot/`
- new config(s): `configs/data/libero_2cam_plus90_reweighted_weaktasks_v2.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v2_3e-5.yaml`
- fix commits: `99826cc`, `b2b49d0`

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
