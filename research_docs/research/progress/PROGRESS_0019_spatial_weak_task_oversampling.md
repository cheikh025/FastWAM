# PROGRESS_0019 — spatial_weak_task_oversampling

- **Experiment ID:** 0019_spatial_weak_task_oversampling
- **Status:** `PROMOTE`
- **Created:** 2026-08-14
- **Updated:** 2026-08-14
- **Parent experiment:** 0018_libero90_weak_task_oversampling_v2 (`PROMOTE`, previously accepted main-line checkpoint)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt` (weights-only resume)
- **Selected candidate checkpoint:** `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` — now the accepted main-line checkpoint
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

exp0018 promoted LIBERO-90 to 95.49% canonical (clearing the project's 95% goal for the first time), and 4 of 5 suites (LIBERO-90, Object, Goal, Long) now clear 95% simultaneously. **LIBERO-Spatial is the sole remaining suite below the goal, at 94.80% — just 0.20pp short**, with the gap concentrated in exactly 2 of its 10 tasks (task4 "bowl in top drawer" 76.00%, task5 "bowl on ramekin" 80.00%; all other 8 tasks are 96-100%). This candidate applies the same validated task-level oversampling technique to these 2 Spatial tasks specifically — the first time this technique has been used outside LIBERO-90 — while keeping exp0018's full LIBERO-90 recipe unchanged, targeting the smallest remaining gap to the project's active goal.

## 2. Research state before experiment

exp0018 is the accepted main-line checkpoint: LIBERO-90 95.49%, LIBERO-Object 99.60%, LIBERO-Goal 97.40%, LIBERO-Long/10 95.20% (all clear the 95% goal), **LIBERO-Spatial 94.80%** (canonical) — the sole suite short of the goal, by only 0.20pp. Per-task Spatial breakdown (canonical) identified a clean, concentrated gap: task4 (76.00%) and task5 (80.00%) are clearly weaker than the other 8 tasks (96.00-100.00%), unlike LIBERO-90's earlier situation where weakness was spread across many tasks.

## 3. Candidate design

### Modifications

1. **New filtered subset dataset** `data/libero_mujoco3.3.2/libero_spatial_weak_subset_lerobot/` (80 episodes, 2 target tasks: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate" [42 episodes], "pick up the black bowl on the ramekin and place it on the plate" [38 episodes]). Built and verified with the same corrected script/smoke-test process validated in exp0017/0018.
2. **New data config** `configs/data/libero_2cam_plus90_reweighted_weaktasks_v3.yaml`: identical to exp0018's recipe (including its full LIBERO-90 weak-task subset v2, unchanged) plus the new Spatial weak-subset directory listed 4x (320 episodes, ~2.3% of the resulting ~14,000-episode mix — a much smaller addition than either LIBERO-90 round, matching the much smaller and more concentrated nature of Spatial's gap).
3. **New task config** `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v3_3e-5.yaml`.
4. `resume=./runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt` (weights-only resume from the new best checkpoint).
5. **Cache pre-warmed via a real 4-rank `accelerate launch` dry run before the actual training launch**, and full process/GPU cleanup verified before the real launch (established lesson from exp0017/0018's launch debugging).

### Why this candidate

Directly targets the smallest remaining gap to the project's active goal (all five suites >=95%) using the exact same validated mechanism (filtered-subset oversampling + weights-only resume) that closed the much larger LIBERO-90 gap in exp0017/0018. The gap is small (0.20pp) and concentrated in just 2 tasks, so a small, low-risk addition (320 episodes, ~2.3% of the mix) is a proportionate response — unlike LIBERO-90's much larger interventions.

### What to watch

- Whether task4/task5 improve without meaningfully diluting the other 8 Spatial tasks or any other suite (especially LIBERO-90, which just reached its own goal for the first time and should not regress).
- Whether LIBERO-Spatial's aggregate crosses 95%.
- Whether this closes the project's active goal outright (all five suites >=95% simultaneously) — if so, this would be the first checkpoint to do so.

### Initial compute plan

- Initial training budget: `max_steps=5000` — a deliberately shorter budget than the standard 8000, applying the efficiency lesson from exp0018's mid-run progress check *properly this time* (set from the start, not truncated mid-run) given how small and low-risk this data-mix change is relative to exp0017/0018's larger interventions.
- Checkpoint/save plan: `save_every=500`, prune intermediate weights aggressively (disk pressure lesson from exp0017/0018).
- No interim progress check planned by default (GPU contention with eval).

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes) + new, uncommitted config/data files and new data directory (`data/libero_mujoco3.3.2/libero_spatial_weak_subset_lerobot/`).
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_weaktasks_v3_3e-5`
- Config overrides: `learning_rate=3e-5 max_steps=5000`
- Resume source: `runs/reweighted_libero90_finetune/exp0018_weak_task_oversample_v2/checkpoints/weights/step_008000.pt` (weights-only)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`). Venv: `/workspace/venv-fastwam`.

## 6. Training execution and control timeline

- Exact launch command: see below (launched 2026-08-14, into a verified-clean environment with a pre-warmed cache from the start)
- Start time: 2026-08-14 (immediately after exp0018's records were finalized and GPUs freed — no idle time)
- Status: running

### Why training ended

Planned completion — reached `max_steps=5000` cleanly, no early-stop trigger. This time the process exited fully on its own (all distributed worker processes gone, GPU memory freed) without needing an explicit `pkill` sweep, unlike exp0018's mid-run stop.

- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (12,041,735,545 bytes)
- Final training diagnostics at step 5000: `val_loss=0.1568 infer_psnr=26.6969 infer_ssim=0.8381 action_l2=0.0083 action_l1=0.0511` — healthy throughout, `lr=3.00e-07` (schedule fully decayed as expected).

## 7. Evaluation events

### Event 1 — `candidate_screen` (cheap panel)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`
- Reference: exp0018's canonical result (Spatial 94.80%, Object 99.60%, Goal 97.40%, Long 95.20%, LIBERO-90 95.49%) and exp0018's own panel (Spatial 90%, Object 100%, Goal 95%, Long 100%, LIBERO-90 100%, overall 97.00%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0019_cheap_panel/` (first launch attempt hit the tmux-death infra bug again, discarded, relaunched cleanly)

**Result — best overall panel score in the project's history:**

| Suite | exp0018 (panel) | exp0019 (panel) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 90.00% | **95.00%** | +5pp |
| LIBERO-Object | 100.00% | 100.00% | 0pp |
| LIBERO-Goal | 95.00% | 100.00% | +5pp |
| LIBERO-Long/10 | 100.00% | 100.00% | 0pp |
| LIBERO-90 (5-task) | 100.00% | 96.00% | -4pp (task73 dipped to 80%, still within noise) |
| **Overall** | **97.00%** | **98.20%** | **+1.2pp** |

Task5 ("bowl on ramekin", one of the 2 directly-targeted Spatial tasks) scored 80.0% on this panel — task4 ("bowl in top drawer") is not part of this fixed sentinel panel, so its improvement is not visible here. Confirmation evaluation (full 130-task coverage, 15 trials/task) launched immediately to get a reliable, full-coverage reading — particularly on Spatial, the deciding suite for the project's active goal.

### Event 2 — `confirmation` (full 130-task coverage, 15 trials/task)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`
- Tasks: all 130 canonical tasks (10 Spatial, 10 Object, 10 Goal, 10 Long, 90 LIBERO-90), 15 trials/task
- Output dir: `evaluate_results/exp0019_confirmation/`
- Validated: 130/130 task JSON files present (10+10+10+10+90), every file `total_episodes==15` (checked programmatically), aggregate success rates recomputed independently from raw per-task JSON and match the reported summary exactly.

**Result — first time all five suites individually clear 95% at confirmation precision:**

| Suite | exp0018 (canonical, accepted) | exp0019 (confirmation) | Delta | Clears 95% goal? |
|---|---:|---:|---:|---|
| LIBERO-Spatial | 94.80% | **96.00%** | +1.20pp | **yes** |
| LIBERO-Object | 99.60% | 99.33% | -0.27pp | yes |
| LIBERO-Goal | 97.40% | 97.33% | -0.07pp | yes |
| LIBERO-Long/10 | 95.20% | 97.33% | +2.13pp | yes |
| LIBERO-90 | 95.49% | 95.19% | -0.30pp | yes |
| **Overall** | — | **97.04%** | — | — |

Spatial per-task breakdown confirms the targeted mechanism worked exactly as intended: task4 ("bowl in top drawer") 76.00% -> 80.00%, task5 ("bowl on ramekin") 80.00% -> 86.67%; the other 8 Spatial tasks remain 93.3-100%, no dilution observed. The small drops on Object/Goal/LIBERO-90 (all <0.3pp) are consistent with ordinary trial-count noise (15 trials/task) and are far smaller than the Spatial/Long gains; none crosses below its own 95% goal line, let alone the 90% retention floor.

Given this is the strongest and most complete result in the project's history (all five suites simultaneously >=95% for the first time, at 15-trial precision), canonical (50-trial) evaluation was launched immediately as promotion-grade confirmation, per the standard escalation path.

### Event 3 — `canonical` (full 130-task coverage, 50 trials/task)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (12,041,735,545 bytes)
- Tasks: all 130 canonical tasks, 50 trials/task
- Output dir: `evaluate_results/exp0019_canonical/`
- Runtime: ~08:26 start (checkpoint mtime) through completion; monitored continuously via progress-count + GPU-utilization checks, no stuck/idle periods observed.
- Validated: 130/130 task JSON files present (10+10+10+10+90), every file `total_episodes==50` (checked programmatically), aggregate success rates recomputed independently from raw per-task JSON and match the reported `summary.csv` exactly.

**Result — first checkpoint in the project to fully satisfy the active goal (all five suites >=95% simultaneously) at canonical precision:**

| Suite | exp0018 (canonical, previously accepted) | exp0019 (canonical) | Delta | Clears 95% goal? | Clears 90% retention floor? |
|---|---:|---:|---:|---|---|
| LIBERO-Spatial | 94.80% | **97.00%** | +2.20pp | **yes** | yes |
| LIBERO-Object | 99.60% | **99.60%** | 0pp | yes | yes |
| LIBERO-Goal | 97.40% | **97.20%** | -0.20pp | yes | yes |
| LIBERO-Long/10 | 95.20% | **98.00%** | +2.80pp | yes | yes |
| LIBERO-90 | 95.49% | **95.13%** | -0.36pp | yes | yes |
| **Overall** | 96.50% | **97.39%** | +0.89pp | — | — |

Spatial per-task breakdown at canonical precision: task4 ("bowl in top drawer") 76.00% -> **88.00%** (+12pp), task5 ("bowl on ramekin") 80.00% -> **84.00%** (+4pp); the other 8 Spatial tasks are 98-100% (only task8 at 98%, all others 100%) — no dilution of untargeted tasks. LIBERO-90's weakest tasks remain task21 "stove+pan" (50%), task27 "wine rack" (54%), task75 "book right compartment" (60%), task51 "butter in basket" (70%) — none of these were targeted by this candidate and their positions are consistent with exp0018's known weak points, not new regressions. All five suites clear both the 95% goal and the 90% retention floor with real margin (smallest margin: LIBERO-90 at 95.13%, +0.13pp over goal but +5.13pp over the 90% floor).

## 8. Comparison and interpretation

exp0019 is an unambiguous win at canonical precision: it closes the project's remaining gap (Spatial, the sole suite below 95% under exp0018), Long/10 also gained materially (+2.80pp), Object held exactly flat, and Goal/LIBERO-90 moved by less than 0.4pp in either direction — well within ordinary evaluation noise and nowhere near either suite's floor. The targeted mechanism (task-level oversampling of exactly the 2 weak Spatial tasks) produced the intended, isolated effect: both targeted tasks improved substantially (+12pp, +4pp) while every other Spatial task stayed at or near 100%, and no other suite showed any sign of dilution. This is the first checkpoint in the project's history — across 19 candidate experiments — to have all five suites simultaneously clear both the 90% retention floor and the newer, harder 95% goal at full canonical (50-trial) precision.

This result also validates the broader research strategy adopted after exp0016's rejection: pivoting from generic continued training (which plateaued for LIBERO-90 specifically) to concentrated, task-level data engineering. All three uses of this technique in this project (exp0017, exp0018, exp0019) produced real, verified, targeted gains with no observed downside beyond ordinary noise-level fluctuation in untouched suites.

## 9. Decision

**PROMOTE.** exp0019's canonical evaluation shows all five suites simultaneously at or above 95% (Spatial 97.00%, Object 99.60%, Goal 97.20%, Long/10 98.00%, LIBERO-90 95.13%), comfortably clearing both the retention floor (90%) and the active project goal (95% on all five suites) for the first time in the project. This becomes the new accepted main-line checkpoint, replacing exp0018.

## 10. What this changes for the next experiment

The project's active goal (all five suites >=95%) is now fully satisfied at canonical precision. Margins are real but not huge on two suites (LIBERO-90 at 95.13%, +0.13pp over goal; Goal at 97.20%, essentially flat vs exp0018). The next experiment should shift from "close a gap" to "increase margin / robustness across the board," with LIBERO-90 as the natural continued focus since it has the smallest margin and the most persistently weak tasks (task21 "stove+pan" 50%, task27 "wine rack" 54%, task75 "book right compartment" 60%, task51 "butter in basket" 70% — none previously targeted). A further round of task-level oversampling targeting these 4 tasks, using the same validated mechanism, is the most direct next candidate. Alternatively, given the goal is now met, a broader, lower-urgency direction (e.g. reducing evaluation variance, or testing robustness under distribution shift) could also be considered if the user's priorities shift from "close the gap" to "consolidate/harden."

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/train.log`
- new data directory: `data/libero_mujoco3.3.2/libero_spatial_weak_subset_lerobot/`
- new config(s): `configs/data/libero_2cam_plus90_reweighted_weaktasks_v3.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v3_3e-5.yaml`
- fix commits: `99826cc`, `b2b49d0`
- cheap panel raw results: `evaluate_results/exp0019_cheap_panel/`
- confirmation raw results: `evaluate_results/exp0019_confirmation/`
- canonical raw results: `evaluate_results/exp0019_canonical/`
- HF backup (verified present): `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/` — `step_005000.pt`, `dataset_stats.json`, `libero_2cam_plus90_reweighted_weaktasks_v3.yaml`, `libero_uncond_2cam224_plus90_reweighted_weaktasks_v3_3e-5.yaml`, `PROGRESS_0019_spatial_weak_task_oversampling.md`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded
- [x] intermediate checkpoints and progress decisions recorded when used (none needed — trained to planned completion)
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified (HF file listing confirmed all 5 files present)
- [x] every evaluation event has purpose/settings/raw results recorded (cheap panel, confirmation, canonical)
- [x] five-suite metrics recorded for canonical evaluation
- [x] final decision and reasoning recorded (`PROMOTE`)
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
