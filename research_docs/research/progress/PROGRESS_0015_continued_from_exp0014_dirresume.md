# PROGRESS_0015 — continued_from_exp0014_dirresume

- **Experiment ID:** 0015_continued_from_exp0014_dirresume
- **Status:** `REJECT`
- **Created:** 2026-08-12
- **Updated:** 2026-08-12
- **Parent experiment:** 0014_continued_from_exp0013_dirresume (`PROMOTE`)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/state/step_014000` (full accelerate/deepspeed state, directory resume)
- **Selected candidate checkpoint:** n/a — training in progress
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (both resume fixes applied)

## 1. Result at a glance

Direct continuation of the newly-promoted exp0014 checkpoint via directory/full-state resume (same validated mechanism used for exp0014 itself), extending `max_steps` from 14000 to 22000 (+8000 steps). Goal: continue pushing LIBERO-90 (currently the project's worst-case suite at 91.09% canonical) and LIBERO-Long/10 (93.40%, second-weakest) toward the project's updated >=95%-on-all-five-suites target, under the "more training" lever that has produced consistent, still-unplateaued gains across four consecutive prior extensions (exp0009->0010->0013->0014).

## 2. Research state before experiment

exp0014 was just promoted (canonical): LIBERO-Spatial 97.20%, LIBERO-Object 100.00%, LIBERO-Goal 96.60%, LIBERO-Long/10 93.40%, **LIBERO-90 91.09%** — the first checkpoint to clear 90% on all five suites, satisfying the project's original goal. The active goal has since been raised to >=95% on all five suites, with explicit focus on the current worst-case suite. Worst-case = LIBERO-90 (91.09%); second-weakest = LIBERO-Long/10 (93.40%); Spatial/Object/Goal already clear 95%.

Per-task LIBERO-90 analysis (canonical, exp0014) identifies two distinct failure modes:
1. **Task 51** ("pick up the butter and put it in the basket", 38.00%) has zero training demonstrations — a genuine data gap. Comparing exp0013 (54.0%) -> exp0014 (38.0%) on this task shows no consistent improvement from generic continued training (bounces around, consistent with near-random behavior on a task the model never actually trains on) — this task is **not expected to respond to this candidate** and would need a dedicated data fix in a future candidate.
2. **The "book in caddy compartment" cluster** (tasks 73, 75, 81, 82, 83) and other weak tasks (32, 6, 89) **did show clear, consistent improvement from exp0013->exp0014's generic continued training**: task73 50.0%->66.0% (+16pp), task75 42.0%->66.0% (+24pp), task81 18.0%->46.0% (+28pp), task89 16.0%->56.0% (+40pp), task6 46.0%->54.0% (+8pp), task32 38.0%->44.0% (+6pp). This is direct evidence that generic continued training is still closing real gaps in exactly the categories dragging LIBERO-90 down, not just a diffuse/generic effect — supporting another continuation as the next candidate before considering a more surgical data-level intervention.

## 3. Candidate design

### Modifications

1. `resume=./runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/state/step_014000` (directory/full-state resume of exp0014, not weights-only — preserves optimizer momentum/variance exactly, as validated for exp0014 itself).
2. `max_steps=22000` (extends exp0014's own 14000-step schedule by 8000 steps — similar scale to the exp0009->0010 (+4000) and exp0013->exp0014 (+6000) extensions, slightly larger given the still-strong improvement trend and no plateau signal).
3. No other recipe changes (same reweighted goal5x/long5x mix, same `task=libero_uncond_2cam224_plus90_reweighted_3e-5`).

### Why this candidate

Continued training via directory-resume is now a fully validated, low-risk, low-implementation-cost mechanism (this will be its second real production use, after exp0014). The task-level evidence above shows this exact lever has been directly responsible for closing the specific weak clusters that matter most for the current goal (LIBERO-90's worst-performing categories), not just producing a diffuse average improvement — so it remains the best-evidenced next move before investing in a more complex, higher-risk targeted data intervention (e.g. sourcing/synthesizing demonstrations for task 51, or explicitly reweighting the book-in-caddy category).

### What to watch

- Whether LIBERO-90 continues climbing past 91.09% toward the 95% goal.
- Whether LIBERO-Long/10 (93.40%, second-weakest) also continues improving, or whether it starts to plateau/regress while LIBERO-90 improves (a genuine trade-off risk worth watching given the two are the closest to each other and both below 95%).
- Whether the already-strong suites (Spatial 97.20%, Object 100.00%, Goal 96.60%) hold at or above 95% — any suite dropping below 90% would violate the retention gate outright; dropping below 95% (but staying >=90%) is tolerable per the goal's "worst-case" framing but worth tracking.
- Task 51 specifically: if it continues to show no consistent improvement (as expected, given the data-gap diagnosis), that is confirmatory evidence a future candidate needs a targeted data fix rather than more generic training.
- Whether the book-in-caddy cluster (73, 75, 81, 82, 83) keeps improving at a similar rate, or whether it starts to plateau (which would also motivate a future targeted intervention).

### Initial compute plan

- Initial training budget: `max_steps=22000` (8000 new steps from the resumed `global_step=14000`, ~6.5-7h at established throughput, extrapolating from exp0014's ~5h6min for 6000 steps).
- Checkpoint/save plan: `save_every=500`, only the newest full-state checkpoint kept locally at any time (disk-monitor pattern), preserved (not pruned to zero) at run end for a possible further continuation.
- Progress check: skip unless GPU capacity frees up mid-run (eval cannot run concurrently with training under ZeRO-1) — same reasoning applied successfully for exp0014's own training.

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (both resume fixes)
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `max_steps=22000`
- Resume source: `runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/state/step_014000` (full accelerate/deepspeed state — directory resume)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5, except the two resume-fix commits in `src/fastwam/trainer.py` (`99826cc`, `b2b49d0`). Note: this project requires the dedicated venv `/workspace/venv-fastwam` (not the default `/venv/main`) — see the infra note in `research/NOTES.md` from exp0014's first launch attempt.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  source /workspace/venv-fastwam/bin/activate
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./runs/reweighted_libero90_finetune/exp0014_continued_dirresume/checkpoints/state/step_014000 \
    max_steps=22000 \
    save_every=500 \
    output_dir=./runs/reweighted_libero90_finetune/exp0015_continued_dirresume2 \
    wandb.name=exp0015_continued_from_exp0014_dirresume
  ```
- Start time: 2026-08-12 ~03:28 UTC (immediately after exp0014's canonical evaluation completed, records finalized, and GPUs freed — no idle time)
- End time: 2026-08-12 ~10:23 UTC (`max_steps reached step=22000`, clean exit, no crash, no OOM, no disk-full warning — runtime ~6h55min for the 8000 new steps)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0015_continued_dirresume2/checkpoints/weights/step_022000.pt` (12,041,735,545 bytes)
- Final training diagnostics at step 22000: `val_loss=0.0881 infer_psnr=27.9623 infer_ssim=0.8626 action_l2=0.0105 action_l1=0.0496`
- Training health check: no discontinuity at the resume boundary (resumed correctly at `lr=9.8039e-06`, the mathematically correct position 14000/22000 through the extended schedule). Train loss stayed in the same noisy 0.10-0.21 band throughout the run, val_loss noisy 0.04-0.29 with no trend, action_l2 stayed in the low 0.003-0.04 band throughout (one transient spike to 0.0401 at step 20400, immediately recovered to 0.0052 at the next checkpoint — ordinary noise, nowhere near the ~0.087 "relearning from scratch" signature). Confirms the directory-resume fix continues to work correctly on its second real production use.
- Final full-state checkpoint (`step_022000`) preserved on the scratch disk, keeping a true directory-resume extension option available if needed.

### Why training ended

Planned completion — reached `max_steps=22000` cleanly with no early-stop trigger. No interim progress check was performed (GPUs fully saturated by training the whole run; consistent with the reasoning used for exp0014).

## 7. Evaluation events

### Event 1 — `candidate_screen` (cheap panel)

- Checkpoint: `runs/reweighted_libero90_finetune/exp0015_continued_dirresume2/checkpoints/weights/step_022000.pt`
- Reference: exp0014's step_014000.pt panel result (Spatial 100.00%, Object 100.00%, Goal 100.00%, Long 100.00%, LIBERO-90 92.00%, overall 98.40%)
- Tasks: same 21-task widened panel, 5 trials/task
- Raw results: `evaluate_results/exp0015_cheap_panel_v2/` (note: first launch attempt, `exp0015_cheap_panel/`, hit the same stuck-tmux infra bug documented for exp0014 — see `research/NOTES.md` — discarded with zero real results; re-launched cleanly as `_v2`, verified stable via repeated `tmux list-sessions` checks before trusting it)

**Result — flat/mixed relative to exp0014, within the established panel noise floor:**

| Suite | exp0014 (panel) | exp0015 (panel) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 100.00% | 90.00% | -10pp |
| LIBERO-Object | 100.00% | 100.00% | 0pp |
| LIBERO-Goal | 100.00% | 100.00% | 0pp |
| LIBERO-Long/10 | 100.00% | 95.00% | -5pp |
| LIBERO-90 (5-task) | 92.00% | 92.00% | 0pp |
| **Overall** | **98.40%** | **95.40%** | **-3pp** |

Given exp0014's own panel score (80% on the LIBERO-90 sample, at the earlier stage) understated its true confirmation-level result (91.19%), and this panel result is ambiguous rather than clearly negative, confirmation evaluation (full 130-task coverage, 15 trials/task) was launched to get a precise reading rather than over-interpreting a small, noisy panel.

### Event 2 — `confirmation` (full 130-task coverage, 15 trials/task)

- Command: `run_libero_manager.py` with all 5 suites, `EVALUATION.num_trials=15`, fresh `output_dir=evaluate_results/exp0015_confirmation`
- Runtime: ~1h (2026-08-12 10:41 -> ~11:41 UTC), all 130 tasks completed, 0 failures. Verified: every suite at exactly 15 trials/task, no partial data.

**Result — essentially a plateau relative to exp0014, not a clear further improvement:**

| Suite | exp0014 (canonical, 50 trials) | exp0015 (confirmation, 15 trials) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 97.20% | 97.33% | +0.13pp |
| LIBERO-Object | 100.00% | 100.00% | 0pp |
| LIBERO-Goal | 96.60% | 96.00% | -0.60pp |
| LIBERO-Long/10 | 93.40% | 92.67% | -0.73pp |
| LIBERO-90 (all 90 tasks) | 91.09% | 91.78% | +0.69pp |
| **Overall** | **95.66%** | **95.56%** | **-0.10pp** |

At the aggregate level this is flat/mixed, all deltas well within noise. At the per-task level (previously-weak LIBERO-90 tasks), the picture is genuinely noisy rather than a clean "nothing changed": task89 jumped 56%->86.7% and task6 jumped 54%->73.3%, but task75 dropped 66%->53.3% and task32 dropped 44%->40% — individual tasks swinging significantly in both directions while the aggregate stays flat, consistent with the project's own established training-run noise floor (up to 40pp/task from the exp0006 reproducibility check) rather than a systematic trend.

## 8. Comparison and interpretation

Four consecutive generic continued-training extensions (exp0009->0010->0013->0014) each produced large, consistent LIBERO-90 gains (48%->72%->76%->88.49%->91.09% canonical). This fifth extension (exp0014->0015, same lever, same data mix, +8000 more steps) produced no material further gain and a mild (noise-level) dip on two suites. This is a genuine plateau signal for the "more generic training on the unchanged data mix" lever — not because training became unhealthy (loss/action_l2 diagnostics stayed normal throughout, see Section 6), but because additional steps on an unchanged data distribution appear to have stopped adding new information once the model reached this level of fit. Under the project's updated goal (>=95% on all five suites, worst-case-focused), this plateau matters: LIBERO-90 (~91-92%) and LIBERO-Long/10 (~93%) remain well short of 95%, and simply extending training further on the same mix is not the lever likely to close that gap.

## 9. Decision

- **Decision:** `REJECT`. exp0015 is not a materially better main-line parent than exp0014 — the changes are within noise, mixed in direction, and canonical-grade evaluation is not warranted to resolve a sub-1pp difference that wouldn't change the research direction either way.
- **Retention gate passed:** yes (all four original suites still comfortably above 90%), but this is not sufficient to promote given no material LIBERO-90 gain.
- **exp0014 remains the accepted main-line checkpoint.**
- exp0015's local checkpoint (`step_022000.pt`) is kept locally (not uploaded to HF, since it does not represent a clearly useful, separately-preservable branch) in case a future candidate wants to resume from it rather than exp0014 — no urgency to delete, but not a priority artifact either.

## 10. What this changes for the next experiment

**Generic "more training" on the unchanged reweighted data mix has plateaued after four successful rounds.** The next candidate should use a genuinely different lever rather than a fifth extension of the same kind:

1. **Data-composition investigation performed**: checked actual episode counts per suite in the training mix — `libero_90` already contributes ~3921 episodes (~44% of the total mix) even at its current 1x listing, versus `libero_goal`/`libero_10` at 2165/1940 episodes (5x oversampled) and `libero_spatial`/`libero_object` at ~434-457 episodes (1x). So LIBERO-90 is *not* under-represented by raw volume — it is already the largest single component of the mix. This means naive further suite-level oversampling of LIBERO-90 is a blunt, uncertain lever (and risks diluting Goal/Long, the two suites closest to the retention floor, repeating the "suite-selective forgetting" risk documented for exp0001).
2. **No task-level/category-level oversampling mechanism currently exists** in `RobotVideoDataset`/`dataset_dirs` (checked `src/fastwam/datasets/lerobot/robot_video_dataset.py` — dataset_dirs is a flat list of whole-dataset directories, no per-task filter or per-episode weight support). Building one (e.g. a filtered sub-dataset directory for just the weak-task episodes, reusing the same list-repetition oversampling mechanism) is a viable future direction but requires non-trivial data engineering and was not attempted in this cycle given time constraints.
3. **Chosen next candidate (exp0016)**: an LR warm-restart — fresh weights-only resume from exp0014 with a renewed, meaningfully-high peak LR (rather than another directory-resume extension that only reaches a partially-decayed LR), to test whether the plateau is a shallow-schedule artifact (i.e., each extension's LR spent too little time in a genuinely high-LR regime) versus a true data/capacity limit. This is a different, low-risk, well-understood lever (LR restarts are standard for escaping apparent plateaus) that doesn't carry the retention risk of a suite-reweighting change, while the investigation above is preserved for a future targeted data-engineering candidate if the LR-restart also plateaus.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0015_continued_dirresume2/train.log` (live copy: `/tmp/.../scratchpad/train_exp0015.log`)
- full-state checkpoint (preserved): `/home/claudeuser/fastwam_state_scratch/exp0015_continued_dirresume2/state/step_022000/`
- config(s): unchanged from exp0003/0006/0009/0010/0013/0014
- fix commits: `99826cc`, `b2b49d0`

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
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — confirmation-level evidence was sufficient for REJECT, canonical not warranted)
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
