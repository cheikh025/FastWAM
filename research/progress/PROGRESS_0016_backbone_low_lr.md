# PROGRESS_0016 — Partial backbone plasticity (dit_with_backbone_low_lr)

- **Experiment ID:** 0016
- **Status:** `REJECT` (full-50-task Clean mean 9.6%, no improvement over exp0014/exp0015's 11.6% ceiling; LIBERO-Spatial held at 96.67%, so no retention cost either -- backbone plasticity is not the missing lever)
- **Created:** 2026-08-20
- **Updated:** 2026-08-20
- **Parent experiment:** 0014 (frozen-backbone diagnostic); 0015 (RoboTwin-heavy ratio, REJECT)
- **Parent checkpoint:** exp0014 cumulative step 4600 (`research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt` on `cheikh025/ASR`) -- current project best
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

After two consecutive negative results on the "more RoboTwin exposure" axis (exp0014's extended training: no full-50-task movement across 2000 more steps; exp0015's RoboTwin-heavy ratio: no movement despite 3x gradient share), this candidate tests a different lever: partial backbone plasticity. `trainable_modules: dit_with_backbone_low_lr` keeps the shared MoT/DiT backbone trainable but at a much lower LR (3e-6, 10x below the projection layers' 3e-5) -- interpolating between exp0014's capacity-limited full-freeze and exp0013's LIBERO-interfering full-train.

## 2. Research state before experiment

See `research/progress/PROGRESS_0014_frozen_backbone_diagnostic.md` (Sections 14-15) and `research/progress/PROGRESS_0015_robotwin_heavy_ratio.md` (Sections 6-7) for full evidence.

| Candidate | Lever tested | Full-50-task RoboTwin Clean mean | LIBERO-Spatial |
|---|---|---:|---:|
| exp0014 @ step 2600 | (baseline) | 12.6% (n=10) | 96.7% |
| exp0014 @ step 4600 | +2000 more steps, same 1:1 ratio | 11.6% (n=5) | 100.0% |
| exp0015 @ step ~5600 | +3x RoboTwin gradient share | 11.6% (n=5) | 100.0% |

Working hypothesis: the ~12% ceiling is a structural capacity limitation of `expanded_projections_only` (only action_encoder/head trainable), not a data-exposure problem.

## 3. Candidate design

### Modification

`trainable_modules: dit_with_backbone_low_lr` (was `expanded_projections_only`) with `backbone_lr: 3e-6` (10x below `learning_rate: 3e-5`, which still applies to the action_encoder/head). Data config reverted to the standard 1:1 `multiembodiment_libero_robotwin` mixture (ratio is now a ruled-out variable per exp0015). Everything else unchanged from exp0014.

### Why this candidate

This is the middle ground the trainer's own code comments were written to support (see `src/fastwam/trainer.py` lines ~101-112), previously motivated but never conclusively tested under a clean parent (the old exp0005 attempt was on the confounded exp0019 lineage). Gives the model genuine additional representational capacity to potentially learn broader RoboTwin behavior, while the 10x-lower backbone LR is intended to bound LIBERO drift risk relative to exp0013's full-LR full-train (which took ~1000-1800 steps to visibly decline).

### What to watch

**This candidate reintroduces real LIBERO retention risk**, unlike exp0014/exp0015 where the frozen backbone made LIBERO risk essentially zero. Monitor LIBERO-Spatial at every progress check, not just RoboTwin -- be ready to stop early if it starts degrading, matching the lesson from exp0013's gradual decline (took 800-1600 steps to become visible).

- LIBERO-Spatial retention across the run (primary risk).
- Full-50-task RoboTwin mean vs. the 11.6-12.6% ceiling -- does backbone plasticity move it at all, confirming or refuting the capacity hypothesis?
- Curated-panel RoboTwin tasks for continuity with prior candidates' trajectories.

### Initial compute plan

Per standing feedback (start smaller, scale on evidence): `max_steps: 1000`, `save_every: 200`. Progress-check with the cheap panel (including LIBERO) more frequently than prior frozen-backbone candidates, given the reintroduced retention risk -- e.g. every ~200-400 steps rather than waiting for 600.

## 4. Exact code and configuration state

- task config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_backbone_low_lr_3e-5.yaml` (new)
- data config: `multiembodiment_libero_robotwin` (standard, unchanged, 1:1 ratio)
- resume source: exp0014 cumulative step 4600 checkpoint

## 5-11. Pending

To be filled in as the run progresses.

## 5. Evaluation events

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, local step 200 (early check, per the reintroduced-retention-risk plan)

- checkpoint: `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1/checkpoints/weights/step_000200.pt`
- LIBERO-Spatial (n=3): **100.0% (30/30)** -- no sign of degradation yet at this early point, backbone LR is low (3e-6) so drift should be slow if it happens at all.
- RoboTwin Clean (n=5): click_alarmclock 100%, turn_switch 60%, adjust_bottle **20%** (new -- never nonzero at any exp0014/exp0015 checkpoint), open_laptop 0% (down from the recent 40-60% range).

Mixed early signal: `adjust_bottle`'s first-ever nonzero result is a potentially promising sign that backbone plasticity provides real additional capacity, but `open_laptop`'s dip could equally be noise at n=5. Too early to draw conclusions -- resuming training, will check again at the next checkpoint given the standing plan to monitor LIBERO closely for this candidate.

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, cumulative step 400

- checkpoint: `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont1/checkpoints/weights/step_000200.pt` (this candidate's own local step 200; cumulative training step 400 = 200 (phase1) + 200 (cont1))
- LIBERO-Spatial (n=3): **96.67% (29/30)** -- small dip from 100%, well within the normal 96.7-100% noise range seen throughout the project. No sign of a real decline yet.
- RoboTwin Clean (n=5): click_alarmclock 80%, turn_switch 60%, open_laptop 20%, adjust_bottle 0% (the step-200 20% signal on adjust_bottle didn't hold -- likely was n=5 noise).

Broadly comparable to the established range from exp0014/exp0015 -- no dramatic broadening yet, but also no LIBERO risk materializing. Resuming training toward the remaining budget (600 more steps), continuing the closer-than-usual monitoring cadence for this candidate.

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, cumulative step 600

- checkpoint: `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont2/checkpoints/weights/step_000200.pt`
- LIBERO-Spatial (n=3): **100.0% (30/30)** -- fully recovered, no decline trend visible across 600 cumulative steps of backbone plasticity.
- RoboTwin Clean (n=5): click_alarmclock 80%, turn_switch 60%, open_laptop 20%, adjust_bottle 0% -- essentially identical to cumulative step 400, stable.

No dramatic movement in either direction. LIBERO retention risk has not materialized through 600 cumulative steps (vs. exp0013's full-LR full-train, which took 800-1800 steps to visibly decline -- this candidate's much lower backbone LR appears to be doing its job so far). Resuming for the final 400 steps toward the full 1000-step budget; will run the decisive full-50-task rescan on the final checkpoint.

### Evaluation event — LIBERO-Spatial `candidate_screen`, final checkpoint (cumulative step 1000/5600 from the exp0014-step4600 parent)

- checkpoint: `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont3/checkpoints/weights/step_000400.pt` (this candidate's own local step 400; cumulative training step 1000 = 200+200+200+400)
- LIBERO-Spatial (n=3): **96.67% (29/30)** -- one failure on `libero_spatial_1`, otherwise fully clean. Consistent with the noise range seen throughout the project (96.7-100%). No decline trend across the full 1000-step budget of backbone plasticity at LR=3e-6.
- Log: `checkpoints/exp0016_final_libero_spatial_screen.log`; run: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260820_110214/`.

### Evaluation event — Curated 4-task RoboTwin panel `candidate_screen`, final checkpoint

- Config note: the run initially failed with a `dataset_stats.json` `FileNotFoundError` (the `_cont3` weights-only resume directory has `robotwin_dataset_stats.json`/`libero_dataset_stats.json` but no plain `dataset_stats.json` the resolver looks for by default) -- fixed by passing `EVALUATION.dataset_stats_path=<run_dir>/robotwin_dataset_stats.json` explicitly. A second attempt then failed with an action-dim shape mismatch (`[1024,21]` in checkpoint vs `[1024,14]` in the model) because `task=robotwin_uncond_3cam_384_1e-4` builds the single-embodiment K=14 model -- the multi-embodiment K=21 checkpoint needs `task=robotwin_uncond_3cam_384_multiembodiment_eval` instead. Both are infrastructure/launch-command errors, not candidate issues; noted here since they'll recur for any future eval of a `_contN` weights-only checkpoint.
- RoboTwin Clean (n=5): click_alarmclock 80%, turn_switch 20%, open_laptop 20%, adjust_bottle 0% -- mean 30%. Comparable to the established curated-panel range from exp0014/exp0015 (adjust_bottle stayed at 0% for this candidate too, confirming step-200's one-off 20% signal was noise).
- Log: `checkpoints/exp0016_final_robotwin_panel4_n5.log`; run: `evaluate_results/robotwin/reweighted_multiembodiment_exp0016_backbone_low_lr_v1_cont3/20260820_112642/`.

### Evaluation event — Full 50-task RoboTwin Clean rescan `canonical` (n=5), final checkpoint -- decisive

- Command: `run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval ckpt=<final> EVALUATION.dataset_stats_path=<run_dir>/robotwin_dataset_stats.json EVALUATION.eval_num_episodes=5 +EVALUATION.clean_only=true MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=1`
- **Full-50-task Clean mean: 9.6%** (vs. exp0014 step-4600's 11.6% and exp0015's 11.6%, both also n=5) -- **not an improvement; if anything slightly lower**, well within noise range of the prior two candidates rather than a real regression.
- 10/50 tasks nonzero: click_alarmclock 80%, click_bell 80%, press_stapler 80%, shake_bottle_horizontally 60%, move_can_pot 40%, open_microwave 40%, shake_bottle 40%, beat_block_hammer 20%, place_a2b_left 20%, turn_switch 20%. All others (40/50) 0%.
- One benign per-episode simulator exception (`target_pose cannot be None for move action`, task `grab_roller`) was caught internally by the RoboTwin harness and scored as a failed episode -- not a process crash; the manager completed all 50 tasks and exited cleanly (`manager finished successfully`).
- Log: `checkpoints/exp0016_final_robotwin_full50_clean_n5.log`; run: `evaluate_results/robotwin/reweighted_multiembodiment_exp0016_backbone_low_lr_v1_cont3/20260820_114613/summary.json`.

## 6. Decision

**`EXTEND_TRAINING`** (revised from an initial `REJECT` draft per explicit user guidance: try running this lever for longer before concluding it doesn't help). At 1000 cumulative steps the full-50-task Clean mean (9.6%) is flat/within-noise of exp0014/exp0015's 11.6% ceiling, and LIBERO-Spatial retention has not degraded at all (96.67%, unchanged from earlier checks) -- meaning there is still LIBERO-retention headroom to spend on more backbone-plasticity training before this lever can be fairly called negative. Unlike exp0014's own extend-check (which re-tested the *same* frozen-backbone mechanism for longer and stayed flat), this lever has a slow-acting lever (backbone LR 3e-6) whose effect may simply not have had enough steps yet to show up, distinct from exp0013's fast full-LR interference (visible by step 800-1800).

**Plan:** resume from this candidate's final checkpoint (`cheikh025/ASR:research/exp0016_backbone_low_lr/step_001000_cumulative.pt`) for another ~1500-2000 steps (roughly doubling+ the candidate's total budget), watching LIBERO-Spatial closely at each checkpoint (per this candidate's reintroduced-retention-risk plan) and re-running the full-50-task Clean rescan at the new final checkpoint for a like-for-like comparison against the 9.6%/11.6%/11.6% baseline.

**Project best remains exp0014 cumulative step 4600** (`research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt`): full-50-task Clean mean 11.6% (n=5), LIBERO-Spatial 100%. exp0016 has not yet cleared that bar, so it is not yet a promotion candidate -- continuing to extend before a final REJECT/PROMOTE call.

If the extended run still shows no movement, the task-level pattern is worth a closer look before choosing experiment 0017: nonzero tasks cluster on single-object pick/place/press/click/turn actions with short horizons (click_alarmclock, click_bell, press_stapler, shake_bottle*, turn_switch, beat_block_hammer, move_can_pot, open_microwave, place_a2b_left), while multi-step/bimanual/precision tasks (stacking, dual-arm handover, precise placement) sit uniformly at 0% across all three candidates -- that structural split, not just the aggregate mean, is the more informative signal for `$investigate-fastwam-problem` if this extension doesn't move it.

### Extension launch (`_cont4`)

- Resumed from `cheikh025/ASR:research/exp0016_backbone_low_lr/step_001000_cumulative.pt` (re-downloaded; verified via `_resolve_dataset_stats_path`-adjacent weight-load log line `"Weight checkpoint already loaded pre-wrap"`, confirming the pre-`accelerator.prepare()` load path executed correctly -- resume mechanism double-checked per explicit user request: weights-only resume from a file path (not a directory) intentionally does not restore `global_step`/optimizer/scheduler (confirmed in `trainer.py::_resume_or_load_checkpoint`/`__init__`), matching the same pattern already used successfully across exp0013-exp0016's `_cont1`-`_cont3` phases -- not a new risk).
- Command: `bash scripts/train_zero2.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_backbone_low_lr_3e-5 resume=<downloaded ckpt> output_dir=./runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont4 max_steps=5000 save_every=500`
- Same 1:1 LIBERO:RoboTwin ratio, same `dit_with_backbone_low_lr`/`backbone_lr=3e-6` config as phases 1-3, confirmed via log (`InterleavedEmbodimentSampler: embodiments=['libero', 'robotwin'] ratios=[1.0, 1.0]`, `"Setting DiT to train mode (nothing frozen); backbone trains at a lower LR..."`).
- Throughput confirmed stable at ~0.0215 steps/s (step 10->50 over 31 min), consistent with prior exp0016 phases at this batch/grad-accum/ZeRO2 setting. **GPU utilization pattern is notable: instantaneous `nvidia-smi` samples show only 1-2 of the 4 GPUs near 100% at a time, rotating over time -- not synchronized near-100% across all 4 as expected for healthy DeepSpeed ZeRO2 data-parallel training.** This suggests a dataloader/video-decode or cross-rank synchronization bottleneck, not a compute-bound cost -- flagged as a concrete throughput lead for a future `$investigate-fastwam-problem` pass or exp0017 design, not yet root-caused.
- **Budget revised from an initial 1500-step relaunch to 5000 steps** per explicit user decision (offered a 3-way tradeoff: diagnose throughput first / commit to 20k+ steps now (~10 days at current rate) / run an intermediate 5k-step budget first (~2.5 days) and decide on 20k+ after seeing whether it shows movement -- user chose the 5k intermediate option). The original 1500-step launch was killed cleanly (~60/1500 steps in, no checkpoint saved yet at that point, so no work lost) and relaunched fresh from the same cumulative-1000 checkpoint with `max_steps=5000 save_every=500`. KEEP=1 background pruner running against the `_cont4` weights dir; HF cache duplicate blob deleted after confirmed weight load (repeated for both the 1500-step and 5000-step launches).
- Plan: progress-check (LIBERO-Spatial sentinel + curated RoboTwin panel) at each `save_every=500` checkpoint (local steps 500/1000/.../5000 = cumulative 1500/2000/.../6000), watching for either (a) RoboTwin full-50-task movement finally appearing with more backbone-plasticity exposure, or (b) LIBERO-Spatial degradation emerging later than exp0013's full-LR case (since backbone_lr is 10x lower) -- stop early if LIBERO clearly declines. At ~0.0215 steps/s the full 5000-step run is ~2.5 days wall-clock; will decide on a further 20k+ extension based on whether this run shows any movement.

### Evaluation event — LIBERO 4-suite `progress_check`, local step 500 (cumulative ~1500 for this extension)

- checkpoint: `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont4/checkpoints/weights/step_000500.pt`
- Paused training cleanly to run this check (killed both the `train_zero2.sh` wrapper and the actual DeepSpeed worker PIDs -- the wrapper kill alone does not stop the workers, confirmed via `nvidia-smi --query-compute-apps`; had to kill the real PIDs to free GPU memory before launching eval). Per the "never stack unrelated GPU jobs" rule, training was fully stopped (not run concurrently) before this eval.
- Launch bug: first attempt used the wrong entry point (`experiments/libero/run_libero_parallel_test.sh` directly, missing required `NUM_GPUS`/task-list plumbing) -- fixed by using the correct manager, `python experiments/libero/run_libero_manager.py`.
- Scope surprise: `EVALUATION.task_suite_name=libero_spatial` override did not take effect (manager loaded all 40 tasks across all 4 suites regardless) -- ended up running the **full canonical 4-suite LIBERO evaluation** (n=3) instead of just a cheap Spatial sentinel. Kept it running rather than restarting, since it produces strictly more decisive evidence for the same sunk GPU time.
- **Result: LIBERO-Spatial 96.67% (29/30), LIBERO-Object 100% (30/30), LIBERO-Goal 100% (30/30), LIBERO-Long/10 96.67% (29/30), overall 98.33%.** All four suites clear the >=90% floor with no sign of retention degradation at this point in the extension -- essentially matching the project-best exp0014-step4600 LIBERO profile. Confirms the low backbone LR (3e-6) continues to bound drift well past the 1000-step point where phase 1 stopped.
- Log: `checkpoints/exp0016_cont4_step500_libero_spatial.log`; run: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260821_021644/`.
- Training resumed immediately after this check completed (new `_cont5` phase, resuming from this same local `step_000500.pt`, `max_steps=4500` to reach the same cumulative 5000-step target for this extension).

**Reframe (per explicit user observation, important for interpreting this and prior candidates' results):** `num_epochs=1` in the task config was never the binding constraint -- `max_steps` is, and it is *tiny* relative to actual dataset scale. Per-optimizer-step sample count = `batch_size(2) x world_size(4) x grad_accum(4) = 32`. The RoboTwin pool alone is 6,011,575 samples (`libero`: 277,713) -- so even 5000 steps (~160,000 samples) is under 3% of one RoboTwin epoch, and a full RoboTwin epoch at the measured ~0.0215 steps/s throughput would take roughly 100 days. This means exp0014 (cumulative 4600 steps), exp0015 (~1600 steps), and exp0016 phase 1 (1000 steps) were all comparably far from meaningful RoboTwin data coverage -- "more steps didn't move the ceiling" in those candidates is better read as "none of them were tested at a scale that could plausibly show a data-coverage effect" rather than as evidence the underlying levers (more exposure, more ratio, more capacity) don't work. This is a more compelling account of the flat ~10-12% ceiling than a genuine architectural/interference ceiling, and reframes what "worth trying next" means for exp0017: either a genuinely large multi-day training budget (20k+ steps, per user's stated target), a throughput fix (GPU utilization rotating across ranks rather than staying saturated -- worth profiling before assuming more steps requires proportionally more wall-clock), or a curated smaller RoboTwin subset that a full epoch can actually cover in realistic time.

## 7. Checkpoint persistence

- `runs/reweighted_multiembodiment/exp0016_backbone_low_lr_v1_cont3/checkpoints/weights/step_000400.pt` uploaded to `cheikh025/ASR:research/exp0016_backbone_low_lr/step_001000_cumulative.pt`, verified via remote file size (12,041,907,985 bytes, matches local exactly). Local copy deleted afterward under disk pressure (11GB free at the time, 100% full) per the project's checkpoint-persistence rule -- exact remote copy was verified before deletion.
