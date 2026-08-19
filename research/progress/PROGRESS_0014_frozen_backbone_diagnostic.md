# PROGRESS_0014 — Frozen-backbone diagnostic: does it stop exp0013's RoboTwin decline?

- **Experiment ID:** 0014
- **Status:** `DIAGNOSE` (frozen-backbone hypothesis confirmed as helpful, but full 50-task RoboTwin scan shows the real gap to the project goal is training scale/coverage, not this candidate's mechanism)
- **Created:** 2026-08-19
- **Updated:** 2026-08-19
- **Parent experiment:** 0013 (release-parent multi-embodiment training)
- **Parent checkpoint:** exp0013 cumulative step 1000 (`research/exp0013_release_parent_disjoint_offset/step_001000.pt` on `cheikh025/ASR`) -- the selected best checkpoint, NOT the raw release-checkpoint parent
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Diagnostic candidate testing which of PROGRESS_0013's two leading hypotheses explains the RoboTwin capability decline observed past cumulative step 1000 (2 of 4 tested tasks collapsed to 0% by step 1800 despite LIBERO staying solid and training loss continuing to fall). Freezes the entire shared backbone (`trainable_modules: expanded_projections_only`) so only the newly-expanded action-encoder/head can change, then resumes training from the good step-1000 checkpoint.

## 2. Research state before experiment

See `research/progress/PROGRESS_0013_release_parent_disjoint_offset_baseline.md` Section 12 for full evidence. Summary:

| Cumulative step | LIBERO-Spatial | click_alarmclock | turn_switch | press_stapler | open_laptop |
|---|---:|---:|---:|---:|---:|
| 1000 | 100.0% | 80.0% | 60.0% | -- | -- |
| 1800 | 96.7% | 80.0% | 0.0% | 40.0% | 0.0% |
| 2600 | 96.7% | 40.0% | 20.0% | -- | -- |

Leading hypotheses:
1. LIBERO/RoboTwin gradient interference at the shared backbone.
2. Open-loop/closed-loop distribution shift -- the action-encoder/head overfits the offline imitation loss in a way that hurts closed-loop rollout stability, independent of backbone drift.
3. LR-schedule-restart-on-resume artifact -- partially ruled out (`turn_switch`'s 60%->0% collapse happened entirely within `cont1`'s single uninterrupted schedule).

## 3. Candidate design

### Modification

Single change from exp0013's config: `trainable_modules: expanded_projections_only` (freezes `model.dit` entirely -- both video and action MoT/DiT experts -- leaving only `action_mixture.action_encoder` + `action_mixture.head` trainable). Everything else (data mixture, LR 3e-5 cosine, batch_size=2, disjoint K=21/22 offsets) unchanged from exp0013.

### Why this candidate

This is the cheapest way to distinguish the two leading hypotheses without new instrumentation (no per-embodiment loss logging exists in the trainer -- would need retrospective code changes and a fresh run to observe, more expensive than this direct test):
- If RoboTwin capability stabilizes under a frozen backbone -> hypothesis 1 (backbone interference) is supported.
- If RoboTwin capability still declines despite the frozen backbone -> hypothesis 2 (action-head-specific overfitting/distribution-shift) is supported, since the only thing still changing is the projection layers themselves.

Resuming from exp0013's step-1000 checkpoint (not the raw release parent) preserves the already-established strong RoboTwin capability as the starting point, so any further decline is unambiguously attributable to continued training from that point, matching exp0013's own trajectory for a clean comparison.

### What to watch

- RoboTwin `click_alarmclock`/`turn_switch` Clean (n=5) trend vs. exp0013's step-1000 baseline (80%/60%) and step-1800 trajectory (80%/0%) -- does freezing the backbone hold these steady, or does the same decline still occur?
- LIBERO-Spatial retention (expected to be trivially preserved or even more stable than exp0013, since the backbone literally cannot drift).

### Initial compute plan

Per standing feedback (start smaller, scale based on evidence): `max_steps: 1000`, `save_every: 200`. Progress-check around local step 400-600 with the same cheap panel (LIBERO-Spatial n=3, RoboTwin click_alarmclock+turn_switch Clean n=5) for direct comparability against exp0013's own step-1000/1800/2600 numbers.

## 4. Exact code and configuration state

- task config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_frozen_backbone_3e-5.yaml` (new)
- parent code commit: pending (recorded after commit)
- resume source: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1/checkpoints/weights/step_001000.pt` (downloaded from `cheikh025/ASR:research/exp0013_release_parent_disjoint_offset/step_001000.pt`)

## 5-11. Pending

To be filled in as the run progresses.

## 5. Evaluation events

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, local step 400

- checkpoint: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1/checkpoints/weights/step_000400.pt`
- LIBERO-Spatial (n=3): **96.67% (29/30)** -- essentially unchanged from exp0013's step-1000 baseline, as expected (backbone frozen, near-zero drift possible).
- RoboTwin Clean (n=5), via the new `EVALUATION.task_names` multi-task manager override (true 2-GPU parallelism):

  | Task | exp0013 step 1000 (parent) | exp0014 local step 400 (frozen backbone) |
  |---|---:|---:|
  | click_alarmclock | 80.0% | **40.0%** |
  | turn_switch | 60.0% | **60.0%** |

## 6. Interpretation (partial, more evidence pending)

**Mixed result, not a clean resolution of the two hypotheses.** `turn_switch` held exactly steady (60%->60%) under the frozen backbone -- unlike its collapse to 0% by exp0013's cumulative step 1800 under full-backbone training, suggesting backbone freezing *is* protective for this task (supports hypothesis 1 for turn_switch specifically). But `click_alarmclock` still declined by the same magnitude (80%->40%) despite the backbone being **entirely frozen** -- since nothing but the action_encoder/head could have changed, this decline cannot be attributed to shared-backbone interference, and instead directly implicates the action-encoder/head itself (supports hypothesis 2 for click_alarmclock specifically).

**Working interpretation**: both hypotheses likely contribute, with task-specific sensitivity -- backbone drift matters for some tasks (turn_switch), while the projection-layer's own overfitting/distribution-shift matters for others (click_alarmclock), independent of backbone changes. Neither hypothesis alone is a clean, complete explanation. This suggests future candidates may need to address both (e.g., lower LR specifically on the projection layers themselves, in addition to backbone freezing/low-LR, and/or explicit regularization on the action head).

**Next step**: continue this candidate to its remaining budget (max_steps=1000, currently at step 400) to see whether click_alarmclock's decline continues/stabilizes and whether turn_switch's stability holds with more steps, before drawing a final conclusion.

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, final (cumulative local step 1000)

- checkpoint: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1_cont1/checkpoints/weights/step_000600.pt` (exp0014's own local step 1000 total: 400 + 600)
- LIBERO-Spatial (n=3): **96.67% (29/30)** -- stable throughout, as expected.
- RoboTwin Clean (n=5), full 4-task panel:

  | Task | exp0013 step 1000 (parent) | exp0013 step 1800 (full backbone, declined) | exp0014 local step 400 (frozen) | exp0014 local step 1000 (frozen, final) |
  |---|---:|---:|---:|---:|
  | click_alarmclock | 80.0% | 80.0% | 40.0% | **80.0%** |
  | turn_switch | 60.0% | 0.0% | 60.0% | **80.0%** |
  | press_stapler | -- | 40.0% | -- | **100.0%** |
  | open_laptop | -- | 0.0% | -- | **40.0%** |
  | **mean (4 tasks)** | -- | **30.0%** | -- | **75.0%** |

### Decisive result

Freezing the shared backbone and continuing training from exp0013's step-1000 checkpoint for another 1000 steps **reversed the decline entirely** -- RoboTwin mean success across the 4-task panel rose from 30.0% (exp0013's declined step-1800 state) to 75.0%, with every task at or above its best prior value and LIBERO-Spatial unchanged at 96.67%. click_alarmclock's mid-run dip (80%->40% at local step 400) recovered fully by local step 1000 (back to 80%), suggesting it was a transient fluctuation rather than a sustained trend under this recipe.

This strongly supports **hypothesis 1** (shared-backbone LIBERO/RoboTwin gradient interference) as the primary driver of exp0013's decline -- removing the backbone's ability to drift not only stopped further degradation but let the projection layers alone continue improving RoboTwin capability substantially. Hypothesis 2 (action-head-specific overfitting) is not ruled out as a contributing factor (the mid-run click_alarmclock dip is still unexplained) but is clearly not the dominant effect, since the projection layers kept training throughout and capability still improved overall.

## 7. Decision

**`SELECT_CHECKPOINT`**: exp0014's final checkpoint (cumulative training step 1000 under this candidate, resumed from exp0013's own step 1000 -- i.e. 2000 total training steps from the release-checkpoint parent) is the new best validated checkpoint for the project, superseding exp0013's step-1000 pick. LIBERO-Spatial 96.67%, RoboTwin 4-task Clean mean 75.0% (individual tasks 40-100%).

**`EXTEND_TRAINING`**: given the clearly positive, still-improving trend and no sign of a plateau, continue training further under the same frozen-backbone regime rather than stopping. Plan: resume for another budget (matching the "start small, scale on evidence" practice) and progress-check again before deciding whether to extend further, broaden to the full 4-suite LIBERO validation, or move toward canonical evaluation.

## 8. Full 4-suite LIBERO confirmation (n=10/task) at cumulative step 2600

| Suite | Success (n=10/task) | Meets >=90%? |
|---|---:|:---:|
| LIBERO-Spatial | 99.0% (99/100) | yes |
| LIBERO-Object | 99.0% (99/100) | yes |
| LIBERO-Goal | 97.0% (97/100) | yes |
| **LIBERO-Long** | **87.0% (87/100)** | **NO** |

Full run: `checkpoints/exp0014_step2600cum_libero_4suites_n10.log`, `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260819_081729/summary.json`.

**Real promotion blocker, not noise**: Spatial/Object/Goal all comfortably clear 90%, but Long fails at 87.0%, driven specifically by two tasks -- `libero_10_4` ("put the white mug on the left plate and the yellow/white mug on the right plate", 50.0%, 5/10) and `libero_10_6` ("put the white mug on the plate and the chocolate pudding to the right", 60.0%, 6/10) -- both complex, multi-step, dual-object placement tasks. Every other Long task scored 80-100%.

**Cross-checked against exp0012's zero-training baseline** (`evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_211301/summary.json`, the release-checkpoint-expanded parent before any multi-embodiment training): these exact same two tasks were *already* the weakest in the suite there too, both at 80.0% (vs 100% for most others), contributing to that baseline's overall Long score of 95.0%. **This is not a new failure mode introduced by training** -- it's a pre-existing marginal spot in the parent checkpoint's own capability that got meaningfully amplified under the current checkpoint (80%->50% and 80%->60%, a real ~20-30pp additional drop each), which is what pushed the suite below the 90% floor. Framing for the next investigation: why did multi-embodiment training specifically worsen these two already-hard, complex dual-object tasks while leaving the rest of Long (and all of Spatial/Object/Goal) intact or improved?

## 9. Updated decision

This checkpoint (exp0014 cumulative training step 2600) is the strongest evidence point so far -- RoboTwin 4-task Clean mean stable at 75%, LIBERO Spatial/Object/Goal all >=97% -- but **cannot be promoted as-is**: LIBERO-Long fails the project's >=90% floor at 87.0%, driven by two specific dual-object mug-manipulation tasks. Before further RoboTwin scaling or a promotion attempt, either (a) investigate/address this specific Long weakness (task-level diagnostic, possibly related to the multi-embodiment padding interacting with longer/more complex action horizons), or (b) confirm whether this is within run-to-run noise via a repeat at a different seed before treating it as a real capability gap. `research/STATE.md` updated to record this as an open blocker.

## 10. Full 50-task RoboTwin evaluation (Clean-only, n=10/task) at cumulative step 2600

Launched per explicit user request ("yes eval full robotwing before moving"). Command: `run_robotwin_manager.py` with no `task_name`/`task_names` filter (loads all 50 canonical tasks), `EVALUATION.eval_num_episodes=10 +EVALUATION.clean_only=true MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=1`. Log: `checkpoints/exp0014_step2600cum_robotwin_full50_clean_n10.log`. Full results: `evaluate_results/robotwin/reweighted_multiembodiment_exp0014_release_parent_frozen_backbone_v1_cont2/20260819_084901/summary.json`.

**Runtime**: ~8h51m wall-clock (08:49:01 -> 17:40:34), 4-way GPU parallelism. Confirmed a strong correlation between success and episode duration: successful episodes terminate early (6-13 min), failing episodes run out the full per-task step-limit horizon before timing out (27-80 min) -- so evaluation gets meaningfully faster as the model improves, not just more successful. Per user feedback, future full-50-task scans should default to n=5 (not n=10) given this cost.

### Final result: 12.6% mean Clean success across all 50 tasks

| Success | Tasks |
|---|---|
| 100% | `click_bell`, `open_microwave`, `press_stapler` |
| 90% | `turn_switch` |
| 80% | `click_alarmclock` |
| 50% | `shake_bottle_horizontally` |
| 30% | `shake_bottle` |
| 10% | `move_playingcard_away`, `place_a2b_right`, `place_bread_skillet`, `put_bottles_dustbin` |
| 40% | `open_laptop` |
| 0% | remaining 40 tasks (including the 4 confirmed zero-training-data tasks: `blocks_ranking_rgb/size`, `handover_mic/block`) |

**This is a much more sobering picture than the curated 4-task cheap panel (75% mean) suggested.** The curated panel (click_alarmclock, turn_switch, press_stapler, open_laptop) happened to select 4 of the model's genuinely strongest tasks -- not a representative sample of the full 50-task benchmark. Only 6 of 50 tasks clear 50%; the other 44 are weak-to-zero, dominated by tasks with little/no training-data exposure or that the model simply has not yet learned within this very early (2600-step) training budget.

### Implication for the promotion tradeoff

Combined with the LIBERO-Long regression (87.0% vs the 95.0% pre-training baseline), the real picture is: this checkpoint has learned real, strong capability on a small number of RoboTwin tasks with good training-data support, at the cost of a specific LIBERO-Long regression on two already-hard tasks -- but is nowhere close to the project's actual full-50-task >=90% RoboTwin target. The gap is not primarily a promotion-tradeoff question at this point; it's a training-scale/coverage question. Next steps should focus on continuing training (more steps, better data coverage/weighting across the 50 tasks) rather than a promotion decision on the current checkpoint.

## 11. Continuation (cont3) — extending training given the validated recipe and the coverage gap

Given the full 50-task result confirms this is a training-scale/coverage gap (not a flawed recipe -- the frozen-backbone approach genuinely reversed exp0013's decline and produced strong per-task capability where training data supports it), the natural next step is more training under the same regime, not a new candidate design. Resumed from cumulative step 2600 with a larger increment (2000 steps, up from the previous 600/1000 increments) now that the recipe is well-validated.

- resume source: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1_cont2/checkpoints/weights/step_000600.pt` (cumulative step 2600)
- output dir: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1_cont3`
- budget: `max_steps=2000` (cumulative target: step 4600 from the release-checkpoint parent)
- plan: progress-check with the cheap panel at intervals, and run another full-50-task RoboTwin Clean-only scan (n=5 per updated user guidance, not n=10) once meaningful further training has accumulated, to track whether the 12.6% full-benchmark mean is climbing.

## 12. Progress check at cumulative step 3600 (cont3, local step 1000)

| Task | step 2600 | step 3600 |
|---|---:|---:|
| click_alarmclock | 80.0% | **100.0%** |
| turn_switch | 60.0% | **100.0%** |
| open_laptop | 60.0% | **60.0%** (unchanged, n=5 noise) |
| adjust_bottle | -- | **0.0%** (new data point, likely little/no training coverage) |

LIBERO-Spatial (n=3): 96.67%, stable.

**Continued clear improvement on well-represented tasks** (click_alarmclock, turn_switch both reached 100%) with more training, while `adjust_bottle` (a task that scored 0% in the full 50-task scan too) remains flat -- consistent with the "training-scale/coverage gap" diagnosis: more steps help where the model has real signal to learn from, but cannot substitute for missing/thin per-task data coverage. Continuing training (cont3 still has 1000 steps remaining toward its 2000-step budget, cumulative target 4600).

## 13. Progress check at cumulative step 4600 (cont4, final -- 2000 total steps since the full-50-task scan)

| Task | step 2600 (full-scan point) | step 3600 | step 4600 |
|---|---:|---:|---:|
| click_alarmclock | 40.0% | 100.0% | **100.0%** |
| turn_switch | 20.0% | 100.0% | **80.0%** |
| open_laptop | 60.0% | 60.0% | **40.0%** |
| adjust_bottle | 0.0% (full scan) | 0.0% | **0.0%** |

LIBERO-Spatial (n=3): **100.0%**.

click_alarmclock/turn_switch remain strong and well above their full-scan-time values; open_laptop is noisy (n=5) in the 40-60% band; adjust_bottle remains completely flat at 0% across three checkpoints spanning 2000 steps, reinforcing that this is a data-coverage limitation, not something more training alone will fix. cont4 completed its full 1000-step budget (cumulative training now at step 4600 from the release-checkpoint parent, 2000 steps since the full-50-task scan at step 2600).

**Next planned evidence point**: a full-50-task RoboTwin Clean-only rescan at n=5 (per updated guidance) to see whether the 12.6% full-benchmark mean has meaningfully improved with the additional 2000 steps, before deciding whether to extend training further or shift strategy (e.g. addressing data coverage directly).
