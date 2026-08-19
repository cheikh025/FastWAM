# PROGRESS_0014 — Frozen-backbone diagnostic: does it stop exp0013's RoboTwin decline?

- **Experiment ID:** 0014
- **Status:** `RUNNING` (smoke test / launch in progress)
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
