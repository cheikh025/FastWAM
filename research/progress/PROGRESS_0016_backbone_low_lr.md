# PROGRESS_0016 — Partial backbone plasticity (dit_with_backbone_low_lr)

- **Experiment ID:** 0016
- **Status:** `RUNNING` (smoke test / launch in progress)
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
