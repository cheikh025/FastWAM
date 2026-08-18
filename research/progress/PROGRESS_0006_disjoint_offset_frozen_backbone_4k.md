# PROGRESS_0006 — disjoint-offset frozen backbone, 4x training budget

- **Experiment ID:** 0006
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0004_disjoint_offset_frozen_backbone (rejected on LIBERO floor, but the best-so-far recipe with genuine RoboTwin capability; this candidate directly extends its training budget)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003/exp0004/exp0005 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### Investigation that motivated this candidate (see PROGRESS_0005 Section 9)

exp0005 tested whether partial backbone plasticity (10x lower LR than the
projection layers, same 1000-step budget as exp0004) would preserve retention while
unlocking more RoboTwin capability. It did not: `adjust_bottle` stayed dead
(0.0%/0.0%), and `click_alarmclock` **lost** its randomized-phase success entirely
(33.3%->0.0%) — the opposite of the hypothesis. This pointed at two open
questions:

1. Is exp0004's `click_alarmclock` 33.3%/33.3% itself just n=3-episode noise?
2. If real, is training **budget** (not the backbone LR-ratio dial) the more
   relevant lever — RoboTwin's own action/video pathway received the same ~500
   realized gradient steps regardless of backbone LR (~1:1 LIBERO:RoboTwin
   interleaving over 1000 total steps), so a small, fast-to-saturate 6-tensor
   trainable budget (`action_encoder`/`head`/`proprio_encoder`) may simply need
   more exposure rather than needing the shared backbone to adapt at all.

This candidate directly tests (2): re-run exp0004's *exact* recipe (frozen
backbone, disjoint-offset projections — no new variable) for 4x the training
budget. This also directly follows the standing project instruction ("u migh later
also attempt longer steps 8k or 4k... yes contunie owkrkign") to try longer runs
once a working retention-preserving recipe exists — exp0004's frozen+disjoint
recipe is exactly that starting point, and reusing it exactly (rather than
combining a longer budget with yet another new variable) keeps this a clean,
interpretable single-variable extension.

### Accepted RoboTwin state

- exp0001 (trainable backbone, overlapping K=14): 0.0% everywhere.
- exp0004 (frozen backbone, disjoint K=21/22): `adjust_bottle` 0.0%/0.0%, `click_alarmclock` **33.3%/33.3%** — first non-zero RoboTwin result in this project.
- exp0005 (partial-plasticity backbone, disjoint K=21/22): `adjust_bottle` 0.0%/0.0%, `click_alarmclock` 33.3%/**0.0%** — regression from exp0004.

### Accepted LIBERO retention state

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% (inherited) |
| LIBERO-Spatial | 97.00% (inherited); 96.67% fresh-machine sentinel |
| LIBERO-Object | 99.60% (inherited) |
| LIBERO-Goal | 97.20% (inherited) |
| LIBERO-Long / LIBERO-10 | 98.00% (inherited) |

Five multi-embodiment candidates tried so far, all rejected, none clearing the 90% floor:

| Candidate | Backbone | Projections | Steps | LIBERO-Spatial | RoboTwin (2-task panel) |
|---|---|---|---:|---:|---|
| exp0001 | trainable | overlapping (K=14) | 1000 | 73.33% | 0.0% everywhere |
| exp0002 | frozen | overlapping (K=14) | 1000 | 16.67% | not tested |
| exp0003 | trainable | disjoint (K=21/22) | 1000 | 50.00% | not tested |
| exp0004 | frozen | disjoint (K=21/22) | 1000 | 63.33% | `click_alarmclock` 33.3%/33.3% |
| exp0005 | partial (backbone_lr=3e-6) | disjoint (K=21/22) | 1000 | 63.33% | `click_alarmclock` 33.3%/0.0% (worse) |

## 3. Candidate design

### Modifications

1. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_4k_3e-5.yaml`: byte-for-byte identical to exp0004's `multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml` except `max_steps: 4000` (was 1000) and `save_every: 500` (was 100/200 — widened simply to keep the number of intermediate checkpoints manageable across a 4x longer run; disk headroom is unaffected either way since the `KEEP=1` pruner bounds usage regardless of total step count).
2. No code changes — reuses exp0004's `trainable_modules: expanded_projections_only` mechanism and exp0003/exp0004/exp0005's `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` parent checkpoint unchanged.

### Why this candidate

See "Investigation that motivated this candidate" above. In short: exp0004 is the
best-so-far recipe (highest retention among candidates with any measured RoboTwin
capability), and exp0005 showed that changing the backbone-plasticity dial on the
*same* budget makes things worse, not better — so the next single-variable test is
budget, not another architecture/optimization tweak, using the recipe already known
to work best.

### Multi-embodiment representation/configuration

Identical to exp0003/exp0004/exp0005 (K=21 action / K=22 proprio, LIBERO offset 0,
RoboTwin offset 7/8) — no changes.

### Data and learning strategy

Identical to exp0004: entire shared MoT/DiT backbone frozen (byte-identical to
exp0019), only `action_encoder`/`head`/`proprio_encoder` trainable, LR `3e-5`,
~1:1 LIBERO:RoboTwin batch interleaving — only the total step count changes (1000 -> 4000).

### What to watch

- Primary: does LIBERO-Spatial retention hold near exp0004's 63.33% (or degrade) under 4x more RoboTwin gradient exposure through the same 6 trainable tensors — if the disjoint-offset fix is doing its job, retention should not degrade meaningfully just from more steps through an otherwise-frozen backbone.
- Primary: does RoboTwin capability meaningfully improve with more steps — recover any `adjust_bottle` signal, strengthen or at least maintain `click_alarmclock`'s clean/randomized results (a *degradation* from more steps on the same frozen-backbone recipe would be a genuinely new and important finding, since it would rule out "more budget helps" too).
- Mid-run progress check at step ~2000 (halfway) is planned specifically to catch early divergence/degradation before committing the full ~2.5-hour budget — this is the adaptive-evaluation discipline the project calls for on longer runs, not present in prior 1000-step candidates where a mid-run check wasn't as valuable relative to the total cost.

### Initial compute plan

- initial training budget: 4000 steps (4x exp0001-0005's 1000-step budget)
- checkpoint/save plan: `save_every: 500` (8 checkpoints across the run), `save_full_state: false`, active `KEEP=1` pruner (15s polling)
- expected training speed: exp0004's frozen-backbone recipe ran at `0.45 step/s` (the fastest of any candidate, since gradients are only computed for 6 tensors) — 4000 steps projects to roughly `4000/0.45 ≈ 8900s ≈ 148 min ≈ 2.5 hours`
- when a progress check might be useful: at step ~2000 (halfway), a cheap LIBERO-Spatial sentinel + RoboTwin 2-task progress check to decide `CONTINUE_TRAINING` / `EXTEND_TRAINING` / `STOP_TRAINING` / `SELECT_CHECKPOINT` before committing the remaining budget
- expected total cost: ~2.5h training + evaluation time at the midpoint and end (each LIBERO screen ~35-40min, each RoboTwin progress check ~25-30min) — the most compute-intensive candidate so far, justified by directly testing the standing project's own suggested longer-training-run direction using the best-validated recipe

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + config change together, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `4c481ae` (exp0005 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_4k_3e-5.yaml` (new), this report
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_4k_3e-5.yaml`
- config overrides: `model.redirect_common_files=false`, `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- LIBERO/RoboTwin dataset configs: unchanged from exp0003/exp0004/exp0005
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 1.0/1.0 (unchanged)
- action/state normalization/mask configuration: unchanged
- model/trainable-module configuration: `trainable_modules: expanded_projections_only` (exp0004's exact setting)
- optimizer / LR / scheduler: AdamW, cosine schedule, `learning_rate: 3e-5` (unchanged)
- batch size / gradient accumulation / effective batch: `batch_size: 1` per GPU, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps / epochs / budget: `max_steps: 4000`
- checkpoint/save cadence: `save_every: 500`
- random seed(s): not explicitly controlled (matches all prior candidates)
- resume source and resume type: weights-only resume (`save_full_state: false`) from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate.

## 6. Training execution and control timeline

Not yet launched.

## 7. Evaluation events

None yet.
