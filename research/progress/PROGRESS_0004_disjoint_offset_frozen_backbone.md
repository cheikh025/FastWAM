# PROGRESS_0004 — disjoint-offset projections + frozen backbone

- **Experiment ID:** 0004
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0003_disjoint_action_offset (rejected; this candidate reuses its checkpoint/config machinery with one variable changed)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003 — reused, no re-expansion needed)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### Accepted RoboTwin state

Unchanged from exp0003 — no candidate has yet cleared the LIBERO retention gate to justify RoboTwin evaluation compute beyond exp0001's single 2-task progress-check (0.0%).

### Accepted LIBERO retention state

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% (inherited) |
| LIBERO-Spatial | 97.00% (inherited); 96.67% fresh-machine sentinel |
| LIBERO-Object | 99.60% (inherited) |
| LIBERO-Goal | 97.20% (inherited) |
| LIBERO-Long / LIBERO-10 | 98.00% (inherited) |

Three multi-embodiment candidates tried so far, all rejected, none clearing the 90% floor:

| Candidate | Backbone | Projections | LIBERO-Spatial |
|---|---|---|---:|
| exp0001 | trainable | overlapping (K=14, both at offset 0) | 73.33% |
| exp0002 | **frozen** | overlapping (K=14, both at offset 0) | 16.67% |
| exp0003 | trainable | **disjoint** (K=21/22, LIBERO@0, RoboTwin@7/8) | 50.00% |
| exp0004 (this candidate) | **frozen** | **disjoint** | ? |

This candidate fills the missing cell in the 2x2 design. See
`research/progress/PROGRESS_0003_disjoint_action_offset.md` Section 9 for the full
investigation that motivated it: exp0002 showed backbone plasticity is net-helpful
(freezing made things worse) but confounded that with overlapping projections;
exp0003 isolated the projection fix (proven interference-free in isolation via a
gradient-isolation unit test) but confounded it with a full trainable backbone, and
scored *worse* than exp0001 — showing projection-column overlap is not the
sole/dominant interference mechanism. Neither result cleanly separates "projection
overlap" from "backbone drift" as the dominant cause.

## 3. Candidate design

### Modifications

1. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml`: identical to exp0003's `multiembodiment_libero_robotwin_disjoint_offset_3e-5.yaml` (same data/model config, same K=21/22 disjoint-offset representation, same batch/LR/schedule/save settings) except `trainable_modules: expanded_projections_only` — freezes the entire shared MoT/DiT backbone (byte-identical to exp0019 by construction, verified mechanism from exp0002), trains only `action_encoder`/`head`/`proprio_encoder` (now 21/22-wide instead of exp0002's 14-wide, but the freezing mechanism itself is width-agnostic — verified by reading `Trainer._apply_expanded_projections_only_train_mode`, which operates on module objects, not their dimensions, so no code changes were needed).
2. No other changes — reuses exp0003's `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` parent checkpoint, `ConcatLeftAlign` offset implementation, data configs, and unit tests as-is.

### Why this candidate

Directly targets the open question from exp0003's investigation: does the disjoint-
offset projection fix actually help LIBERO retention when isolated from the
confound of a fully-trainable shared backbone? exp0001 vs exp0002 already
established that backbone freezing, by itself, hurts retention when projections
still overlap (16.67% < 73.33%). This candidate asks the complementary question:
with projections disjoint (removing the projection-overlap interference exp0002
never controlled for), does freezing the backbone now help, hurt, or make no
material difference relative to exp0003's 50.00%?

- If exp0004 clears well above exp0002's 16.67% (toward or above exp0001's 73.33%): disjoint offsets genuinely help retention once isolated from backbone-plasticity effects — the next candidate should combine disjoint offsets with *some* backbone plasticity (e.g. partial/gradual unfreezing, or a lower backbone LR rather than a full freeze) to get the best of both.
- If exp0004 stays near exp0002's floor (~16-20%): backbone-level interference is implicated as the dominant mechanism regardless of projection design — motivating a fundamentally different retention strategy (explicit distillation/EWC-style regularization, LoRA-style low-rank backbone adaptation instead of full fine-tune, or a much lower RoboTwin:LIBERO sampling ratio) rather than further projection-layer tweaks.
- If exp0004 lands close to exp0003's 50.00%: freezing vs. training the backbone makes little difference once projections are disjoint, suggesting the backbone was never the dominant channel for *this* candidate family, and the 50.00% ceiling reflects some other limiting factor (e.g. genuine run-to-run variance, or a property of the disjoint-offset design itself such as the all-fresh-init RoboTwin projection weights' early-training instability).

### Multi-embodiment representation/configuration

Identical to exp0003 (K=21 action / K=22 proprio, LIBERO offset 0, RoboTwin offset
7/8) — see `PROGRESS_0003` Section 3 for full detail. Only `trainable_modules`
differs.

### Data and learning strategy

Identical to exp0003 except trainable/frozen modules: full backbone frozen
(byte-identical to exp0019), only `action_encoder`/`head`/`proprio_encoder`
trainable (matches exp0002's mechanism, applied to the wider K=21/22 tensors).

### What to watch

- Primary: does LIBERO-Spatial clear meaningfully above exp0002's 16.67% floor? This is the direct test of whether disjoint offsets help once isolated from backbone-plasticity confound.
- Secondary: how does it compare to exp0003's 50.00% — does removing backbone plasticity help, hurt, or not matter once projections are disjoint?
- Training stability: with only 6 tensors trainable (~same scale as exp0002), expect faster/lower-variance convergence on those tensors specifically; watch for NaN/instability same as prior candidates (none seen in exp0001-0003).

### Initial compute plan

- initial training budget: 1000 steps (matches exp0001/exp0002/exp0003, for a controlled single-variable comparison)
- checkpoint/save plan: `save_every: 200`, `save_full_state: false`, active `KEEP=1` pruner (15s polling) — same disk-safety pattern as exp0003, given the same ~34-45GB headroom regime
- when a progress check might be useful: not planned mid-run — exp0002's precedent (only 800 of 1000 steps evaluated due to a disk-crisis recovery) showed even a slightly-short run is informative; will evaluate at whatever the final saved checkpoint is
- expected training/evaluation cost: comparable to exp0002 (~67 min training + ~40 min LIBERO screen)

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + the new task config together, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `0ddebe0` (exp0003 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml` (new), this report
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml`
- config overrides: `model.redirect_common_files=false` (baked into `fastwam_multiembodiment.yaml` default), `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- LIBERO/RoboTwin dataset configs: unchanged from exp0003 (`configs/data/multiembodiment_libero_robotwin.yaml`, K=21/22, offsets 0 and 7/8)
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 1.0/1.0 (unchanged)
- action/state normalization configuration: unchanged
- action validity-mask configuration: unchanged (disjoint-offset masks, same as exp0003)
- model/trainable-module configuration: `trainable_modules: expanded_projections_only` — entire shared MoT/DiT backbone frozen, only `action_encoder`/`head`/`proprio_encoder` trainable
- optimizer / LR / scheduler: AdamW, cosine schedule, `learning_rate: 3e-5` (unchanged)
- batch size / gradient accumulation / effective batch: `batch_size: 1` per GPU, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps / epochs / budget: `max_steps: 1000`
- checkpoint/save cadence: `save_every: 200`
- random seed(s): not explicitly controlled (matches all prior candidates)
- resume source and resume type: weights-only resume (`save_full_state: false`) from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same checkpoint exp0003 used)

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate.

## 6. Training execution and control timeline

### Training smoke test (pre-launch validation) — PASS

8 steps, sane losses matching exp0003's smoke test step-1 loss exactly (`1.2713`,
confirming deterministic forward-pass reproducibility given the same resumed
checkpoint), no NaN/Inf. Confirmed via log line `Freezing shared MoT backbone;
trainer.py:341` (appeared twice — pre- and post-`accelerator.prepare()`, matching
the re-application-after-wrap discipline already verified in exp0002) that freezing
was actually applied for this candidate — a different log message than exp0002's
("Setting DiT to train mode...") but the correct one for `trainable_modules:
expanded_projections_only`. Checkpoint verified: `action_encoder.weight` shape
`(1024, 21)`, `head.weight` shape `(21, 1024)`, `proprio_encoder.weight` shape
`(4096, 22)` (K=21/22, matching exp0003's design), zero NaN/Inf. Smoke-test run
directory deleted after verification; log at `checkpoints/exp0004_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0004_disjoint_offset_frozen_backbone_v1 \
    save_every=200 \
    wandb.name=exp0004_disjoint_offset_frozen_backbone
  ```
- start time: 2026-08-18 05:48 UTC (immediately following the smoke test)
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0004_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0004.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log watching for checkpoint-save events and failure signatures

### Training completion

All 1000 steps completed cleanly, no NaN/Inf, no anomalies. Final: `loss=0.9458
loss_action=0.7720 loss_video=0.1738` (lr decayed to `3.00e-07`). Notably
`loss_action` at completion (0.7720) is substantially higher than exp0003's
(0.1009, trainable backbone) — consistent with a frozen backbone limiting how well
the model can fit either embodiment's action prediction through the (unchanged)
shared hidden representations, matching exp0002's analogous pattern (its final loss
was also elevated relative to exp0001). Checkpoint `step_001000.pt`
(12,042,248,481 bytes) verified: shapes `(1024,21)`/`(21,1024)`/`(4096,22)`, zero
NaN/Inf, `step: 1000`. Training log: `checkpoints/exp0004_train.log`.

### Evaluation event — LIBERO-Spatial `candidate_screen`

- benchmark: `libero`
- checkpoint / training step: exp0004, step 1000 (full budget)
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0004_disjoint_offset_frozen_backbone_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0004_disjoint_offset_frozen_backbone_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0019 canonical 97.00%; setup sentinel 96.67%; exp0001 73.33%; exp0002 16.67%; exp0003 50.00%
- result: launched, awaiting completion — log at `checkpoints/exp0004_libero_screen.log`

## 7. Evaluation events

None yet.
