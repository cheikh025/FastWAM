# PROGRESS_0003 — disjoint per-embodiment column offset

- **Experiment ID:** 0003
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0000_parent_baseline (exp0019); diagnosis inherited from 0001 and 0002
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` — re-expanded from `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt` (hash-verified `decbb99c...` before expansion). Expansion verified: LIBERO's original weights are byte-identical at `[0:7]`/`[0:8]`, RoboTwin's new `[7:21]`/`[8:22]` range is non-zero (encoder) / zero (head), matching the deliberate init rule.
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** `8e74590` (implementation commit; checkpoint expansion + training launch follow)

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### Accepted RoboTwin state

- canonical metric(s): full 50-task Aloha-AgileX benchmark, Clean/Randomized tracked separately; not yet measured for any accepted checkpoint. exp0001 (rejected): 0.0% on a 2-task progress-check panel. exp0002 (rejected): not evaluated (already rejected on LIBERO alone).
- important weak tasks/difficulties: none characterized yet — no candidate has reached RoboTwin evaluation with non-collapsed LIBERO retention.

### Accepted LIBERO retention state

Inherited canonical parent record (exp0019, not re-measured this session beyond setup sentinel):

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% (inherited) |
| LIBERO-Spatial | 97.00% (inherited); 96.67% fresh-machine sentinel (30 trials) |
| LIBERO-Object | 99.60% (inherited) |
| LIBERO-Goal | 97.20% (inherited) |
| LIBERO-Long / LIBERO-10 | 98.00% (inherited) |

Both prior multi-embodiment candidates regressed LIBERO-Spatial (the only suite measured for either candidate so far, used as the cheap sentinel):

- exp0001 (full fine-tune, K=14 shared, both embodiments left-aligned at offset 0): 96.67% → 73.33%.
- exp0002 (frozen backbone, same K=14/offset-0 layout, only projection layers trainable): 96.67% → **16.67%**, worse than exp0001 despite the backbone being byte-identical to exp0019 by construction.

exp0002's result refuted the leading hypothesis at the time (unmasked video-denoising loss training the shared backbone on RoboTwin's visual domain) — the backbone was provably frozen, yet retention got worse, not better. This forced re-diagnosis of the actual mechanism (Section 3 below), which is the direct motivation for this candidate.

## 3. Candidate design

### Modifications

1. Add `action_offset`/`state_offset` parameters to `ConcatLeftAlign`
   (`src/fastwam/datasets/lerobot/transforms/action_state_merger.py`), defaulting to
   `0` (fully backward-compatible with every existing single-embodiment and
   exp0001/exp0002 config). `_pad` places the natural-dim data at
   `[offset, offset+native_dim)` within the K-wide tensor instead of always
   `[0, native_dim)`; `_crop` (used in `backward()`/inference decode) mirrors this.
2. Widen the shared multi-embodiment action/proprio dimension from K=14 to
   **K=21 (action) / K=22 (proprio)** = LIBERO's natural dim (7/8) + RoboTwin's
   natural dim (14/14), and assign each embodiment a **disjoint** column range:
   LIBERO stays at offset 0 (`[0:7]`/`[0:8]`, unchanged from exp0001/exp0002 — the
   inherited exp0019 weight positions need no repositioning), RoboTwin moves to
   offset 7/8 (`[7:21]`/`[8:22]`, previously `[0:14]`/`[0:14]`).
3. Re-expand the checkpoint: `research/tools/expand_checkpoint_for_multiembodiment.py`
   run against the **original, unexpanded** exp0019 checkpoint (not
   `exp0019_expanded_k14`) with `--new-action-dim 21 --new-proprio-dim 22`, producing
   `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`. No code changes to the
   tool itself — it already appends new columns after the original ones, which is
   exactly the placement LIBERO needs at offset 0.
4. Update multi-embodiment configs (`configs/data/libero_2cam_multiembodiment.yaml`,
   `configs/data/robotwin_multiembodiment.yaml`,
   `configs/data/multiembodiment_libero_robotwin.yaml`,
   `configs/model/fastwam_multiembodiment.yaml`) to the new K=21/22 dims and offsets.
5. Update the eval-side pad/crop logic in `experiments/libero/eval_libero_single.py`
   and `experiments/robotwin/fastwam_policy/deploy_policy.py` to read
   `action_offset`/`state_offset` from the merger (defaulting to 0) instead of
   assuming offset 0 unconditionally.
6. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3e-5.yaml`
   — otherwise identical to exp0001's `multiembodiment_libero_robotwin_3e-5.yaml`
   (full fine-tune, no backbone freezing; same batch/LR/schedule/save settings).
7. New unit tests: `research/tools/test_action_state_merger_offset.py` — offset
   padding/cropping correctness, a forward/backward roundtrip under disjoint offsets,
   and (the central claim) a gradient-isolation test proving a **shared** `Linear`'s
   weight columns/rows outside an embodiment's own offset range receive exactly zero
   gradient from that embodiment's batches — plus a contrast test confirming the OLD
   offset-0-for-both layout does leak gradient (validating the isolation test's own
   premise, not just an unrelated masking coincidence).

**No changes** to `src/fastwam/models/wan22/action_dit.py`, `fastwam.py`,
`fastwam_idm.py`, `trainer.py`, the dataset/sampler pipeline, or checkpoint
save/load — an earlier in-session attempt at this fix used fully separate
per-embodiment `nn.ModuleDict` weight matrices (threading an `embodiment` string
through `pre_dit`/`post_dit`/`build_inputs`/`training_loss`/dataset tagging/checkpoint
state dicts), which was reverted (`git checkout --`) once the disjoint-offset
approach was worked out, since it achieves the identical interference-elimination
guarantee with a far smaller, lower-risk diff confined entirely to the padding/config
layer.

### Why this candidate

Re-diagnosis of exp0001/exp0002 (see `PROGRESS_0002_frozen_backbone_warmup.md`
Section 9-10): `action_encoder`/`head`/`proprio_encoder` are single shared weight
matrices. Under the exp0001/exp0002 K=14 layout, LIBERO's valid columns are `[0:7]`
(action) / `[0:8]` (proprio) and RoboTwin's valid columns are `[0:14]` (both) —
RoboTwin's range is a strict **superset** of LIBERO's, because both embodiments were
left-aligned at offset 0 via `ConcatLeftAlign`'s only-ever-implemented behavior.

For a `Linear` layer `y = xW^T + b`, the gradient contribution to weight column `j`
from a given batch is proportional to that batch's input value at column `j`. LIBERO
batches have exactly-zero input at columns `[7:14]` (the padded region), so they
never touch those weight columns — that part was always correct. But RoboTwin
batches have **nonzero** input at columns `[0:7]` too (RoboTwin's own real,
non-padded first 7 native channels), so every RoboTwin training step directly
overwrites the same `action_encoder` weight columns (and, symmetrically, `head`
weight rows, protected only by loss masking on the output side but not on the input
side) that LIBERO's forward pass depends on — regardless of the per-channel loss
masking, which correctly zeroes each embodiment's own **padding-only** positions but
cannot protect a position the *other* embodiment considers genuinely valid. exp0002
(freezing everything except these projection layers) made retention markedly worse
(16.67% vs. exp0001's 73.33%) precisely because there was no longer any downstream
plasticity elsewhere in the network to partially absorb/compensate for this direct
overwrite — strong evidence the interference is located at the projection-weight
level, not the shared MoT backbone.

The fix directly targets this mechanism: assign each embodiment a **disjoint**
column range so that, for every training batch, every weight column/row outside that
batch's embodiment's own range has **exactly zero** input (or exactly zero loss
gradient at the output, already true) — making a shared `Linear`'s per-embodiment
isolation property mathematically identical to using fully separate weight matrices,
without the much larger implementation surface (embodiment threading through the
model/training/checkpoint/dataset code) that a true separate-weights design would
require. This is confirmed directly (not just argued) in
`test_shared_linear_gradient_isolation_across_embodiments`.

### Multi-embodiment representation/configuration

- shared action dimension: K=21 (was 14)
- shared proprio/state dimension: K=22 (was 14)
- LIBERO valid channels/mask: action `[0:7]`, proprio `[0:8]` (offset 0, unchanged position from exp0001/exp0002)
- RoboTwin valid channels/mask: action `[7:21]`, proprio `[8:22]` (offset 7/8, moved from `[0:14]`/`[0:14]`)
- normalization/statistics behavior: unchanged — each dataset normalizes in its own natural (un-padded) dimensionality before the merger pads/offsets (LIBERO min/max, RoboTwin z-score), per the existing `ConcatLeftAlign.forward()` ordering.
- checkpoint projection expansion/initialization: re-run `expand_checkpoint_for_multiembodiment.py` against the *original* exp0019 checkpoint (not the K=14 expansion) with `--new-action-dim 21 --new-proprio-dim 22`; LIBERO's inherited weights land unchanged at their original column/row positions (0-6/0-7), RoboTwin's new columns/rows (7-20/8-21) get the same deliberate init as before (encoder input columns: fresh-`nn.Linear`-seeded; head output rows: zero).
- embodiment/control conditioning: unchanged — Qwen-VLA-style textual embodiment description prepended to the instruction (`FastWAMProcessor.augment_instruction()`), same as exp0001/exp0002.
- camera/observation handling: unchanged from exp0001/exp0002 (each embodiment keeps its own native camera count/resolution; `InterleavedEmbodimentSampler` keeps batches embodiment-homogeneous).
- inference slicing/decoding: `eval_libero_single.py`/`deploy_policy.py` now crop/pad using each merger's `action_offset`/`state_offset` (read via `getattr(merger, "action_offset", 0)`) instead of assuming offset 0.

### Data and learning strategy

- LIBERO datasets/tasks: unchanged from exp0001/exp0002 — Spatial/Object/Goal/Long-10 (LIBERO-90 excluded, no lerobot-format training data available; monitored via evaluation only).
- RoboTwin datasets/tasks: unchanged — `./data/robotwin2.0/robotwin2.0`.
- sampling/mixing ratios: unchanged — ~1:1 batch-level interleaving via `InterleavedEmbodimentSampler` (`ratio: 1.0` each).
- per-task/per-dataset weights: unchanged.
- replay/rehearsal strategy: unchanged (LIBERO itself IS the replay stream at ~1:1 ratio; no separate rehearsal buffer).
- loss weights: unchanged (`loss.lambda_action: 1.0`; masked per-channel/per-timestep action loss, video loss unmasked/shared as before).
- retention/distillation/regularization: none added this candidate — isolating the offset fix as the single variable against exp0001's full-fine-tune setting (not exp0002's frozen-backbone setting, since freezing was refuted as a fix).
- trainable/frozen modules: full fine-tune, no freezing (matches exp0001, not exp0002's `trainable_modules: expanded_projections_only`).

### What to watch

- Primary: does LIBERO-Spatial sentinel retention recover toward the 96.67% fresh-machine baseline (or at least clear the >=90% floor), confirming the disjoint-offset fix actually eliminates the regression exp0001/exp0002 both showed?
- Does RoboTwin begin to show any non-zero learning signal (exp0001's 0.0% RoboTwin result was measured on a severely LIBERO-collapsed checkpoint — even if the model is not yet good at RoboTwin, a healthy LIBERO retention run should be evaluated on RoboTwin regardless, since a 0% floor with intact LIBERO would itself be informative)?
- Training stability: no NaNs/OOMs; loss curves for both embodiments' action loss and the shared video loss behave sensibly (no discontinuity at batch-embodiment switches).
- If LIBERO retention is now healthy, watch for how much RoboTwin actually learns in the same 1000-step budget as exp0001 — the *unrelated* interference fix might not by itself be enough compute/signal for strong RoboTwin performance; that would motivate a follow-up candidate (e.g. longer training, per the user's message #8 guidance) rather than a re-diagnosis.

### Initial compute plan

- initial training budget: 1000 steps (matches exp0001, for a controlled single-variable comparison — the offset fix is the only intended difference in candidate design)
- checkpoint/save plan: `save_every: 100`, `save_full_state: false` (same disk-pressure-driven settings as exp0001/exp0002); active pruner during the run (`KEEP` sized to the disk-math rule from `research/NOTES.md`, given `/workspace` is at 45GB free as of this report — see Section 5)
- when a progress check might be useful: a cheap LIBERO-Spatial sentinel (10 tasks x 3 trials, same panel used for exp0001/exp0002) as early as practical once a checkpoint exists (e.g. step 100-200), since the entire point of this candidate is to test whether the interference mechanism is actually fixed — no need to wait for the full 1000 steps if the sentinel already shows clear recovery or clear continued collapse
- expected training/evaluation cost: comparable to exp0001/exp0002 (~1000 steps, same batch/GPU config)

## 4. Exact code and configuration state

- Git commit: `8e74590`
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `e0ac005` (exp0002 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed:
  - `src/fastwam/datasets/lerobot/transforms/action_state_merger.py` (offset support)
  - `configs/data/libero_2cam_multiembodiment.yaml`
  - `configs/data/robotwin_multiembodiment.yaml`
  - `configs/data/multiembodiment_libero_robotwin.yaml`
  - `configs/model/fastwam_multiembodiment.yaml`
  - `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3e-5.yaml` (new)
  - `configs/task/libero_uncond_2cam224_multiembodiment_eval.yaml` (comment only)
  - `configs/task/robotwin_uncond_3cam_384_multiembodiment_eval.yaml` (comment only)
  - `experiments/libero/eval_libero_single.py` (offset-aware pad/crop)
  - `experiments/robotwin/fastwam_policy/deploy_policy.py` (offset-aware pad/crop)
  - `research/tools/test_action_state_merger_offset.py` (new)
  - `research/RUNBOOK.md`, `research/STATE.md` (this candidate's design/status)
- diff summary: see `git diff --stat` at commit time
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3e-5.yaml`
- config overrides: `model.redirect_common_files=false` (standing requirement, dead HF repo — baked into `fastwam_multiembodiment.yaml` default), `resume=<path to checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt>`
- LIBERO dataset config: `configs/data/libero_2cam_multiembodiment.yaml` (K=21/22, offset 0/0)
- RoboTwin dataset config: `configs/data/robotwin_multiembodiment.yaml` (K=21/22, offset 7/8)
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 1.0/1.0 (unchanged)
- action/state normalization configuration: unchanged (LIBERO min/max, RoboTwin z-score, per-dataset before padding)
- action validity-mask configuration: `action_dim_is_pad` now reflects the disjoint offset layout automatically (derived from `ConcatLeftAlign._pad`'s mask output, which already accounts for `offset`)
- model/trainable-module configuration: full fine-tune (`trainable_modules` not set / default `"dit"`, i.e. all parameters trainable — matches exp0001)
- optimizer / LR / scheduler: AdamW (project default), cosine schedule, `learning_rate: 3e-5` (unchanged from exp0001/exp0002)
- batch size / gradient accumulation / effective batch: `batch_size: 1` per GPU, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps / epochs / budget: `max_steps: 1000`
- checkpoint/save cadence: `save_every: 100`
- random seed(s): not explicitly controlled (matches exp0001/exp0002 — not a controlled-seed comparison)
- resume source and resume type: weights-only resume (`save_full_state: false`) from the newly re-expanded `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`

## 5. Hardware and software environment

Not applicable — setup already validated in `PROGRESS_0000_PARENT_BASELINE.md`; no infrastructure changes this candidate beyond the checkpoint re-expansion (Section 3). Disk: `/workspace` at 45GB free / 1.1TB total as of this report (2026-08-18); root filesystem `/` (212GB, ephemeral) available as fallback overflow if needed (see `research/NOTES.md`).

## 6. Training execution and control timeline

### Training smoke test (pre-launch validation) — PASS

Command:
```bash
bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_3e-5 \
  resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
  output_dir=./runs/_smoke_test/multiembodiment_exp0003_v1 \
  max_steps=8 save_every=8 log_every=1 eval_every=999999
```

Result: all 8 steps completed with sane, non-NaN losses (`1.2713, 0.8076, 1.2550, 1.7008,
2.2254, 1.3199, 1.2212, 1.4882`), datasets built correctly (`libero: 277713 samples,
ratio=1.000`, `robotwin: 6011575 samples, ratio=1.000`), checkpoint written and
verified: `action_encoder.weight` shape `(1024, 21)`, `head.weight` shape `(21, 1024)`,
`proprio_encoder.weight` shape `(4096, 22)` (matching the new K=21/22 disjoint-offset
design), zero NaN/Inf across every tensor. The existing 904GB RoboTwin +81MB LIBERO
text-embedding caches (built during exp0001 setup) required no recomputation — text
embeddings depend only on the (unchanged) instruction/embodiment-description strings,
not the action/proprio dimension. Smoke-test run directory deleted after verification
(temporary artifact); log at `checkpoints/exp0003_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0003_disjoint_offset_v1 \
    save_every=200 \
    wandb.name=exp0003_disjoint_offset_baseline
  ```
- start time: 2026-08-18 (immediately following the smoke test)
- number of GPUs/world size: 4 (`scripts/accelerate_configs/accelerate_zero1_ds.yaml`, DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0003_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0003.log`), `KEEP=1`
  (only the most recent weights-only checkpoint retained at a time), 15s polling —
  sized against 34GB free at launch (`(KEEP+1) x ~12GB = 24GB <= 34GB`, per the
  arithmetic rule from `research/NOTES.md` after the exp0001/exp0002 disk-crisis
  recurrences)
- monitoring: persistent `Monitor` on the training log watching for checkpoint-save
  events, prune events, and failure signatures (Traceback/Error/NaN/OOM/disk-write-failure)

### Training progress (check-in at step 350/1000)

Healthy: losses well-behaved (`loss` 0.18-0.88, `loss_action` 0.05-0.70, no NaN/Inf,
no spikes remotely like exp0001's step-5 5.02 outlier), `speed=0.27 step/s`,
`eta=00:40:28` as of step 350. `step_000200.pt` checkpoint exists and is being
correctly retained by the pruner (only one file present, matching `KEEP=1`). Disk
steady at 34GB free.

**Decision on early LIBERO-Spatial screening**: deferred rather than run now.
All 4 GPUs are at 61-65GB/80GB from the active training job (~16-20GB free per
GPU); the LIBERO eval loads the full model per worker (~14-20GB per
`research/RUNBOOK.md`'s sizing note), so a parallel eval risks OOM-crashing either
the eval or, worse, the training job itself. With training healthy and ~40 min from
completing its full 1000-step budget, waiting for completion is safer than risking
an expensive 4-GPU training run for an earlier partial-step checkpoint.

### Training completion

All 1000 steps completed cleanly, no NaN/Inf, no anomalies, no OOM, no disk errors.
Final: `loss=0.2449 loss_action=0.1009 loss_video=0.1440` (lr decayed to `3.00e-07`
per cosine schedule). Checkpoint `step_001000.pt` (12,041,907,641 bytes) written and
verified: `action_encoder.weight` shape `(1024, 21)`, `head.weight` shape
`(21, 1024)`, `proprio_encoder.weight` shape `(4096, 22)` (matches K=21/22 design),
zero NaN/Inf across every `mot` tensor, `step: 1000` metadata correct. Training log:
`checkpoints/exp0003_train.log`. Pruner/monitor stopped cleanly after confirming no
training processes remained and all 4 GPUs returned to 0MiB/0% usage.

### Evaluation event — LIBERO-Spatial `candidate_screen`

- benchmark: `libero`
- checkpoint / training step: exp0003, step 1000 (full budget)
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0003_disjoint_offset_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0003_disjoint_offset_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0019 canonical 97.00%; setup sentinel 96.67% (29/30); exp0001 (same panel) 73.33% (22/30); exp0002 (same panel) 16.67% (5/30)
- result: launched, awaiting completion — log at `checkpoints/exp0003_libero_screen.log`

## 7. Evaluation events

None yet.
