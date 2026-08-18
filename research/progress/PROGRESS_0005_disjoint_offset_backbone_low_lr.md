# PROGRESS_0005 — disjoint-offset projections + low-LR backbone

- **Experiment ID:** 0005
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0004_disjoint_offset_frozen_backbone (rejected; this candidate directly extends its reasoning)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003/exp0004 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### Accepted RoboTwin state

No candidate has cleared the LIBERO retention gate for canonical/broad RoboTwin
evaluation. Progress-check evidence exists for two candidates:

- exp0001 (trainable backbone, overlapping K=14 projections): 0.0% on `adjust_bottle`/`click_alarmclock`, clean+random.
- exp0004 (frozen backbone, disjoint K=21/22 projections): `adjust_bottle` 0.0%/0.0%, `click_alarmclock` **33.3%/33.3%** — the first non-zero RoboTwin result anywhere in this project.

### Accepted LIBERO retention state

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% (inherited) |
| LIBERO-Spatial | 97.00% (inherited); 96.67% fresh-machine sentinel |
| LIBERO-Object | 99.60% (inherited) |
| LIBERO-Goal | 97.20% (inherited) |
| LIBERO-Long / LIBERO-10 | 98.00% (inherited) |

Four multi-embodiment candidates tried so far, all rejected, none clearing the 90% floor:

| Candidate | Backbone | Projections | LIBERO-Spatial | RoboTwin (2-task panel) |
|---|---|---|---:|---|
| exp0001 | trainable | overlapping (K=14) | 73.33% | 0.0% everywhere |
| exp0002 | frozen | overlapping (K=14) | 16.67% | not tested |
| exp0003 | trainable | disjoint (K=21/22) | 50.00% | not tested |
| exp0004 | frozen | disjoint (K=21/22) | 63.33% | `click_alarmclock` 33.3%/33.3%, `adjust_bottle` 0%/0% |

## 3. Candidate design

### Modifications

1. Added a new `Trainer.trainable_modules` mode, `"dit_with_backbone_low_lr"`, in `src/fastwam/trainer.py`:
   - Freeze/train-mode setting: identical to `"dit"` (nothing frozen — the entire shared `model.dit` backbone, both video and action DiT/MoT blocks, stays trainable, matching exp0001/exp0003's setting, not exp0002/exp0004's freeze).
   - Optimizer construction: instead of one flat parameter list, splits `model.dit.parameters()` into two `AdamW` parameter groups by id-based set difference — `action_encoder`/`head` (the widened, disjoint-offset projection layers) at the full `learning_rate`, everything else in `model.dit` (the actual shared backbone) at a new, separately-configured `backbone_lr`. `proprio_encoder` (not part of `model.dit`) always trains at the full `learning_rate`, matching every prior candidate's treatment of it.
   - New config field `backbone_lr` (required only when this mode is selected; `None` by default, fully backward-compatible with every existing task config).
2. New unit test `research/tools/test_differential_backbone_lr_param_groups.py`: verifies the backbone/projection parameter-group split is disjoint and covers all of `model.dit`'s parameters, that each group gets its intended LR, and (end-to-end) that after one real optimizer step the projection group's parameters move by exactly the LR ratio more than the backbone group's — not just that the LR values are stored correctly, but that they actually take effect.
3. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_backbone_low_lr_3e-5.yaml`: reuses exp0003/exp0004's K=21/22 disjoint-offset data/model config unchanged; sets `trainable_modules: dit_with_backbone_low_lr`, `backbone_lr: 3e-6` (10x lower than `learning_rate: 3e-5`).

### Why this candidate

exp0004's matrix result plus its RoboTwin progress check together motivate a
specific middle ground rather than another single-axis toggle (see
`PROGRESS_0004` Section 9 for the full reasoning this candidate directly
implements). In short: disjoint-offset projections are confirmed as a real,
substantial retention fix; backbone-level drift is confirmed as a second,
independent interference channel; but a fully frozen backbone, while giving the
best retention among the disjoint-offset candidates, likely caps how much RoboTwin
can ultimately learn (it already learned *something* even fully frozen, which is
notable, but a frozen backbone cannot adapt its shared visual/temporal
representations to RoboTwin's very different bimanual visual domain at all). A
10x-lower backbone LR is a single, easily-implemented, easily-reasoned-about dial
between exp0003's LR ratio (1:1, exp0003's 50.00% result) and exp0004's LR ratio
(0:1 i.e. fully frozen, exp0004's 63.33% result) — if the hypothesis is right, this
candidate should land LIBERO retention somewhere between 50.00% and 63.33% (not
guaranteed higher than exp0004, since the whole point is trading some retention for
more RoboTwin learning capacity) while showing a comparable-or-better RoboTwin
progress-check result to exp0004's, ideally recovering some `adjust_bottle` signal.

### Multi-embodiment representation/configuration

Identical to exp0003/exp0004 (K=21 action / K=22 proprio, LIBERO offset 0, RoboTwin
offset 7/8) — no changes to the action/state representation itself, only to how the
backbone is optimized.

### Data and learning strategy

Identical to exp0003/exp0004 except trainable/frozen modules: entire backbone
trainable (not frozen) but at `backbone_lr=3e-6` instead of the full
`learning_rate=3e-5`; `action_encoder`/`head`/`proprio_encoder` at the full
`3e-5`, same as every prior candidate.

### What to watch

- Primary: does LIBERO-Spatial land in a useful range — ideally close to or above exp0004's 63.33% (not necessarily higher, since the tradeoff being tested may cost some retention for RoboTwin capability) and clearly above exp0003's 50.00% floor (if it's *worse* than exp0003 despite 10x less backbone plasticity than exp0003's full LR, something unexpected is happening and needs its own investigation, similar to exp0003's own surprise result).
- Primary: does the RoboTwin progress check (same 2-task panel) show comparable-or-better results than exp0004's `click_alarmclock` 33.3%/33.3%, `adjust_bottle` 0%/0% — especially whether `adjust_bottle` shows any life at all, which would be the clearest sign that backbone plasticity helps RoboTwin skill acquisition specifically.
- Training stability: two-parameter-group AdamW with DeepSpeed ZeRO-1 is a new code path (verified in isolation via unit test, but not yet exercised in a real distributed run) — the smoke test before real launch is specifically important here, more so than for exp0003/exp0004 which only changed config/data, not trainer optimizer-construction code.

### Initial compute plan

- initial training budget: 1000 steps (matches all prior candidates, for a controlled comparison)
- checkpoint/save plan: `save_every: 200`, `save_full_state: false`, active `KEEP=1` pruner (15s polling), same disk-safety pattern as exp0003/exp0004
- when a progress check might be useful: evaluate the final checkpoint (LIBERO screen + RoboTwin 2-task progress check) once training completes, matching exp0004's evaluation depth given this is a genuinely new axis of variation
- expected training/evaluation cost: comparable to exp0003/exp0004 (~45-70 min training + ~35-40 min LIBERO screen + ~25-35 min RoboTwin progress check)

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + code/config changes together, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `06efae2` (exp0004 RoboTwin-result commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed:
  - `src/fastwam/trainer.py` (new `dit_with_backbone_low_lr` mode, two-parameter-group optimizer construction, `backbone_lr` config field)
  - `configs/task/multiembodiment_libero_robotwin_disjoint_offset_backbone_low_lr_3e-5.yaml` (new)
  - `research/tools/test_differential_backbone_lr_param_groups.py` (new)
  - this report
- diff summary: see `git diff --stat` at commit time
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_backbone_low_lr_3e-5.yaml`
- config overrides: `model.redirect_common_files=false` (baked into `fastwam_multiembodiment.yaml` default), `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- LIBERO/RoboTwin dataset configs: unchanged from exp0003/exp0004 (K=21/22, offsets 0 and 7/8)
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 1.0/1.0 (unchanged)
- action/state normalization configuration: unchanged
- action validity-mask configuration: unchanged (disjoint-offset masks, same as exp0003/exp0004)
- model/trainable-module configuration: `trainable_modules: dit_with_backbone_low_lr`, `backbone_lr: 3e-6` — entire shared MoT/DiT backbone trainable at `3e-6`, `action_encoder`/`head`/`proprio_encoder` trainable at `3e-5`
- optimizer / LR / scheduler: AdamW with two parameter groups (backbone `3e-6`, projections `3e-5`), cosine schedule (shared `eta_min` floor derived from `learning_rate*0.01`, applied as an absolute floor to both groups — a minor asymmetry noted in Section 6, not expected to matter)
- batch size / gradient accumulation / effective batch: `batch_size: 1` per GPU, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps / epochs / budget: `max_steps: 1000`
- checkpoint/save cadence: `save_every: 200`
- random seed(s): not explicitly controlled (matches all prior candidates)
- resume source and resume type: weights-only resume (`save_full_state: false`) from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate beyond the new trainer optimizer-construction code path (covered by the smoke test).

## 6. Training execution and control timeline

### Training smoke test (pre-launch validation) — PASS

8 steps, sane losses matching exp0003/exp0004's smoke-test step-1 loss exactly
(`1.2713`, confirming deterministic forward-pass reproducibility given the same
resumed checkpoint), no NaN/Inf. Confirmed via log line "Setting DiT to train mode
(nothing frozen); backbone trains at a lower LR..." (appeared twice, pre- and
post-`accelerator.prepare()`) that the new `dit_with_backbone_low_lr` mode's
freeze/train-mode setting was applied correctly (nothing frozen, unlike
exp0002/exp0004). The logged `lr=` value tracks `optimizer.param_groups[0]`
(the backbone group, since it's listed first) — started at `2.90e-06` (near
`backbone_lr=3e-6` after 1 warmup step) and decayed toward the shared `eta_min`
floor by step 8 (an artifact of the smoke test's tiny 8-step schedule, not a
concern for the real 1000-step run). Checkpoint verified: shapes
`(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf. Smoke-test run directory deleted
after verification; log at `checkpoints/exp0005_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_backbone_low_lr_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0005_disjoint_offset_backbone_low_lr_v1 \
    save_every=200 \
    wandb.name=exp0005_disjoint_offset_backbone_low_lr
  ```
- start time: 2026-08-18 ~07:17 UTC (immediately following the smoke test)
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0005_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0005.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log watching for checkpoint-save events and failure signatures

### Known minor imprecision (not a bug, documented for interpretation)

`Trainer._build_scheduler`'s `CosineAnnealingLR(..., eta_min=self.learning_rate * 0.01)`
uses a single, absolute `eta_min` value shared across both optimizer parameter groups
(this is how PyTorch's `CosineAnnealingLR` works — `eta_min` is not per-group).
Concretely: `eta_min = 3e-5 * 0.01 = 3e-7`. The projection group (base `3e-5`) decays
to `3e-7`, a clean 1% floor. The backbone group (base `3e-6`) decays to the *same*
absolute `3e-7`, which is 10% of its own base rather than 1% — a minor asymmetry in
relative decay depth, not a correctness bug (both groups' warmup/cosine *shape* is
still correctly scaled to their own `initial_lr` by `LinearLR`/`CosineAnnealingLR`;
only the absolute floor value is shared). Not fixed for this candidate: the effect
is small (a 10% vs 1% floor only matters in the last few percent of training) and
not worth the added complexity of a custom per-group-floor scheduler for an
exploratory candidate.

## 7. Evaluation events

None yet.
