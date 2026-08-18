# PROGRESS_0004 — disjoint-offset projections + frozen backbone

- **Experiment ID:** 0004
- **Status:** `REJECT`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0003_disjoint_action_offset (rejected; this candidate reuses its checkpoint/config machinery with one variable changed)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003 — reused, no re-expansion needed)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Training completed cleanly (1000/1000 steps, no NaN/anomalies). LIBERO-Spatial
candidate_screen: **63.33% (19/30)** — the best multi-embodiment retention result
after exp0001 (73.33%), and a large jump over exp0002 (16.67%, the other
frozen-backbone candidate). This fills the missing cell of the 2x2 (backbone
trainable/frozen x projections overlap/disjoint) matrix and reveals two clean,
*opposite-direction* effects: freezing the backbone hurts when projections overlap
(73.33%->16.67%) but helps when projections are disjoint (50.00%->63.33%) — evidence
that disjoint offsets are a real, substantial fix, and that backbone-level drift is a
*second, independent* interference channel. Still below exp0001 and far below the
90% floor. A RoboTwin progress check on the same checkpoint (2 tasks, clean+randomized,
3 episodes each) found `click_alarmclock` at **33.3%/33.3%** — the first non-zero
RoboTwin result anywhere in this project (exp0001 measured 0.0% everywhere, including
this exact task); `adjust_bottle` remained at 0.0%/0.0%. **Decision: `REJECT`**
(LIBERO floor not met) but this is the strongest multi-embodiment evidence base so
far for designing exp0005 — both because it isolates the disjoint-offset effect
cleanly and because it is the first candidate to show genuine (if partial) RoboTwin
learning. See Section 7 for the full matrix analysis and Section 9 for next-candidate
reasoning.

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
| exp0004 (this candidate) | **frozen** | **disjoint** | 63.33% |

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
- **result: 63.33% (19/30)** — far above exp0002's 16.67%, above exp0003's 50.00%, but still below exp0001's 73.33% and far below the 90% floor.
- raw results path: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_063043/`
- runtime: ~33 minutes
- validity checks: 10/10 task result files present, correct checkpoint path/step, correct per-embodiment stats file.
- per-task breakdown:

  | Task | baseline | exp0001 (train+overlap) | exp0002 (freeze+overlap) | exp0003 (train+disjoint) | exp0004 (freeze+disjoint) |
  |---|---:|---:|---:|---:|---:|
  | task0 "bowl between plate/ramekin" | 100% | 66.7% | 66.7% | 0% | 66.7% |
  | task1 "bowl next to ramekin" | 100% | 100% | 0% | 33.3% | **0%** |
  | task2 "bowl from table center" | 100% | 100% | 33.3% | 100% | 100% |
  | task3 "bowl on cookie box" | 100% | 100% | 33.3% | 100% | 66.7% |
  | task4 "bowl in top drawer" | 66.7% | 0% | 0% | 0% | **0%** |
  | task5 "bowl on ramekin" | 100% | 33.3% | 0% | 0% | 66.7% |
  | task6 "bowl next to cookie box" | 100% | 66.7% | 0% | 100% | 100% |
  | task7 "bowl on stove" | 100% | 66.7% | 0% | 0% | 100% |
  | task8 "bowl next to plate" | 100% | 100% | 33.3% | 100% | 100% |
  | task9 "bowl on wooden cabinet" | 100% | 100% | 0% | 66.7% | 33.3% |
  | **Overall** | **96.7%** | **73.3%** | **16.7%** | **50.0%** | **63.3%** |

**The full 2x2 matrix**:

  |  | overlapping projections (K=14) | disjoint projections (K=21/22) |
  |---|---:|---:|
  | **trainable backbone** | exp0001: 73.33% | exp0003: 50.00% |
  | **frozen backbone** | exp0002: 16.67% | exp0004: **63.33%** |

This is a real, informative, and somewhat counter-intuitive finding. Two clean,
opposite-direction effects appear depending on the projection design:

- **With overlapping projections**, freezing the backbone *hurts* retention (73.33% -> 16.67%): with no backbone plasticity, there is nothing to absorb/compensate for RoboTwin's gradient corrupting LIBERO's shared projection weights.
- **With disjoint projections**, freezing the backbone *helps* retention (50.00% -> 63.33%): once projection corruption is eliminated, the backbone itself becomes the dominant remaining interference channel — RoboTwin's gradient still drifts the fully-shared 30-layer MoT/DiT backbone away from LIBERO-favorable representations when trainable, and freezing it removes that channel too.

Task4 ("bowl in top drawer of wooden cabinet") is 0% in **every** multi-embodiment
candidate so far, including the baseline's own weakest task (66.7%) — a
persistently hard task independent of any of these design choices, not new evidence
about the candidate itself. Task1 also dropped to 0% here despite being 100% for
both the baseline and exp0001, and 33.3% for exp0003 — consistent with continued
non-trivial noise/instability at the individual-task level on this small panel, even
in the best-so-far frozen+disjoint configuration.

decision enabled by this evidence: `DIAGNOSE`, combined with the matrix pattern above
— disjoint offsets are now confirmed as a real, substantial fix (17%->63% in the
frozen-backbone regime, isolating the effect cleanly), but backbone-level drift is a
second, independent interference channel that a full freeze eliminates at the cost of
whatever adaptation benefit backbone plasticity might otherwise provide (unclear
whether that benefit is for LIBERO retention, RoboTwin skill acquisition, or both —
exp0001, the only candidate with a trainable backbone AND competitive retention,
still has not been shown to produce any real RoboTwin capability either). Still no
candidate exceeds exp0001's 73.33%, and none clear the 90% floor.

## 7. Evaluation events

(LIBERO-Spatial candidate_screen recorded in Section 6 above.)

### Event — RoboTwin `progress_check`

- benchmark: `robotwin`
- checkpoint / training step: exp0004, step 1000
- decision this evaluation was meant to inform: whether a frozen backbone (with disjoint-offset projections) can support ANY RoboTwin capability, given exp0004 is the best-retention frozen-backbone candidate so far and no prior candidate has both non-collapsed LIBERO retention AND a RoboTwin measurement
- exact task/difficulty coverage: 2 of 50 canonical tasks (`adjust_bottle`, `click_alarmclock`), `demo_clean` phase (matching exp0001's panel for direct comparison)
- trials/episodes: 3 per task
- exact command (per-task, pinned to separate GPUs):
  ```bash
  CUDA_VISIBLE_DEVICES=<0|1> python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0004_disjoint_offset_frozen_backbone_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0004_disjoint_offset_frozen_backbone_v1/robotwin_dataset_stats.json \
    EVALUATION.task_name=<adjust_bottle|click_alarmclock> EVALUATION.eval_num_episodes=3 \
    MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1
  ```
  (First attempt, without `EVALUATION.dataset_stats_path`, crashed with
  `FileNotFoundError` — a real, previously-masked infra gap, see
  `research/NOTES.md` "RoboTwin eval gotcha — dataset_stats.json auto-discovery".)
- reference: exp0001's same 2-task panel — 0.0% across every measurement
- **result**:

  | Task | Clean | Randomized |
  |---|---:|---:|
  | `adjust_bottle` | 0.0% (0/3) | 0.0% (0/3) |
  | `click_alarmclock` | **33.3% (1/3)** | **33.3% (1/3)** |

  **`click_alarmclock` is the first non-zero RoboTwin result anywhere in this
  project.** exp0001 (trainable backbone, overlapping projections) measured 0.0%
  across every prior RoboTwin measurement, including this exact same task/panel.
  This is genuine evidence that the frozen-backbone + disjoint-offset recipe (which
  also gave the best LIBERO retention among the three imperfect candidates,
  63.33%) can produce *some* real RoboTwin capability — not just avoid catastrophic
  forgetting. `adjust_bottle` still shows zero capability, consistent with
  exp0001's own diagnostic note that ~500 real per-embodiment gradient steps (given
  ~1:1 interleaving over 1000 total steps) is a small fraction of what a dedicated
  RoboTwin specialist would use, and some tasks may need more exposure/harder skills
  than others before any success emerges at all.
- raw results path: `evaluate_results/robotwin/reweighted_multiembodiment_exp0004_disjoint_offset_frozen_backbone_v1/20260818_064304/{adjust_bottle,click_alarmclock}/_result_{clean,random}.txt`
- runtime: ~23 minutes total (both tasks in parallel, clean+random phases sequential per task)
- infra anomaly: first attempt (both tasks) crashed with `FileNotFoundError` on
  `dataset_stats.json` auto-discovery (see command note above) — not a model/data
  problem, fixed by passing `EVALUATION.dataset_stats_path` explicitly; the
  crashed attempt's empty-result `summary.json` (`evaluate_results/.../20260818_064228/`)
  is retained as a record of the failed attempt, not the actual evidence.
- validity checks: correct checkpoint path in every launch command; `unseen`
  instruction type confirmed in all 4 raw result files; both clean and randomized
  phases completed for both tasks (`manager finished successfully` for each).
- decision enabled by this evidence: informs exp0005's design — the frozen-backbone
  disjoint-offset recipe is not just a retention-preserving no-op, it has already
  begun learning at least one real RoboTwin skill within the same 1000-step budget
  that gave exp0001 zero measurable RoboTwin capability at any task. Combined with
  the LIBERO matrix, this makes "give RoboTwin's projection/action-head more
  capacity to keep learning while limiting how much backbone drift LIBERO retention
  has to absorb" (i.e. partial/gradual backbone plasticity, not a full freeze or
  full train) a well-motivated next step rather than a purely theoretical one.
- reason: progress-check-grade evidence (2 of 50 tasks, 3 trials/phase) — sufficient
  to establish "some RoboTwin capability now exists" but not to characterize
  RoboTwin performance broadly; not canonical, not promotion-relevant on its own.

## 8. Decision

- **Decision:** `REJECT`
- **Canonical RoboTwin evidence available:** no (progress-check grade only, 2 of 50 tasks)
- **All five LIBERO >=90% canonical:** no (63.33% Spatial sentinel, below floor)
- **Reason:** LIBERO retention floor not met, though this is the best frozen-backbone retention result (63.33%, vs exp0002's 16.67%) and the first candidate with any measured non-zero RoboTwin capability (`click_alarmclock` 33.3%/33.3%).
- **Checkpoint/branch to preserve:** none; not promoted. Checkpoint removed after evidence capture (reproducible from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` + this task config).
- **Next main-line parent:** unchanged — exp0019.

## 9. What this changes for the next experiment

The completed 2x2 matrix plus this RoboTwin evidence together motivate a specific,
well-targeted next design rather than another single-axis toggle:

|  | overlapping projections | disjoint projections |
|---|---:|---:|
| trainable backbone | exp0001: 73.33% LIBERO, 0% RoboTwin | exp0003: 50.00% LIBERO, not tested |
| frozen backbone | exp0002: 16.67% LIBERO, not tested | exp0004: 63.33% LIBERO, **33.3%/33.3% on 1 of 2 RoboTwin tasks** |

exp0004 shows a full backbone freeze is not a dead end for RoboTwin skill
acquisition — it already learned something in the same 1000-step budget that gave
exp0001 zero capability everywhere — but it likely caps how much RoboTwin can
ultimately learn (a frozen backbone limits how well the shared visual/temporal
representations can adapt to RoboTwin's very different bimanual visual domain,
which the trainable-backbone exp0001, despite its worse retention, presumably
handles better — though this remains unconfirmed since exp0001 was never evaluated
long enough or with a working projection design to show RoboTwin gains).

**Recommended exp0005 direction**: combine disjoint-offset projections with *partial*
backbone plasticity instead of a full freeze or full train — aiming to keep most of
exp0004's retention advantage while giving the backbone enough room to better adapt
to RoboTwin (potentially recovering `adjust_bottle` and improving `click_alarmclock`
further). Concrete options, roughly in order of implementation simplicity:

1. **Much lower backbone LR than the projection-layer LR** (e.g. backbone at
   `3e-6` or `1e-6`, projections at the existing `3e-5`) — simplest to implement
   (two parameter groups in the optimizer instead of a binary freeze), directly
   dials the amount of backbone drift rather than an all-or-nothing choice.
2. **Partial layer freezing** (e.g. freeze the video-expert DiT blocks entirely,
   leave the action-expert DiT blocks and MoT mixture attention trainable) — more
   surgical, targets the hypothesis that video-domain drift specifically (RoboTwin's
   very different visual scenes) is what hurts LIBERO, while action-relevant
   backbone capacity remains free to adapt.
3. **Gradual/staged unfreezing** (start frozen like exp0004, unfreeze progressively
   over the step budget) — most implementation complexity, likely not worth it
   before options 1-2 are tried.

Option 1 (differential LR) is recommended as the next candidate: it is a single,
easily-verified trainer change (two optimizer parameter groups), directly
interpolates between exp0003 (LR ratio 1:1, full plasticity) and exp0004 (LR ratio
0:1, no plasticity), and its LR ratio can be tuned by a future candidate if the
first choice isn't well-calibrated.
