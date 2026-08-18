# PROGRESS_0009 — LIBERO-only continued-training control

- **Experiment ID:** 0009
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0008_disjoint_offset_3to1_ratio (rejected; this candidate directly answers the open question from its Section 9)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0008 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### The gap this candidate fills

Across all 8 prior candidates, every single one trained on a LIBERO+RoboTwin
mixture at some ratio (1:1 in exp0001-0007, 3:1 in exp0008). **None has established
a LIBERO-only continued-training control** — i.e., resuming from the K=21/22
disjoint-offset expanded checkpoint and continuing training on LIBERO data alone,
with RoboTwin entirely absent from the mixture. This means the project has never
actually confirmed that the LIBERO retention problem (no candidate has approached
the 90% floor; best result is exp0001's 73.33%, most disjoint-offset variants land
in a 36-63% band) is a genuine multi-embodiment interference phenomenon, as opposed
to something more basic about continuing to fine-tune this specific expanded/padded
architecture — independent of RoboTwin ever being present in training.

This gap became apparent after exp0008's surprising result: shifting the mixing
ratio toward *more* LIBERO rehearsal (3:1 instead of 1:1) made LIBERO retention
*worse* (36.67% vs. exp0003's 50.00% at the identical architecture, 1:1 ratio),
the opposite of the naive rehearsal-helps-retention hypothesis. That result is hard
to explain if RoboTwin exposure is the primary driver of forgetting — it becomes
easier to explain if simply continuing to train the K=21/22 architecture (on
*whichever* data) drifts away from exp0019's original solution to some degree, and
RoboTwin's presence is a secondary/contributing factor rather than sufficient
explanation on its own.

### Full evidence picture across 8 prior candidates

| Candidate | Backbone | Projections | Ratio | Steps | LIBERO-Spatial | RoboTwin |
|---|---|---|---|---:|---:|---|
| exp0001 | trainable | overlapping (K=14) | 1:1 | 1000 | **73.33%** | 0.0% |
| exp0002 | frozen | overlapping (K=14) | 1:1 | 1000 | 16.67% | not tested |
| exp0003 | trainable | disjoint (K=21/22) | 1:1 | 1000 | 50.00% | not tested |
| exp0004 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 63.33% | 33.3%/33.3% (n=3) |
| exp0005 | partial-plasticity | disjoint (K=21/22) | 1:1 | 1000 | 63.33% | 33.3%/0.0% |
| exp0006 | frozen | disjoint (K=21/22) | 1:1 | 4000 | 63.33% | 0.0%/0.0% |
| exp0007 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 56.67% | 20.0%/10.0% (n=10, confirmed real) |
| exp0008 | trainable | disjoint (K=21/22) | 3:1 | 1000 | 36.67% | 0.0% (all 4 measurements) |

Inherited canonical exp0019 LIBERO-Spatial: 97.00% (fresh-machine sentinel: 96.67%).

## 3. Candidate design

### Modifications

1. New task config `configs/task/libero_only_disjoint_offset_control_3e-5.yaml`: uses `configs/data/libero_2cam_multiembodiment.yaml` directly (a plain single-embodiment `RobotVideoDataset`, not `build_multi_embodiment_dataset` — no `InterleavedEmbodimentSampler` involved, the simplest possible setup) with `configs/model/fastwam_multiembodiment.yaml` (K=21/22, unchanged). Same LR (`3e-5`), schedule (`cosine`), batch/accumulation (`1`/`4`), and step budget (`1000`) as every recent candidate — matches exp0001/exp0003/exp0008's full-trainable-backbone setting (`trainable_modules` unset, defaults to `"dit"`), so any retention difference is attributable only to RoboTwin's absence, not a confounded hyperparameter change.
2. No code changes.

### Why this candidate

Directly tests the foundational assumption behind every candidate since exp0002:
that the retention problem is caused by *multi-embodiment interference* specifically
(projection overlap, backbone drift, or mixing ratio). This is also the cheapest
candidate in the project — no RoboTwin dataset loading (skips the ~900GB
text-embedding cache and the much larger 6M-sample RoboTwin dataset), no RoboTwin
evaluation needed.

- If LIBERO-Spatial retention holds up (clears or approaches exp0019's inherited ~96.67%, or at minimum clears the 90% floor) under LIBERO-only continued training: this confirms RoboTwin's presence in the mixture is the necessary cause of every prior candidate's degradation — future candidates should continue targeting the mixing/interference axis (e.g. exp0006 Section 9's option 2: an explicit retention/distillation mechanism; or a much smaller RoboTwin ratio than even 1:1).
- If LIBERO-Spatial retention degrades even with zero RoboTwin exposure (lands in a similar 36-73% range as the multi-embodiment candidates): the "multi-embodiment interference" framing needs revisiting. The real driver would be something more basic — continued-training drift from the expanded/padded K=21/22 checkpoint itself, a schedule/LR mismatch relative to exp0019's own original training regime, or general optimization instability from continuing to fine-tune an already-converged solution at this LR — independent of RoboTwin. This would redirect the whole research direction toward stabilizing continued training itself (e.g. a much lower LR, a shorter budget, or a regularization term anchoring to the original exp0019 weights) rather than further architecture/mixing tweaks.

### Multi-embodiment representation/configuration

K=21 action / K=22 proprio (unchanged from exp0003-0008), LIBERO at offset 0
(unchanged position). RoboTwin's disjoint offset range ([7:21]/[8:22]) exists in
the architecture but receives zero gradient this run (never sampled).

### Data and learning strategy

- Dataset: LIBERO only (Spatial/Object/Goal/Long-10, same 4 suites as every prior candidate's LIBERO portion) — no RoboTwin.
- Sampling: plain shuffled DataLoader (no interleaving needed for a single dataset).
- Trainable/frozen modules: full fine-tune, no freezing (matches exp0001/exp0003/exp0008).
- Optimizer/LR/schedule: unchanged (AdamW, cosine, `3e-5`, 1000 steps).

### What to watch

- Primary: LIBERO-Spatial candidate_screen result — the entire point of this candidate. Compare against exp0019's inherited ~96.67-97.00%, the 90% floor, and the 36-73% range every multi-embodiment candidate has landed in.
- No RoboTwin evaluation planned (RoboTwin was never in training; a RoboTwin measurement would trivially be ~0% and add no information).

### Initial compute plan

- initial training budget: 1000 steps (matches all recent candidates for comparability)
- checkpoint/save plan: `save_every: 200`, `save_full_state: false`, active `KEEP=1` pruner
- evaluation plan: LIBERO-Spatial candidate_screen only (standard 3-trial/10-task) — no RoboTwin check needed
- expected cost: likely faster than any prior candidate per-step (LIBERO's smaller dataset/samples vs. RoboTwin's larger ones) — training + one LIBERO screen (~35-40min), no RoboTwin eval time at all

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + config, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `caeecc4` (exp0008 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: `configs/task/libero_only_disjoint_offset_control_3e-5.yaml` (new), this report
- training config: `configs/task/libero_only_disjoint_offset_control_3e-5.yaml`
- config overrides: `model.redirect_common_files=false`, `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- data config: `configs/data/libero_2cam_multiembodiment.yaml` (unchanged, already K=21/22 disjoint-offset from exp0003's setup)
- sampler: none (plain shuffled DataLoader, single dataset)
- model/trainable-module configuration: full fine-tune (`trainable_modules` default `"dit"`)
- optimizer / LR / scheduler: AdamW, cosine, `learning_rate: 3e-5` (unchanged)
- batch size / gradient accumulation: `batch_size: 1`, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps: `max_steps: 1000`
- random seed(s): not explicitly controlled (matches all prior candidates)
- resume source: weights-only resume from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate.

## 6. Training execution and control timeline

Not yet launched.

## 7. Evaluation events

None yet.
