# PROGRESS_0009 — LIBERO-only continued-training control

- **Experiment ID:** 0009
- **Status:** `REJECT`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0008_disjoint_offset_3to1_ratio (rejected; this candidate directly answers the open question from its Section 9)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0008 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** `ff61b1b` (implementation); see Section 7 for follow-on commits

## 1. Result at a glance

Training completed cleanly (1000/1000 steps, no NaN/anomalies, notably low final
loss `0.0878`). LIBERO-Spatial: **23.33% (7/30) — the second-worst result of any
candidate in the project, despite ZERO RoboTwin exposure of any kind.** This is a
decisive finding: it falsifies the "multi-embodiment interference" framing as the
*primary* cause of the retention problem that has driven every candidate since
exp0002 (projection overlap, backbone drift, mixing ratio). Continuing to fine-tune
this expanded/padded K=21/22 checkpoint degrades LIBERO retention substantially
even with no other embodiment present at all. **Decision: `REJECT`** (obviously —
this was never intended as a promotable candidate, only a diagnostic control), but
this redirects the whole research direction. See Section 9 for the new leading
hypothesis (optimizer/LR instability from continued fine-tuning, not embodiment
interference) and exp0010's design.

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

### Training smoke test (pre-launch validation) — PASS

8 steps, sane and notably low, stable losses (`0.1002, 0.1844, 0.1990, 0.1920,
0.3049, 0.1400, 0.1840, 0.1189`) — much lower than any mixed-embodiment candidate's
smoke test, expected since this is pure LIBERO data the model already handles well.
Confirmed via `Setting DiT to train mode...` that the full-trainable-backbone mode
is active. Checkpoint verified: shapes `(1024,21)`/`(21,1024)`/`(4096,22)`, zero
NaN/Inf. Smoke-test run directory deleted after verification; log at
`checkpoints/exp0009_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=libero_only_disjoint_offset_control_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0009_libero_only_control_v1 \
    save_every=200 \
    wandb.name=exp0009_libero_only_control
  ```
- start time: 2026-08-18 ~15:30 UTC (immediately following the smoke test)
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0009_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0009.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log

### Training completion

All 1000 steps completed cleanly, no NaN/Inf, no anomalies. Final: `loss=0.0878
loss_action=0.0251 loss_video=0.0627` — notably much lower than every prior
candidate (even exp0008's LIBERO-heavy 3:1 mix, which still had some RoboTwin batches
pulling the average up), consistent with pure LIBERO fine-tuning on data the model
already handles well. Runtime ~46 minutes (15:31 -> 16:17), actually comparable to
rather than faster than mixed-embodiment runs (LIBERO's per-sample cost isn't
necessarily lower, just its total dataset size). Checkpoint `step_001000.pt`
verified: shapes `(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf, `step: 1000`.

### Evaluation launch

LIBERO-Spatial screen launched on GPUs 0-1, using `dataset_stats.json` (the plain
default filename, since this is a single-embodiment run with no
collision/`stats_filename` override needed).

## 7. Evaluation events

### Evaluation event — LIBERO-Spatial `candidate_screen`

- benchmark: `libero`
- checkpoint / training step: exp0009, step 1000
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0009_libero_only_control_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0009_libero_only_control_v1/dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0019 canonical 97.00% / fresh-machine sentinel 96.67%; 90% floor; every multi-embodiment candidate's 16.67-73.33% range
- **result: 23.33% (7/30) — the second-worst result of any candidate in the project (only exp0002's 16.67% is lower), and this candidate had ZERO RoboTwin exposure of any kind.**
- raw results path: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_161833/`
- runtime: ~36 minutes
- per-task breakdown: 0% on tasks 0, 1, 4, 7, 8, 9 (6 of 10 tasks); 33.3-66.7% on tasks 2, 3, 5, 6. Task4 (the persistently-weakest task across every candidate, including the baseline's own 66.7%) is 0% here too, consistent.

**This is a definitive, decisive finding.** LIBERO-only continued training — with
literally no other embodiment present, no mixing ratio, no projection-overlap
possibility, no shared-backbone cross-embodiment drift — still destroys most of
exp0019's inherited LIBERO-Spatial performance (97.00% -> 23.33%). This directly
falsifies the "multi-embodiment interference" framing as the *primary* cause of the
retention problem that has driven candidate selection since exp0002. Whatever is
happening is a property of continuing to fine-tune this specific expanded/padded
K=21/22 checkpoint — RoboTwin's presence in the mixture (at any ratio tested,
1:1 or 3:1) was never the dominant factor; every multi-embodiment candidate's
retention loss was, at minimum, *substantially* attributable to this same
underlying continued-training instability, not to embodiment interference per se.
- decision enabled by this evidence: `DIAGNOSE` — this redirects the entire research
  direction. See Section 9.

## 8. Decision

- **Decision:** `REJECT` (diagnostic control, never intended for promotion)
- **Canonical RoboTwin evidence available:** n/a (RoboTwin never in training or evaluation for this candidate)
- **All five LIBERO >=90% canonical:** no (23.33% Spatial sentinel, far below floor — second-worst of any candidate)
- **Reason:** confirms the retention problem is not primarily a multi-embodiment interference phenomenon — it persists, severely, with zero RoboTwin exposure. Not a candidate for promotion; its purpose was diagnostic.
- **Checkpoint/branch to preserve:** none. Checkpoint removed after evidence capture.
- **Next main-line parent:** unchanged — exp0019.

## 9. What this changes for the next experiment

This is the most important finding of the research loop so far. Every candidate
since exp0002 has framed the LIBERO retention problem as caused by
*multi-embodiment interference* — shared projection weight-column overlap
(exp0002/0003), backbone drift from RoboTwin's visual domain (exp0004-0006),
backbone-plasticity dosage (exp0005), or mixing ratio (exp0008). exp0009 shows that
framing was, at best, addressing a secondary contributor: **a LIBERO-only continued
training run, with no RoboTwin present at all, degrades LIBERO-Spatial to 23.33% —
worse than 6 of the 8 multi-embodiment candidates.** Whatever is destroying
performance is a property of *continuing to fine-tune this specific
expanded/padded K=21/22 checkpoint*, not of embodiment interference.

### Leading hypothesis: continued-training/optimizer instability, not interference

The most likely concrete mechanism, given the evidence:

1. **Cold-start Adam optimizer state.** Every candidate resumes *weights only*
   (`save_full_state: false`, and more fundamentally the *initial* resume in every
   candidate loads only weights with no prior optimizer state at all — this run
   starts Adam's moment estimates at zero). Adam's bias-correction term
   (`m_hat = m/(1-beta1^t)`, `v_hat = v/(1-beta2^t)`) amplifies effective step size
   for the first ~1/(1-beta) steps after a cold start (here `beta1=0.9`, so
   meaningfully inflated for roughly the first ~10-50 steps, with residual effects
   longer). Applied to an *already-converged* solution (exp0019's LIBERO-only
   optimum) at a non-trivial LR (`3e-5`, matching exp0019's own *continuation*
   fine-tuning LR — but exp0019 presumably had its own warmed/tuned optimizer state
   throughout its training, not a cold restart), this could produce disruptive
   early updates that knock the solution off its optimum before the loss signal has
   a chance to pull it back — especially since the widened action_encoder/head/
   proprio_encoder columns are freshly initialized and initially contribute large,
   noisy gradients that (via any shared computation, e.g. the LayerNorm/residual
   paths inside the DiT blocks) could inject noise into otherwise-converged
   backbone parameters even for LIBERO's own forward pass.
2. **LR itself may simply be too high for continued fine-tuning of an
   already-converged checkpoint**, independent of the cold-start issue — 1000
   steps at `3e-5` with a full trainable backbone is a substantial optimization
   budget relative to how close exp0019 already was to its own local optimum.
3. **The checkpoint expansion/widening itself** (K=7/8 -> K=21/22) may alter loss
   landscape/gradient-norm scale in a way that interacts poorly with fixed
   hyperparameters (LR, warmup fraction, gradient-clip norm) tuned for the
   original K=7/8 architecture, independent of (1)/(2).

None of these are new *architecture* ideas — they are training-recipe/optimization
hypotheses, a different axis entirely from anything tested in exp0001-0008.

### Recommended exp0010

A LIBERO-only run (same setup as exp0009, isolating the true cause from
multi-embodiment confounds) at a **much lower learning rate** (e.g. `3e-6`, 10x
lower, or lower) — directly testing hypothesis (1)/(2) above. This is the simplest,
cheapest, most directly diagnostic next step: if a 10x-lower LR preserves LIBERO
retention close to exp0019's inherited ~96.67-97.00% (or at least clears 90%), that
confirms optimizer/LR instability (not interference) as the real driver, and the
practical fix for the whole project becomes "use a much lower LR for continued
fine-tuning" rather than any of the architecture/mixing interventions tried so far
— which could then be revisited at the corrected LR. If a 10x-lower LR *still*
degrades LIBERO substantially, the cause is something else entirely (e.g. the
checkpoint-expansion process itself, or a more fundamental instability), and a
different diagnostic (e.g. checking gradient norms/per-parameter update magnitudes
directly during a short LIBERO-only run) would be the next step.

## 10. Artifacts

- training log: `checkpoints/exp0009_train.log`
- smoke test log: `checkpoints/exp0009_smoke_train.log`
- LIBERO screen log: `checkpoints/exp0009_libero_screen.log`
- LIBERO screen raw results: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_161833/`
- expanded parent checkpoint used: `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
