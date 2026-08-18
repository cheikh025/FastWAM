# PROGRESS_0008 — disjoint-offset projections, trainable backbone, 3:1 LIBERO:RoboTwin ratio

- **Experiment ID:** 0008
- **Status:** `REJECT`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0007_exp0004_replication (rejected; this candidate acts on its Section 9 recommendation)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0007 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** `081a6c2` (implementation); see Section 7 for follow-on commits

## 1. Result at a glance

Training completed cleanly (1000/1000 steps, no NaN/anomalies). LIBERO-Spatial:
**36.67% (11/30) — worse than exp0003's 50.00%** (identical architecture, 1:1
ratio), the opposite of the hypothesis that more LIBERO rehearsal would protect
retention. RoboTwin progress check: **0.0% on all 4 measurements** (`adjust_bottle`
and `click_alarmclock`, clean and random), confirmed via raw result files — the
first candidate to show zero capability on *both* tested tasks simultaneously.
**Decision: `REJECT`** — this candidate underperforms on both LIBERO and RoboTwin
axes at once. See Section 7 for full results and Section 9 for the investigation
this surprising result triggered (a previously-missing LIBERO-only continued-
training control, proposed as exp0009).

## 2. Research state before experiment

### Full evidence picture across 7 prior candidates

| Candidate | Backbone | Projections | Ratio | Steps | LIBERO-Spatial | RoboTwin (`click_alarmclock`) |
|---|---|---|---|---:|---:|---|
| exp0001 | trainable | overlapping (K=14) | 1:1 | 1000 | **73.33%** | 0.0% (n=3) |
| exp0002 | frozen | overlapping (K=14) | 1:1 | 1000 | 16.67% | not tested |
| exp0003 | trainable | disjoint (K=21/22) | 1:1 | 1000 | 50.00% | not tested |
| exp0004 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 63.33% | 33.3%/33.3% (n=3) |
| exp0005 | partial-plasticity | disjoint (K=21/22) | 1:1 | 1000 | 63.33% | 33.3%/0.0% (n=3) |
| exp0006 | frozen | disjoint (K=21/22) | 1:1 | 4000 | 63.33% | 0.0%/0.0% (n=3) |
| exp0007 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 56.67% | **20.0%/10.0% (n=10, confirmed real)** |

None clear the 90% LIBERO floor. `exp0001` (trainable backbone, *overlapping*
projections — the design later diagnosed as interference-prone) remains the single
best LIBERO retention result of all 7 candidates, ahead of every disjoint-offset
variant despite the disjoint-offset fix being mathematically proven to remove a
real interference mechanism (see `PROGRESS_0003`). All disjoint-offset +
frozen-backbone runs (exp0004/0005/0006/0007) converge on a 56-63% LIBERO band
regardless of training-budget/LR treatment. exp0007 confirmed (at n=10, not just
n=3) that this recipe family's RoboTwin capability, while real, is modest (~10-30%
on one of two tested tasks).

### Why ratio, specifically

Every candidate so far used the same ~1:1 LIBERO:RoboTwin batch-level mixing ratio
(`InterleavedEmbodimentSampler` with `ratio: 1.0`/`1.0`). No candidate has varied
this. exp0007's confirmation that ~500 realized RoboTwin gradient steps (half of
1000 total steps under 1:1) already produces real RoboTwin capability suggests
RoboTwin's own learning need is not acutely step-starved even at a diluted ratio —
opening room to shift more of the total budget toward LIBERO rehearsal without
necessarily giving up all RoboTwin capability.

## 3. Candidate design

### Modifications

1. New data config `configs/data/multiembodiment_libero_robotwin_3to1.yaml`: byte-for-byte copy of `multiembodiment_libero_robotwin.yaml` except LIBERO's `ratio` raised from `1.0` to `3.0` (RoboTwin unchanged at `1.0`) — a 3:1 LIBERO:RoboTwin batch-level mix (verified via `InterleavedEmbodimentSampler`'s generic ratio normalization, `[3.0, 1.0]` -> `[0.75, 0.25]`; no code changes needed, mechanism already unit-tested for non-1:1 ratios).
2. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3to1_3e-5.yaml`: reuses exp0003's K=21/22 disjoint-offset model config and full-trainable-backbone setting (`trainable_modules` unset, defaults to `"dit"` — matches exp0001/exp0003, not exp0002/exp0004-0007's frozen/partial-frozen settings) with the new 3:1 ratio data config.
3. No code changes.

### Why this candidate

Combines the two most-validated positive findings (disjoint-offset projections
proven interference-free; a trainable backbone giving the best retention result so
far) with the one variable never yet tested (mixing ratio), directly targeting the
project's central unresolved problem — LIBERO retention has never approached the
90% floor in any multi-embodiment candidate, while RoboTwin capability, though
modest, is already confirmed achievable with less total gradient exposure than any
candidate has used for RoboTwin so far. This is not a repeat of exp0003 (which used
the same architecture at 1:1 and got 50.00%, worse than exp0001) — the ratio change
is the deliberate, single new variable.

- If LIBERO retention clears meaningfully above exp0003's 50.00% (ideally toward or above exp0001's 73.33%) while RoboTwin capability remains non-zero: the ratio is a real, usable lever, and further increasing it (e.g. 5:1, 8:1) or extending the training budget at this ratio become the next natural steps.
- If LIBERO retention doesn't improve despite 3x less RoboTwin gradient exposure: the interference isn't primarily about *how much* RoboTwin gradient the model sees, but something else about the mixing itself (e.g. the video-denoising loss remaining unmasked across embodiments, a hypothesis raised early in exp0001's investigation and never directly retested since) — motivating a different lever for exp0009.
- If RoboTwin capability drops to a clean 0% at 3:1 (plausible given the lower total RoboTwin step count, ~250 realized steps instead of ~500): informs the ratio/step tradeoff quantitatively for future budget planning.

### Multi-embodiment representation/configuration

Identical to exp0003/0004-0007 (K=21 action / K=22 proprio, LIBERO offset 0,
RoboTwin offset 7/8) — no changes to the representation itself.

### Data and learning strategy

- LIBERO/RoboTwin datasets: unchanged.
- Sampling/mixing ratio: **3:1 LIBERO:RoboTwin** (was 1:1 in every prior candidate) — the sole new variable.
- Trainable/frozen modules: full fine-tune, no freezing (`trainable_modules` default `"dit"`) — matches exp0001/exp0003, not the frozen/partial-frozen family (exp0002, exp0004-0007).
- Optimizer/LR/schedule: unchanged (AdamW, cosine, `3e-5`).

### What to watch

- Primary: LIBERO-Spatial retention vs. exp0001's 73.33% (best-so-far) and exp0003's 50.00% (same architecture, 1:1 ratio) — does the 3:1 ratio move retention toward or past exp0001's level?
- Secondary: RoboTwin `click_alarmclock` capability at the standard 3-episode progress-check panel (not the enlarged n=10 this time, to keep evaluation cost proportionate — a promising result here would justify a later n=10 confirmation, per exp0007's established practice) — does it remain non-zero despite ~3x fewer realized RoboTwin gradient steps?
- Training stability: no new code paths, low risk; standard NaN/OOM checks.

### Initial compute plan

- initial training budget: 1000 steps (matches the majority of prior candidates for comparability)
- checkpoint/save plan: `save_every: 200`, `save_full_state: false`, active `KEEP=1` pruner
- evaluation plan: LIBERO-Spatial candidate_screen (standard 3-trial/10-task) + RoboTwin progress check (standard 3-episode panel, both tasks)
- expected cost: ~40-70min training (trainable backbone is slower than the frozen-backbone family — exp0001/exp0003 took ~50-70min for 1000 steps) + ~35-40min LIBERO screen + ~20-25min RoboTwin check

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + config changes together, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `a8ce8ed` (exp0007 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: `configs/data/multiembodiment_libero_robotwin_3to1.yaml` (new), `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3to1_3e-5.yaml` (new), this report
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_3to1_3e-5.yaml`
- config overrides: `model.redirect_common_files=false`, `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 3.0/1.0 (normalized 0.75/0.25) — the sole new variable this candidate
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

8 steps, sane losses (`0.9352, 0.5128, 0.5894, 1.0796, 1.3958, 0.6361, 1.0164,
0.4416`), no NaN/Inf. Confirmed via `Setting DiT to train mode...` (matches
exp0001/exp0003's full-trainable-backbone log message, not the frozen-backbone
family's `Freezing shared MoT backbone` message) that the correct trainable-module
mode is active. Checkpoint verified: shapes `(1024,21)`/`(21,1024)`/`(4096,22)`,
zero NaN/Inf. Smoke-test run directory deleted after verification; log at
`checkpoints/exp0008_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_3to1_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0008_3to1_ratio_v1 \
    save_every=200 \
    wandb.name=exp0008_disjoint_offset_3to1_ratio
  ```
- start time: 2026-08-18 ~14:02 UTC (immediately following the smoke test)
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0008_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0008.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log

### Training completion

All 1000 steps completed cleanly, no NaN/Inf, no anomalies. Final: `loss=0.1734
loss_action=0.0726 loss_video=0.1008` — notably much lower than exp0001/exp0003's
typical final losses (~0.6-1.5 range), consistent with the 3:1 ratio meaning ~750 of
1000 batches are LIBERO (which the model already handles well from exp0019's
inherited weights) — not directly comparable to prior candidates' loss values given
the different batch-mix composition, but a reasonable, expected pattern (not a sign
of a problem). Checkpoint `step_001000.pt` verified: shapes
`(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf, `step: 1000`. Training log:
`checkpoints/exp0008_train.log`. Runtime ~49 minutes (14:06 -> 14:55), comparable to
exp0001/exp0003's trainable-backbone pace.

### Evaluation launch

LIBERO-Spatial screen launched on GPUs 0-1. RoboTwin progress check will be run
*sequentially* after this finishes (not in parallel via non-zero
`CUDA_VISIBLE_DEVICES`, per the documented gotcha in `research/NOTES.md` that
`run_robotwin_manager.py` ignores that remapping).

## 7. Evaluation events

### Evaluation event — LIBERO-Spatial `candidate_screen`

- benchmark: `libero`
- checkpoint / training step: exp0008, step 1000
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0008_3to1_ratio_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0008_3to1_ratio_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0001 (73.33%, best-so-far); exp0003 (50.00%, same architecture at 1:1 ratio)
- **result: 36.67% (11/30) — WORSE than exp0003's 50.00%, the opposite of the hypothesis.** Shifting the mixing ratio toward more LIBERO rehearsal (3:1 instead of 1:1), holding architecture fixed (trainable backbone, disjoint-offset projections), made LIBERO retention *worse*, not better.
- raw results path: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_145630/`
- runtime: ~37 minutes
- per-task breakdown vs. exp0003 (same architecture, 1:1 ratio):

  | Task | exp0003 (1:1) | exp0008 (3:1) |
  |---|---:|---:|
  | task0 | 0% | 0% |
  | task1 | 33.3% | 0% |
  | task2 | 100% | 66.7% |
  | task3 | 100% | 33.3% |
  | task4 | 0% | 0% |
  | task5 | 0% | 100% |
  | task6 | 100% | 66.7% |
  | task7 | 0% | 66.7% |
  | task8 | 100% | 0% |
  | task9 | 66.7% | 33.3% |
  | **Overall** | **50.0%** | **36.7%** |

  Not a uniform decline — task5/task7 *improved* substantially (0%->100%, 0%->66.7%)
  while task3/task8 dropped sharply (100%->33.3%, 100%->0%). This scattered,
  task-swapping pattern (not a clean monotonic worsening) is consistent with the
  high per-task variance already seen throughout this project's evidence base
  (n=3 trials, quantized outcomes), though the *aggregate* drop (50.0%->36.7%,
  13.3pp) is large enough that it plausibly reflects a genuine effect, not pure
  noise, on top of that variance — n=2 data points for this exact architecture
  (exp0003 at 1:1, exp0008 at 3:1) is not enough to fully separate the two.
- decision enabled by this evidence: this is a genuinely surprising, important
  negative result that refutes the ratio hypothesis as tested and needs
  interpretation (see Section 9) before choosing exp0009 — not a simple
  confirm/deny of "more rehearsal helps."

### Evaluation event — RoboTwin `progress_check`

- benchmark: `robotwin`
- checkpoint / training step: exp0008, step 1000
- exact command (per-task, `CUDA_VISIBLE_DEVICES=0`/`=1`, run after the LIBERO screen finished, not in parallel — per the documented GPU-pinning gotcha in `research/NOTES.md`):
  ```bash
  CUDA_VISIBLE_DEVICES=<0|1> python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0008_3to1_ratio_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0008_3to1_ratio_v1/robotwin_dataset_stats.json \
    EVALUATION.task_name=<adjust_bottle|click_alarmclock> EVALUATION.eval_num_episodes=3 \
    MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1
  ```
- **result (confirmed via raw `_result_*.txt` files): 0.0% on all 4 measurements** (`adjust_bottle` clean/random, `click_alarmclock` clean/random).
- reference: exp0004 (33.3%/33.3%, frozen backbone, 1:1); exp0007 (20.0%/10.0% confirmed at n=10, frozen backbone, 1:1) — both frozen-backbone-family candidates at 1:1 ratio. exp0008 has BOTH a different backbone treatment (trainable, matching exp0001/exp0003) AND a different ratio (3:1) simultaneously, so this result cannot isolate which factor (backbone plasticity vs. ratio vs. ~3x fewer realized RoboTwin steps, ~250 vs ~500) drove the drop to zero — it is consistent with any of them, or their combination.
- runtime: ~35 minutes
- validity checks: correct checkpoint/stats paths; `unseen` instruction type; all phases completed (`manager finished successfully` for both); results verified directly from raw files.
- decision enabled by this evidence: `DIAGNOSE` — combined with the LIBERO regression, this candidate underperforms on both axes simultaneously. See Section 9.

## 8. Decision

- **Decision:** `REJECT`
- **Canonical RoboTwin evidence available:** no
- **All five LIBERO >=90% canonical:** no (36.67% Spatial sentinel, below floor, worse than exp0003's 50.00% at the same architecture)
- **Reason:** LIBERO retention regressed relative to the same architecture at 1:1 ratio, and RoboTwin capability dropped to zero on both tested tasks — this candidate underperforms on both axes simultaneously, the opposite of its intended purpose.
- **Checkpoint/branch to preserve:** none; not promoted. Checkpoint removed after evidence capture.
- **Next main-line parent:** unchanged — exp0019.

## 9. What this changes for the next experiment

The ratio hypothesis, as tested, is refuted: shifting toward more LIBERO rehearsal
(3:1 vs. 1:1) made LIBERO retention *worse*, not better, while also eliminating
RoboTwin capability. This is worth taking seriously rather than dismissing as pure
noise (13.3pp aggregate drop, and a clean 0% across all 4 RoboTwin measurements
where the frozen-backbone family at 1:1 consistently showed some signal) — but the
mechanism is genuinely unclear from the evidence so far.

**A critical gap in the evidence base, only now apparent**: across all 8 candidates
in this project, *every single one* has trained on a LIBERO+RoboTwin mixture at some
ratio (1:1, or now 3:1) — **none has established a LIBERO-only continued-training
control** using the same K=21/22 disjoint-offset architecture. This means the
project has never actually confirmed that the retention problem is a genuine
*multi-embodiment interference* phenomenon at all, as opposed to a more basic fact
about continuing to fine-tune the expanded/padded checkpoint on *any* data,
including LIBERO's own. If simply resuming from `exp0019_expanded_k21_disjoint` and
continuing to train on LIBERO alone (zero RoboTwin exposure) *also* degrades
LIBERO-Spatial retention below exp0019's inherited ~96.67%, that would mean:

- the "interference" framing that has driven candidate selection since exp0002 (projection overlap, backbone drift, mixing ratio) is at best a secondary factor;
- the primary driver is more likely continued-training drift from *any* gradient updates on the K=21/22 architecture (whether from generic optimization dynamics, a subtly different effective LR/schedule than exp0019's own original training used, or something about the expanded/padded interface itself), independent of RoboTwin;
- exp0001's comparatively strong 73.33% might be better explained by *some other* difference in its setup (e.g. K=14 vs K=21/22 widening degree, or simply favorable variance) rather than its overlapping-projection design being uniquely protective.

If, conversely, a LIBERO-only control *does* reproduce something close to exp0019's
original ~96.67% (or at least clears 90%), that cleanly confirms RoboTwin's presence
in the training mixture — via whichever mechanism (ratio, backbone drift, or
something not yet isolated) — is the actual, necessary cause of the retention
problem, and future candidates should continue targeting the mixing/interference
axis rather than suspecting the architecture or continued-training setup itself.

**Recommended exp0009**: a LIBERO-only continued-training control — resume from
`checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (the same K=21/22
disjoint-offset checkpoint every recent candidate has used), train on LIBERO data
alone (no RoboTwin in the mixture at all) for 1000 steps at the same LR/schedule,
then measure LIBERO-Spatial retention. This is also the cheapest candidate possible
in this project so far — no RoboTwin dataset loading, no RoboTwin evaluation needed,
likely faster per-step than any prior candidate (LIBERO's dataset and per-sample
cost are both much smaller than RoboTwin's) — and it directly answers a foundational
question the whole research loop has been assuming rather than verifying.

## 10. Artifacts

- training log: `checkpoints/exp0008_train.log`
- smoke test log: `checkpoints/exp0008_smoke_train.log`
- LIBERO screen log: `checkpoints/exp0008_libero_screen.log`
- LIBERO screen raw results: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_145630/`
- RoboTwin progress-check logs: `checkpoints/exp0008_robotwin_adjust_bottle.log`, `checkpoints/exp0008_robotwin_click_alarmclock.log`
- RoboTwin progress-check raw results: `evaluate_results/robotwin/reweighted_multiembodiment_exp0008_3to1_ratio_v1/20260818_150838/`
- expanded parent checkpoint used: `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
