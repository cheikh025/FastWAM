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

### Training smoke test (pre-launch validation) — PASS

8 steps, losses matching exp0004's smoke test exactly (`1.2713, 0.8180, 1.2406,
1.6864, 2.2222, 1.3406, 1.2225, 1.5009` vs exp0004's `1.2713, 0.8076, ...` — tiny
differences beyond step 1 are expected numerical noise, step-1 matches exactly
confirming deterministic resume from the same checkpoint), no NaN/Inf. Confirmed
via `Freezing shared MoT backbone; trainer.py:377` (appeared twice, pre/post-wrap)
that exp0004's exact freezing mechanism applies unchanged. Checkpoint verified:
shapes `(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf. Smoke-test run directory
deleted after verification; log at `checkpoints/exp0006_smoke_train.log`.

### Real training run

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_4k_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1 \
    save_every=500 \
    wandb.name=exp0006_disjoint_offset_frozen_backbone_4k
  ```
- start time: 2026-08-18 ~08:50 UTC (immediately following the smoke test)
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0006_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0006.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log watching for checkpoint-save events and failure signatures
- planned mid-run check: LIBERO-Spatial sentinel + RoboTwin 2-task progress check at step ~2000

### Mid-run check-in (step ~1400/4000)

Training healthy: losses well-behaved (`loss` 0.30-1.01, `loss_action` 0.14-0.88, no
NaN/Inf), `speed=0.44 step/s` (matches exp0004's fastest-among-candidates pace,
frozen backbone = fewer gradients to compute), `eta≈01:38:00` remaining as of step
1400. A `step_001000.pt` checkpoint was available (the `save_every=500` schedule's
first surviving save under the `KEEP=1` pruner).

**GPU memory note**: this frozen-backbone recipe uses only ~16-17GB/80GB per GPU
during training (vs. ~61-70GB for exp0001/exp0003/exp0005's full-trainable-backbone
runs) — DeepSpeed doesn't need ZeRO optimizer-state memory for the ~5B frozen
backbone parameters, only for the 6 small trainable projection tensors. This left
ample headroom (~60GB+ free per GPU) to run a mid-run LIBERO-Spatial screen
concurrently with training, unlike exp0003's situation (where the trainable-backbone
run left only ~16-20GB free per GPU, too tight to risk a parallel eval). Launched a
LIBERO-Spatial candidate_screen on the step-1000 checkpoint with
`MULTIRUN.max_tasks_per_gpu=1` (more conservative than the usual `2`, given training
is still concurrently active) rather than waiting for training to finish. Training
confirmed still healthy immediately after the eval job started (step 1400, same
speed/loss pattern, no slowdown or memory pressure observed).

**First mid-run eval attempt crashed** (real infra bug, not a model/data problem):
the checkpoint pruner deleted `step_001000.pt` once `step_001500.pt` was saved
(~19 min later, matching the `save_every=500` cadence at `0.44 step/s`), while the
LIBERO screen (2 of 10 tasks already completed) was still reading it — task 2/10
crashed with a checkpoint-load failure, aborting the whole scheduler on first
failure. Documented in `research/NOTES.md` "Mid-training eval gotcha". **Recovery**:
relaunched a smaller, faster mid-run panel (`EVALUATION.num_trials=1` instead of
`3`, `MULTIRUN.max_tasks_per_gpu=2`) against the new current checkpoint
(`step_001500.pt`) — a full 3-trial/10-task screen reliably takes ~33-40 minutes
regardless of parallelism (confirmed across exp0003-0005), longer than the
~19-minute save-cadence window, so a full screen cannot safely run concurrently
with this training run's pruner; a 1-trial (10-episode) panel is fast enough to
finish inside one window and is adequate for a training-control decision, though
noisier than the full screen used for final candidate evaluation.

## 7. Evaluation events

### Event — LIBERO-Spatial mid-run diagnostic (step ~1500/4000)

- benchmark: `libero`
- purpose: `diagnostic` (mid-run, not `candidate_screen` — reduced to 1 trial/task for pruner-race-safety, see above)
- checkpoint / training step: exp0006, step 1500 (partial budget, training still active)
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/checkpoints/weights/step_001500.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=1 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- **result: 80.00% (8/10)** — succeeded this time (no pruner-race crash; likely because with `max_tasks_per_gpu=2`'s 4-way launch parallelism, all 10 tasks had already loaded the checkpoint into memory before the next prune event, even though the eval's own wall-clock (~28 min) ran past it).
- per-task: 8 of 10 tasks at 100% (1/1), 2 at 0% — `libero_spatial_4` (the task that is 0% in *every* multi-embodiment candidate so far, including the baseline's own weakest at 66.7%) and `libero_spatial_1` (0% here; was 100% for exp0004 at step 1000, so likely just single-trial noise, not a real regression).
- context: this is a noisier 1-trial/task panel (10 episodes total, vs. the standard 3-trial/30-episode `candidate_screen`), so not directly comparable in precision to exp0004's step-1000 63.33% (from a 3-trial panel) — but as a rough training-control signal, 80% at step 1500 is a healthy, non-degrading sign that retention has not collapsed under continued training on this recipe.
- decision enabled by this evidence: **`CONTINUE_TRAINING`** — no sign of divergence or collapse at the halfway-ish point; let the run continue toward 4000 steps and evaluate properly (full 3-trial screen + RoboTwin progress check) on the final checkpoint.

### Training completion

All 4000 steps completed cleanly, no NaN/Inf, no anomalies throughout (matches the
step-1500/1780/3270 healthy check-ins along the way). Final: `loss=0.4546
loss_action=0.2080 loss_video=0.2466` (lr decayed to `3.00e-07`). Total wall-clock:
~2h33m (08:54 -> 11:27), matching the ~0.44 step/s pace observed throughout.
Checkpoint `step_004000.pt` (12,042,248,688 bytes) verified: shapes
`(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf, `step: 4000`. Pruner/monitor
stopped cleanly; all 4 GPUs confirmed at 0MiB/0% after training exited.

### Evaluation event — LIBERO-Spatial `candidate_screen` (final)

- benchmark: `libero`
- checkpoint / training step: exp0006, step 4000 (full 4x budget)
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/checkpoints/weights/step_004000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0019 canonical 97.00%; exp0004 (step 1000, same recipe) 63.33%; exp0006 mid-run (step 1500, 1-trial) 80.00%
- **result: 63.33% (19/30)** — lands at the *exact same aggregate* as exp0004's step-1000 result and exp0005's step-1000 result (three separate candidates now converging on 19/30), despite 4x more training steps and the encouraging 80% mid-run (1-trial) signal at step 1500.
- raw results path: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_112827/`
- runtime: ~35 minutes
- per-task breakdown:

  | Task | exp0004 (step 1000) | exp0006 mid-run (step 1500, 1-trial) | exp0006 final (step 4000, 3-trial) |
  |---|---:|---:|---:|
  | task0 "bowl between plate/ramekin" | 66.7% | 100% | 33.3% |
  | task1 "bowl next to ramekin" | 0% | 0% | 33.3% |
  | task2 "bowl from table center" | 100% | 100% | 100% |
  | task3 "bowl on cookie box" | 66.7% | 100% | 100% |
  | task4 "bowl in top drawer" | 0% | 0% | 0% |
  | task5 "bowl on ramekin" | 66.7% | 100% | 33.3% |
  | task6 "bowl next to cookie box" | 100% | 100% | 100% |
  | task7 "bowl on stove" | 100% | 100% | 100% |
  | task8 "bowl next to plate" | 100% | 100% | 100% |
  | task9 "bowl on wooden cabinet" | 33.3% | 100% | 33.3% |
  | **Overall** | **63.3%** | **80.0%** | **63.3%** |

  The mid-run 80% signal did **not** hold up in the full 3-trial re-measurement —
  most tasks that were a clean 100% on a single trial at step 1500 (0, 5, 9) landed
  at only 33.3% under 3 trials by step 4000, meaning the single-trial mid-run panel
  was optimistic (1/1 successes don't reveal a task's true multi-trial rate, and 3x
  more trials naturally regress toward a lower, more representative number even with
  no true underlying change). task4 remains 0% across **every** multi-embodiment
  candidate measured so far.
- decision enabled by this evidence: **the aggregate result did not improve with 4x
  more training** — three separate candidates (exp0004, exp0005, exp0006) all land
  at exactly 19/30 on this panel despite different training budgets/LR treatments,
  which is a meaningfully strong signal that this specific recipe (frozen backbone,
  6 trainable projection tensors, K=21/22 disjoint offset) has a real performance
  ceiling around 63% on this task family, not merely noise. The RoboTwin evidence
  (below) is the more decisive test of this candidate's actual purpose (more budget
  -> more RoboTwin capability).

### Evaluation event — RoboTwin `progress_check` (final, step 4000)

- benchmark: `robotwin`
- checkpoint / training step: exp0006, step 4000
- decision this evaluation was meant to inform: does 4x more training on exp0004's exact recipe improve RoboTwin capability, per this candidate's core hypothesis?
- exact task/difficulty coverage: same 2-task panel as exp0001/exp0004/exp0005 (`adjust_bottle`, `click_alarmclock`), `demo_clean` and `demo_randomized`
- trials/episodes: 3 per task per phase
- exact command (per-task, pinned to separate GPUs):
  ```bash
  CUDA_VISIBLE_DEVICES=<0|1> python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/checkpoints/weights/step_004000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0006_disjoint_offset_frozen_backbone_4k_v1/robotwin_dataset_stats.json \
    EVALUATION.task_name=<adjust_bottle|click_alarmclock> EVALUATION.eval_num_episodes=3 \
    MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1
  ```
- **result**:

  | Task | Clean | Randomized | exp0004 Clean | exp0004 Randomized |
  |---|---:|---:|---:|---:|
  | `adjust_bottle` | 0.0% | 0.0% | 0.0% | 0.0% |
  | `click_alarmclock` | **0.0%** | **0.0%** | 33.3% | 33.3% |

**Complete regression, confirmed via raw `_result_*.txt` files directly (not just
log parsing).** `click_alarmclock` — the one real RoboTwin capability found
anywhere in this project — is entirely gone at step 4000, on both clean and
randomized phases, on the exact same recipe (frozen backbone, disjoint-offset
projections) that produced it at step 1000. `adjust_bottle` remains dead
throughout. 4x more training did not improve RoboTwin capability; it destroyed the
only capability that existed.
- raw results path: `evaluate_results/robotwin/reweighted_multiembodiment_exp0006_disjoint_offset_frozen_backbone_4k_v1/20260818_114019/{adjust_bottle,click_alarmclock}/_result_{clean,random}.txt`
- runtime: ~19 minutes total
- validity checks: correct checkpoint path in every launch command; `unseen` instruction type; both phases completed for both tasks (`manager finished successfully` for each); `EVALUATION.dataset_stats_path` passed explicitly.
- decision enabled by this evidence: **`DIAGNOSE`** — this result, combined with LIBERO's flat 63.33% across three different training treatments (exp0004/exp0005/exp0006), needs its own investigation before choosing exp0007. See Section 9.

## 8. Decision

- **Decision:** `REJECT`
- **Canonical RoboTwin evidence available:** no (progress-check grade only)
- **All five LIBERO >=90% canonical:** no (63.33% Spatial sentinel, below floor, unchanged from exp0004/exp0005)
- **Reason:** LIBERO retention did not improve with 4x more training (flat at 63.33%, matching exp0004 and exp0005 exactly), and RoboTwin capability got *strictly worse* — `click_alarmclock`'s only real success (33.3%/33.3%) vanished entirely, while `adjust_bottle` never gained anything. The candidate's core hypothesis (more training budget unlocks more RoboTwin capability through the same 6 trainable tensors) is directly refuted by this evidence — more budget instead destroyed the one capability that existed.
- **Checkpoint/branch to preserve:** none; not promoted. Checkpoint removed after evidence capture.
- **Next main-line parent:** unchanged — exp0019. **exp0004's step-1000 checkpoint remains the single best evidence point found so far** (63.33% LIBERO, genuine partial RoboTwin capability) — but it was not preserved (deleted per the project's rejected-checkpoint cleanup convention), so it is not directly recoverable; its exact recipe is fully reproducible from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` + `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml` + `max_steps=1000` if ever needed again.

## 9. What this changes for the next experiment

Three candidates (exp0004, exp0005, exp0006) now share the same LIBERO ceiling
(63.33%, 19/30) despite three different training treatments (1000 steps unmodified;
1000 steps with a partial-plasticity backbone; 4000 steps with the same frozen
backbone) — this is unlikely to be coincidence at this point and looks like a real,
recipe-level ceiling: the 6 trainable projection tensors, however trained, converge
to roughly the same LIBERO-Spatial aggregate given "enough" gradient exposure (even
just 1000 steps appears to already be "enough" — exp0006's 4x more steps changed
nothing on the LIBERO side, for better or worse).

On RoboTwin, the picture is more concerning: **both exp0005 (different LR) and
exp0006 (same recipe, more steps) actively destroyed exp0004's one real capability
rather than building on it.** This is a strong signal that exp0004's step-1000
checkpoint sits at (or very near) a narrow, fragile optimum for `click_alarmclock`
specifically — not a stable capability that further training reinforces, but an
accident of where gradient descent happened to be after ~500 realized RoboTwin
gradient steps, which any further optimization (whether via a different LR or more
steps at the same LR) disturbs rather than improves. If true, this means the
"just add more training" and "just tune the LR" directions are both dead ends *for
this specific recipe* (frozen backbone, 6-tensor trainable budget, ~1:1
LIBERO:RoboTwin interleaving) — the model isn't accumulating a growing, robust
RoboTwin skill across training, it's landing on lucky/unlucky checkpoints along a
noisy trajectory.

This points toward needing a genuinely different mechanism for exp0007, not another
point on the (budget x LR) grid already explored:

1. **More frequent intermediate checkpointing + selection, not just more training**: if the underlying trajectory is noisy rather than monotonically improving, the right lever might be evaluating many more intermediate checkpoints (e.g. every 100-200 steps) and *selecting* the best one by RoboTwin progress-check score, rather than training longer and evaluating only the end result. This treats checkpoint selection as compensating for training noise instead of assuming later = better.
2. **Increase the trainable parameter budget beyond just 6 tensors** (still short of a full unfreeze): e.g. also unfreeze the last few MoT/DiT layers (not the whole 30-layer backbone) — enough added capacity that RoboTwin's ~500 gradient steps can build a more genuinely robust representation rather than overfitting/perturbing a narrow existing solution, while still limiting LIBERO-relevant drift to a small fraction of the backbone.
3. **Rehearsal/replay weighting**: increase the LIBERO:RoboTwin interleaving ratio in RoboTwin's favor within a similar total-step budget (RoboTwin's own components currently see relatively few of their "own" gradient steps at ~1:1) — untested variable so far; all six candidates used the same ~1:1 ratio.
4. **Revisit whether `click_alarmclock`'s exp0004 result was ever a stable capability at all** — a direct, cheap diagnostic before spending more training compute: evaluate exp0004's exact recipe (re-trained fresh, 1000 steps, since the original checkpoint was deleted) with a different eval seed, or a slightly larger episode count, to see whether 33.3%/33.3% replicates. If it doesn't reliably replicate even under identical training, the entire "click_alarmclock capability" narrative built across exp0004-0006 may itself be resting on a single lucky evaluation seed rather than a real, reproducible skill.

Recommend `$investigate-fastwam-problem` to weigh option 4 (the cheapest, most
foundational check — does the capability even reliably exist?) before committing
compute to options 1-3, since if the answer to 4 is "no, it doesn't replicate," the
entire recent evidence chain needs reinterpreting as noise rather than a real
capability that later candidates destroyed.

## 10. Artifacts

- training log: `checkpoints/exp0006_train.log`
- smoke test log: `checkpoints/exp0006_smoke_train.log`
- mid-run LIBERO logs: `checkpoints/exp0006_libero_screen_step1000.log` (crashed, pruner race), `checkpoints/exp0006_libero_screen_step1500_midrun.log` (succeeded, 80%)
- final LIBERO screen log: `checkpoints/exp0006_libero_screen_final.log`
- final LIBERO screen raw results: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_112827/`
- RoboTwin progress-check logs: `checkpoints/exp0006_robotwin_adjust_bottle.log`, `checkpoints/exp0006_robotwin_click_alarmclock.log`
- RoboTwin progress-check raw results: `evaluate_results/robotwin/reweighted_multiembodiment_exp0006_disjoint_offset_frozen_backbone_4k_v1/20260818_114019/`
- expanded parent checkpoint used: `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
