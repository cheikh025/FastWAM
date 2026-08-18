# PROGRESS_0002 — frozen_backbone_warmup

- **Experiment ID:** 0002_frozen_backbone_warmup
- **Status:** `REJECT`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0001_padded_multiembodiment_baseline
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k14/step_005000.pt` (same expanded-exp0019 base as exp0001 — this candidate resumes from the same starting point, not from exp0001's step_001000, to give a clean like-for-like comparison)
- **Selected candidate checkpoint:** TBD

## 1. Result at a glance

TBD.

## 2. Research state before experiment

### Accepted RoboTwin state

exp0001 (parent): 0.0% across every completed measurement (2 of 50 tasks, clean+random, 3 trials each) — see `PROGRESS_0001` Event 2. Not surprising after only 1000 steps of a full-embodiment interleave; the open question is whether the training recipe itself is the blocker.

### Accepted LIBERO retention state

| Suite | exp0019 canonical | setup sentinel (3-trial) | exp0001 (3-trial, same panel) |
|---|---:|---:|---:|
| LIBERO-Spatial | 97.00% | 96.67% (29/30) | **73.33% (22/30)** |

exp0001's regression is concentrated on exp0019's already-weakest tasks (task4 "bowl in drawer" 66.7%->0%, task5 "bowl on ramekin" 100%->33.3%) — see `PROGRESS_0001` Event 1 for full per-task breakdown.

## 3. Candidate design

### Modifications

1. Add a new `Trainer.trainable_modules` config option (`"dit"` [existing default, unchanged] | `"expanded_projections_only"` [new]) that freezes the entire shared MoT backbone (both video and action DiT blocks) and trains only the newly-expanded `action_encoder`/`head` (action expert) and `proprio_encoder` — the exact parameters that needed to grow from exp0019's 7/8-dim shapes to the shared K=14. Verified directly on the real model: exactly 6 trainable parameter tensors (`action_encoder.{weight,bias}`, `head.{weight,bias}`, `proprio_encoder.{weight,bias}`), 1841 frozen.
2. New task config `configs/task/multiembodiment_libero_robotwin_frozen_backbone_3e-5.yaml`, identical to exp0001's task config in every other respect (same data mixing, same batch/LR/schedule, same `save_full_state=false`/`eval_every` disabled) — isolating `trainable_modules` as the single changed variable for a clean comparison.

No other changes — this candidate deliberately reuses everything else from exp0001 (K=14 padding, two-level masked loss, embodiment-conditioned prompts, ~1:1 batch interleaving) since those mechanisms are independently unit-tested and verified correct; the investigation (`PROGRESS_0001`, Section 10 equivalent below) concluded the likely cause of exp0001's regression is the *unmasked, always-shared* video-denoising loss training the backbone on RoboTwin's very different visual domain while fully fine-tuning with no freezing — not a bug in the padding/masking/expansion mechanisms themselves.

### Why this candidate

Directly evidence-driven from `$investigate-fastwam-problem`'s diagnosis of exp0001: the video loss is not per-embodiment-masked (unlike the properly-masked action loss), so RoboTwin's visual domain trains the same shared backbone LIBERO depends on; exp0001 additionally chose full fine-tuning with no freezing (a deliberate baseline-isolation choice, now shown to have a real retention cost). This matches the literature review's own findings from setup (Qwen-VLA's staged/progressive training; the forgetting-resistance paper's finding that backbone updates, not action-head changes, drive most forgetting in pretrained VLA fine-tuning). Freezing the backbone during an initial warm-up is the most direct, literature-grounded test of this hypothesis and simultaneously gives the newly-added action/proprio projections a chance to learn to read/write the new channels without corrupting anything LIBERO depends on.

### Multi-embodiment representation/configuration

Unchanged from exp0001 (K=14, same masks/normalization/conditioning) — see `PROGRESS_0001` Section 3.

### Data and learning strategy

Unchanged from exp0001 (same ~1:1 LIBERO:RoboTwin batch interleaving, same 4 LIBERO suites, no LIBERO-90 replay) except:

- trainable/frozen modules: **only** `action_encoder`/`head`/`proprio_encoder` trainable; the entire shared MoT backbone (video + action DiT blocks) frozen.

### What to watch

- Whether LIBERO retention holds substantially better than exp0001's 73.33% (the primary question this candidate answers).
- Whether RoboTwin shows any learning signal at all with only the projection layers trainable — plausible this is *harder* to learn RoboTwin capability from than exp0001's full fine-tune (a real tradeoff to watch, not assumed away), since the frozen backbone's action-token representations were never trained on RoboTwin-scale bimanual actions.
- If LIBERO retention holds AND RoboTwin still shows literally nothing, that's evidence for a longer warm-up or unfreezing schedule (progressive, not evidence against the frozen-warm-up idea itself).

### Initial compute plan

Same budget as exp0001 (1000 steps) for a direct, matched comparison — not a promise to extend automatically. A cheap LIBERO-Spatial sentinel (same 10-task/3-trial panel) immediately after training, before any RoboTwin evaluation, since LIBERO retention is the primary question this candidate is testing.

## 4. Exact code and configuration state

- Git branch: `autoresearch/robotwin-multiembodiment-v1`.
- Git commit: recorded once implementation is committed (this report created alongside the `Trainer.trainable_modules` implementation, in the same commit).
- files changed: `src/fastwam/trainer.py` (`trainable_modules` config field, `_apply_expanded_projections_only_train_mode`, `_set_dit_only_train_mode` now dispatches on the configured mode so it survives re-application at the start of `train()`, optimizer's `trainable_params` list now matches the configured mode exactly rather than hardcoding `model.dit.parameters()`).
- training config: `task=multiembodiment_libero_robotwin_frozen_backbone_3e-5`.
- resume source: weights-only file, `checkpoints/exp0019_expanded_k14/step_005000.pt` (same as exp0001).

## 5. Hardware and software environment

Identical to `PROGRESS_0001` Section 5.

## 6. Training execution and control timeline

- exact launch command (final, after a disk-full crash mid-run — see "Training anomalies"):
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_frozen_backbone_3e-5 \
    resume=runs/reweighted_multiembodiment/exp0002_frozen_backbone_warmup/checkpoints/weights/step_000200.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0002_frozen_backbone_warmup_v2 \
    max_steps=800 save_every=100
  ```
  (first attempt: `resume=checkpoints/exp0019_expanded_k14/step_005000.pt`, `max_steps=1000`, crashed disk-full at step ~300; resumed from the valid step_200 weights with `max_steps=800` to preserve the total 1000-step budget)
- start time: 2026-08-18 02:33:04 UTC (first attempt); 02:49:57 UTC (resumed attempt)
- end time: 2026-08-18 03:30:26 UTC
- wall-clock runtime: ~26 min (first attempt, to step 200) + ~41 min (resumed, 200->800) = ~67 min total for 1000 effective steps
- exit code/status: clean (`max_steps reached step=800`, no errors)
- steps completed: 1000/1000 effective (200 + 800 across two launches)
- throughput: ~0.36-0.42 step/s (similar to exp0001, as expected — see Section 8)
- important losses/diagnostics: loss `0.6486` at step 800 (fluctuating 0.4-2.2 over the run, generally declining — action_encoder/head/proprio_encoder converging from a partially-adapted state), no NaN/Inf in the final checkpoint (1649/1649 mot tensors clean).
- training log: `checkpoints/exp0002_train_v2.log`

### Training anomalies

Same disk-full checkpoint-save crash pattern as exp0001, recurring for a specific, identified reason: the reused pruner script's `KEEP=2` was too generous for the ~34GB headroom actually available (2 kept x 12GB + 1 being written x 12GB = 36GB > 34GB). Fixed by using `KEEP=1` and a faster 15s poll interval (vs. 60s) for the resumed run; recovered by resuming from the last valid checkpoint (step_200, verified loadable first) rather than restarting from exp0019. Full detail and the general `(KEEP+1) x checkpoint_size` sizing rule now in `research/NOTES.md` "Disk crisis".

## 7. Evaluation events

### Event 1 — `candidate_screen` (LIBERO-Spatial retention sentinel — same panel as exp0001)

- benchmark: `libero`
- checkpoint / training step: exp0002, step 800 (1000 effective steps)
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0002_frozen_backbone_warmup_v2/checkpoints/weights/step_000800.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0002_frozen_backbone_warmup_v2/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2
  ```
- reference: exp0019 canonical 97.00%; setup sentinel 96.67% (29/30); **exp0001 (same panel) 73.33% (22/30)**
- **candidate result: 16.67% (5/30) — WORSE than exp0001, not better.** This is the opposite of the leading hypothesis's prediction.
- per-task breakdown:

  | Task | exp0019 setup sentinel | exp0001 | exp0002 (frozen backbone) |
  |---|---:|---:|---:|
  | task0 "bowl between plate/ramekin" | 100% | 66.7% | 66.7% |
  | task1 "bowl next to ramekin" | 100% | 100% | **0%** |
  | task2 "bowl from table center" | 100% | 100% | 33.3% |
  | task3 "bowl on cookie box" | 100% | 100% | 33.3% |
  | task4 "bowl in top drawer" | 66.7% | 0% | 0% |
  | task5 "bowl on ramekin" | 100% | 33.3% | 0% |
  | task6 "bowl next to cookie box" | 100% | 66.7% | 0% |
  | task7 "bowl on stove" | 100% | 66.7% | 0% |
  | task8 "bowl next to plate" | 100% | 100% | 33.3% |
  | task9 "bowl on wooden cabinet" | 100% | 100% | 0% |

- raw results path: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_033143/`
- runtime: ~37 minutes
- validity checks: 10/10 task result files present, correct checkpoint path/step in `summary.json`, correct (freshly recomputed, distinct-filename) LIBERO stats used.
- decision enabled by this evidence: **`DIAGNOSE`** (again) — the leading hypothesis (unmasked video loss corrupting the shared backbone) is not supported by this result: freezing the backbone made retention *worse*, and the backbone was, by construction, held byte-identical to exp0019's for this entire run. Since the backbone is provably unchanged, **whatever is hurting LIBERO here must be entirely within the 6 trainable tensors: `action_encoder`, `head`, `proprio_encoder`.**
- reason: this is a genuinely surprising result relative to the literature-motivated prediction, and demands re-diagnosis before choosing the next candidate — see Section 8/10.

## 8. Comparison and interpretation

**The backbone-freezing hypothesis is refuted by this evidence.** With the entire shared MoT backbone held frozen (byte-identical to exp0019), LIBERO-Spatial retention got *worse* (73.33% -> 16.67%), not better. Since the backbone cannot have changed, the cause must be in the 6 trainable tensors themselves.

**Leading re-diagnosis**: `action_encoder` (Linear 14->1024 in) and `head` (Linear 1024->14 out) are **shared weight matrices whose first 7 columns/rows are used by LIBERO and whose full 14 are used by RoboTwin — these ranges overlap, they are not disjoint.** LIBERO's masked loss correctly prevents *LIBERO's own* gradient from touching columns 7-13 (the padding-only region for LIBERO), but nothing prevents *RoboTwin's* gradient from updating columns 0-6 — the exact columns LIBERO depends on, since RoboTwin's valid range is the full 0-13. Every RoboTwin training step directly overwrites part of the same weight positions LIBERO needs, via plain gradient descent on a shared matrix. In exp0001 (full fine-tune), this same channel overlap existed, but the interference was diluted across ~5B other trainable parameters, and the (also-being-trained) backbone had freedom to adapt around the shifting encoder/head outputs. In exp0002, with *only* these 6 tensors trainable and the backbone frozen (no downstream plasticity to compensate), 100% of both embodiments' gradient signal concentrates on these small shared matrices with nothing to absorb the resulting representational shift — making the direct channel-overlap interference *more* visible, not less.

This reframes the problem: it is not (primarily) a video-backbone/world-model interference issue. It is a **projection-layer weight-sharing** issue — LIBERO's 7 valid channels and RoboTwin's 14 valid channels are not actually independent in the current design, because they occupy overlapping column/row *positions* (both start at index 0) in `action_encoder`/`head`, even though the *loss* is correctly masked per-embodiment.

## 9. Decision

- **Decision:** `REJECT`
- **Canonical RoboTwin evidence available:** no (not evaluated this candidate — LIBERO result alone already rejects it; no point spending RoboTwin compute on a candidate worse than its own rejected predecessor)
- **All five LIBERO >=90% canonical:** no (16.67% sentinel, far below floor)
- **Reason:** worse LIBERO retention than the already-rejected exp0001, with a frozen backbone that provably could not be the cause — the hypothesis this candidate tested is refuted, and the evidence points to a different, more specific mechanism (shared projection-layer columns) that needs a different fix.
- **Checkpoint/branch to preserve:** none; not promoted, diagnostic value fully captured here.
- **Next main-line parent:** unchanged — exp0019 (expanded to K=14).

## 10. What this changes for the next experiment

The next candidate needs to address channel-position overlap in `action_encoder`/`head` directly, not just freeze/unfreeze the backbone. Two concrete directions to weigh:

1. **Per-embodiment gradient masking on the projection layers themselves** (not just the loss): during a LIBERO batch, zero out gradients to `action_encoder`'s/`head`'s weight columns/rows beyond index 7 is already implicit (LIBERO's padded input is exactly zero there, contributing no gradient — this part is fine); the actual gap is the reverse — during a RoboTwin batch, prevent gradient from updating columns 0-6 that LIBERO depends on. This requires an explicit per-embodiment weight mask at the optimizer/gradient level (e.g. a backward hook zeroing specific column/row gradients depending on which embodiment produced the batch), not something `ConcatLeftAlign`'s existing padding/masking touches.
2. **Give each embodiment its own, non-overlapping projection weights** into/out of the shared hidden space (e.g. `action_encoder_libero: Linear(7,1024)`, `action_encoder_robotwin: Linear(14,1024)`, selected per-batch by embodiment, both feeding the *same* shared 1024-dim action transformer) — closer to how several real multi-embodiment systems avoid this exact interference, and avoids needing custom gradient-masking machinery. This is a more direct interface change than the flat K=14-padded-shared-projection approach, but may be the more robust fix.

Option 2 is likely simpler to implement correctly and verify (no custom autograd hooks, easy to unit-test "LIBERO's projection weights are literally never touched by a RoboTwin gradient" the same way the action-dim masking was verified) — recommended as the next candidate's core change, keeping everything else (K=14 shared *transformer* hidden space, per-channel loss masking within each embodiment's own natural dim, embodiment-conditioned prompts, interleaved batch mixing) as-is.

## 11. Artifacts

TBD.

## 12. Reproducibility checklist

- [ ] exact candidate commit and RoboTwin branch recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] multi-embodiment representation/mask/normalization details recorded (unchanged from exp0001)
- [x] dataset mixture and retention strategy recorded (unchanged from exp0001)
- [x] initial training plan recorded
- [ ] training configuration/command recorded
- [ ] hardware/software environment recorded
- [ ] intermediate checkpoints/progress decisions recorded when used
- [ ] logs/checkpoint paths recorded
- [ ] remote checkpoint path verified if HF backup created
- [ ] every evaluation event has benchmark/purpose/settings/raw results
- [ ] canonical RoboTwin metrics recorded when canonical evaluation ran
- [ ] five LIBERO suite metrics recorded when canonical retention evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision/reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
