# PROGRESS_0002 — frozen_backbone_warmup

- **Experiment ID:** 0002_frozen_backbone_warmup
- **Status:** `PLANNED`
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

TBD.

## 7. Evaluation events

TBD.

## 8. Comparison and interpretation

TBD.

## 9. Decision

TBD.

## 10. What this changes for the next experiment

TBD.

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
