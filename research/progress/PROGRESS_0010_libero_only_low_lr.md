# PROGRESS_0010 — LIBERO-only continued-training control, 10x lower LR

- **Experiment ID:** 0010
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0009_libero_only_control (rejected diagnostic; this candidate directly tests its Section 9 hypothesis)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0009 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

## 2. Research state before experiment

### The decisive finding this candidate tests

exp0009 (LIBERO-only continued training, zero RoboTwin exposure, same K=21/22
expanded checkpoint, same LR `3e-5` as every prior candidate) degraded
LIBERO-Spatial to 23.33% — the second-worst result of any candidate in the
project, worse than 6 of 8 multi-embodiment candidates. This falsified the
multi-embodiment-interference framing that drove every candidate since exp0002:
whatever is destroying performance is a property of continuing to fine-tune this
specific expanded/padded checkpoint, not of RoboTwin's presence.

Leading hypothesis (see `PROGRESS_0009` Section 9): cold-start Adam optimizer
instability and/or the LR (`3e-5`) being too high for continued fine-tuning of an
already-converged checkpoint. Every candidate in this project resumes *weights
only* (no prior optimizer state), so Adam's moment estimates restart at zero;
bias-correction amplifies effective step size for roughly the first 10-50 steps,
which combined with a non-trivial LR could disrupt an already-converged solution
before the loss signal pulls it back.

### Full evidence picture across 9 prior candidates

| Candidate | Backbone | Projections | Ratio | Steps | LR | LIBERO-Spatial |
|---|---|---|---|---:|---:|---:|
| exp0001 | trainable | overlapping (K=14) | 1:1 | 1000 | 3e-5 | **73.33%** |
| exp0002 | frozen | overlapping (K=14) | 1:1 | 1000 | 3e-5 | 16.67% |
| exp0003 | trainable | disjoint (K=21/22) | 1:1 | 1000 | 3e-5 | 50.00% |
| exp0004 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 3e-5 | 63.33% |
| exp0005 | partial-plasticity | disjoint (K=21/22) | 1:1 | 1000 | 3e-5 | 63.33% |
| exp0006 | frozen | disjoint (K=21/22) | 1:1 | 4000 | 3e-5 | 63.33% |
| exp0007 | frozen | disjoint (K=21/22) | 1:1 | 1000 | 3e-5 | 56.67% |
| exp0008 | trainable | disjoint (K=21/22) | 3:1 | 1000 | 3e-5 | 36.67% |
| exp0009 | trainable | disjoint (K=21/22) | LIBERO-only | 1000 | 3e-5 | **23.33%** |

Every candidate so far used the same `3e-5` LR — this is the first to vary it.

## 3. Candidate design

### Modifications

1. New task config `configs/task/libero_only_disjoint_offset_control_low_lr_3e-6.yaml`: byte-for-byte identical to exp0009's `libero_only_disjoint_offset_control_3e-5.yaml` except `learning_rate: 3e-6` (was `3e-5`, a 10x reduction).
2. No code changes.

### Why this candidate

The single cheapest, most direct test of the leading hypothesis from exp0009.
Isolates the LR variable completely (identical data, architecture, backbone
treatment, step budget to exp0009) — any retention difference is attributable only
to the LR change.

- If LIBERO-Spatial retention holds up much better than exp0009's 23.33% (ideally clearing or approaching exp0019's inherited ~96.67-97.00%, or at minimum clearing the 90% floor): confirms LR/optimizer instability as the real driver. The practical fix for the whole project becomes "use a much lower LR for continued fine-tuning" — motivating a follow-up that revisits the multi-embodiment mixture (LIBERO+RoboTwin) at this corrected LR, potentially finally clearing the 90% floor while retaining meaningful RoboTwin learning.
- If retention still degrades substantially even at 3e-6: LR alone isn't the (whole) story — the next diagnostic would need to isolate the checkpoint-expansion process itself (e.g. compare against continuing to fine-tune the *original*, unexpanded exp0019 checkpoint on LIBERO alone, to check whether the expansion/widening itself is destabilizing) or examine actual gradient/update-magnitude telemetry during training rather than reasoning from final eval numbers alone.

### Multi-embodiment representation/configuration

Identical to exp0009 (K=21 action / K=22 proprio, LIBERO offset 0). No changes.

### Data and learning strategy

Identical to exp0009 (LIBERO-only, full trainable backbone, 1000 steps) except
`learning_rate: 3e-6` (was `3e-5`).

### What to watch

- Primary: LIBERO-Spatial candidate_screen result vs. exp0009's 23.33%, exp0019's inherited ~96.67-97.00%, and the 90% floor.
- No RoboTwin evaluation needed (RoboTwin never in training).

### Initial compute plan

- initial training budget: 1000 steps (matches exp0009 for a clean comparison)
- checkpoint/save plan: `save_every: 200`, `save_full_state: false`, active `KEEP=1` pruner
- evaluation plan: LIBERO-Spatial candidate_screen only (standard 3-trial/10-task)
- expected cost: comparable to exp0009 (~45min training + ~35-40min LIBERO screen)

## 4. Exact code and configuration state

- Git commit: pending (recorded after committing this report + config, before training launch)
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `9db7086` (exp0009 decisive-finding commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: `configs/task/libero_only_disjoint_offset_control_low_lr_3e-6.yaml` (new), this report
- training config: `configs/task/libero_only_disjoint_offset_control_low_lr_3e-6.yaml`
- config overrides: `model.redirect_common_files=false`, `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- data config: `configs/data/libero_2cam_multiembodiment.yaml` (unchanged)
- optimizer / LR / scheduler: AdamW, cosine, `learning_rate: 3e-6` (10x lower than every prior candidate)
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
