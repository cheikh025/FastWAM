# Current Research State

Status: `RESEARCH_LOOP_ACTIVE — exp0002 running`

## Inherited parent reference

- parent checkpoint name: `exp0019_spatial_weak_task_oversample / step_005000.pt`
- recorded original local path: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`
- durable recorded HF path: `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`
- local verified path (this machine): `/workspace/FastWAM/checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt`
- file identity: 12,041,735,545 bytes, sha256 `decbb99cdd9640a3de5e769801f942940a3d0ab9e2103d10ce5acea632a79c5`
- parent code lineage branch: `autoresearch/libero90-v1`
- checkpoint-producing commit: `b2b49d0` — verified ancestor of `autoresearch/libero90-v1`
- parent branch policy: read-only; not modified/committed/pushed to this session
- new research branch: `autoresearch/robotwin-multiembodiment-v1` — created from exact `b2b49d0`, active, verified before all tracked changes

## Previously validated parent LIBERO metrics

Canonical parent evidence from the previous project (50 trials/task, all 130 tasks):

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% |
| LIBERO-Spatial | 97.00% |
| LIBERO-Object | 99.60% |
| LIBERO-Goal | 97.20% |
| LIBERO-Long / LIBERO-10 | 98.00% |

These are inherited reference metrics, not new-run measurements.

## Fresh-machine parent validation

- exact parent checkpoint downloaded/verified: **done**
- verified local checkpoint path: **done** (see above)
- checkpoint file identity/hash: **done**
- parent commit provenance verified: **done**
- new RoboTwin branch created/active before tracked changes: **done**
- LIBERO smoke/sentinel validation: **done** — LIBERO-Spatial, 10 tasks x 3 trials = 96.67% (29/30), near-exact match to canonical 97.00%, weak task matches parent's own history
- RoboTwin environment/evaluator smoke/reference validation: **done** — full simulator stack (SAPIEN, curobo, aloha-agilex embodiment, 3-camera rendering) verified end-to-end; policy-in-the-loop evaluation not yet run
- training smoke test: **done** — 5 real steps resumed from exp0019, sane non-NaN losses, checkpoint written
- checkpoint reload/resume smoke test: **done** — reloaded smoke checkpoint, 7 more real steps, no shape/key errors
- full Qwen-VLA + relevant multi-embodiment literature review completed: **done** — Qwen-VLA (2605.30280) full text; forgetting-resistance paper (2603.03818) full text; LAP/Qwen-RobotManip abstract-only, no additional mechanisms

## Canonical benchmark definitions

- LIBERO canonical protocol: defined in `research/RUNBOOK.md` — 50 trials/task, all five suites (note: `libero_90` must be explicitly requested, not in the default multirun list)
- RoboTwin canonical protocol/metric(s): defined in `research/RUNBOOK.md` — **confirmed 50-task Aloha-AgileX benchmark**; Clean/Randomized evaluated and tracked separately (`clean_mean_success_rate`, `random_mean_success_rate`); 100 episodes/task/phase; tie-break rule frozen (worst-case-first, then mean, per `GOAL.md`'s LIBERO ranking philosophy)

## Current accepted multi-embodiment checkpoint

None yet. exp0019 (expanded to K=14: `checkpoints/exp0019_expanded_k14/step_005000.pt`) remains the parent for the research loop — exp0001 was rejected, not promoted.

- checkpoint: exp0019, expanded to K=14
- parent: none (original LIBERO-only checkpoint)
- commit: `b2b49d0`

## Current accepted metrics

### RoboTwin

Not yet measured for any accepted checkpoint. exp0001 (rejected) measured 0.0% on a 2-task progress-check panel.

### LIBERO retention

Inherited canonical record (above) stands for the accepted parent (exp0019). exp0001 (rejected) showed LIBERO-Spatial dropping to 73.33% (3-trial sentinel) — see exp0001 below.

## Best useful branch candidates

None yet — `autoresearch/robotwin-multiembodiment-v1` is the sole active branch.

## Latest experiment

`0002_frozen_backbone_warmup` — `RUNNING`. Full report: `research/progress/PROGRESS_0002_frozen_backbone_warmup.md`.

## Experiment history

- `0000_parent_baseline` — `PROMOTE` (setup baseline, not a research candidate).
- `0001_padded_multiembodiment_baseline` — `REJECT`. Full fine-tune (no freezing), K=14 padded interface, ~1:1 LIBERO:RoboTwin interleaving, 1000 steps. Result: LIBERO-Spatial 96.67%->73.33% (real regression, concentrated on exp0019's already-weakest tasks), RoboTwin 0.0% across all measurements. Root-caused (via `$investigate-fastwam-problem`) to the video-denoising loss not being per-embodiment-masked — RoboTwin's very different visual domain directly trains the shared backbone LIBERO's video generation depends on, compounded by no freezing. The padding/masking/expansion mechanisms themselves are independently verified correct (14/14 unit tests) — only the training recipe was rejected. Full report: `research/progress/PROGRESS_0001_padded_multiembodiment_baseline.md`.
- `0002_frozen_backbone_warmup` — `RUNNING`. Freezes the entire shared MoT backbone, trains only the newly-expanded `action_encoder`/`head`/`proprio_encoder` (verified: exactly 6 trainable tensors, 1841 frozen). Same 1000-step budget, same data mixture as exp0001 — isolates `trainable_modules` as the only changed variable, direct test of the interference hypothesis.

## Current research notes

- Implementation groundwork from setup (still current): `ConcatLeftAlign` padding (K=14) is now actively used (was previously unwired); `fastwam.utils.losses.masked_action_loss` implements the two-level per-channel masked loss (wired into both `fastwam.py` and `fastwam_idm.py`); `research/tools/expand_checkpoint_for_multiembodiment.py` grows exp0019's projections to K=14 preserving inherited weights exactly.
- **New since exp0001**: `Trainer.trainable_modules` config option (`"dit"` [default] | `"expanded_projections_only"` [new, exp0002]) for selective backbone freezing.
- **Known open gap**: no downloadable LIBERO-90 lerobot-format training data exists in the public FastWAM dataset (only Spatial/Object/Goal/Long-10) — blocks LIBERO-90 replay/rehearsal design specifically; does not block LIBERO-90 *evaluation*.
- **Known open gap**: RoboTwin's exact per-arm action-channel semantics (delta vs. absolute; per-arm layout) still not confirmed against RoboTwin's own env code.
- **Disk is a hard, recurring constraint**: RoboTwin's 921k unique per-episode instructions require a ~900GB text-embedding cache (already computed, do not delete), leaving very little of the 1.1TB volume for training checkpoints. `save_full_state=false` is required (skips DeepSpeed optimizer-state saves); an active background pruner keeping only 1 checkpoint at a time is required during real training runs (see `research/NOTES.md` "Disk crisis" and its operational addenda — the exact pruner pattern and its race-condition pitfall are documented there).
- Real infra fixes required on every eval/training command: `model.redirect_common_files=false` (dead HF repo otherwise); the eval-side padding/cropping/embodiment-conditioning fixes in `experiments/libero/eval_libero_single.py` and `experiments/robotwin/fastwam_policy/deploy_policy.py` (needed for any K=14 checkpoint, not just exp0001/exp0002); distinct `stats_filename` per embodiment in multi-embodiment data configs (otherwise RoboTwin's stats silently clobber LIBERO's). Full details in `research/NOTES.md`.
- Do not inspect/import `autoresearch/research-docs` as research history for experiment selection.
