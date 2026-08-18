# Current Research State

Status: `RESEARCH_LOOP_ACTIVE — exp0009 training (LIBERO-only continued-training control, no RoboTwin in the mixture at all)`

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

Inherited canonical record (above) stands for the accepted parent (exp0019). exp0001 (rejected): LIBERO-Spatial 73.33%. exp0002 (rejected): LIBERO-Spatial 16.67%, worse than exp0001 — refuted the backbone-interference hypothesis. exp0003 (rejected): LIBERO-Spatial **50.00%** — better than exp0002 but still worse than exp0001, refuting the simple projection-overlap-is-the-sole-cause hypothesis. Best LIBERO-Spatial retention seen so far across any multi-embodiment candidate remains exp0001's 73.33%, still below the 90% floor.

## Best useful branch candidates

None yet — `autoresearch/robotwin-multiembodiment-v1` is the sole active branch.

## Latest experiment

`0008_disjoint_offset_3to1_ratio` — `REJECT`. Full report: `research/progress/PROGRESS_0008_disjoint_offset_3to1_ratio.md`. Combined disjoint-offset projections + trainable backbone (exp0003's architecture) with a 3:1 LIBERO:RoboTwin mixing ratio (was 1:1 in every prior candidate) — testing whether diluting RoboTwin's gradient share protects LIBERO retention. **Result: LIBERO-Spatial 36.67%, WORSE than exp0003's 50.00% at 1:1 (same architecture) — refutes the hypothesis.** RoboTwin: 0.0% on all 4 measurements (confirmed via raw result files), the first candidate to show zero capability on both tested tasks simultaneously. This exposed a critical gap: **no candidate across the whole project has established a LIBERO-only continued-training control** using the K=21/22 disjoint-offset architecture — it's not actually confirmed the retention problem is a multi-embodiment interference phenomenon at all, vs. simply continued-training drift from the expanded/padded architecture independent of RoboTwin. Next: exp0009, a LIBERO-only control (no RoboTwin in the training mixture at all) from the same expanded checkpoint, to isolate this — the cheapest candidate possible in this project (no RoboTwin data/eval needed).

## Experiment history

- `0000_parent_baseline` — `PROMOTE` (setup baseline, not a research candidate).
- `0001_padded_multiembodiment_baseline` — `REJECT`. Full fine-tune (no freezing), K=14 padded interface, ~1:1 LIBERO:RoboTwin interleaving, 1000 steps. Result: LIBERO-Spatial 96.67%->73.33%, RoboTwin 0.0%. Initially attributed to the unmasked video-denoising loss training the shared backbone on RoboTwin's visual domain. Full report: `research/progress/PROGRESS_0001_padded_multiembodiment_baseline.md`.
- `0002_frozen_backbone_warmup` — `REJECT`. Froze the entire shared MoT backbone (byte-identical to exp0019 by construction), trained only `action_encoder`/`head`/`proprio_encoder` (6 tensors). Result: LIBERO-Spatial dropped further to **16.67%** — worse than exp0001, despite the backbone provably being unchanged. **This refutes the backbone-interference hypothesis.** Re-diagnosis: `action_encoder`/`head` are shared weight matrices whose LIBERO-valid columns (0-6) and RoboTwin-valid columns (0-13) *overlap* — RoboTwin's gradient directly overwrites the same weight positions LIBERO depends on, and with the backbone frozen there's no downstream plasticity to absorb the shift (in exp0001 this same overlap existed but was diluted across ~5B other trainable params). Full report: `research/progress/PROGRESS_0002_frozen_backbone_warmup.md`.
- `0003_disjoint_action_offset` — `REJECT`. Gave LIBERO and RoboTwin disjoint, non-overlapping column ranges within a widened shared K=21/22 tensor (LIBERO offset 0, RoboTwin offset 7/8) instead of both left-aligned at 0, via new `ConcatLeftAlign.action_offset`/`state_offset` params — proven interference-free in isolation (gradient-isolation unit test). Same full-fine-tune backbone treatment as exp0001. Result: LIBERO-Spatial **50.00%**, better than exp0002 but *worse* than exp0001's 73.33% — contradicts the simple version of the hypothesis (should have been >= exp0001 if projection overlap were the sole mechanism). Scattered/task-swapping per-task pattern rather than uniform decline. **Refutes projection-overlap-alone as the full explanation**; points toward the shared transformer backbone as an under-isolated interference channel, or meaningful run-to-run variance on the small eval panel. Full report: `research/progress/PROGRESS_0003_disjoint_action_offset.md`.
- `0004_disjoint_offset_frozen_backbone` — `REJECT`. Combined exp0002's backbone freezing (`trainable_modules=expanded_projections_only`) with exp0003's disjoint-offset projections — fills the missing 2x2 matrix cell. Result: LIBERO-Spatial **63.33%** (19/30) — far above exp0002 (16.67%), above exp0003 (50.00%), still below exp0001 (73.33%) and the 90% floor. **Completed matrix**: overlap+trainable=73.33%, overlap+frozen=16.67%, disjoint+trainable=50.00%, disjoint+frozen=63.33%. Freezing has opposite effects by projection design (hurts with overlap, helps with disjoint). A RoboTwin progress check (same 2-task panel as exp0001) found `click_alarmclock` **33.3%/33.3% (clean/random)** — the first non-zero RoboTwin result anywhere in this project (exp0001 was 0.0% everywhere); `adjust_bottle` stayed 0.0%/0.0%. Confirms disjoint offsets are a real fix, backbone drift is a second independent interference channel, and a frozen backbone can still support some genuine RoboTwin learning. Full report: `research/progress/PROGRESS_0004_disjoint_offset_frozen_backbone.md`.
- `0005_disjoint_offset_backbone_low_lr` — `REJECT`. Added `Trainer.trainable_modules="dit_with_backbone_low_lr"`: two-parameter-group AdamW optimizer, entire backbone trainable at `backbone_lr=3e-6` (10x lower than the projection layers' `3e-5`), testing a middle ground between exp0003 (full plasticity) and exp0004 (full freeze). Result: LIBERO-Spatial 63.33% (matched exp0004's aggregate exactly). RoboTwin progress check went the wrong direction: `adjust_bottle` unchanged at 0%/0%, `click_alarmclock` **lost its randomized-phase success entirely** (33.3%->0.0%), keeping only the clean-phase result — partial plasticity cost RoboTwin robustness for no gain, contradicting the hypothesis. Full report: `research/progress/PROGRESS_0005_disjoint_offset_backbone_low_lr.md`.
- `0006_disjoint_offset_frozen_backbone_4k` — `REJECT`. Re-ran exp0004's exact recipe unchanged for 4000 steps (4x budget) instead of 1000, testing whether budget rather than LR ratio was the missing lever. Result: LIBERO-Spatial 63.33% again (three candidates now converge on 19/30 exactly). RoboTwin: `click_alarmclock` **lost its only real capability entirely** (33.3%/33.3% -> 0.0%/0.0%, confirmed via raw result files), `adjust_bottle` stayed dead. A mid-run 1-trial LIBERO check at step 1500 showed a promising 80%, but did not replicate under the full 3-trial re-measurement at step 4000 (single-trial optimism). Also hit and documented a real infra gotcha: the checkpoint pruner raced with and crashed a concurrent full LIBERO screen (`research/NOTES.md` "Mid-training eval gotcha"), recovered via a faster 1-trial mid-run panel. Full report: `research/progress/PROGRESS_0006_disjoint_offset_frozen_backbone_4k.md`.
- `0007_exp0004_replication` — `REJECT`. Re-ran exp0004's exact recipe fresh (no code/config changes), evaluated `click_alarmclock` at n=10 instead of n=3 to settle reproducibility. Result: LIBERO-Spatial 56.67% (close to but not exactly 63.33%, confirming real variance); RoboTwin `click_alarmclock` 20.0% clean / 10.0% random (confirmed via raw result files) — **settles that exp0004's capability was real, not noise**, just noisier/lower than the original single measurement. `adjust_bottle` remains 0% (4 candidates, 12 episodes, zero successes). Reframes exp0005/exp0006's 0% results as ambiguous (genuine regression vs. small-n noise). Also hit and documented a second infra gotcha: `run_robotwin_manager.py` ignores `CUDA_VISIBLE_DEVICES` remapping to non-zero GPU indices (`research/NOTES.md`). Full report: `research/progress/PROGRESS_0007_exp0004_replication.md`.
- `0008_disjoint_offset_3to1_ratio` — `REJECT`. Combined exp0003's architecture (trainable backbone + disjoint-offset K=21/22 projections) with a 3:1 LIBERO:RoboTwin mixing ratio (first-ever test of this variable; every prior candidate used ~1:1). Result: LIBERO-Spatial **36.67%**, worse than exp0003's 50.00% at 1:1 with the identical architecture — refutes the more-rehearsal-helps hypothesis. RoboTwin: **0.0% on all 4 measurements** (confirmed via raw result files) — the first candidate with zero capability on both tested tasks. Exposed a critical gap in the evidence base: no candidate has ever run a LIBERO-only continued-training control on the K=21/22 architecture, so it remains unconfirmed whether the retention ceiling is a genuine multi-embodiment interference effect or simply continued-training drift independent of RoboTwin. Full report: `research/progress/PROGRESS_0008_disjoint_offset_3to1_ratio.md`.

## Current research notes

- Implementation groundwork from setup (still current): `ConcatLeftAlign` padding is now actively used (was previously unwired), now with `action_offset`/`state_offset` support (default 0, exp0003); `fastwam.utils.losses.masked_action_loss` implements the two-level per-channel masked loss (wired into both `fastwam.py` and `fastwam_idm.py`); `research/tools/expand_checkpoint_for_multiembodiment.py` grows exp0019's projections to a target K preserving inherited weights exactly (used for both the K=14-overlap and K=21/22-disjoint expansions, neither promoted).
- `Trainer.trainable_modules` config option (`"dit"` [default] | `"expanded_projections_only"` [exp0002]) for selective backbone freezing — mechanism works correctly (verified 6 trainable/1841 frozen tensors) but did not fix retention alone; kept in the codebase as reusable infrastructure, not the active recipe.
- **Disjoint-offset fix tested and insufficient alone (exp0003)**: giving LIBERO/RoboTwin non-overlapping columns in the shared `action_encoder`/`head`/`proprio_encoder` (proven interference-free in isolation via a gradient-isolation unit test) produced LIBERO-Spatial 50.00% — better than exp0002 (16.67%) but worse than exp0001 (73.33%), so projection-column overlap is evidently not the sole/dominant interference mechanism. **Leading open hypothesis for exp0004**: the shared 30-layer MoT/DiT *backbone* is a real, under-isolated interference channel — no candidate has yet combined a frozen backbone with disjoint-offset projections to test this cleanly (exp0002 froze the backbone but kept the *overlapping* K=14 projections; exp0003 used disjoint projections but kept the full trainable backbone). Also worth weighing: run-to-run variance on the small 3-trial/task eval panel may explain part of the 73.33%->50.00% gap — a repeat run with a controlled seed and/or more trials could help distinguish signal from noise before committing more compute. See `PROGRESS_0003` Section 9 for the full reasoning.
- **Known open gap**: no downloadable LIBERO-90 lerobot-format training data exists in the public FastWAM dataset (only Spatial/Object/Goal/Long-10) — blocks LIBERO-90 replay/rehearsal design specifically; does not block LIBERO-90 *evaluation*.
- **Known open gap**: RoboTwin's exact per-arm action-channel semantics (delta vs. absolute; per-arm layout) still not confirmed against RoboTwin's own env code.
- **Disk is a hard, recurring constraint**: RoboTwin's 921k unique per-episode instructions require a ~900GB text-embedding cache (already computed, do not delete), leaving very little of the 1.1TB volume for training checkpoints. `save_full_state=false` is required (skips DeepSpeed optimizer-state saves); an active background pruner keeping only 1 checkpoint at a time is required during real training runs (see `research/NOTES.md` "Disk crisis" and its operational addenda — the exact pruner pattern and its race-condition pitfall are documented there).
- Real infra fixes required on every eval/training command: `model.redirect_common_files=false` (dead HF repo otherwise); the eval-side padding/cropping/embodiment-conditioning fixes in `experiments/libero/eval_libero_single.py` and `experiments/robotwin/fastwam_policy/deploy_policy.py` (needed for any K=14 checkpoint, not just exp0001/exp0002); distinct `stats_filename` per embodiment in multi-embodiment data configs (otherwise RoboTwin's stats silently clobber LIBERO's). Full details in `research/NOTES.md`.
- Do not inspect/import `autoresearch/research-docs` as research history for experiment selection.
