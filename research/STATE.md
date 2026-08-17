# Current Research State

Status: `SETUP_COMPLETE — READY FOR FIRST CANDIDATE`

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

None yet. exp0019 is the accepted starting parent, not yet a RoboTwin-capable multi-embodiment checkpoint.

- checkpoint: exp0019 (see above)
- parent: none (original LIBERO-only checkpoint)
- commit: `b2b49d0`

## Current accepted metrics

### RoboTwin

Not yet measured — no policy-in-the-loop RoboTwin evaluation has been run yet (only environment-stack verification without a checkpoint).

### LIBERO retention

Inherited canonical record (above) stands; fresh-machine 3-trial Spatial sentinel = 96.67%, consistent with it.

## Best useful branch candidates

None yet — `autoresearch/robotwin-multiembodiment-v1` is the sole active branch, currently at the setup-integration commit (research-control files only, no model/training code changes yet).

## Latest experiment

`0000_parent_baseline` — `PROMOTE` (setup baseline, not a research candidate). Full report: `research/progress/PROGRESS_0000_PARENT_BASELINE.md`. Ledger: `research/EXPERIMENTS.jsonl`.

## Current research notes

- Setup is complete. Next step: `$choose-fastwam-experiment` to select the first real multi-embodiment candidate.
- Key implementation groundwork already identified (see `research/RUNBOOK.md` "Shared padded action representation" and `research/NOTES.md`): the `ConcatLeftAlign` padding mechanism already exists but is unused (`action_target_dim`/`state_target_dim` are `null` in both data configs); the per-channel loss-masking wiring gap is precisely located in `fastwam.py::training_loss` (mirror in `fastwam_idm.py`); no automatic checkpoint-expansion path exists for growing `action_encoder`/`head`/`proprio_encoder` from LIBERO's 7/8-dim to a shared K=14 — must be written.
- Literature-derived recommendations for the first candidate: K=14 shared channels; precise two-level per-channel masked loss (mask-then-mean-per-channel, then mean over valid-channel-count, not all K); Qwen-VLA-style textual embodiment-conditioning prompt (not a learned embedding); roughly 1:1 new(RoboTwin):replay(LIBERO) data mixing; consider a short warm-up training only the expanded projection layers + instruction pathway before unfreezing the shared MoT backbone.
- **Known open gap**: no downloadable LIBERO-90 lerobot-format training data exists in the public FastWAM dataset (only Spatial/Object/Goal/Long-10) — blocks LIBERO-90 replay/rehearsal design specifically; does not block LIBERO-90 *evaluation* (uses the official LIBERO simulator/benchmark directly). Resolve or explicitly work around before finalizing the first candidate's data mixture.
- **Known open gap**: RoboTwin's exact per-arm action-channel semantics (delta vs. absolute; per-arm layout) not yet confirmed against RoboTwin's own env code — the data config alone suggests absolute (no `delta_action_dim_mask` set), but this should be verified before finalizing the masking/normalization design.
- Real infra fixes required on every future eval/training command: `model.redirect_common_files=false` (required standing override, dead HF repo otherwise); GPU memory sizing (`max_tasks_per_gpu<=2` for eval workers; `batch_size=1`+`>=4 GPUs` as the training smoke-test floor, real training needs proper tuning). Full details in `research/NOTES.md`.
- Do not inspect/import `autoresearch/research-docs` as research history for experiment selection.
