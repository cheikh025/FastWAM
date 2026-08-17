# PROGRESS_0000 — Parent baseline / fresh-machine setup

- **Experiment ID:** 0000_parent_baseline
- **Status:** `PROMOTE` (setup validated; exp0019 is the accepted starting parent for the multi-embodiment line)
- **Created:** 2026-08-17
- **Updated:** 2026-08-17
- **Parent experiment:** none (inherited from the prior LIBERO project's exp0019, not produced by this project)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (original path, prior project); durable copy `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`
- **Selected candidate checkpoint:** exp0019 itself (no new training performed in this setup phase)
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** created from `b2b49d0` (verified ancestor of `autoresearch/libero90-v1`)

## 1. Result at a glance

Fresh-machine setup for the FastWAM RoboTwin multi-embodiment project is complete and validated. The exp0019 LIBERO checkpoint was downloaded and identity-verified (exact byte size + sha256), the exact checkpoint-producing commit `b2b49d0` was verified as an ancestor of the read-only parent branch, and the new research branch was created from that exact commit. Both LIBERO and RoboTwin environments were installed, verified, and smoke-tested with real (not synthetic) evidence: a LIBERO-Spatial 3-trial smoke eval scored 96.67% (29/30), closely matching exp0019's own canonical 97.00%, with the one weak task landing within its known historical range; the full RoboTwin 2.0 simulator stack (SAPIEN, curobo motion planning, bimanual aloha-agilex embodiment, 3-camera rendering) was verified end-to-end; a training smoke test resumed from exp0019 and ran real optimizer steps with sane, non-NaN losses, wrote a checkpoint, and that checkpoint was successfully reloaded and used to continue training. The full Qwen-VLA paper plus one directly relevant forgetting-resistance paper were read in full, yielding concrete, actionable mechanisms for the initial multi-embodiment candidate (padded action interface via an already-implemented-but-unused `ConcatLeftAlign`, a precise two-level per-channel masked loss formula, a textual embodiment-conditioning template, and a replay-ratio recommendation). Several real infrastructure bugs were found and fixed along the way (dead HF redirect repo, unpinned/broken curobo HEAD, a LIBERO packaging bug, a LIBERO PyTorch-2.6 compatibility bug, and GPU-memory sizing constraints for concurrent workers) and are documented for reuse.

## 2. Research state before experiment

### Accepted RoboTwin state

- canonical metric(s): not yet measured — this is setup, no RoboTwin policy-in-the-loop evaluation with a checkpoint has been run yet (only environment-stack verification without a policy).
- important weak tasks/difficulties: unknown yet.

### Accepted LIBERO retention state

Inherited canonical reference (prior project, 50 trials/task, all 130 tasks — not re-measured at full precision this session):

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% |
| LIBERO-Spatial | 97.00% |
| LIBERO-Object | 99.60% |
| LIBERO-Goal | 97.20% |
| LIBERO-Long / LIBERO-10 | 98.00% |

These are inherited facts, not new-run measurements. Fresh-machine sentinel validation (LIBERO-Spatial, 10 tasks, 3 trials/task): **96.67% (29/30)** — see Section 7.

## 3. Candidate design

Not applicable — this is the parent-setup baseline record, not a candidate. See `AUTORESEARCH.md`/`CLAUDE.md` for the setup requirements this record satisfies.

### Multi-embodiment representation/configuration

Not yet implemented. Verified facts recorded in `research/RUNBOOK.md` ("Shared padded action representation" section) and `research/NOTES.md`:

- shared action dimension: K=14 proposed (RoboTwin's natural dim; LIBERO pads 7->14, 8->14 for proprio).
- LIBERO valid channels/mask: first 7 of 14 action channels (delta-eef x6 + gripper x1), first 8 of 14 proprio channels.
- RoboTwin valid channels/mask: all 14 action and 14 proprio channels valid.
- normalization/statistics behavior: per-dataset, unchanged from current (LIBERO min/max, RoboTwin z-score) — confirmed aligned with Qwen-VLA's per-dataset normalization principle.
- checkpoint projection expansion/initialization: not yet implemented — confirmed necessary (no automatic path; `strict=False` load tolerates missing keys but not shape mismatches).
- embodiment/control conditioning: not yet implemented — literature review recommends a textual prompt template (Qwen-VLA-style), not a learned embedding.
- camera/observation handling: unchanged per-dataset (LIBERO 2-cam 224px horizontal-concat, RoboTwin 3-cam 384px).
- inference slicing/decoding: `ConcatLeftAlign.backward()` already implements this, currently unused.

### Data and learning strategy

Not yet implemented — recommendations recorded in `research/RUNBOOK.md`/`research/NOTES.md` from the literature review (roughly 1:1 new:replay mixing per batch, not proportional to raw dataset size; possible short warm-up freezing the shared backbone before full unfreeze).

### What to watch

N/A for this baseline record.

### Initial compute plan

N/A for this baseline record — see `$choose-fastwam-experiment` for the first real candidate.

## 4. Exact code and configuration state

- Git commit: created from `b2b49d0` (`Fix directory-resume LR schedule corruption when max_steps is extended`, `src/fastwam/trainer.py`), verified ancestor of `autoresearch/libero90-v1` via `git merge-base --is-ancestor`.
- Git branch: `autoresearch/robotwin-multiembodiment-v1`, created via `git switch -c autoresearch/robotwin-multiembodiment-v1 b2b49d0`, active and verified via `git branch --show-current` before any tracked change.
- parent code commit: `b2b49d0` (exp0019's own progress report records the same commit as its exact code state).
- working tree clean/dirty before launch: clean at `b2b49d0`; this setup phase adds the research-control files (`CLAUDE.md`, `AUTORESEARCH.md`, `SOURCES.md`, `research/`) and the `experiments/robotwin/fastwam_policy` symlink as new tracked additions (see `git diff --stat` below); no FastWAM source files modified.
- files changed: new files only — `CLAUDE.md`, `AUTORESEARCH.md`, `SOURCES.md`, `research/**`, `third_party/RoboTwin/policy/fastwam_policy` (symlink). No existing tracked file was edited.
- training config(s): not applicable to setup itself; smoke tests used `task=libero_uncond_2cam224_1e-4` (see `research/RUNBOOK.md`).
- resume source and resume type: exp0019 weights-only file resume, verified working (see Section 6/7).

## 5. Hardware and software environment

### GPU

- GPU count: 4
- GPU model(s): NVIDIA A100-SXM4-80GB
- memory per GPU: 80GB (81920 MiB)
- NVIDIA driver: 570.211.01
- CUDA runtime/toolkit: driver reports CUDA 12.8; system nvcc at `/usr/local/cuda/bin/nvcc` also 12.8
- `CUDA_VISIBLE_DEVICES` / world size: varied per smoke test (2-4 GPUs); full training world size not yet decided

### CPU / RAM / storage

- relevant disk total/free before run: `/workspace` persistent volume, 1.1TB total, 951GB free after setup (peaked ~gigabytes lower mid-download, all freed after cleanup of temporary archives/smoke-test runs)
- see `research/progress/system_0000_parent_baseline.json` for full captured snapshot (CPU model, RAM, etc.)

### Software

- Python executable/version: dedicated venv `/workspace/venvs/fastwam`, Python 3.10.20
- PyTorch version: 2.7.1+cu128
- PyTorch CUDA version: 12.8
- DeepSpeed version: 0.18.5 (per FastWAM's pinned deps)
- Accelerate version: 1.12.0 (per FastWAM's pinned deps)
- FastWAM repository commit: base `b2b49d0` + untracked research-file additions (this setup)
- RoboTwin repository/revision: vendored at `third_party/RoboTwin`, upstream commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- LIBERO repository/revision: `Lifelong-Robot-Learning/LIBERO` cloned separately to `/workspace/third_party_src/LIBERO`, commit `8f1084e`
- environment identifier: `/workspace/venvs/fastwam` (shared by FastWAM + LIBERO + RoboTwin — required, see `research/RUNBOOK.md` "important version compatibility notes")
- dependency snapshot path: `research/progress/system_0000_parent_baseline.json`

### Setup validation (this record)

- exact parent checkpoint identity/download verification: **done** — `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`, 12,041,735,545 bytes (exact match to size recorded in the prior project's own progress report), sha256 `decbb99cdd9640a3de5e769801f942940a3d0ab9e2103d10ce5acea632a79c5`, loads cleanly under the installed environment.
- parent commit/branch provenance verification: **done** — `b2b49d0` confirmed ancestor of `autoresearch/libero90-v1` via `git merge-base --is-ancestor`.
- creation/activation of `autoresearch/robotwin-multiembodiment-v1` before tracked changes: **done** — verified via `git branch --show-current` before any file was added/edited.
- Qwen-VLA + relevant literature review completion: **done** — full Qwen-VLA (arXiv 2605.30280) read; LAP/Qwen-RobotManip abstract-only (no additional mechanisms extracted); "Pretrained VLA Models Are Surprisingly Resistant to Forgetting" (arXiv 2603.03818) read in full, directly relevant. See `research/RUNBOOK.md` "Literature/implementation understanding gate".
- verified LIBERO/RoboTwin action/state/normalization semantics: **done** — LIBERO action_dim=7/proprio_dim=8 (min/max norm), RoboTwin action_dim=14/proprio_dim=14 (z-score norm), both confirmed from two independent sources (data configs AND checkpoint tensor shapes). RoboTwin's exact per-arm channel semantics (delta vs. absolute) remains an open item flagged for follow-up, not blocking.
- LIBERO evaluation smoke/sentinel command/result/artifacts: **done** — 96.67% (29/30) on LIBERO-Spatial, 10 tasks x 3 trials; `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260817_221019/`.
- RoboTwin evaluator/environment smoke/reference command/result/artifacts: **done** — full simulator stack verified (SAPIEN render + full env instantiation with aloha-agilex, curobo motion planning, 3-camera rendering); `checkpoints/robotwin_env_smoke_test.py`. Policy-in-the-loop evaluation not yet run (needs the multi-embodiment or specialist checkpoint, a follow-up step).
- training smoke command/result/artifacts: **done** — 5 real steps resumed from exp0019, sane losses, checkpoint written; `checkpoints/train_smoke2.log`.
- checkpoint reload/resume smoke command/result/artifacts: **done** — reloaded the smoke-test checkpoint, ran 7 further steps, confirmed weights-only resume restarts the step counter (as documented) and full-state directory checkpointing also works; `checkpoints/reload_smoke.log`.

## 6. Training execution and control timeline

Not applicable in the traditional sense — no real research training occurred in this setup phase, only smoke tests (see Section 5's "Setup validation" and Section 7).

### Why training ended

N/A — setup phase, no candidate training.

### Training anomalies

Two real, diagnosed-and-fixed infra issues during smoke testing (not swept under the rug):

1. `model.redirect_common_files` (default `true`) points VAE/text-encoder downloads at a dead HF repo (`DiffSynth-Studio/Wan-Series-Converted-Safetensors`, confirmed 404) — crashes every eval/training run by default. **Fix: `model.redirect_common_files=false` is now a required standing override on every command.**
2. `MULTIRUN.max_tasks_per_gpu=5` (LIBERO eval) and `batch_size=2` (training, 2 GPUs) both OOM on an 80GB A100 given this model's per-worker memory footprint (~14-20GB for eval; ZeRO-1 optimizer-state allocation is larger for training). **Fix: `max_tasks_per_gpu<=2` for eval; `batch_size=1` with >=4 GPUs for the training smoke floor** (real training will need proper batch/GPU-count tuning once throughput requirements are known).

See `research/NOTES.md` for full details and exact fixes, plus two additional environment-level (not FastWAM-repo) bugs: a LIBERO packaging bug (empty `find_packages()` result) and a LIBERO/PyTorch-2.6 `torch.load` compatibility bug, and the curobo version-pin + warp-lang-version issue for RoboTwin.

## 7. Evaluation events

### Event 1 — `diagnostic` (LIBERO fresh-machine sentinel)

- benchmark: `libero`
- checkpoint / training step: exp0019, step 5000
- decision this evaluation was meant to inform: confirm the fresh-machine setup faithfully reproduces the parent's known behavior before starting research
- exact task/suite/difficulty coverage: all 10 LIBERO-Spatial tasks
- why this panel was used: cheap, fast, and Spatial's known per-task weak point (task4) gives a specific, checkable signal beyond a bare aggregate
- parent/reference checkpoint and matching result: exp0019 canonical (50-trial) Spatial = 97.00%; confirmation-precision (15-trial) = 96.00%
- candidate result: **96.67% (29/30)** — 9/10 tasks at 100% (3/3), `libero_spatial_4` ("bowl in top drawer") at 66.67% (2/3)
- trials/episodes per task: 3
- exact command/config: see `research/RUNBOOK.md` "LIBERO evaluation smoke/sentinel"
- raw results path: `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260817_221019/` (`summary.json`, `summary.csv`, `task_success_rates.csv`)
- runtime: ~32 minutes (10 tasks x 3 trials, 2 workers/GPU x 2 GPUs)
- validity checks: 10/10 task result files present, aggregate recomputed matches `summary.json`'s reported 96.67%
- decision enabled by this evidence: setup validated — proceed past setup to the research loop
- reason: near-exact match to canonical/confirmation-precision Spatial numbers, and the one weak task matches the parent's own documented weak-task history — strong evidence this is a faithful reproduction, not a broken or coincidentally-passing setup

### Event 2 — `diagnostic` (RoboTwin environment stack, no policy)

- benchmark: `robotwin`
- checkpoint / training step: none (deliberately excludes the policy — pure simulator-stack verification)
- decision this evaluation was meant to inform: confirm the RoboTwin 2.0 simulator (SAPIEN, curobo, assets, aloha-agilex embodiment) works end-to-end on this machine
- exact task/suite/difficulty coverage: `click_alarmclock` task, `demo_clean` config, `aloha-agilex` embodiment (representative smoke case, not all 50 tasks individually audited)
- why this panel was used: exercises the full simulator stack (robot loading, motion planning for both arms, camera rendering) without needing a trained checkpoint
- parent/reference checkpoint and matching result: n/a (no policy involved)
- candidate result: **pass** — `setup_demo()`/`get_obs()` completed cleanly, correct camera shapes/keys for all 3 cameras, both arms' curobo motion planners initialized
- trials/episodes per task: n/a (single instantiation + observation, not an episode rollout)
- exact command/config: see `research/RUNBOOK.md` "RoboTwin evaluator/environment smoke"
- raw results path: `checkpoints/robotwin_env_smoke_test.py` (the smoke script, kept for reuse)
- runtime: a few minutes
- validity checks: no crash, correct output structure
- decision enabled by this evidence: RoboTwin simulator infra confirmed ready for real policy evaluation (follow-up step, not yet done)
- reason: exercises the actual failure-prone components (SAPIEN Vulkan rendering, curobo motion planning, bimanual embodiment loading) that a bare import-check would miss

### Event 3 — `diagnostic` (training + reload/resume smoke)

- benchmark: `libero`
- checkpoint / training step: exp0019 -> 5 smoke steps -> reload -> 7 more smoke steps
- decision this evaluation was meant to inform: confirm the training/checkpoint pipeline works correctly on a fresh machine before any real candidate training
- exact task/suite/difficulty coverage: `libero_uncond_2cam224_1e-4` base training path (no multi-embodiment path yet — that doesn't exist until the first candidate)
- why this panel was used: exercises dataset loading, distributed launch (DeepSpeed ZeRO-1), forward/backward, optimizer step, checkpoint save, and checkpoint reload — the full training-infra surface
- parent/reference checkpoint and matching result: n/a (infra check, not a performance comparison)
- candidate result: **pass** — training: 5 steps, losses `0.2085->0.3249->0.3274->0.2167->0.3261`, no NaN; checkpoint written (`step_000005.pt`). Reload: 7 further steps, losses `0.7086->1.3866->0.6076->0.2169->0.5582->0.5121->0.4213`, no NaN, no shape/key errors; confirmed weights-only resume restarts the step counter (expected/documented behavior); full-state directory checkpoint also validated.
- trials/episodes per task: n/a
- exact command/config: see `research/RUNBOOK.md` "Training smoke"/"Reload/resume smoke"
- raw results path: `checkpoints/train_smoke2.log`, `checkpoints/reload_smoke.log` (run directories cleaned up after confirming results, per retention policy)
- runtime: a few minutes total
- validity checks: loss values sane and non-NaN throughout both runs; checkpoint files confirmed present before cleanup
- decision enabled by this evidence: training/checkpoint infra confirmed ready for real candidate training
- reason: real optimizer updates with sane losses on both the initial run and the reloaded-checkpoint continuation is direct, non-circumstantial evidence the pipeline works

### Task-level evidence

LIBERO-Spatial per-task: `libero_spatial_4` ("bowl in top drawer") is the only sub-100% task (66.67%, 2/3) — this exactly matches exp0019's own documented history of this being its targeted weak task (76%->88% across evaluation precisions in the parent project). No other task showed unexpected weakness.

### Evaluation validity

All requested tasks completed (10/10 LIBERO-Spatial), checkpoint identity was verified independently (byte size + hash) before use, output files were validated against the manager's own summary (aggregate recomputed matches). RoboTwin/training smoke tests are infra checks, not performance evidence, and are labeled as such.

## 8. Comparison and interpretation

This is the parent baseline — there is no prior candidate to compare against. The key finding is that fresh-machine setup faithfully reproduces the inherited parent's real behavior (LIBERO-Spatial near-exact match, including reproducing the exact same weak task at a consistent rate), which is strong evidence that training/evaluation results going forward will be trustworthy and not artifacts of a broken environment.

## 9. Decision

- **Decision:** `PROMOTE` (as the accepted starting parent/baseline for the research line — not a new candidate)
- **Canonical RoboTwin evidence available:** no (not yet run; setup only verified the simulator stack, no policy-in-the-loop RoboTwin evaluation yet)
- **All five LIBERO >=90% canonical:** not evaluated this session (inherited canonical record from the prior project shows all five >=90%; a 3-trial Spatial sentinel this session scored 96.67%, consistent with that record)
- **Reason:** exp0019 is verified, its code lineage is verified, and the fresh machine faithfully reproduces its behavior — ready to serve as the starting point for the multi-embodiment research loop.
- **Checkpoint/branch to preserve:** exp0019 (`cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`, already durably stored); branch `autoresearch/robotwin-multiembodiment-v1`.
- **Next main-line parent:** exp0019, until the first multi-embodiment candidate is evaluated and promoted.

## 10. What this changes for the next experiment

The setup is fully validated; `$choose-fastwam-experiment` can now select the initial candidate. The literature review points strongly toward: (1) setting `action_target_dim=14`/`state_target_dim=14` in both data configs to activate the already-implemented `ConcatLeftAlign` padding, (2) implementing the precise two-level per-channel masked loss (mask-then-mean-per-channel, then mean over valid-channel-count, not all K channels) in `fastwam.py::training_loss` (mirrored in `fastwam_idm.py`), (3) a custom checkpoint-expansion step to grow `action_encoder`/`head`/`proprio_encoder` from exp0019's 7/8-dim shapes to 14/14 while preserving inherited weights exactly, (4) a Qwen-VLA-style textual embodiment-conditioning prompt, and (5) roughly 1:1 new(RoboTwin):replay(LIBERO) data mixing. The known LIBERO-90 data gap (no downloadable lerobot-format training data for LIBERO-90, only Spatial/Object/Goal/Long-10) is an open blocker for full LIBERO-90 replay and should be resolved or explicitly worked around in the first candidate's data-mixing design. RoboTwin's exact per-arm action-channel semantics (delta vs. absolute) also remains to be confirmed before finalizing the masking/normalization design.

## 11. Artifacts

- training log: `checkpoints/train_smoke2.log`, `checkpoints/reload_smoke.log` (smoke only; no real candidate training yet)
- intermediate checkpoints: none preserved (smoke-test checkpoints deleted after verification, per retention policy)
- selected checkpoint: exp0019 (`checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt`, local; durable copy already on `cheikh025/ASR`)
- HF remote checkpoint path + verification: `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`, verified present via `list_repo_files`, downloaded, byte-size + sha256 confirmed
- configs: none new yet (multi-embodiment configs not yet created)
- action/mask/normalization implementation tests: none yet (planned for the first candidate)
- raw RoboTwin evaluations: `checkpoints/robotwin_env_smoke_test.py` (environment-only, no policy)
- raw LIBERO evaluations: `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260817_221019/`
- parsed task/suite metrics: `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260817_221019/summary.json`
- videos/rollouts used for diagnosis: per-episode `.mp4` files under the same `evaluate_results/libero/...` run directory
- hardware/software snapshot: `research/progress/system_0000_parent_baseline.json`
- dependency snapshot: same file
- diagnostic scripts/results: `checkpoints/robotwin_env_smoke_test.py`

## 12. Reproducibility checklist

- [x] exact candidate commit and RoboTwin branch recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed (research-control file integration only — no FastWAM source changes)
- [x] multi-embodiment representation/mask/normalization details recorded (planned, not yet implemented)
- [x] dataset mixture and retention strategy recorded (planned, not yet implemented)
- [ ] initial training plan recorded — n/a, this is setup not a candidate
- [x] training configuration/command recorded (smoke tests)
- [x] hardware/software environment recorded
- [x] intermediate checkpoints/progress decisions recorded when used (smoke tests only)
- [x] logs/checkpoint paths recorded
- [x] remote checkpoint path verified (exp0019, pre-existing)
- [x] every evaluation event has benchmark/purpose/settings/raw results
- [ ] canonical RoboTwin metrics recorded — n/a, not yet run
- [ ] five LIBERO suite metrics recorded (canonical) — n/a this session, inherited record used; 3-trial Spatial sentinel recorded
- [x] task-level evidence preserved when relevant
- [x] final decision/reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
