# PROGRESS_0000 — BASELINE

- **Experiment ID:** 0000_baseline
- **Status:** `PROMOTE`
- **Created:** 2026-08-07
- **Updated:** 2026-08-08
- **Parent experiment:** none
- **Parent checkpoint:** none (released FastWAM checkpoint)
- **Selected candidate checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (released, unmodified)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `45d8e1458921d83f8ad6cf9ce993d371208dabd0` (FastWAM fork `cheikh025/FastWAM`, matches upstream `yuantianyuan01/FastWAM` at time of clone)

## 1. Result at a glance

Measures the released FastWAM LIBERO checkpoint (`libero_uncond_2cam224.pt`), unmodified, on all five target suites using this project's canonical evaluation protocol (50 trials/task). This is the fixed comparison point for every future candidate.

| Suite | Success | Paper (reported) | Target |
|---|---:|---:|---:|
| **LIBERO-90** | **15.40% (693/4500)** | n/a | >=90% |
| LIBERO-Spatial | 96.60% (483/500) | 98.2% | >=90% |
| LIBERO-Object | 99.40% (497/500) | 100.0% | >=90% |
| LIBERO-Goal | 96.80% (484/500) | 97.0% | >=90% |
| LIBERO-Long / LIBERO-10 | 94.60% (473/500) | 95.2% | >=90% |

All four original suites reproduce within ~1.6 points of the published paper numbers (strong local-pipeline sanity check) and comfortably clear the 90% retention floor. LIBERO-90 zero-shot is at 15.4% — well below target but not zero: 25/90 tasks show meaningful transfer (mostly object pick-and-place into a basket/tray, resembling LIBERO-Object's task pattern, and stove/microwave/drawer-closing actions resembling LIBERO-Goal), while 65/90 tasks — including the entire "put a book in a caddy compartment" category (tasks 73-89, 16 tasks) — show zero transfer. See Section 7 task-level evidence.

## 2. Research state before experiment

No prior state — this is the first evaluation of the project. Public Fast-WAM paper numbers (Spatial 98.2, Object 100.0, Goal 97.0, Long 95.2) provide context but are not substituted for this local measurement, per project rules.

## 3. Candidate design

### Modifications

None — this is the unmodified released checkpoint, used to establish the local baseline before any research intervention.

### Why this candidate

Required first step per `CLAUDE.md`/`AUTORESEARCH.md`: the locally measured baseline is authoritative and must be established before any model-improvement experiment.

### What to watch

- Whether the four original suites reproduce close to the published paper numbers on this machine/evaluation setup (sanity check on the local pipeline).
- LIBERO-90 zero-shot success rate — expected to be low/near-zero since the released checkpoint was never trained on these 90 tasks; this is the headline number the whole project is trying to raise to >=90%.

### Initial compute plan

- Initial training budget: none (no training in this experiment).
- Checkpoint/save plan: not applicable.
- When a progress check might be useful: not applicable — this is a single fixed canonical evaluation, not a training run.
- Expected training/evaluation cost: see Section 6/7 — full canonical 5-suite protocol, 50 trials/task, 130 tasks total (10+10+10+10+90), run on 4x A100-80GB with 2 concurrent eval workers/GPU (8 total). Estimated ~10-15 hours wall-clock, dominated by LIBERO-90 (90 zero-shot tasks, mostly running to the full step budget rather than terminating early on success).

## 4. Exact code and configuration state

- Git commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: dirty — see Files changed (local-only fixes to the separate LIBERO checkout and repo data assets, not to FastWAM's tracked source; FastWAM repo itself is clean at this commit for this evaluation).
- Files changed (all local-environment/checkout fixes, not committed to FastWAM):
  - `LIBERO/libero/__init__.py` created (empty) — fixes a missing top-level `__init__.py` in the official `Lifelong-Robot-Learning/LIBERO` repo that made `pip install -e .` silently install nothing (`setuptools.find_packages()` found zero packages).
  - `LIBERO/libero/libero/benchmark/__init__.py:164` — added `weights_only=False` to `torch.load(init_states_path)`, required for PyTorch >=2.6 compatibility with LIBERO's own trusted init-state files.
  - Neither fix touches FastWAM's own source tree.
- Training config(s): not applicable (no training).
- Config overrides for evaluation: `task=libero_uncond_2cam224_1e-4`, `MULTIRUN.num_gpus=4`, `MULTIRUN.max_tasks_per_gpu=2`, `MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10,libero_90]`, `EVALUATION.num_trials=50`.
- Dataset config(s): not applicable to evaluation (no training data loaded); eval uses live LIBERO env rollouts + the released `libero_uncond_2cam224_dataset_stats.json` for action/state normalization.
- Sampler/mixing configuration: not applicable.
- Model/trainable-module configuration: not applicable (inference only, no gradient updates).
- Optimizer / LR / scheduler: not applicable.
- Batch size / gradient accumulation / effective batch: not applicable (single-episode rollouts, `env_num=1`).
- Initial training steps / epochs / budget: not applicable.
- Checkpoint/save cadence: not applicable.
- Random seed(s): LIBERO's own fixed initial-state set per task (official benchmark semantics, not separately seeded by us).
- Resume source: not applicable — `ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` loaded fresh for each eval worker via `FastWAM.load_checkpoint`.

## 5. Hardware and software environment

### GPU

- GPU count: 4
- GPU model(s): NVIDIA A100-SXM4-80GB
- Memory per GPU: 79.25 GiB
- NVIDIA driver: 580.105.08
- CUDA runtime/toolkit: system nvcc 13.0 (unused directly); PyTorch wheel cu128 (torch.version.cuda=12.8)
- `CUDA_VISIBLE_DEVICES` / world size: all 4 GPUs visible; evaluation uses `run_libero_parallel_test.sh`'s own per-task `CUDA_VISIBLE_DEVICES=<gpu_id>` assignment, 2 concurrent single-GPU eval processes per GPU (8 total concurrent workers) — empirically determined safe ceiling (see Section 6 anomalies note: 3+ concurrent workers/GPU intermittently approached or exceeded the 79.25GiB limit during model construction+rollout).

### CPU / RAM / storage

- CPU model: AMD EPYC 7532 32-Core Processor (128 logical CPUs)
- Logical CPU count: 128
- System RAM: 1877.9 GiB
- Relevant disk total/free before run: 1016 GiB total, ~956 GiB free before launch

### Software

- OS / kernel: Ubuntu 24.04.4 LTS, kernel 6.8.0-90-generic
- Python executable and version: `/workspace/venv-fastwam/bin/python`, 3.10.20
- PyTorch version: 2.7.1+cu128
- PyTorch CUDA version: 12.8
- cuDNN version: 90701
- DeepSpeed version: 0.18.5 (not used for evaluation, only training)
- Accelerate version: 1.12.0 (not used for evaluation)
- FastWAM repository commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Environment/venv identifier: `/workspace/venv-fastwam` (dedicated venv built for this project; base-image `/venv/main` left untouched)
- Dependency snapshot path: `research/progress/system_baseline.json` (+ `research/progress/system_baseline.pip_freeze.txt`)

### Setup validation (baseline report only)

- **Evaluation smoke test**: `python experiments/libero/eval_libero_single.py task=libero_uncond_2cam224_1e-4 ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 EVALUATION.num_trials=1 gpu_id=0` — PASSED (1/1 success, 124s, `results.json` written and parsed correctly). Required fixing the two LIBERO bugs noted in Section 4 first.
- **Training smoke test**: `bash scripts/train_zero1.sh 4 task=libero_uncond_2cam224_1e-4 resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt max_steps=6 save_every=3 ...` — PASSED (6/6 steps completed, checkpoints written at step 3 and 6). Note: the same run with only 2 GPUs OOM'd during the optimizer step — this ~6.5B-param model (5.00B video expert + 1.02B action expert) needs >=4 GPUs under ZeRO-1 on 80GB A100s, matching the README's expectation of an 8-GPU node. Also required first running `scripts/preprocess_action_dit_backbone.py` (README "Model Preparation" step 2, not needed for eval).
- **Checkpoint reload/resume smoke test**: resumed from `checkpoints/state/step_000006` with `max_steps=8` — PASSED (optimizer/scheduler/dataloader-sampler/RNG state all restored, training continued cleanly at step 7 then 8).

Full detail and exact fix commits: `research/RUNBOOK.md` (Setup smoke tests section), `research/NOTES.md` (Setup smoke test fixes section).

## 6. Training execution and control timeline

Not applicable — no training in this experiment (baseline = unmodified released checkpoint).

### Intermediate checkpoints and progress decisions

Not applicable.

### Why training ended

Not applicable — no training.

### Training anomalies

Not applicable (training). For **evaluation** parallelism, empirical GPU-memory testing before the full run found:
- 1 concurrent eval worker/GPU: ~25 GiB peak (steady state during rollout, higher than the ~10-20GiB seen during model construction alone).
- 2 concurrent workers/GPU: ramped safely to ~28-50 GiB observed during the actual production run's ramp-up — used as the production setting.
- 3 concurrent workers/GPU: survived one isolated test at ~76 GiB peak (out of 79.25 GiB) — only ~3 GiB headroom, judged too risky for an unattended multi-hour run (a single OOM aborts the *entire* `run_libero_parallel_test.sh` scheduler, not just the offending task).
- 4, 5, and 6 concurrent workers/GPU: reliably OOM'd (all concurrent processes in the batch crashed simultaneously when combined memory hit ~79 GiB during model construction, before any of them reached steady-state rollout memory).
- Final production setting: `MULTIRUN.max_tasks_per_gpu=2` (8 total concurrent workers across 4 GPUs).

## 7. Evaluation events

### Event 1 — `canonical`

- Checkpoint / training step: released `libero_uncond_2cam224.pt` (unmodified)
- Decision this evaluation was meant to inform: establish the fixed baseline comparison point for the whole project; determine the true starting point (expected near-zero) for the primary LIBERO-90 target.
- Exact task subset / suite coverage: all 130 tasks across the 5 target suites (libero_spatial=10, libero_object=10, libero_goal=10, libero_10=10, libero_90=90)
- Why this panel was used: this is the canonical protocol itself (full fixed five-suite evaluation), the only evidence type that can establish/promote a checkpoint per project rules.
- Parent/reference checkpoint and matching result: none (first evaluation); public paper numbers for context only (Spatial 98.2, Object 100.0, Goal 97.0, Long 95.2, four-suite avg 97.6 — not a substitute for this measurement).
- Candidate result: **COMPLETE** — all 130 tasks (6500 episodes), see Section 1/7 for full scores.
- Trials per task: 50
- Exact command/config (initial 5-suite launch, 2026-08-08 02:34:06 UTC):
  ```bash
  python experiments/libero/run_libero_manager.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
    MULTIRUN.num_gpus=4 \
    MULTIRUN.max_tasks_per_gpu=2 \
    MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10,libero_90] \
    EVALUATION.num_trials=50 \
    EVALUATION.output_dir=./evaluate_results/baseline_0000
  ```
  **Interruption**: the entire process tree (including the tmux server the eval scheduler depends on) was torn down by an external session/container restart at approximately 2026-08-08 04:17-04:47 UTC (not a task failure — `failed_tasks.txt` was empty, log simply stops mid-status-print with 8 healthy tasks running). At the time of interruption, Spatial/Object/Goal/Long were already fully complete (40/40, results safely on disk) and LIBERO-90 had 0/90 complete (8 tasks in flight, no partial results for those). Relaunched at 2026-08-08 04:47:54 UTC targeting only the remaining suite (same `output_dir`, so results merge into the same directory; the 4 completed suites were not re-run since `run_libero_manager.py` generates its task list fresh from `MULTIRUN.task_suite_names` with no re-run of already-complete tasks when a suite is omitted entirely):

  **Second issue (found and fixed 2026-08-08 ~10:48-10:55 UTC)**: `run_libero_parallel_test.sh`'s completion check counts *all* `gpu*_task*_results.json` files anywhere under `output_dir`, not scoped to the current invocation's task list — see `research/NOTES.md` "Bug found in `run_libero_parallel_test.sh`'s completion check". Because the LIBERO-90 relaunch reused the same `output_dir` that still had the 4 other suites' 40 old result files, the scheduler falsely declared `"All tasks are complete!"` at 51/90 real LIBERO-90 results (40 old + 51 new = 91 >= 90), running `summarize_results.py` and exiting while 6 tasks were still legitimately in-flight and 39 had never been launched. Verified the true missing set directly from the result files (`glob`+regex against `range(90)`), let the 6 in-flight tasks (50-54, 56) finish naturally (tmux session survived the parent script's exit), confirmed 57/90 genuinely done, then discovered a **second, unrelated tmux issue**: re-invoking the scheduler under the same hardcoded `SESSION_NAME="libero_test_v3"` killed the only remaining tmux session, which caused the tmux **server itself** to exit (`"server exited unexpectedly"`), and stale socket state at `/tmp/tmux-1002/default` (left over from the original container-restart interruption) prevented a clean new server from starting — the scheduler's log looked like it was launching 33 new tasks but zero actual GPU processes were running. Fixed by `rm -rf /tmp/tmux-1002/` and verifying a fresh `tmux new-session`/`kill-session` cycle worked before retrying. Relaunched the 33 genuinely-missing tasks (57-89) directly via `run_libero_parallel_test.sh` (bypassing `run_libero_manager.py`, which cannot filter to a task subset) with a **separate `output_dir`** (`evaluate_results/baseline_0000_libero90_remainder/`) specifically to avoid repeating the false-completion bug; will merge its `libero_90/gpu*_task*_results.json` files into `evaluate_results/baseline_0000/libero_90/` once complete, then re-run `summarize_results.py` over the merged directory for final numbers.
  ```bash
  python experiments/libero/run_libero_manager.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
    MULTIRUN.num_gpus=4 \
    MULTIRUN.max_tasks_per_gpu=2 \
    MULTIRUN.task_suite_names=[libero_90] \
    EVALUATION.num_trials=50 \
    EVALUATION.output_dir=./evaluate_results/baseline_0000
  ```
- Raw results path: `evaluate_results/baseline_0000/` (per-task `gpu*_task*_results.json` per suite, merged from the main run + the `libero_90` remainder recovery run — see interruption notes above; `tasks.txt`, `manager_config.yaml`)
- Runtime: ~12.5 hours wall-clock total from initial launch (2026-08-08 02:34 UTC) to final completion (~2026-08-08 15:00 UTC), including the mid-run interruption/recovery overhead (~40 min lost to the session restart + false-completion/tmux-crash recovery detailed above). Pure eval compute: 4 suites ~57 min wall-clock (2000 episodes, 8-way parallel); LIBERO-90 ~10 hours wall-clock (4500 episodes, 8-way parallel, dominated by near-100%-failure-rate full-700-step episodes).
- Validity checks: all 130/130 tasks present with exactly 50/50 episodes each (verified via direct file count + regex task-ID coverage check, not just the scheduler's own — buggy — completion message); `failed_tasks.txt` empty in both the main and remainder runs; four-suite scores cross-checked against published paper numbers (within ~1.6pp, confirms pipeline correctness).
- Decision enabled by this evidence: sets the fixed baseline for all subsequent PROMOTE/REJECT/BRANCH/RETEST decisions; the four original suites additionally serve as the local sanity check against the published paper numbers, now confirmed passing.
- Reason: baseline established; hands off to `$choose-fastwam-experiment` for the first LIBERO-90 candidate.

### Task-level evidence

Final suite-level scores (50 trials/task, all 130 tasks complete, aggregated from raw per-task `results.json`):

| Suite | Success | Paper (reported) |
|---|---:|---:|
| LIBERO-Spatial | 483/500 = **96.6%** | 98.2% |
| LIBERO-Object | 497/500 = **99.4%** | 100.0% |
| LIBERO-Goal | 484/500 = **96.8%** | 97.0% |
| LIBERO-Long / LIBERO-10 | 473/500 = **94.6%** | 95.2% |
| LIBERO-90 | 693/4500 = **15.4%** | n/a (not in paper) |

The four original suites track within ~1.6 points of the published paper numbers (good sanity check on the local evaluation pipeline) and are all comfortably above the 90% retention threshold.

**LIBERO-90 per-task breakdown** — 25/90 tasks have nonzero zero-shot success, 65/90 are exactly 0%:

| task_id | success | description |
|---:|---:|---|
| 9 | 26% | put the black bowl on the plate |
| 12 | 4% | put the black bowl at the back on the plate |
| 19 | 96% | put the moka pot on the stove |
| 20 | 38% | turn on the stove |
| 22 | 82% | close the bottom drawer of the cabinet |
| 24 | 100% | put the black bowl in the bottom drawer of the cabinet |
| 28 | 76% | close the top drawer of the cabinet |
| 33 | 86% | close the microwave |
| 38 | 90% | put the right moka pot on the stove |
| 44 | 26% | turn on the stove |
| 46 | 70% | pick up the alphabet soup and put it in the basket |
| 47 | 82% | pick up the cream cheese box and put it in the basket |
| 50 | 54% | pick up the alphabet soup and put it in the basket |
| 51 | 34% | pick up the butter and put it in the basket |
| 54 | 26% | pick up the tomato sauce and put it in the basket |
| 55 | 100% | pick up the alphabet soup and put it in the tray |
| 56 | 34% | pick up the butter and put it in the tray |
| 57 | 98% | pick up the cream cheese and put it in the tray |
| 59 | 18% | pick up the tomato sauce and put it in the tray |
| 60 | 100% | pick up the black bowl on the left and put it in the tray |
| 64 | 8% | stack the right bowl on the left bowl and place them in the tray |
| 68 | 54% | put the yellow and white mug on the right plate |
| 70 | 74% | put the chocolate pudding to the right of the plate |
| 72 | 8% | put the white mug on the plate |
| 77 | 2% | pick up the book and place it in the back compartment of the caddy |

Qualitative pattern: strong transfer on "pick up X and put in basket/tray" tasks (visually/linguistically close to LIBERO-Object's own task template) and on stove/microwave/drawer-closing single-step actions (close to LIBERO-Goal's template). **Zero transfer on the entire "put a book in a caddy compartment" category** (tasks 73-89 except task 77 at 2% — 16 tasks, all involving a caddy object/scene not represented in the 4 training suites) and on most black-bowl-on-top-of-cabinet / drawer-opening / mug-arrangement tasks. Full per-task results (all 90) preserved in `evaluate_results/baseline_0000/libero_90/gpu*_task*_results.json`.

### Evaluation validity

Confirmed: 130/130 tasks present, 50/50 episodes each, zero entries in either run's `failed_tasks.txt`, checkpoint identity verified via the eval log (`Loaded checkpoint via model.load_checkpoint: ./checkpoints/fastwam_release/libero_uncond_2cam224.pt`), four-suite scores cross-validated against the published paper numbers. One known irregularity: the LIBERO-90 suite's results were assembled from two separate scheduler invocations (main run tasks 0-56ish + remainder run tasks 57-89, see interruption notes) due to the mid-run session restart and a scheduler completion-check bug — verified by direct task-ID-coverage check (not the scheduler's own completion message) that all 90 task IDs are present exactly once with no duplicates or gaps before merging. No other duplicate/missing-task risk identified.

## 8. Comparison and interpretation

No prior checkpoint to compare against — this is the first measurement. Interpretation of the standalone result:

- **The four original suites are healthy and well above the retention floor**, closely matching the paper's reported numbers (max deviation 1.6pp on Spatial). This validates both that the released checkpoint loaded/runs correctly in this environment and that the local evaluation pipeline (action de-normalization, gripper handling, replan/chunking, LIBERO env semantics) is implemented correctly — a necessary precondition for trusting any future candidate's numbers.
- **LIBERO-90 zero-shot success (15.4%) is well below the 90% target but far from zero**, and the failure pattern is not uniform: it concentrates by *task category*, not randomly. Tasks whose language/object/action pattern closely resembles the 4 trained suites (pick-and-place into a basket/tray — echoing LIBERO-Object; single-step stove/microwave/drawer actions — echoing LIBERO-Goal) already transfer substantially (many >70%, several >=90%). Tasks involving object/scene combinations absent from the training mix (the entire book-in-caddy category; most black-bowl-on-cabinet, drawer-opening, and mug-arrangement tasks) transfer at or near 0%.
- This suggests the gap is not purely a matter of "more training on the same recipe" — it's plausibly concentrated in specific under-represented scene/object categories, which is useful signal for how a LIBERO-90 training candidate's data mix or task coverage should be designed.

## 9. Decision

- **Decision:** not applicable — a baseline is not promoted/rejected/branched, it establishes the starting point.
- **Retention gate passed:** yes, for the released checkpoint against itself (all four original suites >=90%, trivially — no modification has been made yet).
- **Reason:** baseline established with full canonical evidence; ready to hand off to the normal research loop.
- **Checkpoint/branch to preserve:** the released checkpoint remains the parent for the first real candidate experiment; no new checkpoint is produced by this baseline measurement.
- **Next main-line parent:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (until a candidate is promoted past it)

## 10. What this changes for the next experiment

- The four original suites reproduce the paper closely — no evaluation-pipeline concerns going into candidate work.
- LIBERO-90 needs to close an ~75-point gap (15.4% -> >=90%), but the per-task pattern shows this isn't uniform: roughly 28% of tasks (25/90) already have partial-to-strong zero-shot competence, while categories entirely absent from the 4-suite training data (book/caddy interactions especially) show zero transfer and will need real training signal on those specific scenes/objects, not just more of the same data. This should inform the first LIBERO-90 training candidate's data-mixing/coverage design (a `$choose-fastwam-experiment` decision).
- Per-episode cost on LIBERO-90 is high while success is low (near-100%-failure-rate episodes run the full 700-step budget, ~54 min/task/50-trials, and trials run strictly sequentially per task with no vectorized-env parallelism in the current eval code) — worth keeping in mind when planning progress-check/candidate-screen panel sizes so evaluation doesn't dominate the research budget; a fine-tuned checkpoint with higher LIBERO-90 success should evaluate noticeably faster (more early episode termination) than this baseline did.
- Two infra issues were found and fixed/worked around during this run (LIBERO packaging bug, LIBERO torch.load compat, `run_libero_manager.py`'s output_dir-reuse false-completion bug, and a tmux stale-socket issue) — all documented in `research/NOTES.md`; future evaluation invocations should use a fresh `output_dir` per invocation rather than reusing one across separate suite-subset runs.

## 11. Artifacts

- training log: not applicable (no training)
- intermediate checkpoint(s): not applicable
- selected checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (unmodified, not re-uploaded — already the canonical release, already on `yuanty/fastwam`)
- Hugging Face remote checkpoint path: not applicable (unmodified release checkpoint, not a research output)
- config(s): `evaluate_results/baseline_0000/manager_config.yaml`, `evaluate_results/baseline_0000/tasks.txt`
- raw progress-evaluation outputs: not applicable (no training progress checks)
- raw candidate/confirmation/canonical evaluation outputs: `evaluate_results/baseline_0000/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json` (libero_90 merged from `evaluate_results/baseline_0000_libero90_remainder/` — that directory preserved as-is for provenance, not deleted)
- parsed task/suite metrics: `experiments/libero/summarize_results.py --output_dir=evaluate_results/baseline_0000` output captured in Section 1/7 above; re-derivable any time from the raw per-task JSONs.
- videos/rollouts used for diagnosis: none specifically extracted for this report; episode MP4s were written per-episode under each suite's `videos/` subdir by `eval_libero_single.py` and remain available on disk for later diagnosis if needed.
- hardware/software snapshot: `research/progress/system_baseline.json`, `research/progress/system_baseline.pip_freeze.txt`
- dependency snapshot: see above (same file)
- diagnostic scripts/results created specifically for this experiment: none beyond the informal parallelism-calibration probes described in Section 6 (not preserved as formal artifacts, their output directories were cleaned up after use)

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded (n/a — this is the parent)
- [x] intentional modifications listed (none — unmodified release)
- [x] initial training plan recorded (n/a — no training)
- [x] training configuration and command recorded (n/a — no training)
- [x] hardware/software environment recorded
- [x] intermediate checkpoints and progress decisions recorded when used (n/a)
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created (n/a — unmodified release checkpoint)
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
