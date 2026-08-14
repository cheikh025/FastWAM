# FastWAM Research Runbook

This file is filled and verified by `$setup-fastwam-research` from the actual repository and machine.

Do not guess commands when the repository already defines them.

## Repository

- repository root: `/workspace/fastwam_libero90/FastWAM`
- origin (working fork): `https://github.com/cheikh025/FastWAM.git`
- upstream (official): `https://github.com/yuantianyuan01/FastWAM.git`
- commit at research start: `45d8e1458921d83f8ad6cf9ce993d371208dabd0` (origin/main, "Merge pull request #20 from yuantianyuan01/dev/fix_gpu_oom")
- research branch: `autoresearch/libero90-v1` (created from the commit above)

## Environment

- Python/conda/venv: dedicated venv at `/workspace/venv-fastwam` (Python 3.10, created via `uv venv --python 3.10`; base-image `/venv/main` (Python 3.12, torch 2.12/cu130) is left untouched since FastWAM pins an older, narrower dependency set that would otherwise conflict). Activate with `source /workspace/venv-fastwam/bin/activate`.
- uv cache/python-install dirs redirected to `/workspace/.uv/{cache,python_install,python_bin}` because `/venv` and `/.uv` are root-owned (no passwordless sudo on this instance) — always export `UV_CACHE_DIR=/workspace/.uv/cache` before `uv pip` calls in this venv, or use plain `pip`.
- Core install commands actually run:
  ```bash
  export UV_CACHE_DIR=/workspace/.uv/cache
  uv venv /workspace/venv-fastwam --python 3.10
  source /workspace/venv-fastwam/bin/activate
  uv pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
  cd /workspace/fastwam_libero90/FastWAM && uv pip install -e .
  ```
- required environment variables: `HF_TOKEN` (HF Hub auth, already set; verified via `huggingface_hub.whoami()` -> account `cheikh025`). `HF_HOME=/workspace/.hf_home` (instance default). No other required env vars found in the codebase (wandb is `enabled: false` by default in `configs/train.yaml`, so no `WANDB_API_KEY` needed for smoke tests).
- GPU count/model: 4x NVIDIA A100-SXM4-80GB, driver 580.105.08 (system CUDA 13.0; PyTorch wheel brings its own cu128 runtime, compatible via minor-version compatibility).
- important dependency versions (pinned by `pyproject.toml`, installed as-is): torch==2.7.1+cu128, torchvision==0.22.1+cu128, transformers==4.49.0, accelerate==1.12.0, deepspeed==0.18.5, hydra-core==1.3.2, numpy==1.26.4, datasets==3.6.0, huggingface-hub==0.29.2.
- LIBERO/MuJoCo setup: `<fill after LIBERO install completes — see task #4>`. Plan: official `Lifelong-Robot-Learning/LIBERO` repo cloned to `/workspace/fastwam_libero90/LIBERO`; install `libero` package itself via `pip install -e .` (its `setup.py` has `install_requires=[]`, so no deps forced), then manually install compatible deps into `/workspace/venv-fastwam` alongside FastWAM's pins rather than LIBERO's `requirements.txt` (which pins python 3.8-era versions that conflict with FastWAM, e.g. numpy==1.22.4, transformers==4.21.1, hydra-core==1.2.0): `robosuite==1.4.0` (verified via PyPI setup.py: requires `mujoco>=2.3.0`, no upper bound — already migrated off `mujoco-py` to native `mujoco` bindings in 1.4, so compatible with `mujoco==3.3.2`), `bddl==1.0.1`, `easydict`, `cloudpickle`, plus whatever `gym`/`gymnasium` LIBERO's env wrappers actually import (verify at install time), and `mujoco==3.3.2` per FastWAM README. `eval_libero_single.py`/`run_libero_manager.py` only import `libero.libero.{benchmark,get_libero_path}` and `libero.libero.envs.{OffScreenRenderEnv,SubprocVectorEnv}` — no `robomimic`/`wandb`/lifelong-learning-algo deps needed for our use.

## Starting checkpoint

- checkpoint path: `/workspace/fastwam_libero90/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` (~11.2 GB, downloaded from `yuanty/fastwam` via `hf download`)
- dataset stats path: `/workspace/fastwam_libero90/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- task/config used by released checkpoint: `task=libero_uncond_2cam224_1e-4` (`configs/task/libero_uncond_2cam224_1e-4.yaml`, uses `data: libero_2cam`, `model: fastwam`)
- checkpoint format: single consolidated `.pt` dict `{"mot": <state_dict>, "step":…, "torch_dtype":…, "proprio_encoder": <optional>}`, loaded via `FastWAM.load_checkpoint` (`src/fastwam/models/wan22/fastwam.py`) — the SAME loader/format is used both for resuming training from a weights-only path and for loading `ckpt=` at LIBERO eval time (see Training section below).

## Data

- four-suite training data: downloaded and extracted from HF dataset `yuanty/LIBERO-fastwam` (LeRobot v2.1, MuJoCo 3.3.2, generated for the Fast-WAM paper — NOT an official upstream LIBERO dump) into `data/libero_mujoco3.3.2/{libero_spatial,libero_object,libero_goal,libero_10}_no_noops_lerobot/`. Total ~4.7GB. Config: `configs/data/libero_2cam.yaml` (`dataset_dirs` lists all four).
- LIBERO-90 data/config: **not shipped by FastWAM's dataset release.** Candidate replacement identified and being staged: `IPEC-COMMUNITY/libero_90_no_noops_lerobot` on HF (LeRobot v2.1, `no_noops`-filtered via the OpenVLA regenerate-and-filter convention — same naming/filtering convention as yuanty's 4 suites). Schema verified via `meta/info.json`: `observation.images.image` + `observation.images.wrist_image` (video, 256x256 vs the other 4 suites' 512x512 — harmless, since the image `raw_shape` assertion in `base_lerobot_dataset.py` is commented out and all images get resized to 224x224 by `train_transforms` regardless of source resolution), `observation.state` (8-dim), `action` (7-dim) — exact match to `libero_2cam.yaml`'s `shape_meta`. `robot_type: franka`, `total_episodes: 3921`, `total_tasks: 73` (unique language strings; LIBERO-90's 90 task IDs are expected to reuse repeated instruction text across scenes — **must verify this dataset's `meta/tasks.jsonl` covers all 90 canonical `libero_90` benchmark task IDs, by matching against `libero.libero.benchmark.get_benchmark_dict()["libero_90"]` task language, before using it for training** — not yet done, blocked on LIBERO install). Fallback candidate if coverage is incomplete: `jesbu1/libero_90_lerobot` (not yet inspected).
- LIBERO-90 **evaluation** requires no new data or code: `libero.libero.benchmark.get_benchmark_dict()["libero_90"]` is the official LIBERO package's own benchmark registry (bddl task files ship with the `libero` package itself), and `experiments/libero/eval_libero_single.py` already has `"libero_90": 700` wired into `_get_max_steps` (same horizon as libero_10/Long) and `summarize_results.py` already lists `libero_90` among its suites. Evaluating libero_90 only requires adding `libero_90` to `MULTIRUN.task_suite_names` (`configs/sim_libero.yaml`) or passing it as a Hydra override — no repo modification needed.
- preprocessing/text-embedding requirements: training requires a precomputed T5 text-embedding cache (`scripts/precompute_text_embeds.py`), written to `data/text_embeds_cache/libero` (per `libero_2cam.yaml: text_embedding_cache_dir`). It reads each dataset dir's `meta/tasks.jsonl` `"task"` field, and `RobotVideoDataset` raises `FileNotFoundError` at train time if the cache for a prompt is missing. Evaluation does NOT use this cache — `eval_libero_single.py` loads a live text encoder (`model.load_text_encoder: true` in `configs/sim_libero.yaml`) and encodes the instruction on the fly each rollout.
- normalization/statistics behavior: if `data.train.processor... ` — more precisely, `RobotVideoDataset` computes fresh min/max/mean/var/quantile stats from the training data and writes `{output_dir}/dataset_stats.json` unless a `pretrained_norm_stats` path is supplied in the data config (none of the LIBERO data configs currently set this, so every LIBERO training run computes stats fresh from whatever `dataset_dirs` are active — this matters if a LIBERO-90 candidate changes `dataset_dirs`, since stats will differ from the released checkpoint's `libero_uncond_2cam224_dataset_stats.json`). The downloaded `libero_uncond_2cam224_dataset_stats.json` is exactly this same schema and is what eval uses via `EVALUATION.dataset_stats_path` (auto-discovered by searching the checkpoint's parent directories if not passed explicitly).

## Training

Canonical training entry point:

```bash
# LIBERO, 8 GPUs originally; NPROC_PER_NODE is fully configurable (not hardcoded)
bash scripts/train_zero1.sh <NPROC_PER_NODE> task=libero_uncond_2cam224_1e-4 <extra hydra overrides>
# internally: accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
#             --num_processes <NPROC_PER_NODE> scripts/train.py output_dir=./runs/<task>/<run_id> ...
```

Required arguments/overrides:

- parent checkpoint: pass `resume=<path>` — a `.pt` file loads weights only (no optimizer/step restore, used to initialize from the released checkpoint or any other candidate's weights); a directory loads full accelerate/deepspeed training state (optimizer/step/dataloader position) for continuing an interrupted run of the SAME candidate.
- output directory: `output_dir=./runs/<task_basename>/<run_id>` — auto-derived by `train_zero1.sh` from `task=` and a timestamp (or `RUN_ID` env var); pass `output_dir=...` explicitly to control it.
- resume behavior: see above (`resume: null` default in `configs/train.yaml`).
- save/checkpoint behavior: `save_every` (steps) controls checkpoint frequency; task config `libero_uncond_2cam224_1e-4.yaml` sets `save_every: 2000`, `eval_every: 200`, `log_every: 10`, `num_epochs: 10` (step-based training loop; `num_epochs` only feeds a total-step estimate when `max_steps` is null).
- distributed launcher: `accelerate launch` with `scripts/accelerate_configs/accelerate_zero1_ds.yaml` (DeepSpeed ZeRO stage 1, no offload, `mixed_precision` controlled by the `Trainer`'s `Accelerator(mixed_precision=...)` — `train.yaml` sets `mixed_precision: "bf16"`, not by the accelerate config file itself).

How to preserve/load intermediate checkpoints:

- checkpoint naming/path: under `<output_dir>/checkpoints/weights/step_XXXXXX.pt` (weights-only, same format as the released checkpoint — usable directly for LIBERO eval `ckpt=`) and `<output_dir>/checkpoints/state/step_XXXXXX/` (full accelerate/deepspeed state dir, for `resume=<dir>`).
- save cadence controls: `save_every=<N>` (steps), override per experiment as needed.
- how to resume from an intermediate checkpoint: `resume=<output_dir>/checkpoints/state/step_XXXXXX` (full resume, same run) or `resume=<output_dir>/checkpoints/weights/step_XXXXXX.pt` (weights-only fresh start, e.g. to branch a new candidate from an intermediate point).

## Setup smoke tests

These are run during initial environment setup only, not before normal experiments. Repeat them only after a material environment/repository-interface/checkpoint-format change.

**Two upstream LIBERO bugs had to be fixed locally before any of this worked** (both are local checkout/venv fixes, not committed anywhere upstream — redo if `/workspace/fastwam_libero90/LIBERO` is ever re-cloned):
1. `LIBERO/libero/__init__.py` was missing (top-level package dir has no `__init__.py`, only its subpackages do), which made `setuptools.find_packages()` return nothing and `pip install -e .` silently install an empty package. Fixed with `touch libero/__init__.py` before installing.
2. `LIBERO/libero/libero/benchmark/__init__.py:164` (`get_task_init_states`) calls `torch.load(init_states_path)` with no `weights_only` argument; PyTorch >=2.6 defaults to `weights_only=True` and raises `UnpicklingError` on LIBERO's own trusted init-state files. Fixed by passing `weights_only=False` explicitly on that line.

Also required before training smoke tests: the README's "Model Preparation" step 2 (not needed for eval, since `configs/sim_libero.yaml` sets `skip_dit_load_from_pretrain: true`, but training's default model config does not):
```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```
This also triggers the (first-time-only, then cached) ~18.6GB Wan2.2-TI2V-5B ModelScope download into `checkpoints/Wan-AI/Wan2.2-TI2V-5B/`.

Evaluation smoke command (PASSED, 2026-08-07):

```bash
source /workspace/venv-fastwam/bin/activate
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 EVALUATION.num_trials=1 gpu_id=0 \
  EVALUATION.output_dir=./evaluate_results/smoke_test
```
Result: checkpoint loaded (5.00B video expert + 1.02B action expert params, matches paper), full 430-step rollout completed, task 0 succeeded 1/1, `evaluate_results/smoke_test/libero_spatial/gpu0_task0_results.json` written and parses correctly. A harmless `EGLError`/`libGLU.so.0 missing` warning prints during post-result context teardown (no sudo available to install `libglu1-mesa`) — occurs strictly after the results file is written, non-blocking.

Training smoke command (PASSED, 2026-08-07 — note: **2 GPUs OOM'd** on this ~6.5B-param model under ZeRO-1 even on 80GB A100s; needs at least 4 GPUs at `batch_size=1`, matching the README's expectation of an 8-GPU node):

```bash
source /workspace/venv-fastwam/bin/activate
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  max_steps=6 save_every=3 eval_every=100 log_every=1 \
  batch_size=1 num_workers=2 \
  output_dir=./runs/smoke_test/train2
```
Result: all 6 steps completed (loss logged each step, e.g. step1 loss=0.119 ... step6 loss=1.91 — noisy/small-batch, expected), checkpoints written at step 3 and step 6 to both `checkpoints/weights/step_00000{3,6}.pt` and `checkpoints/state/step_00000{3,6}/`, clean `max_steps reached` exit.

Reload/resume smoke command (PASSED, 2026-08-08):

```bash
source /workspace/venv-fastwam/bin/activate
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  resume=./runs/smoke_test/train2/checkpoints/state/step_000006 \
  max_steps=8 save_every=100 eval_every=100 log_every=1 \
  batch_size=1 num_workers=2 \
  output_dir=./runs/smoke_test/train2
```
Result: full accelerate/deepspeed state (optimizer/step/dataloader position) restored from `step_000006`, training continued to `step=8` (2 more steps) with no shape/key mismatch errors.

Smoke-test pass criteria and artifact paths:

- evaluation: checkpoint loads, env resets, rollout completes (success or failure outcome both acceptable), `evaluate_results/.../gpu*_task*_results.json` written and parseable. **VERIFIED.**
- training: forward/backward/optimizer step completes without error for a handful of steps, `checkpoints/weights/step_XXXXXX.pt` written. **VERIFIED** (requires >=4 GPUs for this model size).
- reload/resume: training resumes from the smoke checkpoint and continues without shape/key mismatch errors. **VERIFIED.**

## Canonical evaluation

Canonical FastWAM/LIBERO evaluation entry point:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=<checkpoint.pt> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  MULTIRUN.num_gpus=<N> \
  MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10,libero_90] \
  EVALUATION.num_trials=<trials_per_task>
```
This drives `experiments/libero/run_libero_parallel_test.sh`, which fans out one `eval_libero_single.py` process per `(suite, task_id)` across GPUs via `tmux`, and finishes by calling `experiments/libero/summarize_results.py --output_dir=<dir>` for suite-level aggregation.

Suites included in a full promotion evaluation:

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long / LIBERO-10
- LIBERO-90 (added to `MULTIRUN.task_suite_names`; requires no code change — see Data section)

Canonical rollout/trial settings (repository defaults, `configs/sim_libero.yaml`):

- trials per task: `EVALUATION.num_trials`, default `50` — `<confirm the trial count actually used for the canonical baseline once decided; default 50 matches the FastWAM README's released-checkpoint eval example>`
- seeds/initial-state behavior: `env.set_init_state(initial_state)` per episode, drawn from the LIBERO benchmark's own fixed initial-state set for each task (official LIBERO semantics, unmodified).
- episode horizon: `_get_max_steps(task_suite_name)` — 400 steps for spatial/object/goal, 700 for libero_10 and libero_90.
- prompt/instruction behavior: raw LIBERO task language, encoded live by the model's text encoder each rollout (`model.load_text_encoder: true`).
- action replan behavior: `EVALUATION.replan_steps` (default 10), `EVALUATION.num_steps_wait` (default 30), `EVALUATION.binarize_gripper: true`; gripper sign flip + `invert_gripper_action` applied after de-normalization (see `eval_libero_single.py`).
- metric aggregation: per-task success rate over `num_trials` episodes, aggregated to suite-level by `summarize_results.py`.

## Adaptive evaluation plan

Defined 2026-08-08 from the measured baseline (`research/progress/PROGRESS_0000_BASELINE.md`). Baseline cost data: 4 original suites (2000 episodes) took ~57 min wall-clock at 8-way parallelism; LIBERO-90 (4500 episodes, near-100% failure rate, full 700-step budget per episode, no early termination) took ~10 hours. Canonical full-protocol runs are expensive and should not be used for routine progress checks.

### Cheap progress / candidate panel

- task subset: a fixed 21-task panel — **5 LIBERO-90 tasks** spanning the observed transfer spectrum (task 24 "put black bowl in bottom drawer" — 100% baseline, easy; task 19 "put moka pot on stove" — 96% baseline, easy; task 46 "pick up alphabet soup, put in basket" — 70% baseline, medium; task 9 "put black bowl on plate" — 26% baseline, hard; task 73 "pick up book, place in front compartment of caddy" — 0% baseline, from the entirely-untransferred caddy category, hardest/most diagnostic of real new learning) + **16 retention sentinel tasks**, 4 per original suite (out of each suite's 10 tasks): libero_spatial {0,3,5,8}, libero_object {0,3,5,8}, libero_goal {0,3,5,8}, libero_10 {0,3,5,8}.
- why these tasks are informative: the 5 LIBERO-90 tasks span the full observed baseline range (0%, 26%, 70%, 96%, 100%) so a candidate's movement on this panel is informative about both whether genuinely-new categories (task 73) are being learned and whether already-transferring tasks (24, 19) are preserved/improved, not just whether the easy end saturates. **Widened from 2 to 4 sentinel tasks/suite on 2026-08-08** after experiment 0001 revealed that forgetting can be highly non-uniform *within* what looked like comparable suites (e.g. LIBERO-Goal collapsed far more than LIBERO-Spatial despite near-identical baseline scores and episode counts) — 2 tasks/suite is enough to detect a large, uniform-within-suite effect (as it did for 0001; the signal there was unambiguous, ~10⁻⁷ chance of being noise) but risks a coverage blind spot where a suite looks fine on its 2 sampled tasks while several of its other 8 tasks have quietly regressed. 4/10 tasks per suite roughly halves that blind spot at a still-modest cost.
- retention sentinel tasks/suites: 4 tasks/suite from each of the 4 original suites (16 total) — as listed above.
- trials per task: 5 (enough to detect a gross regression or a clear improvement direction; not for precise percentages — a large, unambiguous effect like 0001's doesn't need more, but do not over-interpret small percentage differences at this trial count).
- command/config: run via a custom task-file through `run_libero_parallel_test.sh` with `EVALUATION.num_trials=5`, task list = `[(libero_90,24),(libero_90,19),(libero_90,46),(libero_90,9),(libero_90,73),(libero_spatial,0),(libero_spatial,3),(libero_spatial,5),(libero_spatial,8),(libero_object,0),(libero_object,3),(libero_object,5),(libero_object,8),(libero_goal,0),(libero_goal,3),(libero_goal,5),(libero_goal,8),(libero_10,0),(libero_10,3),(libero_10,5),(libero_10,8)]` (21 tasks x 5 trials = 105 episodes).
- expected cost/runtime: roughly 1.6x the previous 13-task panel's cost given the added sentinel tasks are mostly fast (when passing) or bounded by the 400/700-step cap (when failing); expect ~50-60 min serial-equivalent, well under an hour with a handful of concurrent workers. Still cheap relative to a multi-hour training run.
- accepted-checkpoint baseline on this exact panel: task 24=100%, task19=96%, task46=70%, task9=26%, task73=0% (LIBERO-90); newly-added sentinel tasks 3/8 per suite: spatial 96%/96%, object 100%/100%, goal 88%/100%, long 96%/92% (goal task 3 at 88% is the only individual task below 90% anywhere in the baseline, though the suite average is 96.8% — a reminder that the retention gate is suite-level, not task-level; exact values in `evaluate_results/baseline_0000/*/gpu*_task*_results.json`).

This panel may be used for `progress_check` or `candidate_screen`. Direct checkpoint comparisons must use matching settings.

### Progress-check guidance

Do not use a fixed universal interval. Run a progress check when its cost is small relative to remaining training cost and it could change a decision. Given LIBERO-90 training runs will likely span many thousands of steps, a natural cadence is every 1-2k steps or at each `save_every` checkpoint, but this is a default to override, not a rule.

**Second local filesystem available for scratch (found 2026-08-09):** this instance has a second, independent filesystem beyond `/workspace` (1016GB, the persistent RAID10 volume all research artifacts live on) — the container's own root overlay (`df -h /`), ~499GB free, writable at e.g. `/home/claudeuser/...` or `/tmp/...`. It is almost certainly **ephemeral** (lost on container restart/recycle, unlike `/workspace`), so never put anything there that needs to survive — but that makes it ideal for the large, deliberately-disposable DeepSpeed ZeRO-1 full-`state/` checkpoints (~80GB each) that have twice caused `/workspace` to fill up during training (see disk-full crash note below). Plan: for future training runs, symlink `<output_dir>/checkpoints/state` to a directory on this second filesystem (e.g. `/home/claudeuser/fastwam_state_scratch/<run_id>/state`) before launching, so DeepSpeed's state saves land there instead of consuming `/workspace` — while `checkpoints/weights/*.pt` (the artifacts that actually matter and get evaluated/uploaded) stay on `/workspace` as before. No downside: state checkpoints were already being treated as disposable (superseded ones deleted immediately after each new save), and an external container restart already proved capable of destroying an in-progress run regardless of which filesystem the state was on, so ephemeral storage adds no new risk for this specific artifact type.

**Operational note (learned 2026-08-08, experiment 0002):** training under ZeRO-1 on this model saturates all 4 GPUs' memory (~57-58GB/GPU observed at `batch_size=4`) — there is no free GPU headroom to run eval workers concurrently with an active training process. Launching an eval job while training is still running will OOM the eval workers (training itself was not observed to crash from this, but do not rely on that). A "progress check at step N" therefore means: wait until training reaches a natural pause point (the run completing, or a deliberate stop) before launching any evaluation against that step's saved weights-only `.pt` checkpoint — the checkpoint file itself remains on disk and evaluable at any later time, so there is no need to interrupt a healthy run just to evaluate an intermediate checkpoint immediately. Plan progress-check timing around this constraint (e.g. evaluate several saved checkpoints back-to-back once GPUs are free, rather than trying to check mid-flight).

### Broader confirmation

Purpose: before spending a full canonical run, confirm a promising cheap-panel signal generalizes.
- task/suite coverage: all 90 LIBERO-90 tasks at a reduced trial count (10-15/task) + all 40 original-suite tasks at a reduced trial count (10-15/task) — i.e. the full task set, reduced trials, rather than the reverse.
- trials per task: 10-15 (vs. 50 canonical) — enough to distinguish a real ~10+ point movement from noise without the canonical run's full cost.
- command/config: same as canonical (`run_libero_manager.py` with all 5 suites) but `EVALUATION.num_trials=10` or `15`, into a **fresh `output_dir`** (see NOTES.md — never reuse an output_dir across separate invocations).
- expected cost/runtime: roughly trials_ratio x canonical cost, i.e. ~1/5 to ~1/3 of the ~12.5h canonical run ≈ 2.5-4h, less if the candidate's improved LIBERO-90 success rate causes more early episode termination (likely, since baseline's cost was inflated by near-100% failure).
- accepted-checkpoint baseline on this exact panel: not yet run; when needed, subsample the existing baseline's raw per-task results to the same trial count for a fair comparison, or re-run at reduced trials for a clean baseline-panel reference point.

### Canonical promotion evaluation

Use the full fixed five-suite protocol above (50 trials/task, all 130 tasks). Only this evidence can promote a checkpoint. **Always use a fresh, unused `output_dir`** — do not reuse `evaluate_results/baseline_0000/` or any other prior run's directory (see the false-completion bug documented in `research/NOTES.md`).

### Targeted diagnostic panels

Defined ad hoc per `$investigate-fastwam-problem` use. One concrete candidate already identified: a panel restricted to the 16-task book-in-caddy category (tasks 73-89) to specifically probe whether a candidate has learned this entirely-untransferred category, separate from the general LIBERO-90 signal.

### Avoid overfitting cheap panels

The default cheap panel (13 fixed tasks above) is a compute-allocation tool, not a miniature benchmark. If repeated use makes it unrepresentative (e.g. a candidate overfits specifically to task 73's caddy scene without generalizing to the other 15 caddy tasks), broaden/rotate it and document the change here.

## Durable checkpoint storage

Hugging Face model repository: `cheikh025/ASR`

Authentication:

- source: `HF_TOKEN` environment variable
- setup authentication verified: **yes** (`huggingface_hub.whoami()` -> account `cheikh025`)
- authenticated account: `cheikh025`
- never record the token value

Verified upload command/API (tested 2026-08-08 with a small text file, confirmed via `HfApi().list_repo_files`, commit visible at `https://huggingface.co/cheikh025/ASR/commit/c51dc098f508aaffd249c208cfb7c1659c2830f6`):

```bash
hf upload cheikh025/ASR <local_path> <path_in_repo>
# or, for a directory:
hf upload cheikh025/ASR <local_dir> <path_in_repo> --repo-type model
```

Remote path convention:

- promoted checkpoints: `promoted/<experiment_id>_<short_desc>/step_XXXXXX.pt` (+ dataset_stats.json, config)
- branch checkpoints: `branch/<experiment_id>_<short_desc>/step_XXXXXX.pt`
- continuation/resume checkpoints: `continuation/<experiment_id>_<short_desc>/step_XXXXXX.pt` or full state dir
- associated reports/configs: same remote directory as the checkpoint, e.g. `<...>/PROGRESS_XXXX_*.md`, `<...>/task_config.yaml`

Remote verification method:

```text
hf api list-repo-files cheikh025/ASR  (or huggingface_hub.HfApi().list_repo_files("cheikh025/ASR"))
```

Retention policy: as specified in CLAUDE.md — keep local checkpoints after upload by default; verify remote existence before any local deletion; keep the active parent/continuation checkpoint locally whenever practical.

## Output locations

- training logs: `runs/<task>/<run_id>/` (stdout via `accelerate launch`; consider redirecting to a log file per run)
- checkpoints: `runs/<task>/<run_id>/checkpoints/{weights,state}/`
- evaluation logs: `evaluate_results/<run_id or custom output_dir>/<suite>/task_logs/*.log`
- parsed metrics: `evaluate_results/<...>/<suite>/gpu*_task*_results.json`, aggregated by `summarize_results.py`
- progress-evaluation results: recorded manually into `research/progress/PROGRESS_XXXX_*.md` per experiment

## Metric extraction

Record at minimum for canonical evaluation:

```text
libero_90
libero_spatial
libero_object
libero_goal
libero_long
```

Also preserve per-task results when available (`summarize_results.py` output).

## Failure checks

- how a successful training run is recognized: process exits 0, `checkpoints/weights/step_XXXXXX.pt` exists for the final/expected step, loss logged and finite in stdout/wandb.
- how a failed training run is recognized: nonzero exit, traceback in stdout, or a training/data-loading exception; deepspeed/NCCL errors surface as process crashes across ranks.
- useful log tail command: `tail -100 <captured training stdout log>` (training is launched via `accelerate launch`; redirect stdout explicitly when launching, e.g. `... 2>&1 | tee runs/.../train.log`).
- OOM indicator: `CUDA out of memory` / `torch.cuda.OutOfMemoryError` in stdout, or DeepSpeed/NCCL abort shortly after a forward/backward call.
- missing-checkpoint indicator: expected `checkpoints/weights/step_XXXXXX.pt` absent after `save_every` steps have elapsed per the log.
- evaluation failure indicator: `run_libero_manager.py` reports task(s) in `failed_tasks.txt` under the eval output dir; `eval_libero_single.py` per-task log tail in `evaluate_results/.../task_logs/*.log` shows the traceback; missing `gpu*_task*_results.json` for a launched task after it should have finished.
