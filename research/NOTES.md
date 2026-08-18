# Research Notes

Keep this file concise and forward-looking. This is a **fresh project**.

Do not copy the previous LIBERO AutoResearch notes, failed experiments, hypothesis rankings, or progress history into this file.

Use it for reusable facts discovered during the new RoboTwin multi-embodiment project, such as:

- verified LIBERO/RoboTwin action/state/control semantics;
- data/statistics and normalization facts;
- reusable multi-embodiment implementation details;
- training/evaluation infrastructure findings;
- robust failure signatures or fixes;
- literature mechanisms that concretely affect the current FastWAM design.

Setup seeds only the inherited parent checkpoint/metrics already recorded in `research/STATE.md` and `research/GOAL.md`.

## Confirmed action/state dimensions (two independent sources: data configs + checkpoint tensors)

- LIBERO: action_dim=7 (delta eef pose x6 + absolute gripper x1), proprio_dim=8 (eef pose x6 + gripper x2), norm=min/max.
- RoboTwin: action_dim=14, proprio_dim=14 (bimanual, likely 2x7 per-arm; not confirmed delta vs absolute — RoboTwin config has no `delta_action_dim_mask`, unlike LIBERO), norm=z-score.
- Implies shared padded channel dim **K=14** is the natural choice (RoboTwin already at 14, LIBERO pads 7->14 / 8->14).
- Exact RoboTwin per-arm channel semantics (joint-space vs eef-space, gripper encoding) still needs confirming against RoboTwin's own env code/docs before finalizing the masking design — flagged, not yet resolved.

## Multi-embodiment implementation gap (exact, code-located)

- `src/fastwam/datasets/lerobot/transforms/action_state_merger.py::ConcatLeftAlign` already implements the Qwen-VLA-style padding scheme (`_pad(x, dim)` -> `(x_padded, mask)`, produces `action_dim_is_pad`/`state_dim_is_pad`, `backward()` crops back) but is **unused** — both LIBERO's and RoboTwin's data configs leave `action_target_dim`/`state_target_dim` as `null`.
- `fastwam_processor.py::preprocess` forwards `action_dim_is_pad`/`proprio_dim_is_pad` into the sample dict, but `fastwam.py::FastWAM.build_inputs`/`training_loss` **never reads it** — the action loss does `F.mse_loss(...).mean(dim=2)` over ALL channels unconditionally (~`fastwam.py` lines 550-556). This is the precise wiring gap for masked per-channel loss. Mirror the fix in `fastwam_idm.py`'s analogous path.
- A separate, unwired `embodiment_datasets`/`max_action_dim`/`max_state_dim` Hydra-resolver scaffold exists (`src/fastwam/utils/config_resolvers.py`, `normalizer.py::search_dataset_stats_cache_json`) — no config or trainer code currently consumes it. Available tooling, not a working path.
- `RobotVideoDataset` takes one flat `dataset_dirs` list under one shared `shape_meta`; LIBERO (`image`/`wrist_image` keys) and RoboTwin (`cam_high`/`cam_left_wrist`/`cam_right_wrist` keys) use incompatible raw camera-key schemas, so the two **cannot** be loaded as one dataset instance as-is — need two dataset instances + a weighted concat/sampler, or new code.
- `FastWAM.load_checkpoint()` -> `mot.load_state_dict(strict=False)`: tolerates missing/unexpected keys but **raises on shape mismatch** for keys present in both dicts. Loading exp0019 (7/8-dim) into a widened (14/14-dim) model will **crash**, not silently degrade — custom weight-expansion code required (copy old rows/cols into new larger Linear layers at original index positions, init new ones fresh) before any such load.
- `ActionDiT.from_pretrained` (used only for the separate Wan22 backbone pretrain load, not exp0019) already skips `action_encoder.`/`head.` prefixes via `ACTION_BACKBONE_SKIP_PREFIXES` — precedent that these two layers are expected to be handled per-dataset/specially.

## RoboTwin canonical eval — Clean/Randomized confirmed in code

`experiments/robotwin/run_robotwin_manager.py` runs two phases per task (`EVALUATION.task_config=demo_clean` / `demo_randomized`), reported as separate `clean_mean_success_rate`/`random_mean_success_rate` — directly matches the project goal's "evaluate Clean and Randomized separately." Task list itself depends on `third_party/RoboTwin/task_config/_eval_step_limit.yml`, which requires the full RoboTwin asset/environment setup to generate (in progress as of this note).

## Known data gap

Public `yuanty/LIBERO-fastwam` dataset has no `libero_90_no_noops_lerobot` archive (only spatial/object/goal/10). No LIBERO-90 raw->lerobot conversion script exists in this repo. Blocks training-time LIBERO-90 replay/rehearsal design (does NOT block evaluation, which uses the official `libero` package's simulator/benchmark directly, not pre-recorded data).

## Disk crisis — RoboTwin's text-embedding cache is ~900GB, saturates the 1.1TB volume

RoboTwin's 921,032 unique per-episode instructions (see below) each need a cached T5
context tensor at `[context_len=128, text_dim=4096]` bf16 ≈ 1MB/file — **921k files ≈
900GB total** (`data/text_embeds_cache/robotwin/`, confirmed via `du`). Combined with
RoboTwin's raw 75GB training data, LIBERO's 8.9GB, and ~26GB of essential model
components (Wan-AI T5/VAE + the expanded exp0019 checkpoint), this leaves very little
of the 1.1TB volume for anything else — **a real DeepSpeed full-state checkpoint save
(optimizer shards) during exp0001's first live training smoke test filled the disk to
100% and crashed mid-save** (`PytorchStreamWriter failed writing file: file write
failed`), even though all 8 training steps themselves completed successfully with
sane losses.

**Trimming the cache is not a safe fix**: `RobotVideoDataset.__getitem__` has a
single-retry fallback (catches a cache-miss `FileNotFoundError`, retries once with a
random other index) — but freeing meaningful space would require deleting the large
majority of the 921k files, making a *double*-miss (uncaught, crashes training) likely
rather than rare. The retry is only enough headroom for occasional gaps, not bulk
deletion.

**Actual fix applied**: (1) added `save_full_state: bool` to `Trainer` (default
`true`, preserves existing behavior everywhere) — when `false`, `save_checkpoint()`
skips `accelerator.save_state()` (the large optimizer-shard save) and only writes the
much smaller weights-only `.pt` file; set `save_full_state: false` in
`configs/task/multiembodiment_libero_robotwin_3e-5.yaml` for this disk-constrained
exploratory candidate (weights-only resume is well-tested, see "Reload/resume smoke"
above; losing exact optimizer-momentum continuation is an acceptable tradeoff for a
short, low-LR exploratory run, not for a long production run). (2) Freed 24GB by
deleting `checkpoints/exp0019_parent_hf/` and `checkpoints/fastwam_release/` — both
safely re-downloadable (durably backed up on `cheikh025/ASR` / `yuanty/fastwam`
respectively; exact re-download commands are in `research/RUNBOOK.md`). (3) Deleted
disposable `runs/_smoke_test/` artifacts after extracting their log evidence.

**For future candidates using RoboTwin**: budget disk carefully — the 900GB text
cache is a fixed, unavoidable cost once computed (do not delete it casually, it's
expensive multi-GPU-hours to regenerate), so essentially all remaining volume space
must go to raw data + one candidate's live checkpoints at a time. Prune old
intermediate weights checkpoints aggressively as new ones are written (matches the
disk-pressure lesson already noted in exp0019's own history), and prefer
`save_full_state: false` unless a specific candidate genuinely needs exact
optimizer-state continuation.

## Hardware sizing — max concurrent full-model workers per A100-80GB

Each FastWAM eval worker (LIBERO or presumably RoboTwin) loads a full model instance using ~14-20GB. `MULTIRUN.max_tasks_per_gpu=5` (the manager's apparent default) OOMs on an 80GB A100; `max_tasks_per_gpu=2` is safe (confirmed empirically — 2 workers x ~20GB = ~41GB used, comfortable headroom). Use `<=2` per GPU for LIBERO/RoboTwin eval on this hardware; treat as a starting point for sizing concurrent multi-embodiment training workers too.

## FastWAM model-loading gotcha — dead redirect repo breaks eval/training out of the box

`configs/model/fastwam.yaml` (and `fastwam_idm.yaml`/`fastwam_joint.yaml`) default `redirect_common_files: true`. In `src/fastwam/models/wan22/helpers/loader.py::_resolve_configs`, this remaps the VAE (`Wan2.2_VAE.pth`) and T5 text-encoder (`models_t5_umt5-xxl-enc-bf16.pth`) `model_id`/pattern to a dead HF repo, `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (confirmed 404 — `RepositoryNotFoundError`). This download happens **unconditionally** in `load_wan22_ti2v_5b_components` (`vae_config.download_if_necessary()`, and `text_config.download_if_necessary()` when `load_text_encoder=true`), regardless of `skip_dit_load_from_pretrain` (that flag only skips the separate video-DiT weights load). **Every LIBERO/RoboTwin eval run and every fresh training run will crash on this by default.**

**Fix**: pass `model.redirect_common_files=false` as a Hydra override on every eval/training command. This makes the loader pull the VAE/text-encoder straight from `Wan-AI/Wan2.2-TI2V-5B` (the live, correct source) instead. Pre-populate the local cache once via `fastwam.models.wan22.helpers.loader._resolve_configs(model_id='Wan-AI/Wan2.2-TI2V-5B', tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B', redirect_common_files=False)` + `.download_if_necessary()` on each returned config (VAE/text/tokenizer) to avoid repeated downloads — lands at `checkpoints/Wan-AI/Wan2.2-TI2V-5B/` and `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/`. This must be added to every canonical/training/eval command in the runbook, not just smoke tests.

## RoboTwin install gotcha — curobo version pin required

RoboTwin's own `script/_install.sh` does `git clone https://github.com/NVlabs/curobo.git` with **no ref**, which pulls current upstream HEAD. As of this setup (2026-08), HEAD has been restructured to a flat `curobo/types.py` single file, but RoboTwin's own `envs/robot/planner.py` does `from curobo.types.math import Pose`, which requires the **old** `src/curobo/types/math.py` package layout — only present back at tag **`v0.7.6`** (2024-11-22), not at HEAD or other recent tags. This is upstream drift in RoboTwin's unpinned install script, not specific to our vendored copy. **Fix: checkout `envs/curobo` to tag `v0.7.6` before `pip install -e . --no-build-isolation`.** Note v0.7.6 has real CUDA/C++ extensions (unlike HEAD's warp-lang-only build) so the build genuinely takes real compile time (confirmed ~5 min on this hardware, not the ~2min of a silently-wrong HEAD build), matching RoboTwin's docs' 20-40 min estimate on slower hardware.

**Second-order pin required**: curobo v0.7.6's `setup.cfg` only pins `warp-lang>=0.9.0` (no upper bound), so a fresh install pulls current warp-lang (1.16.0 as of this setup), which dropped the `warp.torch.device_from_torch` API that curobo v0.7.6's collision-checker calls at motion-planner init — this crashes at first use, not at install time. **Fix: pin `warp-lang==1.4.2`** (matches curobo v0.7.6's actual build era). Both pins (`curobo==0.7.6` tag + `warp-lang==1.4.2`) are hard requirements — record them in any reproduction/environment-setup automation, since a naive `_install.sh` run will silently produce a broken environment otherwise (imports succeed, only fails at actual motion-planner init).

**Full RoboTwin sim stack verified working end-to-end** (not just imports): bimanual aloha-agilex robot loads, curobo motion planner initializes for both arms, 3-camera RGB rendering works (`get_obs()` returns correct shapes). Smoke test script preserved at `checkpoints/robotwin_env_smoke_test.py`. SAPIEN also needed `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` for Vulkan ICD discovery (persisted in `${WORKSPACE}/.env`) despite the ICD file already existing at that path.

## LIBERO package install gotcha

Official `Lifelong-Robot-Learning/LIBERO`'s `setup.py` uses `find_packages()` from repo root, which finds **zero** `libero*` packages (no top-level `libero/__init__.py` — real package root is nested `libero/libero/`). A PEP 660 editable install (`pip install -e .`) therefore produces an empty import-finder mapping and `import libero` silently fails. Fix: skip setuptools packaging, just add the repo root to a `.pth` file in site-packages so `libero` resolves as an implicit PEP 420 namespace package instead.

**Second LIBERO gotcha**: `libero/libero/benchmark/__init__.py:164`'s `get_task_init_states()` calls `torch.load(init_states_path)` with no `weights_only` argument — breaks under PyTorch 2.6+, which changed the default to `weights_only=True`. Patched to `torch.load(init_states_path, weights_only=False)` directly in the cloned LIBERO repo (`/workspace/third_party_src/LIBERO`, not FastWAM's tracked source).
