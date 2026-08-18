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

## Dataset-stats collision — multi-embodiment training silently lost LIBERO's normalization stats

`RobotVideoDataset.__init__` always saves computed/loaded normalization stats to a **hardcoded** `{work_dir}/dataset_stats.json`. `build_multi_embodiment_dataset` constructs both embodiments' `RobotVideoDataset`s in the same `work_dir`; RoboTwin's (constructed second, loads its stats from the fixed `data/robotwin2.0/dataset_stats.json`) silently **overwrote** LIBERO's freshly-computed stats before they were ever durably persisted. Training itself was unaffected (each embodiment's own in-memory `processor.set_normalizer_from_stats()` call used the correct stats at construction time — confirmed the run's `dataset_stats.json` was byte-identical to RoboTwin's own stats file, proving the overwrite), but **evaluating exp0001 with the saved `dataset_stats.json` would silently use RoboTwin's normalization for LIBERO** — wrong, and not obviously so (no crash, just quietly incorrect denormalized actions). Caught by comparing the saved file's byte content to RoboTwin's own known stats file before trusting it for evaluation, not by any error.

**Fix**: added a `stats_filename: str = "dataset_stats.json"` constructor param to `RobotVideoDataset` (default preserves existing single-embodiment behavior everywhere), threaded into both `save_dataset_stats_to_json(...)` call sites. `configs/data/multiembodiment_libero_robotwin.yaml` now sets distinct `stats_filename: libero_dataset_stats.json` / `robotwin_dataset_stats.json` per embodiment. exp0001's own lost LIBERO stats were recovered by recomputing them (a deterministic, non-random full pass over the fixed dataset — `research/tools/recompute_libero_multiembodiment_stats.py`), numerically identical to what training actually used.

**Lesson, generalizes the in-loop-eval one above**: anywhere `RobotVideoDataset`/`Trainer` assumes "one dataset per `work_dir`" is a latent multi-embodiment bug, not necessarily one that crashes — some (like this one) fail silently and only surface as subtly wrong evaluation numbers. Audit other `work_dir`-relative hardcoded paths before trusting them for a multi-embodiment run.

## In-loop eval crash — Trainer.evaluate() assumes a plain RobotVideoDataset

`Trainer.evaluate()` (`src/fastwam/trainer.py:484`) does `processor = self.val_dataset.lerobot_dataset.processor`, assuming `val_dataset` is a plain `RobotVideoDataset`. For the multi-embodiment config, `data.val: null` -> `val_dataset = train_dataset` (a `ConcatDataset`), which has no `.lerobot_dataset` attribute — **crashed exp0001's real training run entirely at the first `eval_every` boundary (step 200), with zero checkpoint saved** (the in-loop eval runs before the `save_every` check in the training loop, so all 200 steps of progress were lost, not just the eval). Likely more single-embodiment assumptions exist deeper in `evaluate()` (rollout video shape, camera count) that a one-line fix wouldn't catch. **Fix applied for exp0001**: set `eval_every` very large (999999) to disable this in-loop qualitative PSNR/SSIM check entirely — real RoboTwin/LIBERO evaluation evidence comes from `$evaluate-fastwam-multiembodiment`'s dedicated managers, not this sanity-glance path, so disabling it costs little. Making `Trainer.evaluate()` multi-embodiment-aware (pick one embodiment's held-out sample explicitly, use its own shape_meta/processor) is a candidate for later, separate work — do not assume it's already handled by the padding/masking changes in `research/tools/`.

**Lesson**: any single-embodiment code path not exercised by the loss/dataset/checkpoint-expansion unit tests (which is most of `Trainer`'s non-`training_loss` methods) should be treated as unverified for the multi-embodiment case until actually exercised end-to-end. `eval_every`/`save_every` boundaries are exactly where such gaps surface, often expensively (lost training progress), not gracefully.

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

**Operational addendum from exp0001's actual training run**: even with `save_full_state=false`, weights-only checkpoints (~12GB each at K=14) still exhaust the ~25-58GB of remaining headroom within 2-4 saves at `save_every=100`. Disk hit 100% (5.5GB free) again mid-run before this was caught by manual inspection (not by the log-tailing Monitor — rich's console line-wrapping means a literal substring like `[ckpt]` doesn't reliably appear on a single physical output line for `grep --line-buffered` to match, so checkpoint-save events were mostly silent to the monitor). **Fix**: run a simple background pruner (`while ps -p <pid>; do sleep 60; ls -1t weights/step_*.pt | tail -n +3 | xargs rm -f; done`) alongside any real multi-checkpoint training run on this hardware, keeping only the N most recent weights files, rather than relying on manual or log-monitor-triggered pruning.

**Recurrence in exp0002**: the exact same crash happened again on exp0002's first launch — the background-pruner pattern from exp0001 was reused with `KEEP=2`, but only ~34GB was free at launch time (2 kept x 12GB + 1 being actively written x 12GB = 36GB > 34GB available), so the checkpoint write itself failed mid-write once the 3rd file appeared. **Do the arithmetic explicitly before choosing `KEEP`**: required headroom is `(KEEP + 1) x checkpoint_size_GB` (the `+1` for the checkpoint currently being written) — with ~12GB checkpoints, `KEEP=1` needs ~24GB free, `KEEP=2` needs ~36GB free. Also poll faster than the gap between saves (15s, not 60s) to close the race window between "a 3rd checkpoint file appears" and "disk fills up" — a 12GB write can complete in well under 60s on this hardware. Recovered by resuming from the last valid checkpoint (verified loadable first) rather than restarting from scratch, with `max_steps` reduced by however many steps had already completed, preserving the candidate's total intended step budget.

**Fallback overflow space**: the container's root filesystem (`/`, `overlay`, 212GB) is separate from `/workspace` (the 1.1TB persistent volume, `/dev/md0`) and sits essentially untouched (~1.2GB used, ~211GB free) — everything so far has been written to `/workspace`. `/` is **ephemeral** (wiped on container recycle/destroy, unlike `/workspace`), so never durably store checkpoints or anything meant to survive there, but it's a legitimate temporary overflow valve (e.g. staging a checkpoint before HF upload, or breathing room if `/workspace` pruning ever falls behind) if `/workspace` hits zero again.

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
