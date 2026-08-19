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

**exp0013 real-training recurrence**: with only ~35GB free at launch time (worse than exp0002's ~34GB), the arithmetic (`(KEEP+1) x 12GB`) forced `KEEP=1` (~24GB headroom) rather than `KEEP=2` (~36GB, would not have fit). This means only the single most-recent checkpoint survives locally at any time during training -- any checkpoint worth keeping for evaluation, continuation, or promotion must be evaluated/copied out or uploaded to `cheikh025/ASR` before the *next* `save_every` cadence overwrites it. Used the PID-based pruner form (`while kill -0 $TRAIN_PID; do ...; done`) rather than string-matching the run's `pgrep -f` pattern -- a `pgrep -f "<pattern>"` check matches the grep invocation's own argv when the pattern string appears in it, causing a false "still running" read after the real process has already exited (harmless here since it only delays the pruner's own clean exit, but worth avoiding for correctness).

## RoboTwin training data covers 46/50 canonical tasks, not all 50

`data/robotwin2.0/robotwin2.0` is exactly the `yuanty/robotwin2.0-fastwam` HF dataset -- FastWAM's own official preprocessed RoboTwin training data (README: "The preprocessed RoboTwin dataset used by Fast-WAM"), 27,500 episodes / 6,075,103 frames. Verified by keyword-matching all 27,500 episode instructions (`meta/episodes.jsonl`) against the 50 canonical task names in `third_party/RoboTwin/task_config/_eval_step_limit.yml`: **4 of the 50 canonical tasks have zero episodes in the training data** -- `handover_block`, `handover_mic`, `blocks_ranking_rgb`, `blocks_ranking_size` (0 hits each across the full dataset; every other task has >=1 keyword hit, `hanging_mug` confirmed present with 550 episodes on a full rescan after a small-sample false negative).

**Implication**: any candidate trained on this data (including exp0013+) will have essentially zero training signal for these 4 tasks and should be expected to score near-0% on them at canonical 50-task RoboTwin eval time, regardless of training quality elsewhere -- this is a data-coverage ceiling inherited from FastWAM's own release, not a bug in our training. When interpreting the 50-task canonical RoboTwin average, keep this in mind (the achievable ceiling is closer to 46/50 tasks' worth of real capability plus incidental generalization on the other 4, not 50/50).

## Operational pattern — preserving a checkpoint across a stop-for-eval/resume cycle under KEEP=1

exp0013's step-1000 progress check (`PROGRESS_0013`) surfaced a disk-arithmetic gap in the stop-for-eval/resume pattern: the `KEEP=1` pruner's headroom math (`(KEEP+1) x checkpoint_size_GB`, see the "Disk crisis" section above) assumes *only* the active run's own checkpoints occupy the reserved space. But stopping training to evaluate a checkpoint and then deliberately keeping that file around (its own pruner exits with the training process, so it's no longer auto-pruned) silently consumes a full checkpoint's worth of the headroom the *next* run's pruner arithmetic was counting on -- confirmed twice in immediate succession (the preserved `step_001000.pt`, then the preserved `step_000800.pt` from the resulting continuation run), each time dropping free space from ~24GB towards ~12GB, below the safe `KEEP=1` threshold.

**Standing rule for every future stop-for-eval/resume cycle on this hardware**: as soon as a checkpoint's progress-check evidence is recorded and the decision is `CONTINUE_TRAINING` (i.e. the raw weights file itself is no longer needed locally beyond being the resume source), immediately (1) upload it to `cheikh025/ASR` via `HfApi().upload_file(...)`, (2) verify the remote copy exists with `api.model_info(..., files_metadata=True)` and matching byte size, (3) delete the local copy, all *before* launching the next training phase -- do not let an evaluated-and-kept checkpoint sit unpruned while a new training phase starts. If the checkpoint is still needed locally as the immediate `resume=` source for the very next launch, do the upload+delete in parallel with the new run's early steps (there is normally a `save_every`-steps-long window before the first new save) rather than blocking the relaunch, but still delete before that first new save lands.

Separately: weights-only `resume=<file>` restarts the trainer's step counter at 1 (documented resume semantics), so every resume of this kind must go into a **new output directory** (`_cont1`, `_cont2`, ...) to avoid the new run's own step numbering colliding with and silently overwriting a preserved checkpoint filename from the phase before it.

**Re-downloading a checkpoint from `cheikh025/ASR` for a diagnostic re-check also needs the same disk accounting**: `huggingface_hub.hf_hub_download()` writes into its own content-addressed cache (`${HF_HOME}/hub/models--.../blobs/<hash>`) *in addition to* wherever the file is then copied for actual use -- a single 12GB checkpoint briefly needs ~24GB (cache blob + copy) unless the cache blob is deleted afterward. Confirmed during exp0013's diagnostic re-eval of `step_001800_cumulative.pt`: disk dropped 24GB -> 12GB from the download alone, recovered by `rm`-ing both the cache symlink under `snapshots/` and the real blob under `blobs/` (the symlink alone doesn't free the space -- the blob is the actual file).

**This is easy to forget when the resume path points directly at the HF cache location** (skipping a separate copy step, which seems like the disk-frugal choice) -- confirmed recurring in exp0014: two separate resume downloads left two 12GB cache blobs behind (the training process only reads the resume file at startup, then never touches it again, so the blob becomes pure dead weight for the rest of the run but nothing prompts its removal). This caused a genuine near-miss -- free space silently fell to 4.3GB during a routine mid-training progress check, well past the danger threshold, without any single action that looked risky in isolation. **Standing rule: immediately after any `hf_hub_download()` call finishes being used as a training `resume=` source (i.e. once the training process has started and logged "Weight checkpoint already loaded"), delete the blob under `${HF_HOME}/hub/models--.../blobs/<hash>` -- do not wait for a disk check to prompt it.** Treat every `hf_hub_download()` call as needing an explicit, immediate cleanup step, the same way a checkpoint upload needs an explicit verify-then-delete step.

## LIBERO eval manager gotcha — stale tmux session crashes the server, all launches silently no-op

`experiments/libero/run_libero_parallel_test.sh` (called by `run_libero_manager.py`) manages worker processes via a `libero_test_v3` tmux session: it does `tmux has-session -t libero_test_v3 && tmux kill-session -t ...` (cleanup) then `tmux new-session -d -s libero_test_v3`. If an orphaned `libero_test_v3` session already exists (e.g. left over from an earlier interrupted/killed eval attempt) and happens to be the tmux server's *only* session, killing it makes the server itself exit ("server exited unexpectedly") -- then the following `tmux new-session` and every subsequent `tmux send-keys`/`select-pane` call silently fail ("no server running on /tmp/tmux-1002/default"), while the Python scheduler's own bookkeeping still reports tasks as "Running" (it tracks GPU-load-file state, not actual process existence). **Symptom: scheduler status shows N/8 tasks "Running" indefinitely, but `nvidia-smi` shows 0% GPU/0MB used on every GPU, no per-task log files are created, and no worker processes exist in `ps aux`.** Confirmed exp0013's first step-1000 LIBERO-Spatial progress-check hit this exactly.

**Fix**: `kill -9` the manager process, `tmux kill-server` (clears any stale server/session state), verify `tmux ls` reports no sessions and a fresh manual `tmux new-session -d -s <test>` succeeds, then relaunch the eval command unchanged. **Diagnostic check before assuming an eval is progressing**: cross-reference the scheduler's "Running: N" claim against actual `nvidia-smi` GPU utilization/memory -- if GPUs are idle while the scheduler claims tasks are running, suspect this tmux failure mode rather than waiting longer.

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

## RoboTwin eval gotcha — dataset_stats.json auto-discovery expects the OLD unrenamed filename

`experiments/robotwin/eval_robotwin_single.py::_resolve_dataset_stats_path` auto-discovers
stats by searching checkpoint parent directories for a file literally named
`dataset_stats.json`. This predates the distinct-per-embodiment `stats_filename` fix
(see "Dataset-stats collision" above) that made multi-embodiment runs save
`libero_dataset_stats.json`/`robotwin_dataset_stats.json` instead of a shared, collision-prone
`dataset_stats.json`. Consequence: any RoboTwin eval on a checkpoint from a run that used the
fixed, distinctly-named stats files crashes with `FileNotFoundError: Failed to locate
dataset_stats.json` unless `EVALUATION.dataset_stats_path` is passed explicitly.

**This was silently masked for exp0001**: its run directory happens to contain a leftover,
unrenamed `dataset_stats.json` from partway through the stats-fix rollout (which happened to
hold RoboTwin's own stats, since RoboTwin's dataset was constructed last and — pre-fix —
overwrote the shared file), so exp0001's RoboTwin eval command worked by accident without
`dataset_stats_path`. exp0002/exp0003/exp0004 (post-fix, no plain `dataset_stats.json` file at
all) hit the crash directly when this was first tried for exp0004.

**Fix**: always pass `EVALUATION.dataset_stats_path=<run_dir>/robotwin_dataset_stats.json`
explicitly on every RoboTwin eval command for a multi-embodiment checkpoint — do not rely on
auto-discovery. (The eval script's auto-discovery logic itself was not modified — fixing it to
also try the renamed filenames would be a reasonable follow-up but wasn't needed once the
explicit override is used.)

## Mid-training eval gotcha — checkpoint pruner races with a concurrent LIBERO screen

Running a LIBERO candidate_screen (10 tasks, 3 trials, `MULTIRUN.max_tasks_per_gpu=1`
or `2`) **concurrently** with an active training run's `KEEP=1` checkpoint pruner is
unsafe: a full 10-task/3-trial screen genuinely takes ~33-40 minutes wall-clock
end-to-end (confirmed across exp0003-0005's screens) regardless of
`max_tasks_per_gpu`, since per-task simulation+inference time is the bottleneck, not
raw worker parallelism. If the training run's `save_every` cadence (at its observed
`step/s`) produces a new checkpoint faster than the eval finishes, the pruner deletes
the checkpoint the eval is actively reading mid-run once the newer one appears —
confirmed directly in exp0006 (mid-run screen against `step_001000.pt`; pruner
deleted it at a ~19-minute mark once `step_001500.pt` was saved, ~4 min into the
eval; 2 of 10 tasks had already completed, the 3rd crashed with a checkpoint-load
failure, aborting the whole `run_libero_manager.py` scheduler on the first failure).

**Fix used for exp0006's recovery**: for a *mid-run* progress check specifically
(not the final/candidate_screen evaluation), use a much smaller, faster panel —
`EVALUATION.num_trials=1` instead of `3` (10 episodes instead of 30, ~3x faster,
reliably finishes inside a typical ~19-minute save-cadence window) — rather than
fighting the pruner (increasing `KEEP` risks exceeding the ~34-45GB disk headroom
this project operates under; stopping the pruner for the eval's duration risks the
same). A 1-trial/task panel is noisier but adequate for a
`CONTINUE_TRAINING`/`STOP_TRAINING`-level decision; it is not a substitute for the
full 3-trial `candidate_screen` run against the *final* checkpoint once training
finishes (when there's no longer a moving pruner target to race against).

General rule: only run a full-size LIBERO/RoboTwin eval concurrently with active
training when the eval is expected to finish well inside one `save_every` interval
at the observed `step/s`; otherwise use a smaller mid-run panel, or wait for
training to complete.

## RoboTwin eval gotcha — run_robotwin_manager.py ignores CUDA_VISIBLE_DEVICES remapping

Every prior successful RoboTwin progress-check launch in this project pinned each
task with `CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1` — which happens to be
a no-op remap (physical GPU 0/1 map to local index 0/1 either way), so this never
actually tested whether the manager *honors* the restriction. Attempting
`CUDA_VISIBLE_DEVICES=2`/`=3` for the first time (to run two RoboTwin tasks on
different physical GPUs than a concurrently-running LIBERO screen occupying GPUs
0-1) failed: both jobs' Hydra overrides showed `gpu_id=0` and both actually
allocated on **physical GPU 0** (per the OOM traceback's process list, matching
LIBERO's own workers already there), not physical GPU 2/3 as the env var intended.
`run_robotwin_manager.py`/`eval_robotwin_single.py` appears to resolve its GPU index
independently of `CUDA_VISIBLE_DEVICES` (exact mechanism not yet traced) — the env
var is not a reliable way to place a RoboTwin eval job on a specific non-zero
physical GPU with this manager.

**Consequence**: do not attempt to run a RoboTwin progress check on GPUs 2/3
(or any non-0/1 pinning) concurrently with another job occupying GPUs 0/1 by
setting `CUDA_VISIBLE_DEVICES` to a non-zero value — it will silently target
GPU 0 anyway and can OOM against whatever's already there. Stick to the proven
pattern (`CUDA_VISIBLE_DEVICES=0` and `=1` for the two RoboTwin tasks) and either
run RoboTwin sequentially after any other GPU-occupying job finishes, or verify
the manager's actual GPU-selection code path before trying a different pinning
scheme.

## Training gotcha — removing `embodiment_description` stales the RoboTwin text-embedding cache

`PROGRESS_0011` removed `embodiment_description` (a self-introduced textual conditioning
mechanism) from the multi-embodiment data configs. This is correct for LIBERO/RoboTwin
*evaluation* (which calls the text encoder live, no cache involved) but it silently broke
*training*: the RoboTwin text-embedding cache (`./data/text_embeds_cache/robotwin/`, ~921k
files, ~900GB) had been precomputed by a dedicated custom tool
(`research/tools/precompute_multiembodiment_text_embeds.py`) that specifically replicated
`augment_instruction()`'s *embodiment_description-prefixed* string before hashing it for the
cache filename. With `embodiment_description` removed, `augment_instruction()` now produces
the plain (unprefixed) instruction at training time, so every RoboTwin prompt's SHA256 hash
changed -- the entire RoboTwin cache became stale in one commit, not a small gap.

Symptom: `scripts/train.py` crashes a few minutes into a real training run (past model
construction, mid-dataloading) with `FileNotFoundError: Missing text embedding cache: ...`.
The dataset's own fallback (`RobotVideoDataset.__getitem__` retries once with a random
index on any exception) is not robust to a near-100%-stale cache -- it just fails on the
retry too and crashes the whole distributed job.

LIBERO's cache (`./data/text_embeds_cache/libero/`) was unaffected -- it was originally
computed for single-embodiment LIBERO training, which never used `embodiment_description`,
so it already matched the plain format.

**Fix**: recompute RoboTwin's cache with the plain prompt format. Since `embodiment_description`
is now removed, the *generic*, existing `scripts/precompute_text_embeds.py` tool (which never
included embodiment conditioning to begin with) now produces the correct hashes directly --
no need for the custom multiembodiment-specific script anymore:
```bash
torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py task=<multiembodiment_task_name>
```
Note this recomputes *all* discovered prompts (including LIBERO's, redundantly but harmlessly,
since `overwrite=True` in the current script) -- budget the full ~35-40 minutes the original
precompute took, not a quick backfill.

**Lesson for future config changes**: any change to `FastWAMProcessor.augment_instruction()`'s
output (which text conditioning is prepended, dropped, or reworded) invalidates the text-embedding
cache for every dataset using that processor config, since the cache key is a hash of the exact
runtime prompt string. Check whether a cache needs recomputing *before* launching a real training
run whenever `embodiment_description` or similar instruction-augmentation settings change --
don't assume the cache is still valid just because the underlying raw task/instruction data
didn't change.

### Second gotcha found while fixing the above — `scripts/precompute_text_embeds.py` cross-writes every prompt into every discovered cache directory

Running the recompute against the *combined* multi-embodiment task config
(`multiembodiment_libero_robotwin*`) nearly filled the entire 1.1TB `/workspace` volume in
under two minutes (27GB free -> 740K free) and had to be killed. Root cause: `_collect_dataset_settings`
unions *all* dataset_dirs and *all* cache_dirs discovered anywhere in `cfg.data` into two flat
lists, then the write loop does `for cache_dir in cache_dirs: ...` for *every* encoded prompt --
so every LIBERO prompt gets written into RoboTwin's cache dir and every RoboTwin prompt gets
written into LIBERO's, not just each embodiment's own directory. Against the combined config
(921,072 total prompts, ~912k of them RoboTwin's) this meant writing roughly 2x the intended
volume, growing far faster than the expected ~900GB single-pass total. The custom
`research/tools/precompute_multiembodiment_text_embeds.py`'s own docstring already documented
this exact limitation of the shared script ("a flat prompt-list x cache-dir-list design, fine
when every dataset node wants the identical prompt set, but not when different embodiments need
different... prompts cached to different directories") -- worth re-reading before reaching for
the generic script against a multi-embodiment config again.

**Fix**: run the precompute once per embodiment, using a task config scoped to only that
embodiment's data (e.g. `task=robotwin_uncond_3cam_384_multiembodiment_eval`, which composes
only `data: robotwin_multiembodiment`, not the combined config) -- this way
`_collect_dataset_settings` only discovers one dataset_dir set and one cache_dir per invocation,
and there is no cross-writing.

**Also recovered from**: the stale (embodiment_description-prefixed) RoboTwin cache and the
partial cross-written mess from the killed run were both cleared with
`rm -rf ./data/text_embeds_cache/robotwin` before relaunching scoped -- safe because every
filename in that directory is content-addressed (SHA256 of the prompt string), so nothing
outside that directory could have referenced them, and a clean re-run reconstructs exactly what's
needed. LIBERO's cache directory picked up some harmless-but-wasted extra files from the same
killed cross-write (RoboTwin prompts cached under `./data/text_embeds_cache/libero/`, ~23GB) --
left in place since they don't break anything, just waste some space; safe to clean up later if
disk pressure returns.

**Standing lesson**: when running any large batch/precompute job for the first time against a
new/changed config, actively monitor `df -h` during the run (not just after), especially for any
job whose write volume scales with a large discovered list -- don't assume "the same script
worked fine before" transfers to "it'll behave the same against a different config shape."
