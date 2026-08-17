# FastWAM RoboTwin Multi-Embodiment Research Runbook

This file is filled and verified by `$setup-fastwam-research` from the actual repository and fresh machine.

Do not guess commands when the repository already defines them. Do not copy the old project's runbook or research history wholesale.

## Bootstrap/workspace

- bootstrap package root: `/workspace/fastwam_robotwin` (unpacked package; kept as-is, not the working repo)
- FastWAM repository root after clone: `/workspace/FastWAM` (on the persistent `/workspace` volume; `workspace_is_volume=true`)
- research-control files integrated into repository: yes (`CLAUDE.md`, `AUTORESEARCH.md`, `SOURCES.md`, `.claude/`, `research/` copied into `/workspace/FastWAM`)
- FastWAM README preserved (not overwritten by bootstrap README): yes — `/workspace/FastWAM/README.md` is FastWAM's own upstream README, untouched

## Repository and lineage

- working fork: `https://github.com/cheikh025/FastWAM`
- `origin` URL: `https://github.com/cheikh025/FastWAM.git`
- upstream repository: `https://github.com/yuantianyuan01/FastWAM`
- `upstream` URL: `https://github.com/yuantianyuan01/FastWAM.git`
- parent lineage branch: `autoresearch/libero90-v1`
- expected checkpoint-producing commit: `b2b49d0`
- parent commit verified/reachable from expected lineage: **yes** — `git merge-base --is-ancestor b2b49d0 autoresearch/libero90-v1` succeeds; `b2b49d0` = "Fix directory-resume LR schedule corruption when max_steps is extended" (`src/fastwam/trainer.py`, +47 lines), directly relevant to a resumed/extended training run and consistent with exp0019's promoted PROGRESS report which records `Git commit: b2b49d0 (both resume fixes applied)`
- exact commit used to create new branch: `b2b49d0` (`git switch -c autoresearch/robotwin-multiembodiment-v1 b2b49d0`)
- research branch: `autoresearch/robotwin-multiembodiment-v1` (active; verified via `git branch --show-current`)
- remote research branch pushed/verified: not yet pushed (local only so far; will push before/at first tracked-change commit)

Hard branch guard:

- `autoresearch/libero90-v1` is read-only for this project;
- no tracked change is made before `autoresearch/robotwin-multiembodiment-v1` is active;
- before any expensive run or commit, verify the current branch is the RoboTwin branch.

Old research-memory guard:

- do not checkout/read/import `autoresearch/research-docs` to choose experiments;
- the inherited parent checkpoint/metrics in `GOAL.md`/`STATE.md` are sufficient starting facts.

## Environment

- Python/conda/venv: dedicated `uv`-managed venv at `/workspace/venvs/fastwam` (Python 3.10.20, isolated from the base image's `/venv/main` which carries an incompatible torch 2.11). Activate with `source /workspace/venvs/fastwam/bin/activate`.
- required environment variables: `UV_PYTHON_INSTALL_DIR=/workspace/.uv/python_install`, `UV_CACHE_DIR=/workspace/.uv/cache` (default `/.uv/*` is not writable in this container; both persisted in `${WORKSPACE}/.env`); `LIBERO_CONFIG_PATH` not set — LIBERO uses its default `~/.libero/config.yaml` (pre-created, see LIBERO/MuJoCo setup below); `MUJOCO_GL=egl` required for headless offscreen rendering; `HF_TOKEN` supplied by platform, authenticated as `cheikh025`.
- GPU count/model: 4x NVIDIA A100-SXM4-80GB, driver 570.211.01, all idle/free at setup time.
- important dependency versions: `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128`, `transformers==4.49.0`, `accelerate==1.12.0`, `deepspeed==0.18.5`, `hydra-core==1.3.2`, `numpy==1.26.4`, `einops==0.8.1` — all exactly per FastWAM's `pyproject.toml` (`pip install -e .` in the venv, no conflicts encountered for the core package).
- distributed launcher/world size: `accelerate launch` via `scripts/train_zero1.sh <NPROC_PER_NODE> ...` wrapping DeepSpeed ZeRO-1 (`scripts/accelerate_configs/accelerate_zero1_ds.yaml`). 4 GPUs available on this machine (FastWAM's own README used 8 for LIBERO, 64 for RoboTwin — we will scale down and rely on gradient accumulation / longer wall time).
- LIBERO/MuJoCo setup: official `Lifelong-Robot-Learning/LIBERO` cloned to `/workspace/third_party_src/LIBERO` (commit `8f1084e`, "Add support for dataset download from huggingface"). Installed with `pip install -e . --no-deps` (its own `requirements.txt` pins conflicting old versions — hydra-core 1.2.0, numpy 1.22.4, transformers 4.21.1 — and was deliberately NOT used verbatim). **Known packaging bug**: `setup.py`'s `find_packages()` finds zero top-level `libero*` packages (no `libero/__init__.py` at repo root — the real package root is the nested `libero/libero/`), so PEP 660 editable installs produce an empty import-finder mapping and `import libero` fails silently. Fixed by adding a plain `.pth` file (`echo /workspace/third_party_src/LIBERO > $SITE_PACKAGES/_libero_path.pth`) so `libero` resolves as an implicit PEP 420 namespace package instead — `from libero.libero import benchmark` then works. Additional deps installed individually with `--no-deps` to avoid downgrading FastWAM's pins: `robosuite==1.4.0`, `bddl==1.0.1`, `robomimic==0.2.0`, `easydict==1.9`, `thop==0.1.1.post2209072238`, `gym==0.25.2`, `cloudpickle==2.1.0` (downgraded from 3.1.2 — watch for incompatibility, none observed yet), `future==0.18.2`, `mujoco==3.3.2`, plus transitively-needed `glfw`, `PyOpenGL`/`PyOpenGL-accelerate`, `numba`, `llvmlite` (needed for EGL offscreen rendering and robosuite's numba-accelerated transform utils, not in LIBERO's own requirements.txt). `~/.libero/config.yaml` pre-created (points `datasets` at `/workspace/FastWAM/data/libero_mujoco3.3.2`, `benchmark_root`/`bddl_files`/`init_states`/`assets` at the cloned repo) to skip LIBERO's interactive first-import prompt. Verified: `from libero.libero.envs import OffScreenRenderEnv` imports cleanly under `MUJOCO_GL=egl`.
- RoboTwin/SAPIEN/CuRobo/other simulator setup: all installed into the same `/workspace/venvs/fastwam` venv (required — see "important version compatibility notes"). `uv pip install -r <RoboTwin's script/requirements.txt, fetched from upstream at pinned commit bf44be51cf5717a5595ce59447f2cf5263d2aa95, missing from our vendored copy> -c <constraints pinning torch==2.7.1+cu128, torchvision==0.22.1+cu128, huggingface-hub==0.29.2, av==16.0.1>` (excluded torch/torchvision/huggingface_hub/av/wandb/termcolor/imageio/ffmpeg from the upstream requirements to avoid downgrading FastWAM's pins). Installed: `sapien==3.0.0b1`, `mplib==0.2.1`, `pytorch3d==0.7.8` (built from git `stable` branch, CUDA ops verified), `nvidia-curobo==0.7.6` (built from source — **must pin to tag `v0.7.6`, not HEAD**, see "RoboTwin install gotcha" in `research/NOTES.md`), `warp-lang==1.4.2` (**must downgrade from the default-resolved 1.16.0** — curobo v0.7.6's collision-checker calls `warp.torch.device_from_torch`, dropped in newer warp-lang; 1.4.2 matches curobo v0.7.6's actual build era), plus ~260 transitive deps (scipy, transforms3d, gymnasium, trimesh, open3d, etc.). Source patches applied to installed packages per RoboTwin's `script/_install.sh`: sapien's `wrapper/urdf_loader.py` (UTF-8 encoding on two `open()` calls) and mplib's `planner.py` (removed `or collide` from a screw-plan-failure condition) — both backed up as `.orig` alongside. `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` required for SAPIEN's Vulkan ICD discovery (persisted in `${WORKSPACE}/.env`); `vulkaninfo` confirms all 4 A100s visible. `MUJOCO_GL`/EGL not relevant to SAPIEN — Vulkan was the actual blocker, now fixed. Missing `third_party/RoboTwin/task_config/*.yml` and `assets/_download.py` fetched from upstream at the same pinned commit; `task_config/_eval_step_limit.yml` confirms **exactly 50 canonical RoboTwin tasks**. Assets (`background_texture.zip` ~11GB, `objects.zip` ~3.7GB, `embodiments.zip` ~0.22GB, from HF dataset `TianxingChen/RoboTwin2.0`) downloaded+extracted to `third_party/RoboTwin/assets/` (~16GB); `script/update_embodiment_config_path.py` run afterward (materializes 6 per-embodiment curobo `*.yml` configs from `*_tmp.yml` templates by substituting the assets path — confirmed necessary, not a no-op; confirms `aloha-agilex` embodiment present). SAPIEN offscreen-render smoke test (`script/test_render.py`) passed ("Render Well", clean exit) after the Vulkan fix. `experiments/robotwin/fastwam_policy` symlink verified intact.
- important version compatibility notes: FastWAM's venv is fully separate from the base image's `/venv/main`; do not `source /venv/main/bin/activate` for any FastWAM/LIBERO/RoboTwin work. `pip` on `PATH` resolves to the system `/usr/bin/pip`, not the venv's — always use `uv pip` (with the two env vars above) or the venv's `python3 -m pip` to avoid installing into the wrong environment. nvcc 12.8 present at `/usr/local/cuda/bin/nvcc`, matches the torch cu128 wheel, needed for RoboTwin's CUDA source builds (pytorch3d, curobo). GPU render capability confirmed available (`gl`, `optix`, `vulkan` all true per `vast-capabilities`). **`model.redirect_common_files=false` is a required standing override on every eval/training command** (default `true` points at a dead HF repo — see `research/NOTES.md`). **`MULTIRUN.max_tasks_per_gpu<=2` for LIBERO eval on these A100-80GBs** (each full-model worker uses ~14-20GB; 5/GPU OOMs, 2/GPU is safe) — likely also relevant to sizing concurrent RoboTwin eval workers and multi-embodiment training batch/worker counts later. LIBERO evaluator's `RoboTwin` counterpart (SAPIEN) shares the same venv and was verified compatible.

## Inherited parent checkpoint

Recorded parent:

`runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`

Recorded durable location:

`cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`

Fill/verify:

- HF repository/revision: `cheikh025/ASR`, path `promoted/0019_spatial_weak_task_oversampling/`, main branch (no pinned revision needed — the promoted path is treated as immutable by project convention)
- verified local checkpoint path: `/workspace/FastWAM/checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt` (also fetched alongside: `dataset_stats.json`, `libero_2cam_plus90_reweighted_weaktasks_v3.yaml`, `libero_uncond_2cam224_plus90_reweighted_weaktasks_v3_3e-5.yaml`, `PROGRESS_0019_spatial_weak_task_oversampling.md`)
- file size: 12,041,735,545 bytes — **exact match** to the size recorded in the HF repo listing and to the byte count recorded in the parent project's own `PROGRESS_0019...md` ("Final weights checkpoint: ... (12,041,735,545 bytes)")
- hash/checksum or other strong identity: sha256 `decbb99cdd9640a3de5e769801f942940a3d0ab9e2103d10ce5acea632a79c5` (computed locally after download)
- checkpoint format/keys: `torch.save` dict with top-level keys `mot` (combined video+action MoT state dict, 1649 tensors), `step` (=5000), `torch_dtype` (`torch.bfloat16`), `proprio_encoder` (separate 2-key state dict: `weight`/`bias`). No optimizer state (weights-only checkpoint, as expected for a `.../weights/step_XXXXXX.pt` path).
- checkpoint loads under exact parent code state: **yes** — `torch.load(..., weights_only=False)` succeeds under the installed FastWAM env; full `FastWAM.load_checkpoint()` load (not just raw `torch.load`) not yet exercised — planned as part of the reload/resume smoke test.
- matching stats/config required for LIBERO: `dataset_stats.json` (39,182 bytes) downloaded alongside; the exact training data-mix config/data-subset directory that produced exp0019 (`configs/data/libero_2cam_plus90_reweighted_weaktasks_v3.yaml` + a new `libero_spatial_weak_subset_lerobot/` data directory) was **never committed to git** (confirmed absent at commit `b2b49d0`; exp0019's own progress report explicitly says "new, uncommitted config/data files and new data directory"). The config YAML itself was preserved via the HF upload and is available at the local checkpoint path above; the raw 80-episode weak-subset data directory was not uploaded and is not reproducible from this record. This does not block using the checkpoint (we only need forward inference / continued fine-tuning, not exact retraining of exp0019), but exact bit-for-bit reproduction of the exp0018->exp0019 training step is not possible from currently available artifacts.
- checkpoint expansion procedure for the multi-embodiment action interface: not yet implemented — see "Shared padded action representation" below for the confirmed target dimensions (K=14) and the identified code gap.
- method used to preserve inherited weights exactly where dimensions are unchanged: not yet implemented (planned: copy old `action_encoder`/`head`/`proprio_encoder` weight sub-tensors into newly-sized `nn.Linear` layers at the original index positions, zero/small-random-init the new rows/columns, verified by a numerical test that a LIBERO-shaped padded-zero input reproduces the exact pre-expansion output on the valid channels).

Previously validated parent LIBERO reference (not rerun evidence):

- LIBERO-Spatial: 97.00%
- LIBERO-Object: 99.60%
- LIBERO-Goal: 97.20%
- LIBERO-Long / LIBERO-10: 98.00%
- LIBERO-90: 95.13%

## Literature/implementation understanding gate

Before first multi-embodiment modification:

- full Qwen-VLA paper read: **yes** (arXiv 2605.30280, full HTML fetched and read, not abstract-only)
- Qwen-VLA sections most relevant to this implementation: unified padded action-tensor interface, per-channel/per-timestep masked loss, per-dataset normalization, textual embodiment conditioning, staged/progressive training, joint vision-language auxiliary loss for forgetting mitigation (all summarized below)
- other multi-embodiment/generalist papers read and why they matter: LAP (2602.10556) and Qwen-RobotManip (2606.17846) — abstract-only accessible (no HTML full text / PDF too large), not load-bearing for implementation, no additional mechanism extracted beyond abstracts (cross-embodiment natural-language action representation; alignment-across-dimensions + human-to-robot data scaling, respectively). **"Pretrained VLA Models Are Surprisingly Resistant to Forgetting"** (arXiv 2603.03818, found via search, HTML fetched) — directly relevant to our exact scenario (fine-tuning a converged, promoted VLA on a new embodiment while holding 5 LIBERO benchmarks >=90%); findings below.
- repository code paths inspected for LIBERO actions/states: `configs/data/libero_2cam.yaml`/`libero_2cam_plus90_reweighted_v3.yaml`, checkpoint tensors (`action_encoder`/`head`/`proprio_encoder` shapes)
- repository code paths inspected for RoboTwin actions/states: `configs/data/robotwin.yaml`, released specialist checkpoint tensors
- action loss/masking code paths inspected: `fastwam.py::training_loss` (~lines 550-556), `fastwam_processor.py::preprocess`, `action_state_merger.py::ConcatLeftAlign`
- checkpoint load/expand code paths inspected: `FastWAM.load_checkpoint()`/`save_checkpoint()`, `ActionDiT.from_pretrained` (`ACTION_BACKBONE_SKIP_PREFIXES`)
- inference decode/slicing paths inspected: `ConcatLeftAlign.backward()` (crops padded output back to natural per-embodiment dim) — not yet exercised end-to-end

### Verified transferable mechanisms (Qwen-VLA, arXiv 2605.30280)

1. **Unified padded action tensor**: target `Y ∈ R^(H×K)`, H=horizon, K=fixed shared channel dim across all control modes. A mode using `c<=K` channels places valid values in the **leading** `c` dims, remaining `K-c` zero-padded. This is *exactly* what FastWAM's already-implemented-but-unused `ConcatLeftAlign` does — confirms the implementation task is to actually **set** `action_target_dim=14`/`state_target_dim=14` in both LIBERO's and RoboTwin's data configs, not write new padding code. The paper doesn't prescribe how to pick K numerically; here K=14 falls out naturally (RoboTwin's own natural dim).
2. **Per-channel masked loss — two-level averaging (a meaningful, non-obvious design choice)**: (1) per-channel MSE masked-averaged over valid timesteps only: `l_k = sum_h(M[h,k]*err^2) / sum_h(M[h,k])`; (2) then averaged **uniformly over the `c` active channels** (NOT over all `K`). This prevents an embodiment with more valid channels (RoboTwin, c=14) from implicitly getting more loss weight than one with fewer (LIBERO, c=7) purely from channel count. **FastWAM's current `action_loss_token = F.mse_loss(...).mean(dim=2)` averages over all K channels including padding — this must change to mask-first-per-channel, then mean over `c` (not `K`)**, exactly matching the identified wiring gap, now with the precise correct formula (not just "mask it somehow").
3. **Normalization**: per-dataset quantile normalization (1st/99th percentile, linear map to [-1,1], clipped) in the paper — but this is consistent with FastWAM's existing **per-dataset** (not global) normalization choice already in place (LIBERO min/max, RoboTwin z-score); no change needed, just confirms the existing per-dataset-stats-file approach is right.
4. **Embodiment conditioning — textual prompt, not a learned embedding**: Qwen-VLA conditions on embodiment purely via a **prepended natural-language description** in the instruction: "The robot is {robot_tag} with {single arm/dual arms}[, waist][, mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size} control actions to execute: {instruction}." This is their *sole* embodiment-conditioning mechanism. FastWAM's `fastwam_processor.py::augment_instruction()` already builds instruction strings from task/coarse_task — recommended approach: extend this to prepend an embodiment/control-frequency description per this template, reusing the existing T5 text-embedding pipeline. **No architecture change needed** — cheap, directly evidenced first candidate ingredient.
5. **Staged/progressive training**: Qwen-VLA's 4-stage pretraining-from-scratch recipe isn't directly applicable (we start from an already-converged LIBERO specialist, not pretraining), but the underlying principle — don't jointly unfreeze everything on new-embodiment data from step 1 — is a candidate direction: a short warm-up training only the expanded `action_encoder`/`head` columns + new instruction-prompt pathway before unfreezing the shared MoT backbone.
6. **Forgetting mitigation via auxiliary loss + uniform replay**: joint loss `L = lambda_act*L_act + lambda_vl*L_vl` keeps a vision-language next-token loss active during embodied co-training specifically to prevent perception/reasoning forgetting; every minibatch mixes all task families at a fixed ratio (replay-by-construction every step, not phase-separated periodic replay).

### Additional finding — forgetting-resistance paper (arXiv 2603.03818), directly on-point for this project

- Pretrained/converged VLAs forget far less than from-scratch models when fine-tuned on a new task/embodiment (near-zero/negative backward-transfer vs. 0.19-0.29 for from-scratch) — relevant since our parent (exp0019) is itself a converged, heavily-tuned LIBERO specialist, not a generic pretrain.
- **Simple experience replay is highly effective even at small scale**: ~2% replay-buffer size (~100 samples/task) gives substantial forgetting prevention; ~20% replay -> near-zero forgetting. Recommended mixing ratio: **roughly 1:1 new-task:replay-of-past-tasks per batch**, not proportional to raw dataset size (raw RoboTwin data is ~79GB vs. LIBERO's ~4.7GB — naive proportional mixing would nearly starve LIBERO).
- **Component-specific finding**: vision-language *backbone* updates drive most forgetting, not action-head changes — supports freezing/lightly-tuning the shared MoT backbone initially, training only the expanded action_encoder/head + instruction pathway first (aligned with Qwen-VLA's staged-freeze idea above).
- **Recovery is fast** when regression does occur: pretrained models that do regress recover to peak in just 6-10% of original training steps — useful for `$run-fastwam-training` progress-check pacing (a LIBERO dip early in RoboTwin training may not need immediate rollback; check whether it self-corrects within a small step budget before treating it as a real forgetting failure).

### Bottom-line recommendation for the initial multi-embodiment candidate

1. Set `action_target_dim=14`, `state_target_dim=14` in both LIBERO's and RoboTwin's data configs (mechanism already exists via `ConcatLeftAlign` — zero new padding code needed).
2. Implement per-channel masked loss in `fastwam.py::training_loss` (mirror in `fastwam_idm.py`) using the already-produced-but-unused `action_dim_is_pad`/`proprio_dim_is_pad`: mask-then-mean-per-channel, then mean over the valid-channel-count `c` (not the full `K`) — the precise two-level Qwen-VLA formula, not a naive flat masked mean.
3. Custom checkpoint-expansion step (confirmed necessary, no automatic path exists): grow `action_encoder` (Linear 7->14 in), `head` (Linear 14 out — need to re-verify exact in/out after committing to K=14 for both), and `proprio_encoder` (Linear 8->14 in) by copying exp0019's trained weight rows/columns into the new larger tensors at the original index positions, init the new ones fresh.
4. Prepend a Qwen-VLA-style embodiment/control-frequency text description to the instruction string in `augment_instruction()`.
5. Data mixing: roughly 1:1 new(RoboTwin):replay(LIBERO) per batch, not proportional to raw dataset size. **Blocked by the known LIBERO-90 data gap noted above** — only Spatial/Object/Goal/Long-10 are currently available for replay; LIBERO-90 replay needs that gap resolved first (or the initial candidate can replay only the 4 available suites and treat LIBERO-90 retention as evaluation-only risk until resolved).
6. Consider (candidate-level refinement, not required for the initial baseline) a short warm-up phase training only the expanded action_encoder/head/instruction pathway with the shared MoT backbone frozen, before full unfreeze.

## Data and embodiment semantics

### LIBERO

- dataset source/path/config: `configs/data/libero_2cam_plus90_reweighted_v3.yaml` (parent-family config; exp0019's exact variant `libero_2cam_plus90_reweighted_weaktasks_v3.yaml` was never committed, see checkpoint section above). `dataset_dirs` under `./data/libero_mujoco3.3.2/` (downloaded from `yuanty/LIBERO-fastwam`): `libero_spatial_no_noops_lerobot`, `libero_object_no_noops_lerobot`, `libero_goal_no_noops_lerobot` (listed 7x = oversampled), `libero_10_no_noops_lerobot` (listed 10x), `libero_90_no_noops_lerobot`. Task-mix ratio is implemented purely by **repeating dataset-dir entries in the list** — no explicit weight field.
- camera names/count/resolution: `image`, `wrist_image`; raw `[3,512,512]` → resized to `[3,224,224]`; `num_output_cameras: 2`; horizontally concatenated to `video_size: [224,448]`.
- observation history/frame handling: `num_frames: 33` per training window, `action_video_freq_ratio: 4` (32 action steps per 9 video-latent frames), `context_len: 128`.
- state/proprio representation: `proprio_output_dim: 8` — confirmed by checkpoint tensor `proprio_encoder.weight` shape `(4096, 8)`. Composition (from config semantics): eef pose (6) + gripper (2, absolute).
- action dimension: `7` — confirmed by checkpoint tensors `mixtures.action.action_encoder.weight` `(1024, 7)` and `mixtures.action.head.weight` `(7, 1024)`.
- action channel semantics: eef delta-pose (6) + gripper (1, absolute). `delta_action_dim_mask.default: [T,T,T,T,T,T,F]` — first 6 channels are delta values, last (gripper) is absolute/non-delta.
- units/control convention: delta end-effector pose (position+orientation) + absolute gripper command.
- control frequency: not yet separately verified beyond the frame/chunk structure above (no explicit Hz value inspected yet).
- action chunk/prediction horizon: 32 action steps per training window (see `action_video_freq_ratio` above); `replan_steps: 10` at LIBERO evaluation time (`configs/sim_libero.yaml`).
- temporal padding behavior: not yet separately verified; existing per-channel (not just temporal) padding infra exists but is unused, see "Shared padded action representation" below.
- normalization/statistics source and behavior: `norm_default_mode: min/max`; `use_stepwise_action_norm: False`; `pretrained_norm_stats` field present in the data-config schema (set to the checkpoint's `dataset_stats.json` for continued fine-tuning; `null` on a first fresh training run, auto-computed and cached to `runs/{task}/{run_id}/dataset_stats.json`).
- task/instruction representation: natural-language task instruction, T5-embedded via `scripts/precompute_text_embeds.py` and cached (`text_embedding_cache_dir`, e.g. `./data/text_embeds_cache/libero`).
- training task composition: full LIBERO-90 + LIBERO-Spatial/Object/Goal/Long-10, with task-level oversampling for weak tasks (see `dataset_dirs` repetition above); exact per-suite trial/task counts for canonical eval given below.
- **known gap**: the public `yuanty/LIBERO-fastwam` dataset (downloaded to `data/libero_mujoco3.3.2/`) contains only 4 archives — `libero_10`, `libero_goal`, `libero_object`, `libero_spatial` — **no `libero_90_no_noops_lerobot`**, even though the parent's data configs reference a `libero_90_no_noops_lerobot` directory and no LIBERO-90 raw->lerobot conversion script exists in this repo's `scripts/`. This does not block **evaluation** (canonical LIBERO-90 eval uses the official `libero` package's own benchmark/init-state files via simulation rollouts, not pre-recorded demonstration data — already installed and working), but it **does block training-time LIBERO-90 replay/retention data** for any candidate that wants to rehearse LIBERO-90 explicitly during multi-embodiment fine-tuning. Needs resolution (find/regenerate LIBERO-90 lerobot data from LIBERO's own official demonstration set) before designing a replay-based retention candidate; flag for `$investigate-fastwam-problem` or the first `$choose-fastwam-experiment` pass if replay is the chosen retention strategy.

### RoboTwin

- dataset source/path/config: `configs/data/robotwin.yaml`, task `robotwin_uncond_3cam_384_1e-4`. `dataset_dirs: [./data/robotwin2.0/robotwin2.0]` — a single preprocessed directory spanning all training tasks internally (downloaded from `yuanty/robotwin2.0-fastwam`, 79.1GB compressed across 8 tar parts, now extracted at `/workspace/FastWAM/data/robotwin2.0/`).
- official/current RoboTwin code/assets revision: vendored at `third_party/RoboTwin`, upstream commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95` (`RoboTwin-Platform/RoboTwin`, MIT license) — see `third_party/RoboTwin/README.vendor.md`. Full simulator (SAPIEN/CuRobo/assets) install/verification delegated to a dedicated setup pass — result pending.
- camera names/count/resolution: `cam_high`, `cam_left_wrist`, `cam_right_wrist`; raw `[3,480,640]` → resized to `[3,240,320]`; `num_output_cameras: 3`; `concat_multi_camera: "robotwin"` layout; `video_size: [384,320]`.
- observation history/frame handling: same `num_frames: 33` / `action_video_freq_ratio: 4` / `context_len: 128` structure as LIBERO.
- state/proprio representation: `proprio_output_dim: 14` — confirmed both by the training data config AND directly from the FastWAM-released RoboTwin specialist checkpoint (`yuanty/fastwam:robotwin_uncond_3cam_384.pt`, `proprio_encoder.weight` shape `(4096, 14)`). Bimanual Aloha-AgileX: consistent with 2 arms x 7 channels.
- action dimension: `14` — confirmed identically from config (`action_output_dim: 14`) and from the released specialist checkpoint tensors (`action_encoder.weight` `(1024, 14)`, `head.weight` `(14, 1024)`).
- action channel semantics: no `delta_action_dim_mask` set in the data config (defaults to `None`/not-delta) — **channels are not marked as delta**, unlike LIBERO. This suggests absolute joint-space (or absolute end-effector) values per arm rather than LIBERO's delta-eef convention, but this is inferred from config, not yet confirmed against RoboTwin's own action-space documentation/code (`envs/_base_task.py` or similar) — flagged as not fully resolved, follow up before finalizing the masking/normalization design.
- units/control convention: likely absolute per-arm joint or eef values (2 x 7 channels: 6 arm DoF + 1 gripper per arm) — needs the confirmation above.
- control frequency: not yet separately verified.
- action chunk/prediction horizon: same window structure as LIBERO (32 action steps per training window); `replan_steps: 24` at RoboTwin evaluation time (`configs/sim_robotwin.yaml`), differing from LIBERO's `replan_steps: 10`.
- temporal padding behavior: not yet separately verified; see channel-padding gap below.
- normalization/statistics source and behavior: `norm_default_mode: "z-score"` — **differs from LIBERO's min/max**, an explicit per-dataset normalization difference to preserve when mixing. `pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json` (ships with the FastWAM-preprocessed dataset, present after extraction). `val_set_proportion: 0.01` (RoboTwin config sets a validation split; the inspected LIBERO config did not).
- task/instruction representation: same T5-embedding-cache mechanism as LIBERO, cached separately (`./data/text_embeds_cache/robotwin`); README notes FastWAM evaluates with **unseen** instructions by default (`EVALUATION.instruction_type=unseen`), vs. a `seen` alternative used by another baseline (Lingbot-VA) — freeze at `unseen` per project rule against changing eval semantics after the fact.
- training task composition: single combined preprocessed directory; exact per-task episode counts not yet inspected.
- easy/hard or other domain-randomization split semantics: RoboTwin evaluation runs **two phases per task**, `EVALUATION.task_config=demo_clean` and `EVALUATION.task_config=demo_randomized` — this **is** the project goal's "Clean and Randomized" split, confirmed directly in `experiments/robotwin/run_robotwin_manager.py`/`configs/sim_robotwin.yaml`. Results are reported as two separate means (`clean_mean_success_rate`, `random_mean_success_rate`) with no default blended scalar — matches the goal's requirement to evaluate them separately.

### Shared padded action representation — initial implementation

Ground-truth dimensions (confirmed from both configs and checkpoint tensors, two independent sources): LIBERO action_dim=7 / proprio_dim=8; RoboTwin action_dim=14 / proprio_dim=14.

- shared channel dimension K: **K=14** proposed (RoboTwin's natural dimension is the larger of the two for both action and proprio — LIBERO's 7/8 pad up to 14/14). Not yet finalized in code.
- LIBERO valid channels and mask: first 7 of 14 action channels valid (delta-eef x6 + gripper x1), remaining 7 padding; first 8 of 14 proprio channels valid.
- RoboTwin valid channels and mask: all 14 action and 14 proprio channels valid (no padding needed for RoboTwin under K=14).
- time-step validity mask interaction: not yet designed — needs to compose with the existing temporal `action_is_pad` [B,T] mask already used by the loss.
- loss reduction/averaging rule: not yet implemented — planned: masked mean over only valid (channel, timestep) pairs, not a plain `.mean(dim=2)` over all channels.
- dataset-specific normalization before padding/after decoding: apply each dataset's own normalization (LIBERO min/max, RoboTwin z-score) in its natural (un-padded) dimensionality first, then pad; decode by cropping to the natural dimension before un-normalizing at inference.
- parent action-encoder weight expansion/copy rule: not yet implemented — planned: build new `(1024,14)`/`(14,1024)`/`(4096,14)` Linear layers, copy exp0019's `(1024,7)`/`(7,1024)`/`(4096,8)` weights into the first 7/7/8 columns/rows respectively, initialize the remaining new columns/rows fresh (small-random or zero, TBD by experiment design).
- parent output-head weight expansion/copy rule: same expansion approach as the action encoder (see above), applied to `mixtures.action.head`.
- initialization of new channels/parameters: TBD — will be decided as part of implementing the first multi-embodiment candidate, informed by the literature-review findings.
- inference slicing/decoding by embodiment: crop the model's 14-channel output back to 7 (LIBERO) or 14 (RoboTwin) channels depending on which embodiment is being run, before un-normalizing.
- embodiment/control conditioning used, if any: not yet decided — TBD from literature review (Qwen-VLA-style embodiment id, or implicit channel-position encoding).
- tests proving padded channels do not influence valid-channel gradients/loss: not yet written — required before expensive training per project rules.
- tests proving inherited LIBERO channel behavior is preserved before training: not yet written — required before expensive training per project rules.

**Critical finding — an unwired padding mechanism already exists in the codebase.** `src/fastwam/datasets/lerobot/transforms/action_state_merger.py::ConcatLeftAlign` already implements almost exactly this Qwen-VLA-style scheme (`_pad(x, dim)` right-pads to a target dim and returns `(x_padded, mask)`; produces `batch["action_dim_is_pad"]`/`batch["state_dim_is_pad"]`; `backward()` crops back to the natural per-embodiment dim) but is **currently unused** — both LIBERO's and RoboTwin's data configs leave `action_target_dim`/`state_target_dim` as `null` (no padding applied today, since each currently trains as a single-embodiment run). `fastwam_processor.py::preprocess` already forwards `action_dim_is_pad`/`proprio_dim_is_pad` into the sample dict, **but `fastwam.py::FastWAM.build_inputs` and `training_loss` never read `action_dim_is_pad`** — the current loss does `F.mse_loss(...).mean(dim=2)` over *all* action channels unconditionally (`fastwam.py` ~lines 550-556). This is the exact, precisely-located wiring gap: thread `action_dim_is_pad` through `build_inputs`/`training_loss` (and the analogous path in `fastwam_idm.py`) to mask the per-channel MSE before the `dim=2` mean. There is also a separate, unwired `embodiment_datasets`/`max_action_dim`/`max_state_dim` Hydra-resolver scaffold (`src/fastwam/utils/config_resolvers.py`, `normalizer.py::search_dataset_stats_cache_json`) intended for exactly this multi-embodiment case, with no current config or trainer code path that consumes it — available tooling to build on, not a working path today.

**Checkpoint-loading hazard confirmed**: `FastWAM.load_checkpoint()` calls `mot.load_state_dict(payload["mot"], strict=False)` — `strict=False` tolerates *missing/unexpected keys* but **still raises on a shape mismatch for a key present in both** state dicts. Naively loading exp0019's 7/8-dim checkpoint into a widened 14/14-dim model will crash, not silently degrade — confirms custom weight-expansion code (described above) is required before any checkpoint load into the widened model, not just `strict=False`. Separately, `ActionDiT.from_pretrained` (used only for loading the Wan22-interpolated action-backbone pretrain, not the exp0019 finetune) already explicitly skips `action_encoder.`/`head.` prefixes via `ACTION_BACKBONE_SKIP_PREFIXES` — useful precedent confirming the project's existing design expects these two layers to be handled specially per-dataset.

## Training

Canonical FastWAM training entry point:

```bash
# precompute T5 text-embedding cache first (once per task config)
python scripts/precompute_text_embeds.py task=<task_name>
# or multi-GPU:
torchrun --standalone --nproc_per_node=<N> scripts/precompute_text_embeds.py task=<task_name>

# training (wraps `accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml` -> scripts/train.py, Hydra-based)
bash scripts/train_zero1.sh <NPROC_PER_NODE> task=<task_name> resume=<path> output_dir=./runs/<task>/<run_id> [hydra overrides...]
```

Required arguments/overrides:

- parent checkpoint: `resume=<path>` — a **file** path = weights-only load (via `model.load_checkpoint()`, applied before `accelerator.prepare()` to avoid DeepSpeed ZeRO master-copy staleness); a **directory** path = full training-state resume (`accelerator.load_state()`: optimizer/scheduler/step) followed by `_resync_scheduler_to_global_step()` (the `b2b49d0` fix, rebuilds the scheduler in the current shape after `max_steps` changes).
- output directory: `output_dir=./runs/<task>/<run_id>`.
- resume behavior: see above (file=weights-only, directory=full-state).
- save/checkpoint behavior: `save_every` (task config field, e.g. 500/2500) controls cadence; separate `eval_every` controls any in-loop eval.
- distributed launcher: `accelerate launch` + DeepSpeed ZeRO-1, `scripts/train_zero1.sh <NPROC_PER_NODE> ...`. This machine has 4 GPUs (FastWAM's own recipe used 8 for LIBERO / 64 for RoboTwin) — expect to scale batch/grad-accum or accept longer wall time.

Multi-embodiment controls:

- LIBERO dataset config(s): `configs/data/libero_2cam_plus90_reweighted_v3.yaml` family (task-level oversampling via directory-list repetition).
- RoboTwin dataset config(s): `configs/data/robotwin.yaml` (single combined preprocessed directory).
- dataset/sample mixing mechanism: **not yet implemented for cross-embodiment mixing.** `RobotVideoDataset` (`src/fastwam/datasets/lerobot/robot_video_dataset.py`) takes one flat `dataset_dirs` list under one shared `shape_meta`/processor; LIBERO (`image`/`wrist_image` raw camera keys) and RoboTwin (`cam_high`/`cam_left_wrist`/`cam_right_wrist`) use incompatible raw camera-key schemas, so the two embodiments **cannot** currently be loaded as one `RobotVideoDataset` instance — will need either two dataset instances combined via a weighted concat/sampler, or new code. This is a required first-candidate implementation task, not yet done.
- per-dataset/per-task sampling weights: LIBERO uses directory-repetition (see above); no existing cross-embodiment weighting mechanism.
- action/state normalization handling: per-dataset (`pretrained_norm_stats` field) — LIBERO min/max vs. RoboTwin z-score, must remain distinct per dataset even when mixed (normalize before padding to shared K, per "Shared padded action representation" above).
- per-embodiment loss/mask handling: **not yet implemented** — this is the precise gap described above (`action_dim_is_pad` computed but unused in `fastwam.py::training_loss`).
- trainable/frozen modules: not yet decided for the multi-embodiment candidate (default today: full fine-tune, no freezing).
- embodiment/task conditioning: not yet decided — TBD from literature review.

How to preserve/load intermediate checkpoints:

- checkpoint naming/path: weights checkpoints at `{output_dir}/checkpoints/weights/step_{global_step:06d}.pt` (matches the parent's own path convention, e.g. `.../step_005000.pt`); full training-state checkpoints at `{output_dir}/checkpoints/state/step_{global_step:06d}/` plus a `trainer_state.json` (global_step, epoch, batch_in_epoch).
- save cadence controls: `save_every` in the task config.
- full-state checkpoint location/cost: `{output_dir}/checkpoints/state/step_XXXXXX/` — includes optimizer/scheduler/dataloader-position state, larger than weights-only.
- weights-only checkpoint location/cost: `{output_dir}/checkpoints/weights/step_XXXXXX.pt` — `torch.save({"mot": ..., "proprio_encoder": ..., "step":, "torch_dtype":})`, ~12GB for the current model size (confirmed from exp0019's file size).
- how to resume exactly from an intermediate checkpoint: pass the full-state **directory** as `resume=`.
- how to resume weights-only for a new stage: pass the weights **file** as `resume=` (this is how the multi-embodiment run will bootstrap from exp0019).

Dataloader resume support: `ResumableEpochSampler` (`src/fastwam/utils/samplers.py`) supports mid-epoch resume via `set_resume_batch_offset`.

## Setup smoke tests

Run during initial setup only and repeat only after a material infrastructure/interface/checkpoint-format change.

### LIBERO evaluation smoke/sentinel

Command (the working, final form — two earlier attempts hit real issues, both resolved, see below):

```bash
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=/workspace/FastWAM/checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt \
  EVALUATION.dataset_stats_path=/workspace/FastWAM/checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/dataset_stats.json \
  EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
  model.redirect_common_files=false
```

- tasks/suites/trials: all 10 LIBERO-Spatial tasks, 3 trials/task (30 episodes total).
- parent reference result under this exact smoke/sentinel panel: no exact 3-trial reference exists, but canonical (50-trial) exp0019 Spatial = 97.00%, and the confirmation-precision (15-trial) run recorded 96.00% — this smoke result should land in the same neighborhood.
- **result: 96.67% (29/30) — 9 of 10 tasks at 100% (3/3), one task (`libero_spatial_4`, "pick up the black bowl in the top drawer of the wooden cabinet", the exact task exp0019's own training history targeted) at 66.67% (2/3)**. This is a near-exact match to the canonical 97.00% and confirmation-precision 96.00% Spatial numbers, and `libero_spatial_4`'s specific weakness matches exp0019's own recorded per-task history (76%->88% across evaluation precisions) — strong positive evidence the fresh-machine setup faithfully reproduces the parent's real behavior, not a coincidence or a broken/lucky setup.
- pass criteria: met — no near-zero/crash pattern, per-task results consistent with the inherited canonical record.
- artifact path: `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260817_221019/` (`summary.json`, `summary.csv`, `task_success_rates.csv`, per-task logs+videos).

Two earlier attempts failed for real, diagnosed reasons (not silently ignored):
1. **First attempt** crashed instantly (~16s) on all 10 tasks: `model.redirect_common_files` defaults `true` and points the VAE/text-encoder loader at a dead HF repo (`DiffSynth-Studio/Wan-Series-Converted-Safetensors`, confirmed 404) regardless of `skip_dit_load_from_pretrain`. **Fix**: always pass `model.redirect_common_files=false` on every eval/training command (see "FastWAM model-loading gotcha" in `research/NOTES.md`) — this is now a required standing override, not just a smoke-test workaround.
2. **Second attempt** (after the fix) ran for real but hit `CUDA out of memory` on 2 of 10 tasks: `MULTIRUN.max_tasks_per_gpu=5` packs 5 full model instances (~14-20GB each) per GPU, exceeding 80GB. **Fix**: `MULTIRUN.max_tasks_per_gpu=2` (or lower) is required when running multiple full-model eval workers per A100-80GB — record this as a standing constraint for this hardware, not just this smoke test.

The purpose is to catch a broken fresh-machine setup, not to replace the inherited full canonical parent record. **Both purposes served**: infra is confirmed working, and per-task fidelity to the parent's own history is strong positive evidence beyond a bare pass/fail.

### RoboTwin evaluator/environment smoke

Two levels, both **PASS**:

1. Bare SAPIEN render smoke (`third_party/RoboTwin/script/test_render.py`): required `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` (sapien's own Vulkan ICD auto-detection fails without it despite the ICD file existing at that exact path; persisted in `${WORKSPACE}/.env`). Result: `Render Well`, clean exit.
2. Full environment instantiation (custom script mirroring `script/eval_policy.py`'s arg construction, minus the policy/checkpoint — saved at `checkpoints/robotwin_env_smoke_test.py` for reuse): loaded `task_config/demo_clean.yml` + `_embodiment_config.yml` + `_camera_config.yml`, instantiated `envs.click_alarmclock.click_alarmclock`, called `setup_demo(embodiment=[aloha-agilex])`, then `get_obs()`. **Exercises the full simulator stack**: table/wall creation, bimanual aloha-agilex robot loading, curobo motion-planner init for both arms, camera setup, actor loading + stability check, 3-camera RGB rendering. Result:
   ```
   Embodiment: aloha-agilex  left_robot_file=./assets/embodiments/aloha-agilex/
   Task env class instantiated: <class 'envs.click_alarmclock.click_alarmclock'>
   setup_demo() completed without error
   get_obs() keys: ['observation', 'pointcloud', 'joint_action', 'endpose']
     head_camera rgb shape: (240, 320, 3)
     left_camera rgb shape: (240, 320, 3)
     right_camera rgb shape: (240, 320, 3)
   ROBOTWIN_ENV_SMOKE_TEST_PASSED
   ```
- checkpoint used for smoke/reference: none — this level deliberately excludes the policy/checkpoint (pure simulator-stack verification); FastWAM policy evaluation with a real checkpoint is a follow-up, not yet run.
- task/difficulty/instruction mode: `click_alarmclock` task, `demo_clean` config, `aloha-agilex` embodiment (one of the 50 canonical tasks, chosen as a representative smoke case — not all 50 individually audited; a per-task asset gap would surface during real evaluation, not this generic smoke test).
- pass criteria: met — no crash, correct camera shapes/keys, both arms' motion planners initialize.
- artifact path: `checkpoints/robotwin_env_smoke_test.py` (the smoke script itself, kept for reuse).

Do not treat the RoboTwin specialist as the research parent. (The FastWAM RoboTwin specialist checkpoint, `checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`, was already downloaded earlier as an evaluator reference — an actual policy-in-the-loop RoboTwin eval run with it has not yet been executed as of this note.)

### Training smoke

Command (final working form — first attempt OOM'd, see below):

```bash
bash scripts/train_zero1.sh 4 task=libero_uncond_2cam224_1e-4 \
  resume=/workspace/FastWAM/checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt \
  output_dir=./runs/_smoke_test/exp0019_resume \
  model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.redirect_common_files=false \
  max_steps=5 save_every=5 log_every=1 eval_every=999999 batch_size=1
```

- data path exercised: LIBERO base path (`libero_uncond_2cam224_1e-4`, the 4 available suites — Spatial/Object/Goal/Long-10, no LIBERO-90 due to the known data gap).
- multi-embodiment path exercised: not yet — this is the base single-embodiment training path, confirming infra correctness before the multi-embodiment implementation begins.
- optimizer update verified: **yes** — 5 real training steps completed, losses `0.2085 -> 0.3249 -> 0.3274 -> 0.2167 -> 0.3261` (action/video sub-losses stayed in a healthy range throughout, no NaN, no crash).
- checkpoint write verified: **yes** — `runs/_smoke_test/exp0019_resume/checkpoints/weights/step_000005.pt` written (confirmed present in the training log; run directory since cleaned up per retention policy — logs are the durable evidence for this smoke run).
- artifact path: `checkpoints/train_smoke2.log` (final successful attempt).

**First attempt failed for a real, diagnosed reason**: `batch_size=2` on 2 GPUs completed step 1 successfully (loss=0.1745, action/video sub-losses sane, checkpoint resume worked with no shape errors) but OOM'd on step 2's DeepSpeed ZeRO-1 `optimizer.step()` (master fp32 weight/gradient partition allocation exceeded 80GB). **Fix**: `batch_size=1` with 4 GPUs (more ZeRO-1 sharding) succeeded cleanly. Record `batch_size<=1` and `>=4` GPUs (for ZeRO-1 sharding headroom) as a starting point for sizing real training runs on this hardware — full training will need proper tuning (gradient accumulation, possibly ZeRO-2/3) once real batch/throughput requirements are known, this is just the smoke-test-safe floor.

Separately fixed: LIBERO's own `get_task_init_states()` (`libero/libero/benchmark/__init__.py:164`, in the separately-cloned `/workspace/third_party_src/LIBERO` — not FastWAM's tracked source) calls `torch.load()` without `weights_only=False`, incompatible with PyTorch 2.6+'s changed default. Patched to `torch.load(init_states_path, weights_only=False)`. This is an environment-level patch (like the LIBERO namespace-package `.pth` fix), not a FastWAM repo change.

### Reload/resume smoke

Command: same training entry point, `resume=./runs/_smoke_test/exp0019_resume/checkpoints/weights/step_000005.pt` (weights-only file), `output_dir=./runs/_smoke_test/reload_resume`, `max_steps=7`.

- checkpoint type tested: weights-only file resume (the type used to bootstrap the multi-embodiment candidate from exp0019).
- load/resume behavior verified: **yes** — loaded without shape/key errors, ran 7 further real steps (losses `0.7086 -> 1.3866 -> 0.6076 -> 0.2169 -> 0.5582 -> 0.5121 -> 0.4213`, no NaN/crash). **Confirmed empirically: weights-only `resume=<file>` restarts the step counter at 1 (not continuing global_step from 5)** — matches the documented file-vs-directory resume semantics (file=weights-only, directory=full-state-with-step-continuation) in the "Training" section above; a full-state directory checkpoint was also written and validated during this test (optimizer shards + scheduler + `trainer_state.json`) as the path for exact-continuation resume.
- artifact path: `checkpoints/reload_smoke.log`.

All smoke-test `runs/_smoke_test/*` directories (~183GB of checkpoints/logs) were deleted after confirming results, per the retention policy (reproducible temporary artifacts, not the exp0019 parent or any promoted checkpoint) — logs above are the preserved evidence. `/workspace` has 951GB free after cleanup.

## Canonical LIBERO evaluation

Canonical FastWAM/LIBERO entry point:

```bash
python experiments/libero/run_libero_manager.py task=<task_name> ckpt=<ckpt_path> \
  [EVALUATION.dataset_stats_path=<stats.json>] [EVALUATION.task_suite_name=libero_90] \
  [MULTIRUN.num_gpus=4]
```

Suites:

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long / LIBERO-10
- LIBERO-90

Fixed settings:

- trials per task: `EVALUATION.num_trials: 50` (matches the inherited "50 trials/task" canonical record).
- task membership: **`MULTIRUN.task_suite_names` default list is `[libero_10, libero_goal, libero_spatial, libero_object]` — `libero_90` is NOT included by default** and must be requested explicitly (`task_suite_name=libero_90`, or add to the multirun override list) to cover all five suites. Must fix this explicitly every canonical run.
- seeds/initial-state behavior: not yet separately verified (uses LIBERO's own benchmark init-state files by default).
- episode horizon: not yet separately verified (`num_steps_wait: 30` warmup steps recorded; full episode horizon not yet read from code).
- prompt/instruction behavior: not yet separately verified beyond the T5 text-embedding pipeline shared with training.
- action/replan behavior: `replan_steps: 10`, `binarize_gripper: true`, `use_action_ensembler: false` (`configs/sim_libero.yaml` defaults).
- metric aggregation: per-task success rate; suite aggregate = mean across the suite's tasks. Exact output-file schema (`experiments/libero/summarize_results.py`) not yet read — flagged as a follow-up before fully trusting automated parsing.
- raw result format/path: `./evaluate_results/libero/${task}/${timestamp}/`.

Promotion gate: every suite above must be >=90%.

## Canonical RoboTwin evaluation

Canonical FastWAM/RoboTwin entry point:

```bash
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 ckpt=<ckpt_path> \
  [MULTIRUN.num_gpus=4]
```

Record the actual official/FastWAM benchmark semantics before comparing candidates:

- RoboTwin repository revision/assets: vendored `third_party/RoboTwin`, upstream commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95`; full env/asset install in progress (dedicated setup pass), result pending.
- task set: loaded dynamically from `third_party/RoboTwin/task_config/_eval_step_limit.yml` (keys = task names) via `run_robotwin_manager.py::_load_all_tasks()`. **Confirmed: exactly 50 tasks** — the file was fetched from upstream RoboTwin at the pinned vendor commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95` (it and other `task_config/*.yml` files were missing from the vendored copy and have now been added). This matches the project goal's "full 50-task Aloha-AgileX benchmark" exactly. `third_party/RoboTwin/envs/` has 53 raw task-env `.py` files (a slightly larger raw count; the 50-task `_eval_step_limit.yml` is the authoritative canonical eval list).
- easy/hard/domain-randomization split(s): **confirmed** — two phases per task, `EVALUATION.task_config=demo_clean` and `EVALUATION.task_config=demo_randomized`. This is exactly the project goal's "Clean and Randomized, evaluated separately" requirement.
- instruction mode/template: `EVALUATION.instruction_type: unseen` (default, per `deploy_policy.yml`/README) — FastWAM's own stated/evaluated protocol; a `seen` alternative exists (used by a different baseline, Lingbot-VA) but is **not** the project's frozen setting. Freeze at `unseen`.
- episodes/trials per task: `EVALUATION.eval_num_episodes: 100` per task per phase (clean and randomized each get 100).
- seeds/scenes/randomization: not yet separately verified beyond the clean/randomized phase distinction.
- episode horizon: not yet separately verified.
- action execution/replan behavior: `replan_steps: 24` (vs. LIBERO's 10); `skip_get_obs_within_replan: true` by default (speed optimization — skips RGB rendering while executing an action chunk within one replan window; does not affect success semantics, only makes saved videos low-FPS; can set `false` for fully rendered video).
- success semantics: per RoboTwin's own task-level success check (not yet independently re-verified from `envs/` code).
- metric aggregation: per-task-per-phase success rate parsed from `evaluate_results/robotwin/{ckpt_tag}/{run_ts}/{task_name}/_result_{clean|random}.txt` (last numeric line), aggregated to `summary.csv`/`summary.json` as **unweighted mean across tasks**, kept separate per phase.
- primary project comparison metric(s): **`clean_mean_success_rate` and `random_mean_success_rate`, tracked and compared separately** — matches the project's explicit "evaluate Clean and Randomized separately" requirement; there is no single blended RoboTwin scalar by default.
- tie-break/worst-case rule if multiple primary metrics exist: **frozen now, before any candidate has been evaluated** (zero RoboTwin results exist yet for any candidate as of this note — legitimate point to freeze). Per the project goal's explicit "evaluating Clean and Randomized separately" instruction, `clean_mean_success_rate` and `random_mean_success_rate` are never blended into a single scalar for ranking. Comparison rule, mirroring `GOAL.md`'s worst-case-margin philosophy for LIBERO: **(1) primary — the candidate with the higher `min(clean_mean_success_rate, random_mean_success_rate)` wins** (rewards balanced improvement, penalizes a candidate that's strong on Clean but collapses on Randomized or vice versa); **(2) tie-break — the candidate with the higher `mean(clean_mean_success_rate, random_mean_success_rate)` wins** if the worst-case values are effectively tied; **(3) further tie-break — the same secondary criteria as the LIBERO ranking rule in `GOAL.md`** (stability/reproducibility, lower unnecessary compute, simpler implementation). This rule is now fixed and must not change after seeing candidate results.
- raw result format/path: `evaluate_results/robotwin/{ckpt_tag}/{run_ts}/{task_name}/_result_{clean|random}.txt`, aggregated `summary.csv`/`summary.json`.
- expected runtime/cost: not yet measured (pending working evaluator).

Freeze this comparison rule before candidate optimization. Do not change it after seeing candidate results. **Status: task list confirmed (50 tasks) and tie-break rule frozen above — both fixed before any candidate optimization. `expected runtime/cost` remains unmeasured (pending a first real policy-in-the-loop RoboTwin evaluation run, not yet executed as of this note).**

## Adaptive dual-benchmark evaluation plan

Evaluation can be used during and after training. Choose exact task/trial counts from measured cost/variance and current weaknesses, not from a generic rule.

### Cheap progress / candidate panel

Purpose: decide whether to continue/extend/stop/select a checkpoint without paying full canonical cost.

RoboTwin progress component:

- tasks/difficulties:
- why informative:
- trials:
- command/config:
- accepted-parent/candidate reference on exact panel:

LIBERO retention sentinel component:

- tasks/suites:
- why informative for forgetting:
- trials:
- command/config:
- inherited/accepted reference on exact panel:

Expected total cost/runtime:

Direct comparisons must use matching settings.

### Progress-check guidance

No universal fixed interval. Run when the evidence can change remaining compute allocation.

Possible outcomes:

- continue training;
- extend training;
- stop early;
- select an intermediate checkpoint;
- train more and recheck;
- diagnose.

### Broader confirmation

RoboTwin coverage:

- tasks/difficulties:
- trials:
- command/config:
- accepted reference:

LIBERO retention coverage:

- suites/tasks:
- trials:
- command/config:
- accepted reference:

Expected cost/runtime:

### Canonical promotion evaluation

A checkpoint can be promoted only after:

1. canonical RoboTwin evaluation under the frozen protocol; and
2. canonical evaluation on all five LIBERO benchmarks showing each >=90%.

### Targeted diagnostic panels

Candidate-specific panels are allowed for concrete questions: embodiment interference, specific task failures, padding/masking correctness, normalization issues, camera mismatch, or checkpoint-stage behavior. Record exact purpose/settings. Diagnostic evidence is not promotion evidence.

### Avoid overfitting cheap panels

Cheap panels are compute-allocation tools. Broaden/rotate/revise when they become unrepresentative, and never compare different panel definitions as if directly equivalent.

## Durable checkpoint storage

Hugging Face model repository: `cheikh025/ASR`

Authentication:

- source: `HF_TOKEN`
- setup authentication verified: **yes** — `HfApi().whoami()` succeeded
- authenticated identity: `cheikh025`
- never record token value

Verified parent download method:

```bash
source /workspace/venvs/fastwam/bin/activate  # or any env with huggingface_hub installed
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cheikh025/ASR', repo_type='model',
    allow_patterns=['promoted/0019_spatial_weak_task_oversampling/*'],
    local_dir='checkpoints/exp0019_parent_hf')
"
```

Verified upload method: not yet exercised this session (no new checkpoint produced yet) — will use `huggingface_hub.upload_file`/`upload_folder` against `cheikh025/ASR` when the first candidate checkpoint is ready to persist, following the same `promoted/<name>/` or a new `branch/<name>/` path convention as the parent.

Remote path convention:

- promoted multi-embodiment checkpoints:
- branch checkpoints:
- continuation/resume checkpoints:
- associated reports/configs:

Remote verification method:

```text
<fill during setup>
```

Retention policy:

- keep local checkpoints after upload by default;
- preserve active/valuable full-state checkpoints when exact continuation may be useful;
- remove reproducible temporary artifacts before checkpoints under disk pressure;
- verify exact remote artifact before deleting local checkpoint;
- never delete the active parent/current continuation checkpoint merely to save space.

## Output locations

- training logs: `runs/{task}/{run_id}/train.log` (by convention, matching exp0019's `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/train.log`)
- checkpoints: `runs/{task}/{run_id}/checkpoints/weights/step_XXXXXX.pt` (weights-only) and `.../checkpoints/state/step_XXXXXX/` (full state); parent/reference checkpoints under `/workspace/FastWAM/checkpoints/` (`exp0019_parent_hf/`, `fastwam_release/`)
- evaluation logs: `evaluate_results/libero/{task}/{timestamp}/`, `evaluate_results/robotwin/{ckpt_tag}/{run_ts}/`
- parsed metrics: LIBERO — TBD exact field names pending `experiments/libero/summarize_results.py` read; RoboTwin — `summary.csv`/`summary.json` with `clean_mean_success_rate`/`random_mean_success_rate` plus per-task `clean_success_rate`/`random_success_rate`
- progress-evaluation results: same `evaluate_results/` tree, distinguished by output_dir naming per run
- system/dependency snapshots: `research/tools/capture_system_info.py` output, to be captured for `PROGRESS_0000_PARENT_BASELINE.md`

## Metric extraction

Canonical LIBERO fields:

```text
libero_90
libero_spatial
libero_object
libero_goal
libero_long
```

Exact output-field names from `experiments/libero/summarize_results.py` not yet independently verified — treat the above as the intended mapping until confirmed.

Canonical RoboTwin fields discovered during setup:

```text
clean_success_rate        # per-task, clean split
random_success_rate       # per-task, randomized split
clean_mean_success_rate   # aggregate across tasks, clean split
random_mean_success_rate  # aggregate across tasks, randomized split
```

No single blended RoboTwin scalar exists by default — clean and randomized are tracked as two separate primary metrics per the project goal. Task list underlying the aggregate is not yet finalized (pending RoboTwin environment/asset setup).

Also preserve per-task results when available.

## Failure checks

- successful training indicator:
- failed training indicator:
- useful log tail command:
- OOM indicator:
- NaN/numerical failure indicator:
- missing-checkpoint indicator:
- LIBERO evaluation failure indicator:
- RoboTwin evaluation failure indicator:
- simulator/asset failure indicators:
- action-dimension/mask mismatch indicators:
- unexpected LIBERO-forgetting indicator:
