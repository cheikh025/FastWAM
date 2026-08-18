# PROGRESS_0001 — padded_multiembodiment_baseline

- **Experiment ID:** 0001_padded_multiembodiment_baseline
- **Status:** `RUNNING`
- **Created:** 2026-08-17
- **Updated:** 2026-08-17
- **Parent experiment:** 0000_parent_baseline
- **Parent checkpoint:** `cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt` (exp0019)
- **Selected candidate checkpoint:** TBD — implementation complete and unit-tested; text-embedding precompute in progress; training not yet launched
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** TBD (recorded once implementation is committed)

## 1. Result at a glance

TBD — filled once training/evaluation evidence exists.

## 2. Research state before experiment

### Accepted RoboTwin state

- canonical metric(s): none yet — no RoboTwin policy-in-the-loop evaluation has been run.
- important weak tasks/difficulties: unknown.

### Accepted LIBERO retention state

Inherited canonical reference (prior project):

| Suite | Success |
|---|---:|
| LIBERO-90 | 95.13% |
| LIBERO-Spatial | 97.00% |
| LIBERO-Object | 99.60% |
| LIBERO-Goal | 97.20% |
| LIBERO-Long / LIBERO-10 | 98.00% |

Fresh-machine sentinel (this session, setup phase): LIBERO-Spatial 96.67% (3 trials/task, 10 tasks).

This candidate is not closing a specific gap — it is the mandated first step (`CLAUDE.md`/`GOAL.md`: "Establish a correct padded multi-embodiment baseline as the initial shared action interface") before any RoboTwin performance exists to improve. The opportunity is that RoboTwin performance is currently zero (no multi-embodiment capability exists), and the entire remaining research loop is blocked until this baseline exists.

## 3. Candidate design

### Modifications

1. Activate the already-implemented-but-unused `ConcatLeftAlign` padded-action mechanism: set `action_target_dim=14`, `state_target_dim=14` in new LIBERO and RoboTwin data configs (K=14 = RoboTwin's natural dim; LIBERO's 7/8 pads up to 14/14).
2. Fix the exact wiring gap identified during setup: thread `action_dim_is_pad` from `RobotVideoDataset._get()` (currently dropped, not included in the returned sample dict despite the processor producing it) through `FastWAM.build_inputs` into `training_loss`, and implement the precise Qwen-VLA two-level per-channel masked-loss formula (mask-then-mean-per-channel over valid timesteps, then mean over the valid-channel-count `c`, not the full `K`) in place of the current `.mean(dim=2)` over all channels. Mirror in `fastwam_idm.py` if it has an analogous path.
3. Write a checkpoint-expansion script that grows exp0019's `action_encoder` (Linear 7->14 in), `head` (Linear 7->14 out), and `proprio_encoder` (Linear 8->14 in) into the new K=14 shapes, copying inherited weights into the original index positions exactly and initializing new parameters deliberately (new encoder input columns: fresh-Linear-equivalent random init, matching what a from-scratch model of the new size would produce; new head output rows: zero-init, so unlearned RoboTwin-specific output channels start at a defined, bounded 0 rather than injecting large random values into the flow-matching target space).
4. Build a dataset-mixing mechanism: since LIBERO (224x448, 2-cam horizontal-concat) and RoboTwin (384x320, 3-cam "robotwin"-layout) produce incompatibly-shaped video tensors, true sample-level batch mixing is not possible with the model's fixed-shape video input per forward call. Implement batch-level interleaving instead: a `torch.utils.data.ConcatDataset` combining a LIBERO `RobotVideoDataset` and a RoboTwin `RobotVideoDataset` (each internally homogeneous, each padded to K=14), plus a new custom sampler (`InterleavedEmbodimentSampler`) that yields batches which are each entirely from one sub-dataset, interleaved across steps according to a configurable ratio — approximately 1:1 RoboTwin:LIBERO per the literature review's replay-ratio recommendation, not proportional to each dataset's raw size (RoboTwin's ~14k+ episodes vs. LIBERO's much smaller replay set).
5. Add a Qwen-VLA-style textual embodiment-conditioning prompt, prepended to the instruction in `FastWAMProcessor.augment_instruction()`, configurable per dataset (LIBERO: single-arm description; RoboTwin: bimanual aloha-agilex description).
6. New configs wiring all of the above: `configs/data/libero_2cam_multiembodiment.yaml`, `configs/data/robotwin_multiembodiment.yaml`, `configs/data/multiembodiment_libero_robotwin.yaml`, `configs/model/fastwam_multiembodiment.yaml` (hardcodes `action_dim=14`/`proprio_dim=14` instead of the single-dataset Hydra interpolation, and bakes in `redirect_common_files=false`/`skip_dit_load_from_pretrain=true`/`action_dit_pretrained_path=null` since this always resumes from an expanded exp0019 checkpoint rather than building fresh from the Wan22 backbone), `configs/task/multiembodiment_libero_robotwin_1e-4.yaml`.
7. Correctness tests, written and passing **before** any real training: (a) numerically confirm the padded+masked loss on a LIBERO-shaped (7-valid-of-14) synthetic batch produces the identical value to running the same batch through the original unwidened path (proves padding+masking doesn't alter LIBERO's effective training signal); (b) numerically confirm the expanded checkpoint reproduces exp0019's exact output on the 7 valid LIBERO channels for identical input (proves the expansion preserves inherited behavior exactly, not just "looks right").

### Why this candidate

This is not a competitively-chosen candidate among alternatives — it is the mandated first step per the project contract (`CLAUDE.md`: "For the initial heterogeneous action interface, use a shared padded action representation... Establish a correct padded multi-embodiment baseline as the initial shared action interface. Later candidates remain evidence-driven..."). Every later candidate in the research loop depends on this existing first. The design choices within it (K=14, two-level masked loss, textual embodiment conditioning, ~1:1 batch-level replay ratio, zero-init new head rows / random-init new encoder columns) are drawn directly from the full Qwen-VLA paper read during setup and cross-checked against the actual FastWAM code (not guessed) — see `research/RUNBOOK.md` "Literature/implementation understanding gate" and "Shared padded action representation" sections.

### Multi-embodiment representation/configuration

- shared action dimension: K=14 (action and state/proprio both).
- LIBERO valid channels/mask: action channels 0-6 valid (delta-eef x6 + gripper x1), 7-13 padding; proprio channels 0-7 valid, 8-13 padding.
- RoboTwin valid channels/mask: all 14 action and all 14 proprio channels valid (no padding — RoboTwin already at K).
- normalization/statistics behavior: unchanged, per-dataset (LIBERO min/max, RoboTwin z-score), applied in each dataset's own natural (un-padded) dimensionality before `ConcatLeftAlign` pads to K=14.
- checkpoint projection expansion/initialization: see Modification 3 above; implemented as a standalone, reusable script, not inline training code.
- embodiment/control conditioning: textual prompt prepended per dataset (Modification 5).
- camera/observation handling: unchanged per-dataset (LIBERO 2-cam 224x448, RoboTwin 3-cam 384x320) — this is exactly why sample-level mixing is infeasible and batch-level interleaving is used instead.
- inference slicing/decoding: `ConcatLeftAlign.backward()` (already implemented) crops the model's 14-channel output back to the natural per-embodiment dimension before un-normalizing.

### Data and learning strategy

- LIBERO datasets/tasks: the 4 available suites (Spatial/Object/Goal/Long-10) — LIBERO-90 excluded due to the known data gap (no downloadable lerobot-format LIBERO-90 training data; does not block evaluation, only replay).
- RoboTwin datasets/tasks: the full preprocessed `data/robotwin2.0/robotwin2.0` directory (spans all training tasks internally).
- sampling/mixing ratios: approximately 1:1 RoboTwin:LIBERO at the batch level (interleaved, not proportional to raw dataset size).
- per-task/per-dataset weights: none beyond the dataset-level 1:1 ratio for this initial baseline (existing LIBERO task-level oversampling in `dataset_dirs` repetition is not carried over for this baseline — using the plain `libero_2cam` composition, not `plus90_reweighted`, since the reweighting was tuned for a LIBERO-only regime and reintroducing it is a later, evidence-driven candidate, not part of the mandated baseline).
- replay/rehearsal strategy: every training step interleaves toward the ~1:1 ratio (replay-by-construction each step, matching the forgetting-resistance literature's finding that even modest, evenly-distributed replay is highly effective for a pretrained/converged model).
- loss weights: unchanged from parent (`loss.lambda_action: 1.0`, implicit `lambda_video: 1.0` default) — no per-embodiment loss reweighting in this initial baseline.
- retention/distillation/regularization: none beyond replay for this initial baseline; the shared-backbone-freezing warm-up idea from the literature review is noted as a candidate refinement, not required for this first baseline.
- trainable/frozen modules: full fine-tune of `self.model.dit` + `proprio_encoder` (matching `Trainer._apply_dit_only_train_mode`'s existing behavior) — no freezing in this initial baseline, to keep the first candidate's design as close to the established single-embodiment training recipe as possible and isolate the effect of the padding/masking/mixing mechanism itself.

### What to watch

- Whether LIBERO retention holds during/after training on the padded, mixed-embodiment data (progress-check LIBERO sentinel evidence) — this is the hard constraint.
- Whether RoboTwin shows any learning signal at all (loss decreasing, later: any nonzero task success) — first evidence the multi-embodiment interface actually works end-to-end.
- Whether the correctness tests (padding-invariance, checkpoint-expansion-invariance) pass before training starts — if either fails, do not proceed to training; fix the implementation first.
- Signs of embodiment interference (e.g., LIBERO loss spiking when RoboTwin batches are interleaved in, which could indicate the masking isn't actually isolating the two embodiments' gradients correctly).

### Initial compute plan

- Initial training budget: small and conservative for a first mixed-embodiment run — enough steps to see stable, non-NaN loss curves for both embodiments and a first LIBERO progress-check sentinel, not a full convergence run. Exact step count to be set once training smoke-tests the new multi-embodiment path (distinct from setup's single-embodiment smoke test).
- Checkpoint/save plan: frequent early saves to allow rollback/diagnosis if something goes wrong with the new mixing/masking code, pruned once stability is confirmed.
- When a progress check might be useful: as soon as the first few hundred steps complete cleanly — a cheap LIBERO sentinel (same panel style as the setup validation) plus a RoboTwin loss/qualitative check, before committing to a longer run.
- Expected training/evaluation cost: TBD, depends on GPU count used (4x A100-80GB available) and batch/GPU sizing learned from setup's OOM lessons (`batch_size<=1` per GPU as a floor for full-model training, ZeRO-1).

This is a plan, not a promise to consume the whole budget.

## 4. Exact code and configuration state

- Git branch: `autoresearch/robotwin-multiembodiment-v1`.
- Git commit: implementation committed on this branch (see commit immediately following this report's creation); working tree was clean at the parent commit before these changes.
- files changed (source): `src/fastwam/models/wan22/fastwam.py` (thread `action_dim_is_pad` through `build_inputs`/`training_loss`, use shared masked-loss helper), `src/fastwam/models/wan22/fastwam_idm.py` (mirror the same fix), `src/fastwam/utils/losses.py` (new — `masked_action_loss`, the two-level Qwen-VLA-style masked average), `src/fastwam/datasets/lerobot/robot_video_dataset.py` (stop dropping `action_dim_is_pad`/`proprio_dim_is_pad` from the returned sample dict), `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` (add `embodiment_description` config field, prepend to `augment_instruction()`'s output), `src/fastwam/datasets/lerobot/multi_embodiment.py` (new — `build_multi_embodiment_dataset`, a Hydra-instantiable factory wrapping `ConcatDataset` and tagging it with per-embodiment names/ratios), `src/fastwam/utils/samplers.py` (new — `InterleavedEmbodimentSampler`), `src/fastwam/trainer.py` (`_build_loader` selects the new sampler when the dataset carries `embodiment_ratios`).
- files changed (configs, all new): `configs/data/libero_2cam_multiembodiment.yaml`, `configs/data/robotwin_multiembodiment.yaml` (standalone single-embodiment K=14-padding diagnostic variants — useful in isolation, not used directly by the combined training config), `configs/data/multiembodiment_libero_robotwin.yaml` (the actual combined training data config — self-contained rather than composed from the two standalone files above, because Hydra's `defaults:` list substitution can't embed two independent config files as list items with correctly-resolving internal interpolations; per-embodiment `shape_meta` is duplicated between the two forms rather than interpolated — keep in sync if either changes), `configs/model/fastwam_multiembodiment.yaml` (hardcodes `action_dim=14`/`proprio_dim=14`, bakes in `redirect_common_files=false`/`skip_dit_load_from_pretrain=true`/`action_dit_pretrained_path=null`), `configs/task/multiembodiment_libero_robotwin_3e-5.yaml`.
- files added (tools/tests): `research/tools/expand_checkpoint_for_multiembodiment.py` + `research/tools/test_expand_checkpoint.py`, `research/tools/test_masked_action_loss.py`, `research/tools/test_interleaved_sampler.py`, `research/tools/precompute_multiembodiment_text_embeds.py` (a new, separate precompute tool — deliberately does not modify the existing `scripts/precompute_text_embeds.py`, to avoid any risk to single-embodiment LIBERO/RoboTwin training; see rationale in that file's docstring).
- diff summary: `6 files changed, 184 insertions(+), 27 deletions(-)` in `src/fastwam/` (verified via `git diff --stat`); all changes are strictly additive to existing code paths (new optional dict keys, new config fields defaulting to `None`/unused) — no single-embodiment training/eval config exercises any of the new code, confirmed by every new field defaulting to backward-compatible behavior (`action_dim_is_pad=None` -> identical to the pre-change flat mean; `embodiment_description=None` -> instruction unchanged; `InterleavedEmbodimentSampler` only activates when a dataset explicitly carries `embodiment_ratios`).
- training config: `task=multiembodiment_libero_robotwin_3e-5` (see `configs/task/multiembodiment_libero_robotwin_3e-5.yaml`).
- config overrides at launch: `resume=<expanded checkpoint path>` (see below).
- LIBERO dataset config: embedded in `configs/data/multiembodiment_libero_robotwin.yaml`'s `train.embodiments[0]` — 4 available suites (Spatial/Object/Goal/Long-10), `action_target_dim=14`/`state_target_dim=14`, `embodiment_description="The robot is a Franka Panda single-arm manipulator with a parallel-jaw gripper."`.
- RoboTwin dataset config: `train.embodiments[1]` — full preprocessed directory, `action_target_dim=14`/`state_target_dim=14` (no-op padding, already natural K=14), `embodiment_description="The robot is a bimanual ALOHA-AgileX manipulator with two arms, each with a parallel-jaw gripper."`.
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio `[1.0, 1.0]` (approximately 1:1 batch-level interleaving, per the literature review's replay-ratio recommendation — see `configs/data/multiembodiment_libero_robotwin.yaml`'s `embodiments[*].ratio`).
- action/state normalization configuration: unchanged per-dataset (LIBERO min/max, RoboTwin z-score), applied before padding.
- action validity-mask configuration: `action_target_dim=14`/`state_target_dim=14` on `ConcatLeftAlign` for both embodiments (LIBERO: real padding, 7->14/8->14; RoboTwin: no-op, already 14).
- model/trainable-module configuration: full fine-tune of `model.dit` + `proprio_encoder` (unchanged from the established single-embodiment recipe, via `Trainer._apply_dit_only_train_mode`) — no freezing in this initial baseline.
- optimizer / LR / scheduler: AdamW (betas 0.9/0.95, per `Trainer.__init__`), `learning_rate=3e-5`, `lr_scheduler_type=cosine`, `weight_decay=1e-2`.
- batch size / gradient accumulation / effective batch: `batch_size=1` (per GPU, the setup-validated OOM-safe floor), `gradient_accumulation_steps=4`, run on 4 GPUs -> effective batch size 16 (4 GPUs x 1 x 4 accum).
- initial training steps / epochs / budget: `max_steps=1000`, `save_every=100`, `log_every=10`, `eval_every=200`.
- random seed(s): `seed=42` (base config default, unchanged).
- resume source and resume type: weights-only file resume from `checkpoints/exp0019_expanded_k14/step_005000.pt` (the checkpoint-expansion script's output — see Section 2/3 above and `research/tools/expand_checkpoint_for_multiembodiment.py`), verified byte-exact-preserving of exp0019's original weights on the valid 7/8 channels and correctly-shaped/zero-initialized on the new channels (see verification output referenced below).

### Checkpoint expansion — exact command and verification

```bash
python research/tools/expand_checkpoint_for_multiembodiment.py \
  --input checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt \
  --output checkpoints/exp0019_expanded_k14/step_005000.pt \
  --new-action-dim 14 --new-proprio-dim 14 --seed 0
```

Verified directly against the two checkpoint files (not just the isolated-layer unit tests): all 1649 `mot` keys checked, zero unexpected changes to any key other than the four action-encoder/head weight/bias tensors; `mot` key set identical between original and expanded; `proprio_encoder.bias`/`action_encoder.bias` byte-identical (unchanged, as expected — bias shape doesn't depend on input width); original 7 action-encoder input columns and original 7 head output rows preserved exactly (`torch.equal`); new head output rows exactly zero; original 8 proprio-encoder input columns preserved exactly; `step` metadata preserved (5000).

### First real end-to-end training smoke test — three attempts, two real bugs found and fixed, then PASS

Command (final working form):

```bash
bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_3e-5 \
  resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k14/step_005000.pt \
  output_dir=./runs/_smoke_test/multiembodiment_exp0001_v3 \
  max_steps=8 save_every=8 log_every=1 eval_every=999999
```

**Attempt 1** crashed in dataset construction: `InstantiationException('Cannot instantiate config of type RobotVideoDataset...')`. Root cause: Hydra's `instantiate()` recursively instantiates nested `_target_` configs by default, so `build_multi_embodiment_dataset`'s `embodiments` argument arrives with each `entry["dataset"]` *already* a constructed `RobotVideoDataset` object, not a config — my code was calling `instantiate()` on it a second time. **Fix**: `src/fastwam/datasets/lerobot/multi_embodiment.py` now uses `entry["dataset"]` directly. (This attempt's log also confirmed, before it crashed, that both embodiments' actual dataset construction — LIBERO's normalization-stats iteration and RoboTwin's 6M+-row data load — completed successfully, which was strong pre-existing evidence the data configs themselves were correct.)

**Attempt 2**, after the fix: all 8 real training steps completed with sane, non-NaN losses (`1.1457, 0.6372, 1.1269, 1.2040, 5.0337, 1.1654, 0.8971, 1.4368`), confirming the entire forward/backward/masking/interleaving pipeline works — but then crashed during checkpoint saving with `PytorchStreamWriter failed writing file: file write failed`. Root cause: `/workspace` disk was at 100% (1.8MB free out of 1.1TB) — RoboTwin's text-embedding cache turned out to be **~900GB** (921,032 files x ~1MB each; a real, unavoidable cost given RoboTwin's per-episode-unique instructions — see previous subsection), leaving no room for `Trainer.save_checkpoint()`'s unconditional full DeepSpeed ZeRO state save (optimizer shards, substantially larger than the weights-only file). See "Disk crisis" in `research/NOTES.md` for full detail. **Fix**: added a `save_full_state: bool` flag to `Trainer` (default `true`, so every other existing config's behavior is unchanged) — `save_full_state=false` skips the large optimizer-state save and writes only the weights-only checkpoint; set to `false` in this candidate's task config. Also freed 24GB by deleting two redundant, safely-re-downloadable checkpoints (`exp0019_parent_hf`, `fastwam_release`) — both durably backed up on HF, exact re-download commands recorded in `research/RUNBOOK.md`.

**Attempt 3**, after both fixes: **full success**. All 8 steps reproduced the same loss trajectory as attempt 2 (`1.1457, 0.6371, 1.1276, 1.2036, 5.0238, 1.1658, 0.8964, 1.4370` — matching within tiny numerical noise, confirming determinism), and the weights-only checkpoint (`step_000008.pt`, 12,041,813,369 bytes) was written successfully with no disk error. Verified directly: `action_encoder.weight` shape `(1024, 14)`, `head.weight` shape `(14, 1024)`, `proprio_encoder.weight` shape `(4096, 14)` (all correctly at the shared K=14), and **zero NaN/Inf values across every tensor in the entire checkpoint**. This is genuine, verified evidence that the complete shared padded multi-embodiment interface — checkpoint expansion, per-channel masked loss, embodiment-conditioned text, and batch-interleaved LIBERO/RoboTwin data — works end-to-end on real data. Smoke-test run directory deleted after verification (temporary artifact, per retention policy); this report and `research/RUNBOOK.md`/`research/NOTES.md` are the preserved evidence.

The step-5 loss spike (5.02 vs. a 0.6-1.4 range elsewhere) appeared identically in both successful attempts (same seed/data order), consistent with a genuinely harder individual batch (plausibly an early RoboTwin batch, whose action semantics/scale the model hasn't seen since this padded interface didn't exist during any of exp0019's training) rather than instability — worth watching during the real training run, not alarming from a single reproducible data point.

### Text-embedding cache — a discovered, unavoidable one-time cost

RoboTwin's `meta/tasks.jsonl` contains **921,032 unique per-episode natural-language instructions** (not ~50 fixed task-name strings) — confirmed by direct inspection, not assumed. Combined with LIBERO's 40 unique tasks and each embodiment's textual conditioning prefix, this candidate requires precomputing 921,072 unique T5 text embeddings before any training step can run (the runtime path `RobotVideoDataset._get_cached_text_context` raises `FileNotFoundError` on a cache miss — no live-encoding fallback). The existing shared `scripts/precompute_text_embeds.py` cannot produce these correctly (it reads raw un-augmented task strings and writes every embedding into every discovered cache dir, incompatible with per-embodiment prefixed prompts cached to per-embodiment directories) — a new, separate tool (`research/tools/precompute_multiembodiment_text_embeds.py`) was written instead, deliberately not modifying the shared script to avoid any risk to existing single-embodiment training. Run via `torchrun --standalone --nproc_per_node=4 research/tools/precompute_multiembodiment_text_embeds.py task=multiembodiment_libero_robotwin_3e-5`; resumable (skips already-cached prompts), ~100 prompts/sec/rank observed on 4x A100-80GB (~38 min wall-clock for the full set from cold).

## 5. Hardware and software environment

Not applicable to change — identical to `PROGRESS_0000_PARENT_BASELINE.md` Section 5 (same venv, same GPUs, same repos) unless noted otherwise in a later update.

## 6. Training execution and control timeline

TBD.

## 7. Evaluation events

TBD.

## 8. Comparison and interpretation

TBD.

## 9. Decision

TBD.

## 10. What this changes for the next experiment

TBD.

## 11. Artifacts

TBD.

## 12. Reproducibility checklist

- [ ] exact candidate commit and RoboTwin branch recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] multi-embodiment representation/mask/normalization details recorded
- [x] dataset mixture and retention strategy recorded
- [x] initial training plan recorded
- [ ] training configuration/command recorded
- [ ] hardware/software environment recorded
- [ ] intermediate checkpoints/progress decisions recorded when used
- [ ] logs/checkpoint paths recorded
- [ ] remote checkpoint path verified if HF backup created
- [ ] every evaluation event has benchmark/purpose/settings/raw results
- [ ] canonical RoboTwin metrics recorded when canonical evaluation ran
- [ ] five LIBERO suite metrics recorded when canonical retention evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision/reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
