# PROGRESS_0013 — First real multi-embodiment training candidate on the release-checkpoint parent

- **Experiment ID:** 0013
- **Status:** `RUNNING` (real training launched, max_steps=4000)
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0012 (release-checkpoint LIBERO/RoboTwin baseline establishment)
- **Parent checkpoint:** `/home/claudeuser/local_cache/fastwam_release_expanded_zeroinit/step_000000.pt` (official FastWAM LIBERO release checkpoint, expanded to K=21/22 with the zero-init fix)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending (recorded after smoke test passes and training launches)

## 1. Result at a glance

First real training candidate on the new parent (release checkpoint, switched from `exp0019` in `PROGRESS_0011`). Reuses `exp0003`'s architecture (full fine-tune, disjoint K=21/22 offsets, 1:1 LIBERO:RoboTwin ratio, LR 3e-5) — the multi-embodiment padding/masking design is checkpoint-agnostic, and this establishes a first real evidence point for the new, verified-clean parent. Also tests a larger batch size (2, up from the historical `batch_size=1` floor) per earlier user feedback about GPU utilization.

## 2. Research state before experiment

### Accepted LIBERO baseline (release checkpoint, expanded K=21/22, from `PROGRESS_0011`/`exp0012`)

| Suite | Success |
|---|---:|
| LIBERO-Spatial | 98.00% |
| LIBERO-Object | 99.00% |
| LIBERO-Goal | 98.00% |
| LIBERO-Long | 95.00% |

All comfortably above the 90% floor, verified via the real closed-loop simulator (n=10/task), with three independent Spatial re-checks (96/97/97/98%) confirming reproducibility. This is the trustworthy pre-training reference point for measuring any retention loss from here forward.

### RoboTwin state

No canonical evidence yet. A minimal sanity check (`click_alarmclock`, n=3, `EVALUATION.clean_only` not yet available at the time) confirmed the RoboTwin evaluation pipeline works end-to-end with this checkpoint format (clean 33.3%, random 100% — not meaningful at n=3 on an untrained-on-RoboTwin checkpoint, just a pipeline sanity check).

## 3. Candidate design

### Modifications

1. Reuse `configs/data/multiembodiment_libero_robotwin.yaml` (combined LIBERO+RoboTwin config), now fixed per `PROGRESS_0011`/commit `66d8d32`: `embodiment_description` removed from both embodiment blocks, LIBERO's `pretrained_norm_stats` now points at the release checkpoint's own paired stats file instead of auto-computing from this training mixture's own subset.
2. New task config `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5.yaml`: identical to `exp0003`'s architecture (disjoint K=21/22 offsets, full trainable backbone, 1:1 ratio, LR `3e-5`, cosine schedule) except `batch_size: 2` (up from `1`).
3. Resume from the zero-init-fixed expanded release checkpoint.

### Why this candidate

The multi-embodiment implementation itself (disjoint-offset padding, masked action loss, checkpoint expansion) is architecture-level and was never the actual problem — `PROGRESS_0011` established the entire prior retention-loss investigation (exp0001-exp0010) was confounded by real bugs plus `exp0019`-specific seed fragility, not a flaw in this design. Re-running the same proven architecture (exp0003's) on the new, verified-clean parent is the correct next step: it isolates whether the design choices that looked good/bad on the old parent (disjoint offsets fixing exp0001/exp0002's collapse) still hold, without any of the confounds that plagued the old evidence.

### Multi-embodiment representation/configuration

- shared action dimension: K=21 (LIBERO `[0:7]`, RoboTwin `[7:21]`)
- shared proprio dimension: K=22 (LIBERO `[0:8]`, RoboTwin `[8:22]`)
- normalization: LIBERO uses `pretrained_norm_stats` (release checkpoint's own paired stats); RoboTwin auto-computes from its own training data (`./data/robotwin2.0/dataset_stats.json`, already correct — no prior-checkpoint calibration issue for a never-before-trained embodiment)
- checkpoint expansion: zero-init (fixed in `PROGRESS_0011`, commit `631df95`)
- embodiment/control conditioning: none (no `embodiment_description`)
- inference slicing/decoding: unchanged (`ConcatLeftAlign` crop-then-denormalize, per-embodiment offset)

### Data and learning strategy

- LIBERO datasets: Spatial/Object/Goal/Long-10 (LIBERO-90 excluded — no downloadable lerobot-format data, and out of scope per the current goal anyway)
- RoboTwin datasets: `./data/robotwin2.0/robotwin2.0` (full mixture)
- sampling/mixing ratio: 1:1 batch-level (`InterleavedEmbodimentSampler`)
- trainable/frozen modules: full fine-tune (`trainable_modules` unset, defaults to `"dit"` — entire MoT, both video and action backbones, trainable)
- loss: two-level masked action loss (per-channel, masked-averaged over valid timesteps, then averaged over each embodiment's own valid channel count)

### What to watch

- Training stability at `batch_size=2` (smoke test first; watch for OOM)
- LIBERO retention (Spatial/Object/Goal/Long, cheap panel) vs. this candidate's own `exp0012` baseline (98/99/98/95%)
- RoboTwin capability emerging (progress-check panel, Clean-only per current simplification)

### Initial compute plan

- smoke test: 6 steps, `batch_size=2`, validate no OOM/crash before committing
- initial training budget: 1000 steps (matches historical practice; may extend based on progress checks)
- checkpoint/save plan: `save_every: 100`, `save_full_state: false`, active pruner discipline
- evaluation plan: LIBERO cheap panel (Spatial, candidate_screen) + RoboTwin Clean-only progress panel during/after training

## 4. Exact code and configuration state

- Git commit: pending
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `9b49b68` (clean_only flag addition, most recent prior commit)
- files changed: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5.yaml` (new), this report
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5.yaml`
- data config: `configs/data/multiembodiment_libero_robotwin.yaml` (fixed in commit `66d8d32`)
- optimizer / LR / scheduler: AdamW, cosine, `learning_rate: 3e-5`
- batch size / gradient accumulation: `batch_size: 2`, `gradient_accumulation_steps: 4` (effective global batch 32, double the historical `batch_size=1` recipe's 16) — pending smoke-test validation
- initial training steps: `max_steps: 1000`
- resume source: weights-only resume from the zero-init-fixed expanded release checkpoint

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate beyond the config fixes already committed.

## 6. Training execution and control timeline

### Smoke test

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5 \
    resume=/home/claudeuser/local_cache/fastwam_release_expanded_zeroinit/step_000000.pt \
    output_dir=./runs/_smoke_test/exp0013_release_parent \
    model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.redirect_common_files=false \
    max_steps=6 save_every=6 log_every=1 eval_every=999999 batch_size=2
  ```
- First attempt failed immediately on a Hydra config error (`Could not override 'data@task.data'`) — the new task config was missing the `# @package _global_` directive at the top of the file (present in every other task config, omitted by mistake when authoring this one). Fixed, no research-design impact.
- Second attempt: crashed a few minutes in (past model construction, mid-dataloading) with `FileNotFoundError: Missing text embedding cache` for RoboTwin samples. Root cause: `PROGRESS_0011`'s removal of `embodiment_description` changed the runtime prompt string `augment_instruction()` produces, invalidating the entire RoboTwin text-embedding cache (originally precomputed *with* the embodiment_description prefix baked into the hash by a dedicated custom script). Full writeup in `research/NOTES.md` ("Training gotcha — removing `embodiment_description` stales the RoboTwin text-embedding cache"). Not a research-design issue — a real, one-time infra consequence of the PROGRESS_0011 fix that nobody had triggered yet (eval doesn't touch this cache).
- Fix in progress: recomputing the full text-embedding cache (all 921,072 discovered prompts, LIBERO redundantly but harmlessly included) via the generic `scripts/precompute_text_embeds.py` (now correct now that `embodiment_description` is gone) — `torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5`, log `checkpoints/exp0013_precompute_text_embeds.log`. Expected ~35-40 minutes (matches the original precompute's timing). Smoke test will be re-attempted once this completes.
- Third precompute attempt caused a near-total disk exhaustion (27GB -> 740KB free in under 2 minutes): root cause was the generic precompute script's cross-write design flaw (writes every discovered prompt into every discovered `cache_dir`) run against the *combined* multiembodiment task config, compounded by not having deleted the stale RoboTwin cache first. Recovered via emergency `rm -rf data/text_embeds_cache/robotwin` + relaunch scoped to the RoboTwin-only task config (`robotwin_uncond_3cam_384_multiembodiment_eval`). Full writeup in `research/NOTES.md`. A disk-monitor script used during recovery had its own bug (an `999999`-sentinel "previous reading" placeholder caused a false-positive anomaly-kill on the very first real check, killing a healthy job) — fixed with an empty-string sentinel that skips the drop-check on the first iteration.
- Second smoke test (`batch_size=2`, 6 steps) completed cleanly once the RoboTwin cache was rebuilt: losses ranged 0.95-1.74 (noisy at 6 steps with a fast-decaying cosine LR, as expected), peak GPU memory ~24GB/80GB (comfortable headroom), checkpoint saved and independently verified: file size 12,041,907,845 bytes (~12.04GB, matching the parent checkpoint's ~12.04GB almost exactly), 1649 `mot` tensors sharing exactly 1 underlying storage buffer (consistent with DeepSpeed's flat-parameter representation), ~6.02B total params, and direct value comparison against the parent checkpoint showed near-identical weights (max abs diff ~0.000122, mean ~5e-5, no NaN/Inf) -- confirming a genuine, correctly-trained, non-corrupted save. (An earlier `ls -la` reading of 312,896 bytes on this same file was a transient artifact of checking mid-write, before the ~12GB flush completed; superseded by the corrected reading above.)
- Smoke-test run directory (`runs/_smoke_test/exp0013_release_parent/`) deleted after verification to reclaim its ~12GB.

### Disk budget for real training

Checkpoints are genuinely ~12GB each. With `save_every=200` over `max_steps=4000` (20 possible saves) and only ~35GB free on `/workspace` (97% full, 1.1TB volume), an unpruned run would need ~240GB -- far more than available. Per the established project pattern (`research/NOTES.md` "Disk crisis" section), required pruner headroom is `(KEEP+1) x checkpoint_size_GB`. With ~35GB free: `KEEP=1` needs ~24GB (safe margin), `KEEP=2` needs ~36GB (too tight, matches the exact failure mode that hit exp0002). **Chose `KEEP=1`.** A background pruner (`while kill -0 $TRAIN_PID; do sleep 15; ls -1t weights/step_*.pt | tail -n +2 | xargs -r rm -f; done`) runs alongside training, polling every 15s (fast enough to close the race window given a ~12GB write completes well under 60s on this hardware). This means only the single most recent checkpoint is retained locally during training -- any checkpoint worth keeping for evaluation/continuation must be evaluated or copied out before the next save arrives, or persisted to `cheikh025/ASR` first.

### Real training launch

- exact launch command:
  ```bash
  source /workspace/venvs/fastwam/bin/activate
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5 \
    resume=/home/claudeuser/local_cache/fastwam_release_expanded_zeroinit/step_000000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1 \
    model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.redirect_common_files=false
  ```
- log: `checkpoints/exp0013_train.log`
- pruner log: `checkpoints/exp0013_pruner.log`
- First launch attempt failed immediately with `accelerate: command not found` -- used the base image's `/venv/main` instead of this project's dedicated venv (`/workspace/venvs/fastwam`, per `research/RUNBOOK.md`'s Environment section: FastWAM's venv is fully separate from `/venv/main`, which carries an incompatible torch 2.11). Fixed and relaunched with `source /workspace/venvs/fastwam/bin/activate`; no research-design impact.
- budget: `max_steps=4000` ceiling, `batch_size=2`, `gradient_accumulation_steps=4` (effective global batch 32), `save_every=200`, `save_full_state=false`, LR `3e-5` cosine, resuming from the zero-init-fixed expanded release checkpoint.
- dataset size: `Train/val dataset size: 6289288/6289288` (combined LIBERO+RoboTwin, 1:1 `InterleavedEmbodimentSampler`).
- VRAM: ~68.7-72.8GB/80GB per GPU across all 4 GPUs at `batch_size=2` (84-89% utilized, comfortable headroom, no OOM risk).
- **Plan revision (user feedback, before step 200)**: `max_steps=4000` was never a hard commitment, just an initial suggested budget (user clarified: "no i did not say 4k is ceiling u can try any number i just suggest"). Rather than running unconditionally, treat step ~1000-2000 as a progress-check-then-decide point (LIBERO sentinel + RoboTwin Clean-only cheap panel); the actual training length may end up shorter or longer than 4000 depending on that evidence (`CONTINUE_TRAINING`/`EXTEND_TRAINING`/`STOP_TRAINING`/`SELECT_CHECKPOINT`). Default for *future* candidates should start smaller (1000-2000 steps) and scale up based on evidence rather than committing a large budget upfront -- saved as standing feedback memory. Given the `KEEP=1` pruner overwrites checkpoints every `save_every=200` cadence, the checkpoint nearest the intended eval point must be evaluated (or copied out) promptly before the next save overwrites it locally.

## 7. Evaluation events

Training was stopped cleanly (SIGTERM to the whole process group, `kill -TERM -$TRAIN_PID`) right after the step-1000 checkpoint finished writing, to run this progress check sequentially per the standing rule against stacking training and eval on the same GPUs. Traceback in the training log at this point is the expected `SignalException` from the intentional stop, not a crash.

**Infra gotcha hit during this progress check**: the LIBERO eval manager's first launch attempt silently did nothing for ~1 minute -- a stale orphaned `libero_test_v3` tmux session (leftover from earlier work) caused the tmux server to crash when the manager tried to kill-then-recreate it, so every subsequent pane launch silently failed while the scheduler's bookkeeping still claimed "8/10 running." Diagnosed by cross-checking `nvidia-smi` (0% util on all GPUs) against the scheduler's claim; fixed with `tmux kill-server` + relaunch. Also hit two RoboTwin launch issues on the first attempt: `EVALUATION.clean_only=true` needs the Hydra `+` prefix (`+EVALUATION.clean_only=true`, key not declared in the base YAML schema), and `VK_ICD_FILENAMES` isn't sourced by `venv activate` in a non-interactive shell (must `export` it explicitly per command, despite being in `.env`). All three documented in `research/NOTES.md`.

### Evaluation event — LIBERO-Spatial `progress_check`

- benchmark: `libero`
- checkpoint / training step: exp0013, step 1000
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: pre-training baseline (exp0012/`PROGRESS_0011`) = 98.00% (n=50/task, canonical); this candidate's own release-checkpoint parent, verified.
- **result: 100.0% (30/30)** -- all 10 tasks 100% (3/3). No retention loss whatsoever at step 1000; comfortably above the 90% floor and even above the canonical pre-training baseline (expected -- n=3 noise near ceiling, not a real improvement claim).
- artifact: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260819_003915/summary.json`
- runtime: ~4 min (4-GPU parallel, 2 tasks/GPU).

### Evaluation event — RoboTwin `progress_check` (Clean-only, 2-task panel)

- benchmark: `robotwin`
- checkpoint / training step: exp0013, step 1000
- purpose: first-ever RoboTwin capability read for the new release-checkpoint lineage, to decide whether to continue training.
- exact command (per-task):
  ```bash
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  CUDA_VISIBLE_DEVICES=0 python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1/robotwin_dataset_stats.json \
    EVALUATION.task_name=<click_alarmclock|turn_switch> EVALUATION.eval_num_episodes=5 +EVALUATION.clean_only=true \
    MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1
  ```
- **result** (n=5 episodes/task, Clean only):

  | Task | Clean (n=5) | exp0004-0007 lineage historical ceiling (old exp0019 parent) |
  |---|---:|---:|
  | `click_alarmclock` | **80.0% (4/5)** | 20-33% |
  | `turn_switch` | **60.0% (3/5)** | not previously tested |

  Both tasks confirmed present in training data (46/50-task coverage, see `research/NOTES.md`).
- artifacts: `evaluate_results/robotwin/reweighted_multiembodiment_exp0013_release_parent_disjoint_offset_v1/20260819_004749/summary.json`, `.../20260819_005713/summary.json`
- runtime: ~10 min/task (single GPU, sequential).
- validity checks: `Render Well` confirmed both runs (no Vulkan ICD issue once fixed), correct checkpoint/stats path, `clean_only` correctly skipped the random phase both times (confirmed via manager log: `manager finished successfully` after 1 phase only).

## 8. Comparison and interpretation

At only step 1000/4000 (25% of the suggested initial budget), this candidate already shows:
- **zero measurable LIBERO retention loss** (100% vs. 98% baseline -- within noise of ceiling);
- **substantially stronger RoboTwin capability than anything achieved under the old exp0019-parent lineage** across exp0001-exp0009 (which topped out around 33% on `click_alarmclock` and 0% on most other tasks tested).

This is strong positive evidence that the parent-checkpoint switch (`PROGRESS_0011`, release checkpoint instead of exp0019) was the right call -- the release checkpoint appears to be a dramatically better starting point for RoboTwin adaptation, not just a fix for exp0019's LIBERO-Spatial fragility.

## 9. Decision

**`CONTINUE_TRAINING`** -- resume from `step_001000.pt`, remaining budget to reach the original 4000-step suggestion (3000 more steps; not a hard ceiling, may extend further depending on the next progress check). No sign of a plateau, forgetting, or instability that would justify `STOP_TRAINING` or `SELECT_CHECKPOINT` at this early point. Next progress check planned around step ~2000-2500 with the same cheap panel (LIBERO-Spatial n=3 + the same 2-task RoboTwin Clean-only panel, for direct comparability), stopping training again beforehand per the same no-stacking discipline.

## 10. What this changes for the next experiment

Nothing yet -- still within exp0013's own training run. If this trajectory holds (RoboTwin capability continuing to rise with no LIBERO cost), the next candidate-level question becomes how far this recipe can be pushed before either benchmark plateaus or regresses, and whether the other three LIBERO suites (Object/Goal/Long) hold as well as Spatial did.

## 11b. Continuation (step 1000 -> 4000 cumulative)

Resumed per the `CONTINUE_TRAINING` decision above. Weights-only `resume=` restarts the trainer's internal step counter at 1 (documented resume semantics, `research/RUNBOOK.md`), so this continuation was launched into a **new output directory** (`..._v1_cont1`) rather than the original one, to avoid the continuation's own step 1000 silently overwriting the already-evaluated `step_001000.pt`. **Cumulative step = 1000 + this run's own logged step** for all reporting purposes going forward. A side effect of weights-only resume: the cosine LR schedule also restarts fresh (warmup -> peak 3e-5 -> decay) over this run's own `max_steps=3000`, rather than continuing along the tail of the original single 4000-step curve -- an accepted, previously-used consequence of this codebase's weights-only resume path (matches exp0002's recovery precedent in `research/NOTES.md`), not a bug.

- exact launch command:
  ```bash
  source /workspace/venvs/fastwam/bin/activate
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5 \
    resume=runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1/checkpoints/weights/step_001000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0013_release_parent_disjoint_offset_v1_cont1 \
    model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.redirect_common_files=false \
    max_steps=3000
  ```
- log: `checkpoints/exp0013_train_cont1.log`, pruner log: `checkpoints/exp0013_pruner_cont1.log` (same `KEEP=1`/15s pattern as the first phase)
- plan: progress-check again around this run's own local step ~1000-1500 (cumulative ~2000-2500), same cheap panel (LIBERO-Spatial n=3, RoboTwin `click_alarmclock`+`turn_switch` Clean n=5) for direct comparability against the step-1000 numbers above.

### Disk near-miss during cont1, second phase (cont2)

Preserving `step_001000.pt` locally for the progress-check evidence (correct) had an unaccounted side effect: its own pruner exited with the stopped training process, so it stopped being pruned and permanently consumed 12GB of headroom the `cont1` run's own `KEEP=1` pruner arithmetic didn't know about. Disk dropped from 24GB to 12GB free by cont1's step 800 (cumulative 1800) -- below the safe `KEEP=1` threshold, risking a truncated write on the next save. Fixed by: uploading `step_001000.pt` to `cheikh025/ASR` (`research/exp0013_release_parent_disjoint_offset/step_001000.pt`), verifying the remote copy (matching 12,041,907,845-byte size), deleting the local copy (restored to 24GB free), pausing training during this to eliminate any write race, then resuming as `cont2` (new output dir, same step-counter-reset reasoning as `cont1`) from `cont1`'s `step_000800.pt` (cumulative 1800) with `max_steps=2200` (to reach the original cumulative 4000 target).

The exact same issue immediately recurred with `cont1`'s own now-orphaned `step_000800.pt` once `cont2` started (same unaccounted-permanent-checkpoint pattern) -- caught proactively this time before it became urgent, uploaded to `cheikh025/ASR` (`research/exp0013_release_parent_disjoint_offset/step_001800_cumulative.pt`), verified, deleted, restoring 24GB free well before `cont2`'s first save. Standing operational rule now recorded in `research/NOTES.md`: upload+verify+delete any preserved/evaluated checkpoint immediately once a `CONTINUE_TRAINING` decision is made, before or in parallel with (never after) launching the next training phase.

Durable copies now on `cheikh025/ASR`:
- `research/exp0013_release_parent_disjoint_offset/step_001000.pt` (cumulative step 1000, the evaluated checkpoint: LIBERO-Spatial 100%, RoboTwin click_alarmclock 80%/turn_switch 60%)
- `research/exp0013_release_parent_disjoint_offset/step_001800_cumulative.pt` (cumulative step 1800, not yet evaluated -- an intermediate save, kept for continuation safety only)

## 11. Artifacts

- smoke test log: `checkpoints/exp0013_smoke_train.log`, `checkpoints/exp0013_smoke_train_v2.log`
- real training log: `checkpoints/exp0013_train.log`
- pruner log: `checkpoints/exp0013_pruner.log`
- task config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5.yaml`
