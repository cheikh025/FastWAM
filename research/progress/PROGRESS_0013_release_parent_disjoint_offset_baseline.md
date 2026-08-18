# PROGRESS_0013 — First real multi-embodiment training candidate on the release-checkpoint parent

- **Experiment ID:** 0013
- **Status:** `RUNNING` (smoke test in progress as this report is created)
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

## 7. Evaluation events

None yet.

## 8. Comparison and interpretation

Pending.

## 9. Decision

Pending.

## 10. What this changes for the next experiment

Pending.

## 11. Artifacts

- smoke test log: `checkpoints/exp0013_smoke_train.log`
- task config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_3e-5.yaml`
