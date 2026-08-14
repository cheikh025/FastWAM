# PROGRESS_0020 — libero90_v3_stove_wine_book_intensify

- **Experiment ID:** 0020_libero90_v3_stove_wine_book_intensify
- **Status:** `RUNNING`
- **Created:** 2026-08-14
- **Updated:** 2026-08-14
- **Parent experiment:** 0019_spatial_weak_task_oversampling (`PROMOTE`, current accepted main-line checkpoint)
- **Parent checkpoint:** `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (weights-only resume)
- **Selected candidate checkpoint:** n/a — planned
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `b2b49d0` (base; new uncommitted config/data files pending)

## 1. Result at a glance

exp0019 satisfied the project's active goal for the first time (all five suites >=95% canonical: Spatial 97.00%, Object 99.60%, Goal 97.20%, Long 98.00%, LIBERO-90 95.13%). LIBERO-90 has the smallest margin over the goal (+0.13pp) and the most persistently weak tasks. This candidate targets LIBERO-90's 3 remaining fixable weak tasks: two never-before-targeted tasks (task21 "stove+pan" 50.00%, task27 "wine rack" 54.00%) get their first round of 4x oversampling, and one previously-targeted-but-still-weak task (task75 "book in right compartment of caddy", oversampled at 4x in both exp0017 and exp0018 yet still only 60.00% canonical under exp0019) gets an intensified 8x oversampling to test whether more concentrated exposure succeeds where two rounds of standard-intensity oversampling did not. task51 ("butter in basket") remains excluded — confirmed on two separate occasions (exp0017, exp0018 build smoke tests) to have zero exact-matching training demonstrations in the source LIBERO-90 dataset, a genuine data gap that oversampling cannot address.

## 2. Research state before experiment

exp0019 is the accepted main-line checkpoint, canonical: LIBERO-Spatial 97.00%, LIBERO-Object 99.60%, LIBERO-Goal 97.20%, LIBERO-Long/10 98.00%, **LIBERO-90 95.13%** (all clear the 95% goal; LIBERO-90 has the smallest margin, +0.13pp). LIBERO-90's weakest tasks (canonical, exp0019): task21 "stove+pan" 50.00%, task27 "wine rack" 54.00%, task75 "book right compartment" 60.00%, task51 "butter in basket" 70.00% (confirmed zero-demo data gap, excluded), task63 "stack bowls" 80.00%, task83/81 "book front/other compartment" 82-84%, task65 "red mug left plate" 86.00%, task30 "bowl on plate" 88.00%.

Verified via direct dataset/benchmark inspection that all three newly-targeted tasks have real training data: task_index 11 ("stove+pan", 76 episodes), task_index 7 ("wine rack", 39 episodes), task_index 0 ("book right compartment", 97 episodes, already 4x-oversampled twice as part of the original 7-task book-caddy cluster subset in exp0017/exp0018 without fully resolving its weakness).

## 3. Candidate design

### Modifications

1. **New filtered subset dataset** `data/libero_mujoco3.3.2/libero_90_weak_subset_v3_lerobot/` covering exactly 2 new target tasks: "turn on the stove and put the frying pan on it" (task21, 76 source episodes) and "put the wine bottle on the wine rack" (task27, 39 source episodes). Built with the same validated parquet-rewriting script/methodology used for exp0017/0018/0019 (rewrites `episode_index`/`index`/`task_index` columns, symlinks video files).
2. **Intensified subset** for task75 ("book in right compartment of caddy", 97 source episodes) — listed at 8x oversampling in the data config (double exp0017/0018's 4x), isolated as its own `dataset_dirs` entry so its intensity can be tuned independently of the other tasks without re-touching the existing v2 subset.
3. **New data config** `configs/data/libero_2cam_plus90_reweighted_weaktasks_v4.yaml`: exp0019's full recipe unchanged (including the Spatial weak-subset and LIBERO-90 weak-subset v2, both untouched) plus the new v3 subset (task21+task27) at 4x, plus the task75-only intensified subset at 8x.
4. **New task config** `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v4_3e-5.yaml`.
5. `resume=./runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (weights-only resume from the current accepted checkpoint).
6. Cache pre-warmed via a real 4-rank `accelerate launch` dry run before the actual training launch; full process/GPU cleanup verified before launch (established lesson from exp0017-0019).

### Why this candidate

Directly targets LIBERO-90's remaining weak tasks using the exact validated mechanism (filtered-subset oversampling + weights-only resume) that has worked in all three prior uses. Two of the three targets (task21, task27) are genuinely new — never previously oversampled — so a standard 4x round is the natural first try. The third (task75) has already received two rounds of standard-intensity oversampling without resolving its weakness, so this candidate tests a distinct hypothesis for it specifically: that the mechanism works but needs higher intensity, rather than that the task is structurally unfixable via this lever (task51's actual situation). This also produces a cleaner read for the next iteration: if task75 still doesn't move at 8x, that is stronger evidence it needs a different intervention (e.g. diagnostic inspection of the failure mode) rather than more oversampling.

### What to watch

- Whether task21/task27 respond to their first oversampling round the way LIBERO-90's other never-before-targeted tasks did in exp0017/0018 (most showed real, often large gains).
- Whether task75 finally moves at 8x, or remains flat — this is the key diagnostic signal for whether "more of the same mechanism" is the right lever for this specific task.
- Whether LIBERO-90's aggregate increases without diluting Spatial (which just gained its own margin in exp0019) or any other suite.
- Whether the goal's margins broaden across the board, consistent with the project's shift from gap-closing to margin-building.

### Initial compute plan

- Initial training budget: `max_steps=5000` (matching exp0019's budget — this is a similarly small, low-risk data-mix addition).
- Checkpoint/save plan: `save_every=500`, prune intermediate weights aggressively (established disk-management lesson).
- No interim progress check planned by default (GPU contention with eval); may add one if training diagnostics look ambiguous.

## 4. Exact code and configuration state

- Git commit: `b2b49d0` (base) + new, uncommitted config/data files and new data directory (`data/libero_mujoco3.3.2/libero_90_weak_subset_v3_lerobot/`).
- Training config: `task=libero_uncond_2cam224_plus90_reweighted_weaktasks_v4_3e-5`
- Config overrides: `learning_rate=3e-5 max_steps=5000`
- Resume source: `runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt` (weights-only)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5. Venv: `/workspace/venv-fastwam`.

## 6. Training execution and control timeline

Both new dataset subsets built and smoke-tested successfully before launch:
- `data/libero_mujoco3.3.2/libero_90_weak_subset_v3_lerobot/` (115 episodes: 76 "stove+pan" + 39 "wine rack")
- `data/libero_mujoco3.3.2/libero_90_book_right_intensify_lerobot/` (97 episodes: "book right compartment")

Full end-to-end smoke test via `hydra.compose` + `instantiate(cfg.data.train)` on the real `libero_uncond_2cam224_plus90_reweighted_weaktasks_v4_3e-5` task config: dataset constructed cleanly (2,164,042 samples across the full ~18-entry `dataset_dirs` mix), 8 random `dataset[i]` calls succeeded with no errors.

Launch command:
```bash
source /workspace/venv-fastwam/bin/activate
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_plus90_reweighted_weaktasks_v4_3e-5 \
  resume=./runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt \
  max_steps=5000 save_every=500 eval_every=200 log_every=10 \
  output_dir=./runs/reweighted_libero90_finetune/exp0020_libero90_v3_stove_wine_book
```
- `checkpoints/state` symlinked to `/home/claudeuser/fastwam_state_scratch/exp0020_libero90_v3_stove_wine_book/state` (ephemeral scratch, established disk-management pattern).
- Start time: 2026-08-14 ~15:19 UTC.
- Pre-launch verification: `ps aux` for `train.py`/`accelerate launch`/`torchrun` empty, `nvidia-smi` 0MB on all 4 GPUs, confirmed clean before launch.
- Launch stability verified via repeated process/GPU checks over ~30s post-launch (no phantom tmux-death pattern).
- Status: running, monitored via change-filtered log tail for step progress / crash signatures.

## 7. Evaluation events

Pending.

## 8. Comparison and interpretation

Pending.

## 9. Decision

Pending.

## 10. What this changes for the next experiment

Pending.

## 11. Artifacts

- new data directory: `data/libero_mujoco3.3.2/libero_90_weak_subset_v3_lerobot/`
- new config(s): `configs/data/libero_2cam_plus90_reweighted_weaktasks_v4.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_weaktasks_v4_3e-5.yaml`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded (base commit; new files pending)
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [ ] training configuration and command recorded (pending launch)
- [ ] hardware/software environment recorded (pending)
- [ ] intermediate checkpoints and progress decisions recorded when used (pending)
- [ ] logs and checkpoint paths recorded (pending)
- [ ] remote checkpoint path recorded and verified if an HF backup was created (pending)
- [ ] every evaluation event has purpose/settings/raw results recorded (pending)
- [ ] five-suite metrics recorded when canonical evaluation ran (pending)
- [ ] final decision and reasoning recorded (pending)
- [ ] `research/EXPERIMENTS.jsonl` updated (pending)
- [ ] `research/STATE.md` updated (pending)
