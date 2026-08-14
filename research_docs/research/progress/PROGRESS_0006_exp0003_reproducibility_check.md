# PROGRESS_0006 — exp0003_reproducibility_check

- **Experiment ID:** 0006_exp0003_reproducibility_check
- **Status:** `RUNNING`
- **Created:** 2026-08-09
- **Updated:** 2026-08-09
- **Parent experiment:** 0005_reweighted_v3_goal_compensated (methodological follow-up, not a design iteration)
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- **Selected candidate checkpoint:** n/a — this is a diagnostic, not a candidate to be promoted/rejected
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `eb0a53bcd18a96ca46147c8595639ad5e29d20c8` (same commit as exp0003 — no config changes, exact rerun)

## 1. Result at a glance

Diagnostic run: rerun experiment 0003's exact training recipe a second time (identical config, identical `dataset_dirs`/weights/LR/steps/seed=42) to directly measure how much two nominally-identical training runs vary in outcome. Motivated by `research/NOTES.md` "Methodological warning": exp0003->0004->0005 showed volatility (an untouched suite regressing, a compensated suite not recovering) that could not be explained by eval-side noise, since a same-checkpoint eval repeat (this same session, prior step) returned **bit-identical results on all 21 panel tasks** — confirming LIBERO evaluation is fully deterministic given a fixed checkpoint. That isolates the mystery variance to the training side. This run tests exactly how large that training-side variance is.

## 2. Research state before experiment

See `research/NOTES.md` "Methodological warning: single-run/n=5-trial weight-tuning comparisons are too noisy" for full context. Key immediately-prior finding: re-evaluating experiment 0003's checkpoint (`step_004000.pt`) a second time on the identical 21-task panel produced **exactly the same success count on every task** (0/21 tasks differed at all) — eval measurement noise is not the explanation for the volatility seen across 0003/0004/0005.

## 3. Candidate design

### Modifications

None — this is a literal rerun of experiment 0003's exact recipe:
- Data config: `configs/data/libero_2cam_plus90_reweighted.yaml` (unchanged, goal 5x / long 5x)
- Task config: `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml` (unchanged, lr=3e-5, batch_size=4, grad_accum=4, max_steps=4000)
- Same `seed: 42` (from `configs/train.yaml`, not overridden)
- Same `resume=<released checkpoint>` starting point
- Only the output directory differs (`runs/reweighted_libero90_finetune/exp0006_rerun/`) so it doesn't overwrite exp0003's artifacts

### Why this candidate

Not a research candidate in the usual sense — a controlled reproducibility check. If two runs of the identical recipe (same seed, same everything) produce meaningfully different per-task/per-suite outcomes, that quantifies the training-side noise floor and tells us how much confidence to place in any single-run comparison like 0003 vs 0004 vs 0005. If the two runs match closely, that would point back toward the specific weight changes as the real explanation for the earlier volatility (and the Object regression / Goal non-recovery would need a different explanation). If they diverge substantially, that confirms multi-GPU/distributed-training non-determinism (unpinned NCCL/cuDNN algorithm selection, dataloader worker ordering, etc. — common even with a fixed seed unless explicit determinism flags are set, which this codebase does not appear to set) is a dominant factor, and future candidate comparisons will need either multiple seeds per candidate or much larger effect sizes to be trusted.

### What to watch

- Per-task and per-suite delta between this rerun and the original exp0003 result, on the same 21-task panel used throughout 0003-0005.
- Whether the magnitude of run-to-run variance here is comparable to, smaller than, or larger than the magnitude of the exp0003->0004->0005 deltas — this directly calibrates how much of the earlier story was "real design effect" vs "noise."

### Initial compute plan

- Training budget: `max_steps=4000` (identical to 0003), no early stopping planned — need the full comparable run.
- Checkpoint/save plan: `save_every=500`, state checkpoints on the second/ephemeral filesystem via symlink (validated disk-management fix).
- Evaluation: same 21-task panel, 5 trials/task, once training completes.
- Expected cost: ~3.3h training + ~35min eval, same as prior experiments.

## 4. Exact code and configuration state

- Git commit: `eb0a53bcd18a96ca46147c8595639ad5e29d20c8` (identical to exp0003/0004/0005's task-config-defining commits — no new files)
- Git branch: `autoresearch/libero90-v1`
- Working tree: clean, no changes needed (literal rerun of existing config)
- Files changed: none
- Training config(s): `task=libero_uncond_2cam224_plus90_reweighted_3e-5` (same as exp0003)
- Dataset config(s): `data=libero_2cam_plus90_reweighted` (goal 5x, long 5x — same as exp0003)
- Model/optimizer/LR/batch/steps: identical to exp0003
- Random seed: `seed=42` (unchanged, not overridden — testing whether "same seed" actually yields the same outcome in this multi-GPU pipeline)
- Resume source: `resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` (identical starting point to exp0003)
- Output directory: `runs/reweighted_libero90_finetune/exp0006_rerun/` (only difference from exp0003's launch)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    output_dir=./runs/reweighted_libero90_finetune/exp0006_rerun \
    wandb.name=exp0006_exp0003_rerun
  ```
- Start time: 2026-08-09 ~18:06 UTC
- End time: 2026-08-09 21:35 UTC (`max_steps reached step=4000`, clean exit, no crash, no OOM, no disk-full warning — training runtime ~3h29m)
- Final weights checkpoint: `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt` (12,041,735,545 bytes ≈ 11.2GB, matches exp0003's checkpoint size)
- Disk management: symlinked-state + auto-cleanup monitor ran the full duration with no warnings (root filesystem oscillated 178-338GB free, workspace declined normally as expected); validated fix held for a second full run.

### Intermediate checkpoints and progress decisions

Not planned — single full run to completion for direct comparability with exp0003's own single-run result.

## 7. Evaluation events

### Event 1 — `candidate_screen` (reproducibility comparison)

- Checkpoint / training step: `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt`
- Decision this evaluation was meant to inform: quantify training-run variance by direct comparison against exp0003's identical-recipe result on the identical panel
- Exact task subset: same 21-task widened panel used throughout 0003-0005 (`libero_90:24,19,46,9,73`; `libero_spatial:0,3,5,8`; `libero_object:0,3,5,8`; `libero_goal:0,3,5,8`; `libero_10:0,3,5,8`)
- Trials per task: 5
- Exact command: `run_libero_parallel_test.sh` with `CKPT=.../exp0006_rerun/checkpoints/weights/step_004000.pt CONFIG=libero_uncond_2cam224_plus90_reweighted_3e-5 NUM_GPUS=4 MAX_TASKS_PER_GPU=2 NUM_TRIALS=5`, dataset_stats overridden to exp0006's own `dataset_stats.json`
- Raw results path: `evaluate_results/exp0006_cheap_panel/`
- Runtime: ~2 min (well under training-scale cost; small panel)
- Reference/parent result: exp0003's original run on the same recipe/panel (`evaluate_results/exp0003_cheap_panel/`)

**Suite-level comparison (5 trials/task, 4-5 tasks/suite):**

| Suite | exp0003 | exp0006 (rerun) | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 80.00% | 70.00% | -10pp |
| LIBERO-Object | 100.00% | 95.00% | -5pp |
| LIBERO-Goal | 70.00% | 55.00% | -15pp |
| LIBERO-Long/10 | 20.00% | 30.00% | +10pp |
| LIBERO-90 (5-task sentinel) | 68.00% | 48.00% | -20pp |
| **Overall (21-task panel)** | **67.60%** | **59.60%** | **-8pp** |

**Task-level comparison (21 tasks, identical recipe/seed=42/checkpoint-step, two independent training runs):**

| Task | exp0003 | exp0006 | Delta |
|---|---:|---:|---:|
| libero_10_0 | 60% | 60% | +0pp |
| libero_10_3 | 0% | 0% | +0pp |
| libero_10_5 | 20% | 60% | **+40pp** |
| libero_10_8 | 0% | 0% | +0pp |
| libero_90_19 | 40% | 0% | **-40pp** |
| libero_90_24 | 100% | 100% | +0pp |
| libero_90_46 | 100% | 80% | -20pp |
| libero_90_73 | 40% | 20% | -20pp |
| libero_90_9 | 60% | 40% | -20pp |
| libero_goal_0 | 0% | 0% | +0pp |
| libero_goal_3 | 80% | 40% | **-40pp** |
| libero_goal_5 | 100% | 80% | -20pp |
| libero_goal_8 | 100% | 100% | +0pp |
| libero_object_0 | 100% | 80% | -20pp |
| libero_object_3 | 100% | 100% | +0pp |
| libero_object_5 | 100% | 100% | +0pp |
| libero_object_8 | 100% | 100% | +0pp |
| libero_spatial_0 | 100% | 100% | +0pp |
| libero_spatial_3 | 100% | 100% | +0pp |
| libero_spatial_5 | 20% | 20% | +0pp |
| libero_spatial_8 | 100% | 60% | **-40pp** |

**9 of 21 tasks (43%) differ** between two runs with identical config, identical seed=42, identical starting checkpoint. Three tasks swing by a full 40 percentage points (2/5 trials flipping outcome). Suite-level deltas range from -20pp (LIBERO-90) to +10pp (Long).

### Validity checks

- Both runs used the identical checkpoint-loading path, dataset_stats format, task/trial set, and trial count — a fair matched comparison.
- This is the same 21-task panel already validated as eval-deterministic in this session (a same-checkpoint repeat-eval of exp0003 returned 0/21 tasks differing) — so this result is not an artifact of nondeterministic evaluation; it reflects genuine differences between the two trained checkpoints.
- All 21 tasks completed in both runs (no missing/failed results).

## 8. Comparison and interpretation

**This is training-side non-determinism, not evaluation noise.** The prior repeat-eval of exp0003's own checkpoint (same weights, evaluated twice) returned bit-identical results on all 21 tasks — ruling out eval-side randomness. Here, two *training* runs with the same seed, same code, same data config produced measurably different final checkpoints, evidenced by a 43% task-level disagreement rate and swings up to 40pp on individual tasks and 20pp at suite level.

**Root cause (not directly verified, but consistent with known behavior):** this codebase sets `seed=42` but does not set explicit determinism flags (`torch.backends.cudnn.deterministic`, `torch.use_deterministic_algorithms`, deterministic NCCL reduction ordering, deterministic dataloader worker scheduling). Under 4-GPU DeepSpeed ZeRO-1 training, floating-point non-associativity in distributed all-reduce order and non-deterministic cuDNN kernel selection are sufficient to produce meaningfully different trained weights from a nominally "identical" run, even from the same starting checkpoint.

**One important exception**: `libero_goal` task 0 ("open the middle drawer of the cabinet") was 0/5 in *both* runs — the only task with a clean, large collapse (100% baseline -> 0%) that reproduced exactly. This is consistent with the separate diagnostic finding recorded in `research/NOTES.md` (LIBERO-90 has abundant "top/bottom drawer" tasks but zero "middle drawer" tasks, a plausible systematic interference mechanism) — a real, stable effect, distinguishable from the noise seen elsewhere in this comparison.

**Direct implication for the 0003->0004->0005 series**: the earlier "Object regressed 100%->85% despite being untouched" and "Goal failed to recover despite exceeding 0003's weight share" observations, which prompted this reproducibility check, are now plausibly (not certainly) explained by this same training-run noise rather than by the specific weight changes tested in 0004/0005. The magnitude of noise measured here (suite deltas up to 20pp, task deltas up to 40pp) is comparable to or larger than the effect sizes those experiments were trying to attribute to their design changes. **The 0004/0005 "zero-sum dilution" narrative should be treated as unconfirmed, not refuted** — the evidence simply cannot currently distinguish a real effect of that magnitude from noise.

## 9. Decision

- **Decision:** not applicable — diagnostic, not a candidate for promotion/rejection.
- **Retention gate passed:** not applicable. (For reference, neither run clears the 90% floor on Goal/Long/90 at step 4000 on this recipe.)

## 10. What this changes for the next experiment

1. **Single-run, n=5-trial comparisons are not reliable for effect sizes below roughly 20-40pp at the task level or ~15-20pp at the suite level** on this pipeline, given the demonstrated noise floor. Future weight-tuning-style candidates should either (a) use a much larger effect size as the bar for a confident conclusion, (b) increase trials/task on the cheap panel, (c) run multiple seeds per candidate before concluding a design change worked or hurt, or (d) investigate whether setting explicit determinism flags would tighten this variance (not yet attempted — would need to be validated to not hurt throughput/correctness under DeepSpeed ZeRO-1).
2. Do not resume iterating on goal/long weight ratios (0003 vs 0004 vs 0005-style tuning) until a more robust comparison protocol is adopted — the last three weight-tuning experiments are not trustworthy evidence for what specific weight combination is best.
3. `libero_goal` task 0 is a confirmed stable failure (not noise) with a credible mechanism (drawer-position instruction interference from oversampled LIBERO-90 top/bottom-drawer tasks) — worth a small targeted diagnostic/fix independent of the broader mix-weight question.
4. Immediately following this check, launched **experiment 0007**: extending exp0006's training (same reweighted recipe, resumed full state) from step 4000 to step 8000, to test whether goal/long/90 continue improving with more training time on this recipe — a question orthogonal to the noise-floor finding and still worth answering regardless of it. See `PROGRESS_0007_extended_reweighted_v1.md`.

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0006_rerun/train.log` (full log at `/tmp/.../scratchpad/train_exp0006.log` used during monitoring)
- final weights checkpoint: `runs/reweighted_libero90_finetune/exp0006_rerun/checkpoints/weights/step_004000.pt` (11.2GB)
- HF backup: `cheikh025/ASR:rejected/0006_exp0003_reproducibility_check/step_004000.pt` (uploaded and existence-verified 2026-08-09, per standing policy of preserving all experiment checkpoints regardless of outcome)
- config(s): unchanged from exp0003 (`configs/data/libero_2cam_plus90_reweighted.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml`)
- comparison data: exp0003's original result (`evaluate_results/exp0003_cheap_panel/`), exp0003 repeat-eval (`evaluate_results/exp0003_repeat_eval/`, confirming eval determinism), exp0006's cheap panel (`evaluate_results/exp0006_cheap_panel/`)
- diagnostic note: `research/NOTES.md` "Diagnostic: `libero_goal` task 0 ... persistent 0% collapse, root cause hypothesis"

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed (none — exact rerun)
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded (unchanged from baseline, no drift)
- [x] logs and checkpoint paths recorded
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] final decision and reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated (next step)
- [ ] `research/STATE.md` updated (next step)
