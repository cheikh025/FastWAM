# PROGRESS_0001 — joint_libero90_finetune

- **Experiment ID:** 0001_joint_libero90_finetune
- **Status:** `REJECT`
- **Created:** 2026-08-08
- **Updated:** 2026-08-08
- **Parent experiment:** 0000_baseline
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- **Selected candidate checkpoint:** `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_004000.pt` (evaluated, rejected — not preserved as a branch)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `2971af01391bdeb70e730c175eea8489b57fcc28`

## 1. Result at a glance

Joint fine-tuning on the 4-suite + LIBERO-90 data mix (lr=3e-5, 4000 steps, full fine-tune of `model.dit`, from the released checkpoint). Cheap 13-task screening panel shows **real LIBERO-90 learning on some tasks (26%->100%, 70%->100%) but severe, suite-concentrated catastrophic forgetting** — LIBERO-Goal and LIBERO-Long collapsed to 0-20% on both sampled sentinel tasks each (from 94-100% baseline), while Spatial partially regressed and Object was unaffected. Retention gate clearly fails; canonical evaluation not warranted. **Decision: REJECT.** Proceeding to `$investigate-fastwam-problem` to understand the suite-selective forgetting pattern before designing the next candidate.

## 2. Research state before experiment

Baseline (0000): LIBERO-90 zero-shot 15.40% (693/4500), highly non-uniform across tasks (25/90 tasks with partial-to-strong transfer, 65/90 near-0%, entire 16-task book-in-caddy category untransferred). Four original suites 94.6-99.4%, comfortably above the 90% retention floor and closely matching the published paper numbers. No training has occurred yet in this project — the released checkpoint has literally never seen LIBERO-90 demonstration data.

## 3. Candidate design

### Modifications

1. New data config `configs/data/libero_2cam_plus90.yaml`: identical to the existing `libero_2cam.yaml` (4-suite mix) except `dataset_dirs` also includes `./data/libero_mujoco3.3.2/libero_90_no_noops_lerobot` (the staged, coverage-verified — 89/90 official task IDs — LIBERO-90 training set from `IPEC-COMMUNITY/libero_90_no_noops_lerobot`).
2. New task config `configs/task/libero_uncond_2cam224_plus90_3e-5.yaml`: copy of `libero_uncond_2cam224_1e-4.yaml` with three deliberate changes tied to this being a *continued* fine-tune of an already-converged checkpoint rather than training from scratch:
   - `learning_rate: 3e-5` (down from the original recipe's `1e-4`) — reduces forgetting risk on the four retention-constrained suites.
   - `batch_size: 4`, `gradient_accumulation_steps: 4` (down from `batch_size: 16`, `gradient_accumulation_steps: 1`) — same effective global batch trajectory shape but a per-GPU batch small enough to fit our 4x A100 setup; `batch_size` in this codebase is per-process, and the original recipe's `16` was calibrated for an 8-GPU node, not verified safe on 4 GPUs at this model size (the training smoke test only validated `batch_size=1`).
   - `max_steps: 4000` (replacing `num_epochs: 10`/`max_steps: null`) — an explicit step budget appropriate for a continued fine-tune rather than the original from-pretrained-init training length; intended as an initial ceiling to be controlled adaptively via progress checks (see Section 3 Initial compute plan), not a commitment to train all 4000 steps regardless of evidence.
   - `save_every: 500` (down from `2000`) — more frequent checkpoints given the shorter overall budget, so a progress-check-driven early stop or checkpoint selection has good granularity.
3. Text embeddings precomputed for the new data mix via `scripts/precompute_text_embeds.py task=libero_uncond_2cam224_plus90_3e-5` (111 unique prompts across all 5 datasets, into the existing shared `data/text_embeds_cache/libero` cache dir — additive, does not disturb the 40 prompts already cached for the original 4-suite recipe).
4. Training launched with `resume=<released checkpoint>.pt` (weights-only init from the released model, not a directory-based full-state resume, since this is a new candidate branching from the released checkpoint, not a continuation of a prior run).

No architecture/loss/frozen-module changes — `model.dit` (video expert + action expert) remains the only trainable component, matching the codebase's existing (and only available) fine-tuning mechanism; no LoRA/adapter support exists in this codebase (verified during setup).

### Why this candidate

This is the first LIBERO-90 training candidate for the project, and the most direct one: **the released checkpoint currently has zero training exposure to LIBERO-90 data at all**, so the most basic, credible-upside intervention is to add that data to the training mix and continue fine-tuning. Simple joint fine-tuning (not sequential/multi-stage) is also supported by recent robot-policy continual-learning literature (see `research/NOTES.md` literature scan) reporting good retention from simple continued fine-tuning with a large pretrained backbone, provided LR/duration are controlled — directly motivating the reduced LR here (3e-5 vs the original 1e-4) as the primary retention-protection lever, since this codebase has no LoRA/adapter isolation to fall back on.

Data-mix composition note: the four original suites total 1712 episodes (277,713 frames); LIBERO-90 alone is 3921 episodes (569,249 frames) — roughly a 70/30 split favoring LIBERO-90 by raw sample count if the datasets are simply concatenated (no reweighting mechanism exists in `RobotVideoDataset`/`MultiLeRobotDataset`). This is directionally acceptable for a first candidate since LIBERO-90 is the primary target, but it does mean the original suites are proportionally under-represented relative to the released checkpoint's original (100% original-suite) training mix — this is the main retention risk to watch, alongside the full-fine-tune-with-no-adapter-isolation risk noted above.

### What to watch

- **Primary**: LIBERO-90 success rate movement on the cheap panel (`research/RUNBOOK.md` adaptive evaluation plan) and eventually canonical eval — especially whether the previously-0%-transfer categories (book-in-caddy, task 73 specifically tracked in the cheap panel) start showing any signal, vs. only the already-partially-transferring tasks improving further.
- **Retention**: the 8 sentinel tasks (2 per original suite) in the cheap panel — any suite trending toward the 90% floor is a stop/reduce-LR signal. This full-fine-tune-with-a-diluted-mix design is exactly the scenario most likely to show forgetting, so retention checks should not be skipped even though initial expectation is that 3e-5 is conservative enough.
- **Training health**: loss curves for `loss_video`/`loss_action` (both components share the mixture-of-transformers backbone; check neither diverges), whether `batch_size=4`/`grad_accum=4` actually fits within our 4x A100-80GB memory budget under ZeRO-1 (not yet verified at this batch size — the smoke test only used `batch_size=1`; first launch should be treated as a scale-verification step, ready to reduce batch size further or add memory optimizations if it OOMs).

### Initial compute plan

- Initial training budget: up to `max_steps=4000` (~4.2 effective epochs over the combined 5633-episode/enlarged mix at global batch 64, given `batch_size=4 x grad_accum=4 x 4 GPUs`), but not committed to running the full budget regardless of evidence.
- Checkpoint/save plan: `save_every=500` — weights-only `.pt` (~2GB) at each save point for cheap-panel screening; full accelerate/deepspeed `state/` checkpoints are much larger (~130GB observed during setup) and should not be kept indiscriminately — prefer deleting older `state/` dirs once a later one supersedes them for resume purposes, keeping only the latest.
- When a progress check might be useful: after the first save point (step 500) as an early sanity check that nothing has silently broken (e.g. the model is actually learning on LIBERO-90 data and not just on the original 4 suites, given the LR is intentionally low), then roughly every 1000 steps or 2 save points thereafter using the cheap 13-task panel, adjusting cadence if results are noisy or clearly trending.
- Expected training/evaluation cost: training cost depends on actual measured step throughput (not yet known at this batch/GPU configuration — to be captured by `$run-fastwam-training`); cheap-panel progress checks are budgeted at ~35-40 min each per `research/RUNBOOK.md`, cheap relative to the training run.

## 4. Exact code and configuration state

- Git commit: `2971af01391bdeb70e730c175eea8489b57fcc28`
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: clean (new config files committed at `2971af0`).
- Files changed:
  - `configs/data/libero_2cam_plus90.yaml` (new)
  - `configs/task/libero_uncond_2cam224_plus90_3e-5.yaml` (new)
- Training config(s): `task=libero_uncond_2cam224_plus90_3e-5`
- Config overrides: none beyond the task config itself and `resume=<released checkpoint path>`
- Dataset config(s): `data=libero_2cam_plus90` (5 dataset_dirs: spatial, object, goal, 10, 90)
- Sampler/mixing configuration: simple concatenation via `MultiLeRobotDataset`, no reweighting (see composition note in Section 3)
- Model/trainable-module configuration: unchanged from released — `model.dit` (video expert 5.00B + action expert 1.02B params) trainable, VAE/text-encoder frozen, no LoRA/adapters
- Optimizer / LR / scheduler: AdamW (betas 0.9/0.95, per `trainer.py`), cosine schedule, `learning_rate=3e-5`, `weight_decay=1e-2`
- Batch size / gradient accumulation / effective batch: `batch_size=4` (per-GPU) x `gradient_accumulation_steps=4` x 4 GPUs = effective global batch 64
- Initial training steps / epochs / budget: `max_steps=4000`, `num_epochs=null`
- Checkpoint/save cadence: `save_every=500`
- Random seed(s): default (`seed: 42` in `configs/train.yaml`, not overridden)
- Resume source: `resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` (weights-only init)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5 (same venv, same 4x A100-80GB machine) — not re-captured here; re-snapshot only if the environment materially changes before/during this run.

### Setup validation (baseline report only)

Not applicable — setup already validated in PROGRESS_0000_BASELINE.md; infrastructure has not materially changed since.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    output_dir=./runs/joint_libero90_finetune/exp0001 \
    wandb.name=exp0001_joint_libero90
  ```
- Start time: 2026-08-08 14:55:34 UTC (first logged step)
- End time: 2026-08-08 18:18:44 UTC (`max_steps reached step=4000`)
- Wall-clock runtime: ~3h 23min
- Exit code/status: 0 (clean exit, `max_steps reached step=4000`, no crash)
- Number of GPUs / distributed world size: 4 (DeepSpeed ZeRO-1, `accelerate launch --num_processes 4`)
- Steps completed: 4000/4000 (full planned budget)
- Throughput / step time: ~2.9s/step steady-state (e.g. steps 3650-3990, 340 steps in ~988s)
- Peak GPU memory: not explicitly logged by DeepSpeed's memory printout beyond the model-construction phase (`MA/Max_MA/CA/Max_CA` lines observed during setup in the earlier smoke test); GPU utilization was 97-100% throughout via `nvidia-smi` spot checks, memory ~55-58GB/GPU at steady state (well within the 79.25GiB budget) — `batch_size=4`/`gradient_accumulation_steps=4` (the untested-at-scale setting flagged in Section 3 "What to watch") turned out to fit comfortably.
- Important training diagnostics: `loss` (combined) started at 1.88 (step 10); `val_loss` (computed periodically per `eval_every=200`) trended down with noise: 0.57 (step 200) → 0.36-0.63 (steps 400-1200, noisy) → 0.22-0.28 (steps 1400-2000) → 0.16-0.30 (steps 2200-3600, generally lower but noisy) → 0.16 (step 4000, final). No divergence, no NaN/Inf warnings observed in the full log.
- Training log path: `runs/joint_libero90_finetune/exp0001/train.log`
- System snapshot path: `research/progress/system_exp_0001.json`
- Dependency snapshot path: `research/progress/system_exp_0001.pip_freeze.txt`

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_004000 (final) | ~3h23min | `progress_check` | pending — cheap 13-task panel to be run next | TBD | TBD |

No intermediate progress check was run *during* this first training pass — the full planned 4000-step budget was short enough (~3.4h) relative to the cost of a cheap-panel progress check (~35-40min) that it was more efficient to let it run to the planned budget and then progress-check the result, rather than interrupting a first, exploratory, already-modest-budget run. Future longer candidates should use intermediate progress checks per the normal loop.

### Why training ended

Planned completion — reached `max_steps=4000` as configured, no early stop triggered (none was being monitored live during this run per the note above).

### Training anomalies

None. One config bug caught before any GPU compute was wasted: the first launch attempt (`num_epochs: null` in the task config) crashed immediately at trainer construction (`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'` at `trainer.py:39`, since `Wan22Trainer.__init__` unconditionally does `int(cfg.num_epochs)` regardless of whether `max_steps` is set) — fixed by setting `num_epochs: 1` (an unused placeholder when `max_steps` drives the step count) and committed as a follow-up commit (`600810e`) before the successful relaunch. Disk usage note: the 8 full-state `checkpoints/state/step_*` deepspeed-ZeRO1 saves were ~80GB each (640GB total) — deleted all but the latest (`step_004000`) immediately after training completed, per the retention plan in Section 3, bringing disk usage back down from 79% to 24%.

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_004000.pt` (final, only checkpoint from this run — no intermediate progress check was performed during training, see Section 6)
- Decision this evaluation was meant to inform: whether this candidate deserves broader confirmation evaluation, or should be rejected/diagnosed.
- Exact task subset / suite coverage: the RUNBOOK's cheap 13-task panel — LIBERO-90 tasks {24,19,46,9,73} + 2 sentinel tasks each from libero_spatial, libero_object, libero_goal, libero_10 ({0,5} in each).
- Why this panel was used: pre-defined in `research/RUNBOOK.md` from baseline evidence, spans the observed LIBERO-90 transfer spectrum plus retention sentinels from all 4 original suites.
- Parent/reference checkpoint and matching result: released checkpoint (`0000_baseline`), same exact task IDs pulled from the canonical baseline's raw per-task results for a fair comparison.
- Candidate result: see table below.
- Trials per task: 5
- Exact command/config:
  ```bash
  CONFIG=libero_uncond_2cam224_plus90_3e-5 \
  CKPT=./runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_004000.pt \
  NUM_GPUS=4 NUM_TRIALS=5 MAX_TASKS_PER_GPU=2 \
  OUTPUT_DIR=./evaluate_results/exp0001_cheap_panel \
  EXTRA_ARGS="EVALUATION.dataset_stats_path=./runs/joint_libero90_finetune/exp0001/dataset_stats.json" \
  bash experiments/libero/run_libero_parallel_test.sh evaluate_results/exp0001_cheap_panel/tasks.txt
  ```
- Raw results path: `evaluate_results/exp0001_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- Runtime: ~25 min wall-clock (launched 18:22, completed before 18:48)
- Validity checks: all 13/13 tasks present, 5/5 episodes each, `failed_tasks.txt` empty, correct checkpoint + this run's own `dataset_stats.json` (normalization differs from the released checkpoint's since it was recomputed fresh over the 5-dataset mix — using the candidate's own stats, not the baseline's, is the correct comparison since the model was trained against its own normalization).

| Suite | Task | Baseline | Candidate (5 trials) | Delta |
|---|---:|---:|---:|---:|
| Spatial | 0 | 98% | 100% | +2 |
| Spatial | 5 | 94% | **40%** | **-54** |
| Object | 0 | 98% | 100% | +2 |
| Object | 5 | 96% | 100% | +4 |
| **Goal** | 0 | 100% | **20%** | **-80** |
| **Goal** | 5 | 100% | **0%** | **-100** |
| **Long/10** | 0 | 94% | **0%** | **-94** |
| **Long/10** | 5 | 100% | **0%** | **-100** |
| LIBERO-90 | 24 | 100% | 80% | -20 |
| LIBERO-90 | 19 | 96% | **0%** | **-96** |
| LIBERO-90 | 46 | 70% | 100% | +30 |
| LIBERO-90 | 9 | 26% | 100% | +74 |
| LIBERO-90 | 73 | 0% | 0% | 0 |

- Decision enabled by this evidence: **DIAGNOSE**, not a direct REJECT/RETEST — the pattern is too specific/surprising (severe collapse concentrated in Goal and Long, while Spatial/Object mostly hold and two LIBERO-90 tasks improve substantially) to design the next candidate from a generic "lower the LR further" guess without understanding the mechanism. Proceeding to `$investigate-fastwam-problem` before making the candidate-level call.
- Reason: 0/5 results on tasks previously at 94-100% success is not plausible sampling noise (binomial p ≈ 3x10⁻⁷ for a single such observation under the baseline's true rate, and it occurred independently on 4 different sentinel tasks across 2 suites) — this is real, substantial catastrophic forgetting on this candidate, concentrated non-uniformly across suites. The retention gate (all original suites >=90%) is clearly failed as measured on this panel; canonical confirmation is not warranted before understanding why the damage is so suite-specific, since that understanding should inform whether the fix is "lower LR further," "reweight the data mix," "something specific about Goal/Long task-description overlap with the newly-dominant LIBERO-90 data," or something else entirely.

### Task-level evidence

Two very different things happened at once: (1) genuine LIBERO-90 learning on at least 2 tasks (9: 26%->100%, 46: 70%->100%), showing the training mechanism *can* work; (2) severe, suite-concentrated forgetting (Goal and Long collapsing to near-0% on both sampled tasks each; Spatial partially so; Object apparently unaffected). This selective pattern is the key puzzle for `$investigate-fastwam-problem` — a uniform "the LR was too high" story would predict roughly proportional damage across all 4 original suites, not near-total collapse in 2 suites while a 3rd is fine.

### Evaluation validity

Confirmed: 13/13 tasks, 5/5 episodes each, zero failures, correct checkpoint and matching stats file. Small trial count (5) means individual percentages are noisy in the 20-80% range, but the multiple independent 0/5 results against a 94-100% baseline are not attributable to that noise.

## 8. Comparison and interpretation

- **LIBERO-90 change**: mixed but with genuine positive signal on some tasks — task 9 (26%->100%) and task 46 (70%->100%) both improved substantially, showing the training mechanism can and does teach LIBERO-90 behavior from the added data. Task 24 regressed mildly (100%->80%, plausibly noise at n=5). Task 19 collapsed (96%->0%) and task 73 (the untransferred book-in-caddy category) remained at 0% — no evidence yet that entirely-new categories are being learned in 4000 steps at this data ratio.
- **Original-suite change (screening evidence only, not canonical)**: Object held essentially perfectly (both sentinels ~unchanged or +4). Spatial mostly held (task 0 unchanged/+2) but task 5 dropped sharply (94%->40%). **Goal and Long both collapsed on both sampled sentinel tasks** (Goal: 100%->20% and 100%->0%; Long: 94%->0% and 100%->0%) — this is the dominant, alarming finding.
- **Task-level pattern**: the forgetting is not uniform across the 4 original suites, which rules out a simple "LR too high, damage proportional everywhere" story. Something about Goal and Long specifically (their task templates, scene overlap with the now-dominant LIBERO-90 data, or their smaller effective share of an already-diluted 4-suite portion of the mix) appears more vulnerable than Object. This needs diagnosis before the next candidate is designed — see `$investigate-fastwam-problem` follow-up.
- **Training progression**: the run trained smoothly to its full planned budget (4000/4000 steps, no crash, decreasing val_loss) — the *training process* was healthy; the *result* is nonetheless unacceptable for promotion. This means the problem is in the candidate design (data mix / LR / duration / interaction with WAM-specific training dynamics), not an implementation bug in this specific run.
- **Retention satisfied**: **no** — Goal and Long are far below the 90% floor on this screening evidence.

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** **no** — LIBERO-Goal and LIBERO-Long both collapsed well below 90% on the cheap screening panel (94-100% baseline down to 0-20%). This is decisive enough (non-noise, per Section 7 reasoning) that canonical evaluation is not needed to confirm the reject; canonical-evaluation compute is better spent on a redesigned candidate.
- **Reason:** severe, suite-concentrated catastrophic forgetting on LIBERO-Goal and LIBERO-Long, despite a reduced learning rate (3e-5 vs the original 1e-4) and full training completing without technical issues. The candidate does show real LIBERO-90 learning capability (2/5 panel tasks with large gains), so the underlying data-and-training approach is not obviously hopeless — but this specific configuration (simple concatenation, no reweighting, lr=3e-5, 4000 steps, batch/grad-accum as configured) is not viable as-is.
- **Checkpoint/branch to preserve:** not preserved as a main-line branch (too damaged on 2 retention-critical suites to be a useful parent), but the checkpoint itself is preserved on `cheikh025/ASR` at `rejected/0001_joint_libero90_finetune/step_004000.pt` per the user's explicit request to keep rejected-candidate checkpoints too.
- **Next main-line parent:** unchanged — `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (0000_baseline remains the accepted main-line checkpoint).

## 10. What this changes for the next experiment

- **Do not repeat simple full-fine-tune joint concatenation at this LR/duration without addressing the Goal/Long-specific vulnerability.** The next candidate needs either: a lower LR still, a shorter duration, an explicit data-mix reweighting that protects the original suites' effective gradient share (rather than just adding LIBERO-90 on top), or a retention-protecting mechanism (e.g. periodic small-scale rehearsal, EWC-style regularization, or reconsidering full-fine-tune vs a more isolated adaptation approach) — but which of these actually addresses the root cause is exactly what `$investigate-fastwam-problem` should clarify before committing compute to another 4000-step run.
- **LIBERO-90 tasks 9 and 46 (and by extension their categories — "put bowl on plate" style placement and "pick up X, put in basket" pick-and-place) respond well to even this flawed recipe** — useful positive signal that the basic data/mechanism works for at least some task categories; task 73 (book-in-caddy) still shows nothing, suggesting entirely-novel-scene categories may need either more targeted data emphasis or more steps/exposure than tasks that already partially resembled the training suites.
- **A cheap progress check during training (not just a single post-hoc screen) would likely have caught this forgetting earlier** and is recommended for the next candidate's training plan, rather than running the full budget blind as this first exploratory candidate did.

## 11. Artifacts

- training log: `runs/joint_libero90_finetune/exp0001/train.log`
- intermediate checkpoint(s): `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_{500,1000,...,3500}.pt` (7 intermediate weights-only checkpoints retained on local disk, not evaluated in this pass — available for a future diagnostic sweep, e.g. to find the step at which Goal/Long forgetting onset, without retraining)
- selected checkpoint: `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_004000.pt` (final; local only, not uploaded to HF — rejected candidate, not worth durable remote storage per `research/RUNBOOK.md` retention policy)
- Hugging Face remote checkpoint path: `cheikh025/ASR/rejected/0001_joint_libero90_finetune/step_004000.pt` — uploaded 2026-08-09 per the user's request to preserve rejected-candidate checkpoints too, not just promoted ones; verified present via `HfApi().list_repo_files`
- config(s): `configs/data/libero_2cam_plus90.yaml`, `configs/task/libero_uncond_2cam224_plus90_3e-5.yaml` (commits `2971af0`, `600810e`)
- raw progress-evaluation outputs: none (no intermediate progress check was run during training, see Section 6)
- raw candidate/confirmation/canonical evaluation outputs: `evaluate_results/exp0001_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- parsed task/suite metrics: table in Section 7 Event 1
- videos/rollouts: per-episode MP4s written under `evaluate_results/exp0001_cheap_panel/<suite>/videos/` by `eval_libero_single.py`, available for qualitative failure inspection if `$investigate-fastwam-problem` needs it
- hardware/software snapshot: `research/progress/system_exp_0001.json` (same machine/env as baseline, re-captured for this experiment)
- dependency snapshot: `research/progress/system_exp_0001.pip_freeze.txt`
- diagnostic scripts/results: none yet — to be created by `$investigate-fastwam-problem`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded
- [x] intermediate checkpoints and progress decisions recorded when used (none used mid-run; intermediate files preserved on disk for future use, noted above)
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created (n/a — not uploaded, rejected)
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — canonical was not run, screening evidence was decisive enough for REJECT)
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
