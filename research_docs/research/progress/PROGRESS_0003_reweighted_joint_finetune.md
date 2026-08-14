# PROGRESS_0003 — reweighted_joint_finetune

- **Experiment ID:** 0003_reweighted_joint_finetune
- **Status:** `REJECT`
- **Created:** 2026-08-08
- **Updated:** 2026-08-08
- **Parent experiment:** 0001_joint_libero90_finetune (same starting point — released checkpoint; 0002 was a continuation of 0001, not a data-design change, and is not this candidate's parent)
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- **Selected candidate checkpoint:** `runs/reweighted_libero90_finetune/exp0003/checkpoints/weights/step_004000.pt` (evaluated, rejected — upload to HF pending, see Section 9)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `61f0fb15a3917e47817024b3d1f6442309011fb4`

## 1. Result at a glance

Oversampling LIBERO-Goal and LIBERO-10 5x each in the training mix (restoring ~their original ~25% share) **dramatically improved retention** vs 0001/0002's unweighted mix: LIBERO-Object fully retained (100% on the 4-task panel sample), LIBERO-Spatial mostly retained (3/4 tasks perfect, one weak), LIBERO-Goal mostly recovered (3/4 tasks 80-100%, one task still fully collapsed), and LIBERO-90 improved *despite* a reduced relative mix share — including the first-ever nonzero result on the previously completely-untransferred book-in-caddy category (task 73: 0%->40%). **LIBERO-Long remains the clear outlier**, still far below the retention floor (~20% panel average) though showing more signal than in 0001/0002 (one task moved 0%->60%). Also hit and recovered from a **disk-full training crash** at the very last checkpoint save (see Section 6) — the weights checkpoint survived intact. **Decision: REJECT** (Long's retention clearly fails, and Goal task 0 and Spatial task 5 are each still fully/mostly collapsed) but this is the strongest, most informative candidate so far and directly motivates a targeted follow-up (push Long's oversampling further, since partial-but-incomplete improvement suggests dilution was part but not all of its problem).

## 2. Research state before experiment

0001 (rejected): joint fine-tune, simple 1x concatenation of LIBERO-90 into the 4-suite mix, lr=3e-5, 4000 steps. Real LIBERO-90 learning on some tasks, but severe forgetting on LIBERO-Goal and LIBERO-Long.

0002 (rejected): extended 0001's training to 8000 steps. Decisively ruled out "just needs more time" — Goal/Long never recovered across the full trajectory, and Spatial's partial recovery reversed with more training.

Investigation (`research/NOTES.md`) identified two structural leads: (a) LIBERO-Goal has direct task-template overlap with several LIBERO-90 tasks (e.g. "turn on the stove" is the literal same instruction in both), plausibly causing persistent representational interference rather than simple dilution; (b) LIBERO-Long is 100% multi-step/long-horizon while every added LIBERO-90 task is single-step, and LIBERO-90 outnumbers Long 10:1 in episodes, plausibly biasing the model toward premature termination. Both explanations point toward the same first fix to try: give Goal and Long more effective weight in the training mix.

## 3. Candidate design

### Modifications

1. New data config `configs/data/libero_2cam_plus90_reweighted.yaml`: identical to `libero_2cam_plus90.yaml` except `libero_goal_no_noops_lerobot` and `libero_10_no_noops_lerobot` are each **listed 5 times** in `dataset_dirs` (verified this is a safe, code-free way to oversample: `MultiLeRobotDataset` stores per-entry `LeRobotDataset` instances in a plain list with no path-uniqueness assumption anywhere in the loading code, so a repeated path simply contributes its episodes to the concatenated pool multiple times). This raises Goal from 433->2165 effective episodes and Long from 388->1940, giving them ~24.3% and ~21.8% of the new 8917-episode mix respectively (close to their original ~25% share in the pre-LIBERO-90 4-suite-only mix), while Spatial/Object stay at 1x (they held up reasonably in 0001/0002 — Object fully recovered, Spatial only partially regressed) and LIBERO-90 stays at 1x (3921 episodes, now ~44.0% of the mix — reduced from ~69.6% but still the largest single share, appropriate for the primary target).
2. New task config `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml`: identical to 0001's task config (`learning_rate=3e-5`, `batch_size=4`, `gradient_accumulation_steps=4`, `max_steps=4000`, `save_every=500`) except pointing at the new reweighted data config — **deliberately kept everything else the same as 0001** so this experiment isolates the data-mix change as the only variable, for a clean comparison against 0001's known-bad result at the same step budget.
3. Text embeddings re-verified via `scripts/precompute_text_embeds.py task=libero_uncond_2cam224_plus90_reweighted_3e-5` (idempotent — same 5 underlying datasets/prompt set as 0001's mix, just different sampling weights, so no new prompts).
4. Training to be launched with `resume=<released checkpoint>.pt` (fresh branch from the released checkpoint, same as 0001 — not a continuation of 0001/0002's now-damaged checkpoint).

### Why this candidate

This is the most direct test of the leading explanation from the investigation: that Goal/Long's forgetting is (at least partly) a gradient-share/dilution problem exacerbated by direct task-template competition with the now-dominant LIBERO-90 data. Oversampling is the simplest possible lever to test this — no new code, no removal of any LIBERO-90 task, isolates the "more relative weight for the vulnerable suites" variable cleanly against 0001's identical-except-data-mix result. If Goal/Long still collapse under this reweighted mix, that would suggest the interference is not primarily about relative gradient share (contradicting part of the leading hypothesis) and point toward a different fix (e.g. explicitly excluding the specific overlapping LIBERO-90 tasks from Goal's competing signal, rather than just outweighting them).

### What to watch

- **Primary**: whether LIBERO-Goal and LIBERO-Long's sentinel tasks (same 6-task diagnostic subset used in 0001/0002's investigation, for direct comparability) hold up this time, compared to their complete collapse under 1x weighting.
- **Secondary**: whether Spatial (which partially regressed in 0001/0002 despite being left at 1x here) also benefits incidentally from the overall reduced LIBERO-90 dominance (69.6%->44.0%), or whether it needs its own oversampling boost in a future iteration if it still struggles.
- **Trade-off risk**: with LIBERO-90's mix share roughly halved (69.6%->44.0%), LIBERO-90 learning could be slower/weaker than in 0001 — check whether tasks 9 and 46 (which showed strong gains in 0001) still improve meaningfully under the reweighted mix, and whether task 73 (book-in-caddy, 0% throughout so far) shows any different behavior.
- **Training health**: same as before — loss curves, no divergence expected given identical optimizer/LR/architecture settings to 0001, which trained cleanly.

### Initial compute plan

- Initial training budget: `max_steps=4000` (matching 0001 exactly, for a clean A/B comparison at the same budget — 0002 already showed more steps doesn't help under the OLD mix, so there's no strong prior for needing more steps under the NEW mix either; extend later via progress check if evidence supports it, same resumable-state mechanism as 0002 demonstrated).
- Checkpoint/save plan: `save_every=500`, same disk-management discipline as before (delete superseded full-`state/` dirs immediately after each new one is confirmed complete, keep all weights-only `.pt` files).
- When a progress check might be useful: **learned from 0002's GPU-contention mistake — schedule the check for after training reaches a clean stopping point (e.g. step 2000, ~half budget) rather than assuming concurrent eval capacity.** At minimum, plan one check around step 2000 using the same 6-task diagnostic subset, directly comparable against 0001/0002's step-2000 result (which was 0/6 across the board under the old mix) to see if reweighting changes the *early* stability-gap dynamic, not just the final outcome.
- Expected training/evaluation cost: ~4000 steps at ~2.9-3s/step ≈ ~3.3 hours training; each diagnostic check ~15-20 min (6 tasks x 5 trials, small subset) once GPUs are free.

## 4. Exact code and configuration state

- Git commit: `61f0fb15a3917e47817024b3d1f6442309011fb4`
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: clean (committed at `61f0fb1`).
- Files changed:
  - `configs/data/libero_2cam_plus90_reweighted.yaml` (new)
  - `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml` (new)
- Training config(s): `task=libero_uncond_2cam224_plus90_reweighted_3e-5`
- Config overrides: `resume=<released checkpoint>`
- Dataset config(s): `data=libero_2cam_plus90_reweighted` (spatial 1x=434, object 1x=457, goal 5x=2165, long 5x=1940, libero_90 1x=3921; total 8917 episodes)
- Sampler/mixing configuration: oversampling via repeated `dataset_dirs` entries (see Modifications) — no custom sampler code, `MultiLeRobotDataset`'s default concatenation + uniform sampling handles it.
- Model/trainable-module configuration: unchanged (`model.dit` only, no LoRA/adapters)
- Optimizer / LR / scheduler: unchanged from 0001 (AdamW, cosine, lr=3e-5, weight_decay=1e-2)
- Batch size / gradient accumulation / effective batch: unchanged from 0001 (4 / 4 / 64 global)
- Initial training steps / epochs / budget: `max_steps=4000` (matching 0001)
- Checkpoint/save cadence: `save_every=500`
- Random seed(s): default (`seed: 42`)
- Resume source: `resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` (fresh branch, weights-only init)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5 — same venv, same 4x A100-80GB machine.

### Setup validation (baseline report only)

Not applicable — setup already validated; infrastructure has not materially changed.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    output_dir=./runs/reweighted_libero90_finetune/exp0003 \
    wandb.name=exp0003_reweighted_joint_libero90
  ```
- Start time: 2026-08-08 22:49:39 UTC
- End: training reached step 4000 (all optimizer steps completed) but the **final checkpoint save crashed** at ~2026-08-09 02:18 UTC due to `/workspace` disk hitting 100% full (see Training anomalies below).
- Wall-clock runtime: ~3h29min to step 4000
- Exit code/status: nonzero (crashed during final checkpoint write), but training compute itself had already completed all 4000 steps before the crash.
- Steps completed: 4000/4000 (optimizer steps); the weights-only checkpoint for step 4000 was confirmed written successfully and valid before the crash occurred in the subsequent full-state save.
- Throughput: consistent with prior experiments (~2.9-3s/step)
- Training log path: `runs/reweighted_libero90_finetune/exp0003/train.log`

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_004000 (final, evaluated after the crash) | ~3h29min | `candidate_screen` | 21-task panel: Object 100%, Spatial 80%, Goal 70%, Long 20%, LIBERO-90 68% (see Section 7) | `REJECT` this exact candidate; design a Long-focused follow-up | n/a — training already complete |

No real-time progress check was performed during training (planned for step 2000, but deliberately deferred to avoid the GPU-contention mistake from exp0002 — the step_002000.pt checkpoint remained valid and could have been evaluated after the fact, but was superseded by the step 4000 result before that became necessary).

### Why training ended

Planned completion of optimizer steps (4000/4000), immediately followed by an unplanned crash during the final checkpoint's full-state save due to disk exhaustion — not a training/candidate-design failure.

### Training anomalies

**Disk-full crash**: `/workspace` reached 100% full (1016G/1016G) during the step_004000 full-state (DeepSpeed ZeRO-1) checkpoint save, causing a `PytorchStreamWriter failed writing file` / `unexpected pos` truncated-write error on all 4 ranks. Root cause: unlike 0001/0002, superseded `checkpoints/state/step_*` dirs were not cleaned up *during* this run (only planned for after completion), so all 8 full-state saves (~80GB each, ~640GB) accumulated on top of ~325GB already in use, exceeding the 1016GB volume. **Recovery**: `save_checkpoint()` writes the weights-only `.pt` file *before* attempting the full-state save (`trainer.py:583-599`), so `step_004000.pt` (~12GB) was already complete and valid on disk when the crash occurred — verified by loading it (`torch.load(..., weights_only=False)` succeeded, correct `step: 4000`, full 1649-key state dict). Only the *resumable* full-state checkpoint for step 4000 was lost (the incomplete/corrupted 44GB partial `state/step_004000` dir was deleted; `state/step_003500`, the latest complete one, was kept). No training compute was lost — all 4000 planned steps had already run. Full detail and the general lesson (always proactively clean up `state/` dirs *during*, not just after, any run with `save_every` frequent enough to accumulate multiple ~80GB saves) recorded in `research/NOTES.md`. Also discovered during recovery: this instance has a second, independent, likely-ephemeral ~499GB filesystem (the container's root overlay, writable under `/home/claudeuser` or `/tmp`) separate from `/workspace`'s 1016GB persistent volume — planned to be used for future runs' disposable `state/` checkpoints specifically to prevent this recurring, per the new `research/RUNBOOK.md` note.

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/reweighted_libero90_finetune/exp0003/checkpoints/weights/step_004000.pt` (final, verified valid despite the crash during its full-state save)
- Decision this evaluation was meant to inform: whether the reweighted data mix resolves the retention failure seen in 0001/0002, and whether this candidate deserves broader confirmation or promotion-track evaluation.
- Exact task subset / suite coverage: the RUNBOOK's widened 21-task panel — 5 LIBERO-90 tasks (24,19,46,9,73) + 16 sentinel tasks (4 per original suite: {0,3,5,8} each).
- Why this panel was used: the panel was widened from 13 to 21 tasks (2->4 sentinels/suite) specifically after 0001's investigation revealed within-suite non-uniformity, making this the first candidate evaluated with the improved-coverage panel.
- Parent/reference checkpoint and matching result: released checkpoint (`0000_baseline`), same exact task IDs from the canonical baseline's raw per-task results.
- Candidate result: see table below.
- Trials per task: 5
- Exact command/config:
  ```bash
  CONFIG=libero_uncond_2cam224_plus90_reweighted_3e-5 \
  CKPT=./runs/reweighted_libero90_finetune/exp0003/checkpoints/weights/step_004000.pt \
  NUM_GPUS=4 NUM_TRIALS=5 MAX_TASKS_PER_GPU=2 \
  OUTPUT_DIR=./evaluate_results/exp0003_cheap_panel \
  EXTRA_ARGS="EVALUATION.dataset_stats_path=./runs/reweighted_libero90_finetune/exp0003/dataset_stats.json" \
  bash experiments/libero/run_libero_parallel_test.sh evaluate_results/exp0003_cheap_panel/tasks.txt
  ```
- Raw results path: `evaluate_results/exp0003_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- Runtime: ~35 min (launched ~02:20, completed before ~02:56 UTC)
- Validity checks: all 21/21 tasks present, 5/5 episodes each, `failed_tasks.txt` empty, correct checkpoint and this run's own `dataset_stats.json`.

| Suite | Task | Baseline | Candidate (5 trials) |
|---|---:|---:|---:|
| Spatial | 0 | 98% | 100% |
| Spatial | 3 | 96% | 100% |
| Spatial | 5 | 94% | **20%** |
| Spatial | 8 | 96% | 100% |
| Object | 0 | 98% | 100% |
| Object | 3 | 100% | 100% |
| Object | 5 | 96% | 100% |
| Object | 8 | 100% | 100% |
| Goal | 0 | 100% | **0%** |
| Goal | 3 | 88% | 80% |
| Goal | 5 | 100% | 100% |
| Goal | 8 | 100% | 100% |
| Long | 0 | 94% | **60%** |
| Long | 3 | 96% | **0%** |
| Long | 5 | 100% | **20%** |
| Long | 8 | 92% | **0%** |
| LIBERO-90 | 24 | 100% | 100% |
| LIBERO-90 | 19 | 96% | 40% |
| LIBERO-90 | 46 | 70% | 100% |
| LIBERO-90 | 9 | 26% | 60% |
| LIBERO-90 | 73 | 0% | **40%** |

- Decision enabled by this evidence: `REJECT` this exact candidate (Long's panel average ~20%, far below the 90% floor; Goal task 0 fully collapsed; Spatial task 5 still weak) but **do not abandon the reweighting direction** — Object and most of Spatial/Goal are strong evidence the mechanism works, and Long's partial-but-incomplete improvement (0%->60% on one task, but two others still at 0%) suggests the next iteration should push Long's weight further rather than redesign from scratch.
- Reason: retention gate requires all four original suites >=90%; Long clearly fails by a wide margin on this panel (not attributable to n=5 noise given two outright 0/5 results), and Goal/Spatial each retain one fully-or-mostly-collapsed task. LIBERO-90 gains (task 73 in particular) are promising enough to keep pursuing the reweighting approach rather than reverting to unweighted concatenation.

### Task-level evidence

Suite averages on this panel (4 sentinel tasks/suite for originals, 5 tasks for LIBERO-90):

| Suite | Panel avg | Pattern |
|---|---:|---|
| Object | 100% | Fully retained, no exceptions |
| Spatial | 80% | 3/4 tasks perfect, task 5 specifically weak (a recurring problem task across 0001/0002/0003) |
| Goal | 70% | 3/4 tasks recovered to 80-100%, task 0 specifically still fully collapsed |
| Long | 20% | Still the dominant problem; some signal now present where there was previously none (task 0), but 3/4 tasks still failing |
| LIBERO-90 (5-task sample) | 68% | Genuine improvement across the board; task 73 (book-in-caddy) shows its first-ever nonzero result |

Two specific tasks recur as stubbornly weak across every reweighting/retention experiment so far: **libero_spatial task 5** and **libero_goal task 0** — both fully or mostly collapsed in 0001, 0002, and now 0003 despite the overall mix improving substantially. This task-specific persistence (rather than resolving with the suite-level fix) suggests these two individual tasks may have their own specific interference source (e.g. direct visual/instruction overlap with a particular LIBERO-90 task) worth a targeted look if they remain outliers in a future candidate.

### Evaluation validity

Confirmed: 21/21 tasks, 5/5 episodes each, zero failures, correct checkpoint despite the crash (verified via direct `torch.load`), matching dataset_stats.json. n=5/task is imprecise for exact percentages but the pattern (multiple 0/5 and 5/5 results) is well beyond noise for the qualitative conclusions drawn here.

## 8. Comparison and interpretation

- **LIBERO-90 change**: positive and broad-based despite LIBERO-90's reduced mix share (69.6%->44.0%) — 4/5 sampled tasks improved or held (task 19 regressed 96%->40% but was already an outlier at baseline for being unusually strong on a hard-transfer suite), and task 73 broke out of the entirely-untransferred book-in-caddy category for the first time across all experiments so far.
- **Original-suite change**: night-and-day difference from 0001/0002. Object fully retained. Spatial and Goal both mostly recovered with one persistent weak task each. Long remains clearly failing but shows more life than under unweighted concatenation.
- **Task-level pattern**: the reweighting fix (addressing gradient-share dilution) clearly helped Object/Spatial/Goal substantially and Long partially — consistent with dilution being the dominant mechanism for Object/Spatial/Goal, but only a partial explanation for Long (which likely also suffers from the single-step-vs-multi-step behavioral bias hypothesized in the original investigation, not fully addressed by more episodes alone since the *episodes* are still short-horizon dominated in aggregate. even with Long itself oversampled).
- **Training progression**: training itself remained healthy through all 4000 steps (the only issue was an unrelated disk-management crash during checkpoint I/O, not a training/optimization problem).
- **Retention satisfied**: no, not for this exact candidate (Long fails badly, two individual tasks in Goal/Spatial remain weak) — but this is the closest any candidate has come, and the remaining gap is well-characterized and actionable.

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** no — Long is far below 90% on this screening evidence, and this is not attributable to trial-count noise.
- **Reason:** despite dramatic improvement over 0001/0002, Long's retention still clearly fails the 90% floor, and Goal task 0 / Spatial task 5 remain individually collapsed. Canonical evaluation is not warranted for this exact checkpoint given the clear Long failure, but the reweighting mechanism is validated strongly enough (Object/Spatial/Goal mostly recovered, LIBERO-90 gains preserved and even improved in one previously-stuck category) to iterate directly rather than abandon the approach.
- **Checkpoint/branch to preserve:** not preserved as a formal branch (Long's failure is too severe for the checkpoint itself to be broadly useful), but **uploaded to `cheikh025/ASR`** at `rejected/0003_reweighted_joint_finetune/step_004000.pt` (2026-08-09) per the user's explicit request to preserve rejected-candidate checkpoints going forward, not just promoted ones. `HF_TOKEN` was restored by the user (via `/workspace/.env`) after being lost partway through the session.
- **Next main-line parent:** unchanged — `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (0000_baseline remains accepted).

## 10. What this changes for the next experiment

- **The reweighting direction is validated and should be iterated on, not abandoned.** The next candidate should push LIBERO-Long's oversampling weight further (e.g. 8-10x instead of 5x) given the partial-but-incomplete improvement suggests dilution was a real but not fully-addressed factor; consider whether Goal also needs a small further boost given task 0 remains fully collapsed while its other 3 sampled tasks recovered well.
- **libero_spatial task 5 and libero_goal task 0 are persistently weak across every experiment so far** (0001, 0002, 0003) despite very different overall mix compositions — worth a targeted diagnostic look (e.g. checking whether either has an unusually close visual/instruction match to a specific dominant LIBERO-90 task) if they remain outliers after the next reweighting iteration.
- **Operational**: always proactively delete superseded `checkpoints/state/*` dirs *during* a run (not just after), and consider redirecting `state/` saves to the second, ephemeral ~499GB filesystem found during this experiment's crash recovery (`/home/claudeuser` or `/tmp`) to remove this failure mode entirely for future runs.
- **HF_TOKEN needs to be restored** in the environment before any further checkpoint uploads can proceed (lost partway through this session; flagged to the user).

## 11. Artifacts

- training log: `runs/reweighted_libero90_finetune/exp0003/train.log`
- intermediate checkpoint(s): `runs/reweighted_libero90_finetune/exp0003/checkpoints/weights/step_{500,...,3500}.pt` retained locally
- selected checkpoint: `runs/reweighted_libero90_finetune/exp0003/checkpoints/weights/step_004000.pt`
- Hugging Face remote checkpoint path: `cheikh025/ASR/rejected/0003_reweighted_joint_finetune/step_004000.pt` — uploaded 2026-08-09
- config(s): `configs/data/libero_2cam_plus90_reweighted.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_3e-5.yaml` (commit `61f0fb1`)
- raw progress-evaluation outputs: none (no real-time progress check, see Section 6)
- raw candidate/confirmation/canonical evaluation outputs: `evaluate_results/exp0003_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- parsed task/suite metrics: table in Section 7
- videos/rollouts: per-episode MP4s under `evaluate_results/exp0003_cheap_panel/<suite>/videos/`
- hardware/software snapshot: unchanged from baseline/0001 (`research/progress/system_exp_0001.json`)
- dependency snapshot: unchanged from baseline/0001
- diagnostic scripts/results: none beyond the panel above

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded (unchanged from baseline/0001)
- [x] intermediate checkpoints and progress decisions recorded when used
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — canonical not run, screening evidence decisive enough for REJECT)
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
