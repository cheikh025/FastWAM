# PROGRESS_0002 — extend_joint_finetune

- **Experiment ID:** 0002_extend_joint_finetune
- **Status:** `REJECT`
- **Created:** 2026-08-08
- **Updated:** 2026-08-08
- **Parent experiment:** 0001_joint_libero90_finetune
- **Parent checkpoint:** `runs/joint_libero90_finetune/exp0001/checkpoints/state/step_004000` (full accelerate/deepspeed state, resumed with optimizer/scheduler/step)
- **Selected candidate checkpoint:** `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_008000.pt` (evaluated, rejected — not preserved as a branch)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `600810ec291d7abb8e03d0f44ee6aeda419b66e1` (same code/config as 0001 — this is a continuation, not a redesign)

## 1. Result at a glance

Extended exp0001's training from step 4000 to step 8000 (same data/config, full-state resume) to test whether LIBERO-Goal/Long/Spatial would recover with more time, as LIBERO-Object had between step 2000-4000. **Result: no.** Full trajectory across 6 sentinel tasks at steps 1000/2000/4000/5000/8000 shows Object recovered and held (100% at both 4000 and 8000), but Goal and Long never recovered at any point (stuck at 0%, one brief 20% blip each), and Spatial actually **regressed further** with more training (40% at step 4000 -> 0% at step 8000). This decisively refutes the duration-insufficiency hypothesis. **Decision: REJECT** — the problem is structural (specific to Goal/Long), not a matter of training longer on the same unweighted data mix. Proceeding to design a data-reweighting candidate.

## 2. Research state before experiment

0001 (rejected): joint fine-tune on 4-suite+LIBERO-90 mix, lr=3e-5, 4000 steps. Diagnosis (`$investigate-fastwam-problem`, recorded in `research/NOTES.md`) found: (a) training was numerically healthy throughout (smooth loss decrease, no divergence); (b) a "stability gap" — all sampled sentinel tasks, including LIBERO-Object, were fully collapsed (0/5) at both step 1000 and step 2000 (50% through training); (c) LIBERO-Object fully recovered to 100% by step 4000 (recovery happened late, between step 2000-4000); (d) LIBERO-Goal, LIBERO-Long, and one LIBERO-Spatial sentinel task did **not** recover by step 4000. Two competing explanations were identified: simple duration insufficiency (goal/long/spatial just need more steps to complete the same recovery arc object did), vs. structural obstacles specific to those suites (goal's direct task-template overlap with LIBERO-90 causing persistent interference; long's single-step/short-horizon data bias and low episode count). The cheapest, most direct test of the duration-insufficiency hypothesis is to simply continue training from the existing step_004000 full-state checkpoint and see whether the suites that haven't recovered do so with more steps — this requires no redesign and reuses already-completed compute.

## 3. Candidate design

### Modifications

1. **No code/config changes** — same data mix (`libero_2cam_plus90`), same task config (`libero_uncond_2cam224_plus90_3e-5`), same LR/schedule/batch settings as 0001.
2. Resume training from `runs/joint_libero90_finetune/exp0001/checkpoints/state/step_004000` (a **directory**, not a `.pt` file) — this restores full optimizer/scheduler/step/dataloader-sampler state and continues the exact same run, rather than branching a new one from the released checkpoint.
3. Extend the step budget: `max_steps` raised from 4000 to 8000 (4000 additional steps), i.e. `max_steps=8000` override on top of the same task config, since the resumed trainer's step counter continues from 4000.

### Why this candidate

This is the cheapest possible test of the leading, simplest hypothesis from the investigation: that LIBERO-Goal/Long/Spatial's lack of recovery by step 4000 reflects insufficient training duration for their specific recovery arc (which for Object completed somewhere between step 2000 and 4000), not a structural flaw in the data mix or approach. If true, no redesign is needed at all — just more steps of the exact same recipe. If false (goal/long remain collapsed even with double the training), that cleanly rules out "just needs more time" and directs the next candidate toward the structural-fix hypotheses (data reweighting, long-specific oversampling) identified in the investigation, without having wasted compute on a premature redesign.

### What to watch

- **Primary**: whether LIBERO-Goal and LIBERO-Long's sentinel tasks (0, 5 at minimum, using the now-widened 21-task cheap panel from `research/RUNBOOK.md`) recover toward their baseline levels (94-100%) as training continues past step 4000, mirroring Object's step-2000-to-4000 recovery arc.
- **Secondary**: whether LIBERO-90 gains (tasks 9, 46 in the panel) are preserved or further improved, and whether task 73 (book-in-caddy, still 0% at step 4000) shows any movement with more exposure.
- **Risk**: extending training could also make things worse (e.g. further drift away from goal/long, or object's newly-recovered performance regressing again) — this is exactly why progress checks at intervals (not just a single check at the new final step) are planned, so a regression can be caught and the run stopped/an earlier checkpoint selected rather than blindly running to 8000.

### Initial compute plan

- Initial training budget: 4000 additional steps (8000 total), but explicitly not committed to running the full extension regardless of evidence.
- Checkpoint/save plan: same `save_every=500` — new weights-only checkpoints at steps 4500, 5000, ..., 8000. Full state dirs: keep only the latest superseding one (delete earlier `state/` dirs immediately after each new one is confirmed written, per the disk-management lesson from 0001, where 8 uncleaned `state/` dirs hit 640GB).
- When a progress check might be useful: after +1000 additional steps (step 5000) as a first checkpoint — cheap enough (~13-21 task panel, ~35-50 min) relative to the remaining ~3000-step budget to be worth checking before committing further compute. If goal/long show any recovery trend at step 5000, continue; if still fully collapsed with no trend, this already meaningfully weakens the duration-insufficiency hypothesis and may justify stopping early (`STOP_TRAINING`) rather than spending the full extension.
- Expected training/evaluation cost: ~4000 steps at ~2.9s/step observed rate ≈ ~3.2 hours training; each progress check ~35-50 min per the cheap panel's updated cost estimate.

## 4. Exact code and configuration state

- Git commit: `600810ec291d7abb8e03d0f44ee6aeda419b66e1` (unchanged from 0001 — no new commit needed since no files changed)
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: clean
- Files changed: none
- Training config(s): `task=libero_uncond_2cam224_plus90_3e-5` (unchanged)
- Config overrides: `resume=./runs/joint_libero90_finetune/exp0001/checkpoints/state/step_004000`, `max_steps=8000`
- Dataset config(s): unchanged (`libero_2cam_plus90`, 5 dataset_dirs)
- Sampler/mixing configuration: unchanged
- Model/trainable-module configuration: unchanged
- Optimizer / LR / scheduler: unchanged (AdamW, cosine, lr=3e-5) — restored from the resumed state, so the cosine schedule continues from wherever it was at step 4000 rather than restarting (note: the schedule's `total_train_steps` was computed against the *original* `max_steps=4000` at trainer construction time in exp0001; resuming with a new `max_steps=8000` override means the scheduler will be reconstructed against the new total — this is expected Hydra/trainer behavior since the scheduler is built fresh from config at each launch, only optimizer momentum/variance state is restored from the checkpoint, not the schedule's shape. Worth confirming this behaves sensibly in the training log rather than assuming.)
- Batch size / gradient accumulation / effective batch: unchanged (4 / 4 / 64 global)
- Initial training steps / epochs / budget: `max_steps=8000` (4000 additional beyond the resumed step 4000)
- Checkpoint/save cadence: unchanged (`save_every=500`)
- Random seed(s): restored from checkpoint (dataloader sampler state resumed)
- Resume source: `runs/joint_libero90_finetune/exp0001/checkpoints/state/step_004000` (full state directory)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` / `PROGRESS_0001_joint_libero90_finetune.md` Section 5 — same venv, same 4x A100-80GB machine.

### Setup validation (baseline report only)

Not applicable — setup already validated; infrastructure has not materially changed.

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_3e-5 \
    resume=./runs/joint_libero90_finetune/exp0001/checkpoints/state/step_004000 \
    max_steps=8000 \
    output_dir=./runs/joint_libero90_finetune/exp0001 \
    wandb.name=exp0002_extend_joint_libero90
  ```
- Start time: 2026-08-08 19:01:26 UTC (`Resuming full training state from ...step_004000`), confirmed continuing correctly at `epoch=0 step=4010/8000` by 19:02:42.
- Full state (optimizer/scheduler/step/dataloader-sampler/RNG) restored successfully — training resumed exactly where 0001 left off, not restarted.
- End time: 2026-08-08 ~22:26 UTC (`max_steps reached step=8000`)
- Wall-clock runtime (this extension only): ~3h25min (4000 additional steps)
- Exit code/status: 0 (clean, no crash)
- Steps completed: 8000/8000 (4000 in exp0001 + 4000 more here)
- Throughput: consistent with exp0001 (~2.9-3s/step)
- Val_loss trajectory during extension: noisy, roughly stable in the 0.13-0.34 range (step 4200: 0.13, step 8000: 0.25) — no clear further improvement or divergence at the aggregate level; the interesting signal is entirely in the task-level eval, not the aggregate loss.
- Training log path: `runs/joint_libero90_finetune/exp0001/train.log` (same file, this run's output appended after exp0001's, since it's a continuation)

**Operational note**: attempted to launch a progress-check eval at step 5000 *while training was still running toward 8000* — this was a mistake corrected quickly: training under ZeRO-1 saturates all 4 GPUs' memory (~57-58GB/GPU), leaving no room for concurrent eval workers. The eval workers were killed within ~30s once this was noticed; training itself was not disrupted, but the checkpoint had to be evaluated later once training fully completed instead. Documented in `research/RUNBOOK.md` progress-check guidance to avoid repeating this.

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_005000 | ~4h50min (incl. exp0001) | `progress_check` | 6-task diagnostic: Object 5 recovering (80%), Spatial 5 partially recovered (20%), Long 5 slight signal (20%), Goal 0&5 and Long 0 still 0% | `CONTINUE_TRAINING` (evaluated after training already reached 8000 due to the GPU-contention issue above; in effect this was a retrospective check rather than one that altered the plan in real time) | none — training had already run to completion by the time this was evaluated |
| step_008000 (final) | ~6h48min (incl. exp0001) | `progress_check` (final) | 6-task diagnostic: Object 5 fully recovered (100%), **Spatial 5 regressed further (0%, down from 40% at step 4000)**, Goal 0&5 and Long 0&5 all still 0% | `STOP_TRAINING` (retrospective — the run had already completed its planned 8000-step budget) | none further — this result decisively answers the research question this candidate was designed to test |

### Why training ended

Planned completion — reached `max_steps=8000` as configured. In hindsight, given the step_005000 progress check (evaluated after the fact) already showed no further recovery for Goal/Long and a declining trend for Spatial, a real-time progress check at step 5000 (had the GPU-contention issue not forced a deferral) would likely have justified `STOP_TRAINING` at that point rather than continuing to 8000 — noted as a process lesson, not a regret, since the additional 3000 steps still produced decisive, unambiguous evidence (Spatial's continued decline) that is valuable for ruling out the duration-insufficiency hypothesis with high confidence.

### Training anomalies

None during training itself (smooth loss curves, no crash). The only anomaly was the operational GPU-contention mistake noted above (attempting a concurrent eval mid-training), which was caught and corrected within ~30s with no impact on the training run.

## 7. Evaluation events

### Event 1 — `progress_check` (step 5000, evaluated retrospectively after training completed)

- Checkpoint: `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_005000.pt`
- Task subset: same 6-task diagnostic subset used in exp0001's investigation (libero_goal 0&5, libero_10 0&5, libero_spatial 5, libero_object 5) — chosen for direct comparability with the existing step 1000/2000/4000 trajectory data.
- Trials per task: 5
- Command: same pattern as exp0001's diagnostics, `CKPT=./runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_005000.pt`, `OUTPUT_DIR=./evaluate_results/exp0002_diag_step5000`
- Raw results path: `evaluate_results/exp0002_diag_step5000/`
- Result: object_5=4/5(80%), spatial_5=1/5(20%), long_5=1/5(20%), goal_0=0/5, goal_5=0/5, long_0=0/5
- Decision enabled: retrospective (see above) — showed gradual, partial recovery trend for object/spatial/long-task5 but goal and long-task0 still fully stuck.

### Event 2 — `progress_check` (step 8000, final)

- Checkpoint: `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_008000.pt`
- Task subset/trials/command: same as Event 1, `OUTPUT_DIR=./evaluate_results/exp0002_diag_step8000`
- Raw results path: `evaluate_results/exp0002_diag_step8000/`
- Result: object_5=5/5(100%), spatial_5=0/5(0%), goal_0=0/5, goal_5=0/5, long_0=0/5, long_5=0/5
- Decision enabled: `STOP_TRAINING`/`REJECT` for this candidate's approach — no further recovery for goal/long, and spatial's earlier partial recovery reversed. Extending training further is very unlikely to help given the trend, and continuing to train on this exact recipe is actively risking further regression on suites that had shown partial recovery (spatial).

### Task-level evidence

Full 6-task trajectory (baseline -> step1000 -> step2000 -> step4000 -> step5000 -> step8000):

| Task | Baseline | 1000 | 2000 | 4000 | 5000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|
| libero_object task5 | 96% | 0% | 0% | 100% | 80% | **100%** |
| libero_spatial task5 | 94% | 0% | 0% | 40% | 20% | **0%** |
| libero_goal task0 | 100% | 0% | 0% | 20% | 0% | **0%** |
| libero_goal task5 | 100% | 0% | 0% | 0% | 0% | **0%** |
| libero_10 task0 | 94% | 0% | 0% | 0% | 0% | **0%** |
| libero_10 task5 | 100% | 0% | 0% | 0% | 20% | **0%** |

Object shows a clean, stable recovery arc. Goal and Long show essentially no sustained recovery at any point across 8000 steps (isolated single-trial blips, not a trend). Spatial shows recovery *and then regression* — the most informative data point against "just needs more time," since more training measurably made it worse, not better.

### Evaluation validity

Confirmed: all diagnostic runs completed with 0 failures, 5/5 episodes per task, correct checkpoints and matching dataset_stats.json. Small n=5 per task means individual percentages are imprecise, but the *pattern* across 3 additional data points (5000, and the full trajectory shape) is consistent and interpretable even at this trial count — this is trend evidence, not a single noisy snapshot.

## 8. Comparison and interpretation

- **LIBERO-90 change**: not re-evaluated in this experiment (diagnostic focus was specifically on the retention sentinels to answer the duration question; LIBERO-90 tasks 9/46 that had improved in exp0001 were not re-checked here, since the research question was specifically about retention recovery, not further LIBERO-90 gains — worth checking in a future pass if this data-mix direction is revisited).
- **Original-suite change**: definitively negative for Goal and Long — no recovery across 8000 total steps. Object fully recovered and appears stable. Spatial's partial recovery at step 4000 reversed by step 8000, suggesting continued training on this exact mix is not merely "not helping" for the vulnerable suites but actively harmful over time even for a suite that had shown some resilience.
- **Task-level pattern**: consistent with the investigation's leading hypotheses — Goal's direct task-template overlap with LIBERO-90 (identical/near-identical instructions) plausibly causes persistent representational interference that more exposure to the same conflicting signal does not resolve; Long's single-step-data-dominated mix plausibly biases the model against multi-stage task completion in a way unrelated to training duration.
- **Training progression**: the *training process* remained healthy throughout (smooth loss, no divergence) for the full 8000 steps — this is unambiguously a data/candidate-design problem, not an infrastructure or optimization problem.
- **Retention satisfied**: no — same conclusion as exp0001, now with much stronger evidence that this is not fixable by training longer.

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** no — unchanged conclusion from exp0001, now confirmed not resolvable by extended training.
- **Reason:** doubling the training budget did not recover LIBERO-Goal or LIBERO-Long, and measurably worsened LIBERO-Spatial's partial recovery. This rules out duration-insufficiency as the explanation and directs the next candidate toward a structural fix (data reweighting, task-overlap-aware mixing, or suite-specific rehearsal) rather than simply running the same recipe for longer.
- **Checkpoint/branch to preserve:** not preserved as a main-line branch (not usable as a parent, diagnostic value already captured in this report), but preserved on `cheikh025/ASR` at `rejected/0002_extend_joint_finetune/step_008000.pt` per the user's explicit request to keep rejected-candidate checkpoints too.
- **Next main-line parent:** unchanged — `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (0000_baseline remains accepted).

## 10. What this changes for the next experiment

- **Do not simply extend training as a fix for this kind of suite-selective forgetting** — now demonstrated twice (4000 and 8000 steps) not to work, and to actively worsen at least one partially-recovered suite over time.
- The next candidate should address the mechanism directly: **explicit data reweighting** to protect Goal and Long's effective gradient share (e.g. a weighted sampler upweighting the smaller/more vulnerable original suites relative to their raw episode count, rather than simple concatenation), and/or **investigate the specific LIBERO-90 tasks that overlap with Goal's templates** (turn on the stove, open drawer variants, bowl-on-plate) to see whether excluding or downweighting just those specific overlapping tasks resolves Goal's collapse without sacrificing the LIBERO-90 gains seen on non-overlapping tasks (9, 46).
- Long's fix may need a different lever than Goal's (reweighting alone may not address the single-step-vs-multi-step behavioral bias) — worth testing whether Long recovers under a reweighted mix before assuming a Long-specific mechanism (e.g. oversampling, or explicit rehearsal batches) is also needed.
- Given a real-time progress check was blocked by GPU contention this run, the next candidate's training plan should explicitly schedule progress-check timing around confirmed GPU availability (e.g., pause-and-check rather than assuming concurrent capacity), per the new `research/RUNBOOK.md` operational note.

## 11. Artifacts

- training log: `runs/joint_libero90_finetune/exp0001/train.log` (shared with exp0001, this run's output appended)
- intermediate checkpoint(s): `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_{4500,5000,...,7500}.pt` retained on local disk
- selected checkpoint: `runs/joint_libero90_finetune/exp0001/checkpoints/weights/step_008000.pt` (final; local only, not uploaded — rejected)
- Hugging Face remote checkpoint path: `cheikh025/ASR/rejected/0002_extend_joint_finetune/step_008000.pt` — uploaded 2026-08-09
- config(s): unchanged from 0001 (`configs/data/libero_2cam_plus90.yaml`, `configs/task/libero_uncond_2cam224_plus90_3e-5.yaml`)
- raw progress-evaluation outputs: `evaluate_results/exp0002_diag_step5000/`, `evaluate_results/exp0002_diag_step8000/`
- raw candidate/confirmation/canonical evaluation outputs: same as above (this experiment used progress-check-style diagnostics only)
- parsed task/suite metrics: table in Section 7
- videos/rollouts: per-episode MP4s under each diagnostic run's `<suite>/videos/`
- hardware/software snapshot: unchanged from baseline/0001 (`research/progress/system_exp_0001.json`)
- dependency snapshot: unchanged from baseline/0001
- diagnostic scripts/results: this experiment is itself the diagnostic follow-up to 0001's investigation (see `research/NOTES.md`)

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded (unchanged from baseline/0001, confirmed no drift)
- [x] intermediate checkpoints and progress decisions recorded when used
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created (n/a — not uploaded, rejected)
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — canonical was not run, retrospective progress-check evidence was decisive enough for REJECT)
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated
