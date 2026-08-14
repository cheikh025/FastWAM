# PROGRESS_0004 — reweighted_v2_long_boost

- **Experiment ID:** 0004_reweighted_v2_long_boost
- **Status:** `REJECT`
- **Created:** 2026-08-09
- **Updated:** 2026-08-09
- **Parent experiment:** 0003_reweighted_joint_finetune
- **Parent checkpoint:** `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- **Selected candidate checkpoint:** TBD (pending training)
- **Git branch:** `autoresearch/libero90-v1`
- **Git commit:** `be987053a6816e38f1f83eb82daac1324783ceb9`

## 1. Result at a glance

Pushing LIBERO-Long's oversampling from 5x to 10x produced a **mixed, not clean-win, result**. Long itself improved modestly (panel avg 20%->35%, still far below the 90% floor) — real dose-response signal, confirming more weight helps, but with diminishing returns. However, this came at a cost: **Goal (70%->45%) and Spatial (80%->65%) both regressed**, despite their own oversampling weight being unchanged — because the total mix grew (8917->10857 episodes) when Long's weight increased, so Goal's *proportional* share shrank slightly (24.3%->19.9%) even with the same absolute episode count. LIBERO-90 also slipped a little (68%->60%, 5-task sample). Object remained perfectly stable (100%->100%). **Decision: REJECT** — this is not a strict improvement over 0003, and the zero-sum dilution dynamic (boosting one suite's weight measurably hurts others, not just LIBERO-90) is an important new finding that should reshape the next candidate's design, away from "just increase Long's multiplier further."

## 2. Research state before experiment

0003 (rejected, but strong progress): oversampling Goal and Long 5x each dramatically improved retention. Object fully retained (100% panel avg), Spatial mostly retained (80%, one weak task), Goal mostly recovered (70%, one fully collapsed task). **LIBERO-Long remained the clear outlier (~20% panel avg)** — better than unweighted concatenation (where it was ~0% throughout) but still failing the retention floor by a wide margin, despite already having its gradient share raised from ~6.9% to ~21.8%.

## 3. Candidate design

### Modifications

1. New data config `configs/data/libero_2cam_plus90_reweighted_v2.yaml`: identical to 0003's reweighted mix except **libero_10 (Long) oversampling increased from 5x to 10x** (388 -> 3880 episodes, ~35.7% of the new 10857-episode mix, now comparable to LIBERO-90's own ~36.1% share). Goal stays at 5x (unchanged from 0003, since it already recovered 3/4 sampled tasks at that weight).
2. New task config `configs/task/libero_uncond_2cam224_plus90_reweighted_v2_3e-5.yaml`: identical to 0003's task config (lr=3e-5, batch_size=4, grad_accum=4, max_steps=4000, save_every=500) except pointing at the new data config — isolates Long's weight as the only variable vs 0003.
3. **Infrastructure fix**: `runs/reweighted_v2_libero90_finetune/exp0004/checkpoints/state` is a symlink to `/home/claudeuser/fastwam_state_scratch/exp0004/state`, on the instance's second, independent, likely-ephemeral ~499GB filesystem (the container's own root overlay — found during 0003's disk-full crash recovery, see `research/NOTES.md`). This moves all large, disposable DeepSpeed ZeRO-1 full-state checkpoints off `/workspace` entirely for this run, removing the disk-full failure mode that hit 0003 without needing to remember manual cleanup mid-run (weights-only `.pt` checkpoints, the artifacts that actually matter, still land on `/workspace` as normal via the existing `checkpoints/weights/` path).
4. Training to be launched with `resume=<released checkpoint>.pt` (fresh branch from the released checkpoint, same starting point as 0001/0003).

### Why this candidate

Directly and minimally tests whether Long's remaining shortfall in 0003 is a matter of insufficient gradient weight (in which case pushing the weight further, as done here, should continue closing the gap) or a structural ceiling unrelated to weight (in which case Long would plateau or barely move despite now having a mix share comparable to LIBERO-90 itself). Either outcome is informative: if Long recovers substantially, reweighting alone is sufficient and a final candidate can be tuned from here; if it doesn't, that's strong evidence for the single-step-vs-multi-step behavioral-bias hypothesis from the original investigation, motivating a different kind of fix (e.g. explicit multi-step curriculum, replan/chunking adjustments specific to Long, or excluding some libero_90 tasks least relevant to the primary target while most likely to reinforce single-step behavior).

### What to watch

- **Primary**: LIBERO-Long's 4 sampled sentinel tasks (0,3,5,8) — does the panel average move meaningfully above 0003's ~20%, ideally toward the 90% floor?
- **Secondary**: whether Goal (unchanged weight) holds steady at 0003's level, and whether Object/Spatial (still at 1x, now an even smaller relative share ~4% each given the larger overall mix) show any new degradation from being further diluted by Long's larger weight.
- **Trade-off risk**: LIBERO-90's mix share drops further (44.0%->36.1%) — check tasks 9/46/73 still show reasonable learning, not regressed from 0003's gains.
- **Training health**: same as before, no expected issues; also specifically confirm the state-checkpoint symlink works as intended (checkpoints/state/step_* subdirectories actually appear under `/home/claudeuser/fastwam_state_scratch/exp0004/state/`, not accidentally on `/workspace`) and that `/workspace` disk usage stays flat/low throughout this run.

### Initial compute plan

- Initial training budget: `max_steps=4000` (matching 0001/0003 for consistency).
- Checkpoint/save plan: `save_every=500`; full state saves now land on the second filesystem via symlink, removing the need for aggressive mid-run cleanup on `/workspace` (though the second filesystem's own capacity, ~499GB, should still be monitored — 8 saves x ~80GB = ~640GB would still exceed it, so periodic cleanup of superseded state dirs there remains prudent, just no longer disk-full-crash-risking `/workspace`).
- When a progress check might be useful: given the GPU-contention lesson from 0002, plan to evaluate only after training reaches a clean stop (completion), not mid-flight — same approach as 0003.
- Expected training/evaluation cost: ~4000 steps at ~2.9-3s/step ≈ ~3.3 hours training; eval panel ~35 min (21-task widened panel, matching 0003 for direct comparability).

## 4. Exact code and configuration state

- Git commit: `be987053a6816e38f1f83eb82daac1324783ceb9`
- Git branch: `autoresearch/libero90-v1`
- Working tree clean/dirty before launch: clean (committed at `be98705`).
- Files changed:
  - `configs/data/libero_2cam_plus90_reweighted_v2.yaml` (new)
  - `configs/task/libero_uncond_2cam224_plus90_reweighted_v2_3e-5.yaml` (new)
- Training config(s): `task=libero_uncond_2cam224_plus90_reweighted_v2_3e-5`
- Config overrides: `resume=<released checkpoint>`
- Dataset config(s): `data=libero_2cam_plus90_reweighted_v2` (spatial 1x=434, object 1x=457, goal 5x=2165, long 10x=3880, libero_90 1x=3921; total 10857 episodes)
- Sampler/mixing configuration: oversampling via repeated `dataset_dirs` entries, unchanged mechanism from 0003, only Long's repeat count changed (5->10)
- Model/trainable-module configuration: unchanged
- Optimizer / LR / scheduler: unchanged from 0001/0003 (AdamW, cosine, lr=3e-5)
- Batch size / gradient accumulation / effective batch: unchanged (4 / 4 / 64 global)
- Initial training steps / epochs / budget: `max_steps=4000`
- Checkpoint/save cadence: `save_every=500`; state saves redirected to `/home/claudeuser/fastwam_state_scratch/exp0004/state` via symlink (see Modifications)
- Random seed(s): default (`seed: 42`)
- Resume source: `resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt` (fresh branch)

## 5. Hardware and software environment

Unchanged from `research/progress/PROGRESS_0000_BASELINE.md` Section 5 — same venv, same 4x A100-80GB machine. Additional note: this run uses a second local filesystem (container root overlay, ~499GB) for checkpoint state storage, first identified during 0003's crash recovery — see `research/RUNBOOK.md`.

### Setup validation (baseline report only)

Not applicable — setup already validated; infrastructure has not materially changed (the state-checkpoint symlink is a storage-location change, not a new dependency).

## 6. Training execution and control timeline

- Exact launch command:
  ```bash
  export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/train_zero1.sh 4 \
    task=libero_uncond_2cam224_plus90_reweighted_v2_3e-5 \
    resume=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    output_dir=./runs/reweighted_v2_libero90_finetune/exp0004 \
    wandb.name=exp0004_reweighted_v2_long_boost
  ```
- Start time: 2026-08-09 02:43 UTC
- End time: 2026-08-09 06:14 UTC (`max_steps reached step=4000`)
- Wall-clock runtime: ~3h31min
- Exit code/status: 0 (clean exit, **no crash this time** — the state-checkpoint symlink to the second filesystem fully prevented the disk-full failure mode that hit exp0003)
- Steps completed: 4000/4000
- Throughput: consistent with prior experiments (~3s/step, slightly slower than 0003 given the larger 10857-episode dataset's normalization/indexing overhead)
- Training log path: `runs/reweighted_v2_libero90_finetune/exp0004/train.log`
- **Disk management**: verified working as designed throughout — root filesystem (`/home/claudeuser/fastwam_state_scratch/exp0004/state`) cycled between ~338-418GB free as each ~80GB state checkpoint was written and the superseded one cleaned up automatically (a periodic monitor checked every 5 minutes and deleted all but the newest state dir); `/workspace` only absorbed the small weights-only `.pt` files (~12GB each), ending at 433GB free (58% used) rather than hitting 100% as in exp0003.

### Intermediate checkpoints and progress decisions

| Checkpoint / step | Training runtime so far | Eval purpose | Progress evidence | Decision | Updated training plan |
|---|---:|---|---|---|---|
| step_004000 (final) | ~3h31min | `candidate_screen` | 21-task panel vs exp0003 on the same tasks: Long improved (20%->35%) but Goal/Spatial/LIBERO-90 regressed (see Section 7) | `REJECT` — mixed result, not a clean improvement over 0003 | n/a — training already complete |

No real-time progress check was performed (same rationale as 0003 — deferred to avoid GPU contention, evaluated the single final checkpoint instead).

### Why training ended

Planned completion — reached `max_steps=4000` cleanly, no early stop triggered.

### Training anomalies

None. This run specifically validated the disk-management fix from `research/NOTES.md`/`research/RUNBOOK.md` (state-checkpoint symlink to the second, ephemeral ~499GB filesystem) — no disk-full crash occurred this time, confirming the fix works.

## 7. Evaluation events

### Event 1 — `candidate_screen`

- Checkpoint / training step: `runs/reweighted_v2_libero90_finetune/exp0004/checkpoints/weights/step_004000.pt` (final)
- Decision this evaluation was meant to inform: whether pushing Long's oversampling weight further (5x->10x) improves on exp0003's result without regressing the suites that were already working.
- Exact task subset / suite coverage: identical 21-task widened panel used in exp0003, for direct comparability.
- Why this panel was used: same panel as 0003 specifically to enable a clean paired comparison.
- Parent/reference checkpoint and matching result: both the released checkpoint (`0000_baseline`) and exp0003's result on the same exact tasks (three-way comparison).
- Candidate result: see table below.
- Trials per task: 5
- Exact command/config: same pattern as 0003, `CKPT=./runs/reweighted_v2_libero90_finetune/exp0004/checkpoints/weights/step_004000.pt`, `OUTPUT_DIR=./evaluate_results/exp0004_cheap_panel`
- Raw results path: `evaluate_results/exp0004_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- Runtime: ~35 min
- Validity checks: 21/21 tasks present, 5/5 episodes each, `failed_tasks.txt` empty, correct checkpoint and this run's own `dataset_stats.json`.

| Suite | Task | Baseline | exp0003 (5x/5x) | exp0004 (5x/10x) |
|---|---:|---:|---:|---:|
| Spatial | 0 | 98% | 100% | 100% |
| Spatial | 3 | 96% | 100% | 100% |
| Spatial | 5 | 94% | 20% | **60%** |
| Spatial | 8 | 96% | 100% | **0%** |
| Object | 0 | 98% | 100% | 100% |
| Object | 3 | 100% | 100% | 100% |
| Object | 5 | 96% | 100% | 100% |
| Object | 8 | 100% | 100% | 100% |
| Goal | 0 | 100% | 0% | 0% |
| Goal | 3 | 88% | 80% | **60%** |
| Goal | 5 | 100% | 100% | **20%** |
| Goal | 8 | 100% | 100% | 100% |
| Long | 0 | 94% | 60% | **80%** |
| Long | 3 | 96% | 0% | **20%** |
| Long | 5 | 100% | 20% | **40%** |
| Long | 8 | 92% | 0% | 0% |
| LIBERO-90 | 24 | 100% | 100% | 100% |
| LIBERO-90 | 19 | 96% | 40% | **20%** |
| LIBERO-90 | 46 | 70% | 100% | 100% |
| LIBERO-90 | 9 | 26% | 60% | 60% |
| LIBERO-90 | 73 | 0% | 40% | **20%** |

- Decision enabled by this evidence: `REJECT` — not a strict improvement over 0003. Long shows genuine dose-response improvement (confirming more weight helps, with diminishing returns), but Goal and Spatial regressed on tasks that were previously working, and Long is still far short of the retention floor even after this boost.
- Reason: the total mix growing when Long's weight increased (8917->10857 episodes) diluted Goal's *proportional* share (24.3%->19.9%) even though its absolute oversampling multiplier was unchanged — demonstrating that oversampling one suite is not a free lever, it measurably costs share (and apparently performance) from the others, not just from LIBERO-90.

### Task-level evidence

**Key new finding: the mix is zero-sum, and Goal is unexpectedly sensitive to small share changes.** A ~4.4 percentage-point reduction in Goal's proportional mix share (24.3%->19.9%) coincided with a 25-point drop in its panel average (70%->45%), a much larger effect than the share change alone would naively predict — either Goal sits near a sharp sensitivity threshold in this dynamic, or single-run training variance is contributing meaningfully to these numbers (each experiment here is one training run with no repeated seeds, so this cannot be fully disentangled from noise without a repeat run).

Long's improvement, while real (three of four sampled tasks moved up), is still far from sufficient — task 8 ("put both moka pots on the stove") remains stuck at 0% across both 5x and 10x weightings, suggesting that task specifically may have its own additional obstacle beyond generic Long-suite underweighting (worth a future targeted look, similar to the persistently-weak `libero_goal task 0` and (until this run) `libero_spatial task 5`).

### Evaluation validity

Confirmed: 21/21 tasks, 5/5 episodes each, zero failures, correct checkpoint, matching dataset_stats.json. n=5/task remains imprecise for exact percentages, but the qualitative regression pattern (multiple suites moving in different directions simultaneously, matched against 0003 on identical tasks) is a meaningful signal, not noise-level.

## 8. Comparison and interpretation

- **LIBERO-90 change**: slight regression from 0003 (68%->60% on the 5-task sample), though the two strongest tasks (24, 46) held perfectly; task 73's earlier breakthrough (0%->40% in exp0003) partially reversed (->20% here).
- **Original-suite change**: mixed — Object stable, Long improved but still failing badly, Goal and Spatial both regressed from 0003's level despite unchanged absolute weight for Goal.
- **Task-level pattern**: this is the clearest evidence yet that the training mix behaves as a genuine zero-sum resource allocation problem, not simply "more weight always helps the suite it's given to, with no cost elsewhere." The next candidate needs to account for this rather than continuing to push one suite's multiplier in isolation.
- **Training progression**: training itself was fully healthy (no crash, and this run specifically validated the disk-management fix works).
- **Retention satisfied**: no — Goal (45%) and Long (35%) both fail badly, Spatial (65%) also below floor.

## 9. Decision

- **Decision:** `REJECT`
- **Retention gate passed:** no.
- **Reason:** not a strict improvement over 0003 — Long's modest gain is outweighed by Goal's and Spatial's regressions, and Long remains far short of the retention floor regardless. The zero-sum dilution dynamic revealed here should inform a smarter next design rather than repeating "increase one suite's multiplier further."
- **Checkpoint/branch to preserve:** not preserved as a main-line branch, but uploaded to `cheikh025/ASR` per the now-standing policy of preserving rejected-candidate checkpoints (see Section 11).
- **Next main-line parent:** unchanged — `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (0000_baseline remains accepted).

## 10. What this changes for the next experiment

- **Do not simply increase one suite's oversampling multiplier in isolation** — this measurably steals proportional share from the others (demonstrated here for Goal, which regressed despite unchanged absolute weight). The next candidate should either (a) increase Long's weight while *also* compensating Goal/Spatial with a small additional boost to hold their shares roughly constant, or (b) reduce LIBERO-90's share further (rather than growing the total mix) to make room for Long's increase without diluting the other original suites.
- **Long-specific**: task 8 ("put both moka pots on the stove") stayed at 0% across both 5x and 10x weighting — worth checking whether it has a specific obstacle (e.g. a two-object multi-step composition that's unusually far from anything in the current mix) separate from generic Long-suite underweighting.
- **Consider whether single-seed training runs are introducing too much noise to reliably compare candidates at this granularity** — the Goal sensitivity finding in particular would benefit from a repeat-seed check before being taken as a strong effect rather than partly noise, if compute allows.
- **Checkpoints for rejected candidates are now uploaded to HF as standing policy** (per the user's explicit instruction) — this experiment's checkpoint follows that convention.

## 11. Artifacts

- training log: `runs/reweighted_v2_libero90_finetune/exp0004/train.log`
- intermediate checkpoint(s): `runs/reweighted_v2_libero90_finetune/exp0004/checkpoints/weights/step_{500,...,3500}.pt` retained locally
- selected checkpoint: `runs/reweighted_v2_libero90_finetune/exp0004/checkpoints/weights/step_004000.pt`
- Hugging Face remote checkpoint path: `cheikh025/ASR/rejected/0004_reweighted_v2_long_boost/step_004000.pt` — uploaded and verified 2026-08-09
- config(s): `configs/data/libero_2cam_plus90_reweighted_v2.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_v2_3e-5.yaml` (commit `be98705`)
- raw progress-evaluation outputs: none (no real-time progress check)
- raw candidate/confirmation/canonical evaluation outputs: `evaluate_results/exp0004_cheap_panel/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/gpu*_task*_results.json`
- parsed task/suite metrics: table in Section 7
- videos/rollouts: per-episode MP4s under `evaluate_results/exp0004_cheap_panel/<suite>/videos/`
- hardware/software snapshot: unchanged from baseline/0001 (`research/progress/system_exp_0001.json`)
- dependency snapshot: unchanged from baseline/0001
- diagnostic scripts/results: none beyond the panel above; the state-checkpoint symlink + periodic disk monitor used here are a reusable infra pattern, documented in `research/RUNBOOK.md`

## 12. Reproducibility checklist

- [x] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded
- [x] hardware/software environment recorded
- [x] intermediate checkpoints and progress decisions recorded when used
- [x] logs and checkpoint paths recorded
- [x] remote checkpoint path recorded and verified if an HF backup was created
- [x] every evaluation event has purpose/settings/raw results recorded
- [x] five-suite metrics recorded when canonical evaluation ran (n/a — canonical not run)
- [x] task-level evidence preserved when relevant
- [x] final decision and reasoning recorded
- [x] `research/EXPERIMENTS.jsonl` updated
- [x] `research/STATE.md` updated

## 11. Artifacts

- training log: TBD
- intermediate checkpoint(s): TBD
- selected checkpoint: TBD
- Hugging Face remote checkpoint path: TBD (blocked on HF_TOKEN restoration, see `research/STATE.md`)
- config(s): `configs/data/libero_2cam_plus90_reweighted_v2.yaml`, `configs/task/libero_uncond_2cam224_plus90_reweighted_v2_3e-5.yaml`
- raw progress-evaluation outputs: TBD
- raw candidate/confirmation/canonical evaluation outputs: TBD
- parsed task/suite metrics: TBD
- videos/rollouts: TBD
- hardware/software snapshot: unchanged from baseline
- dependency snapshot: unchanged from baseline
- diagnostic scripts/results: none yet

## 12. Reproducibility checklist

- [ ] exact candidate commit recorded
- [x] parent checkpoint recorded
- [x] intentional modifications listed
- [x] initial training plan recorded
- [x] training configuration and command recorded (config side; exact launch command to be recorded by `$run-fastwam-training`)
- [ ] hardware/software environment recorded (unchanged, will confirm no drift at launch)
- [ ] intermediate checkpoints and progress decisions recorded when used
- [ ] logs and checkpoint paths recorded
- [ ] remote checkpoint path recorded and verified if an HF backup was created
- [ ] every evaluation event has purpose/settings/raw results recorded
- [ ] five-suite metrics recorded when canonical evaluation ran
- [ ] task-level evidence preserved when relevant
- [ ] final decision and reasoning recorded
- [ ] `research/EXPERIMENTS.jsonl` updated
- [ ] `research/STATE.md` updated
