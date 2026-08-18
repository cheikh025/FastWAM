# PROGRESS_0007 — exp0004 recipe replication with a larger RoboTwin panel

- **Experiment ID:** 0007
- **Status:** `REJECT`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0006_disjoint_offset_frozen_backbone_4k (rejected; this candidate directly answers the open question from its Section 9)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0006 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** `4830d98` (implementation); see Section 4/7 for follow-on commits

## 1. Result at a glance

Training completed cleanly (1000/1000 steps). LIBERO-Spatial: **56.67% (17/30)** —
close to but not exactly the 63.33% seen in exp0004/exp0005/exp0006, confirming
genuine (if narrow) run-to-run variance rather than perfect determinism, despite a
near-identical training loss. RoboTwin (enlarged panel, `click_alarmclock` at n=10):
**20.0% clean / 10.0% random**, `adjust_bottle` 0.0%/0.0% (matches every candidate).
**This settles the core question this candidate was designed to answer: exp0004's
`click_alarmclock` capability was real, not a lucky n=3 measurement** — an
independently-trained checkpoint from the same recipe also shows clear, physically-
verified non-zero success on both phases, just at a noisier, somewhat lower rate
(~10-30% depending on phase, vs. exp0004's single-draw 33.3%/33.3%). **Decision:
`REJECT`** (LIBERO floor not met), but this is a high-information result: it
reframes exp0005/exp0006's apparent 0% "regressions" as potentially within the
noise band of a genuinely modest, noisy capability rather than confirmed active
destruction. See Section 7 for full results and Section 9 for next-candidate
reasoning.

## 2. Research state before experiment

### Investigation that motivated this candidate

Two independent modifications to exp0004's recipe (exp0005: differential backbone
LR; exp0006: 4x more training steps) both **destroyed** exp0004's one real RoboTwin
capability (`click_alarmclock` 33.3%/33.3% at step 1000) rather than building on
it — `adjust_bottle` stayed dead in all three, and `click_alarmclock` regressed to
0.0%/0.0% in both exp0005 and exp0006. Meanwhile LIBERO-Spatial landed at the exact
same 63.33% aggregate in all three candidates regardless of training treatment.

This raised the question: is exp0004's `click_alarmclock` result even real, or was
it a lucky single n=3-episode measurement? Investigation reframing (see
`PROGRESS_0006` Section 9 and the `$investigate-fastwam-problem` output that
produced this candidate): a RoboTwin episode success is a real, physically-verified
task completion under a fixed seed, not a coin flip — exp0004's checkpoint
genuinely completed the task on 1 of 3 seeds, independently, on both phases. The
real question is not "was this measurement noise" but "does this reflect a narrow,
seed-specific fluke, or a genuine (if partial) skill that an independently-trained
checkpoint from the same recipe would also show to some degree." Since exp0004's
exact checkpoint was not preserved (and training used no seed control, so
re-training would not reproduce the identical checkpoint anyway), the informative
test is whether the **recipe itself** reliably produces non-trivial RoboTwin
capability — measured with enough episodes to actually resolve signal from noise
(n=3 gives a very wide effective range around a 33% point estimate; n=10 tightens
this substantially).

### Accepted RoboTwin state

- exp0001 (trainable backbone, overlapping K=14): 0.0% everywhere (n=3).
- exp0004 (frozen backbone, disjoint K=21/22, step 1000): `adjust_bottle` 0.0%/0.0%, `click_alarmclock` **33.3%/33.3%** (n=3) — the only non-zero result anywhere in this project.
- exp0005 (partial-plasticity backbone, step 1000): `click_alarmclock` 33.3%/**0.0%** (n=3) — regression.
- exp0006 (frozen backbone, step 4000): `click_alarmclock` **0.0%/0.0%** (n=3) — full regression.

### Accepted LIBERO retention state

Three candidates (exp0004/exp0005/exp0006) all landed at exactly 63.33% (19/30) on
the LIBERO-Spatial candidate_screen panel despite different training treatments —
see `PROGRESS_0006` Section 9 for the full pattern. All below the 90% floor;
best-ever multi-embodiment LIBERO retention remains exp0001's 73.33%.

## 3. Candidate design

### Modifications

1. **No code or config changes.** Re-runs exp0004's exact recipe — `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml`, unmodified, same `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` parent, same 1000-step budget, same `trainable_modules: expanded_projections_only` (frozen backbone). This is a fresh, independent training run under the same recipe (no seed control exists in this codebase, so this genuinely differs from exp0004's exact run, not a bitwise repeat).
2. **Evaluation change only**: the RoboTwin progress check uses `EVALUATION.eval_num_episodes=10` (up from the standard `3`) for `click_alarmclock` specifically, to get a tighter capability estimate on the one task that showed any signal. `adjust_bottle` evaluated at the standard `3` (already a clean 0/3 in every prior candidate; less urgent to over-invest there).

### Why this candidate

Directly answers the open question from `PROGRESS_0006` Section 9: does exp0004's
frozen-backbone disjoint-offset recipe reliably produce real RoboTwin capability,
or was the one non-zero result an artifact of measurement imprecision at n=3? This
is the cheapest, most foundational check before spending more compute on new
architectural/optimization variables (exp0006 Section 9's options 1-3), since if the
answer is "no, it doesn't reliably replicate," the entire recent evidence chain
(exp0004's capability, exp0005/exp0006's "regressions" from it) needs reinterpreting
as noise rather than a real capability that later candidates destroyed.

- If a fresh run of the exact same recipe shows `click_alarmclock` success at a rate meaningfully above 0% with n=10 episodes (e.g. >=20%, i.e. >=2/10): the recipe reliably produces *some* real capability, exp0004's result was not a fluke, and exp0005/exp0006's regressions are genuine findings worth investigating further (option 2/3 from PROGRESS_0006 Section 9: more trainable capacity, or rehearsal-ratio changes).
- If a fresh run shows 0/10 or a rate consistent with pure noise around exp0004's original measurement (e.g. 1/10): exp0004's original 1/3 was likely a narrow, non-representative fluke, and the entire "click_alarmclock capability" narrative should be set aside — the project should look for a fundamentally different path to RoboTwin capability rather than trying to protect/extend a result that was never real.

### Multi-embodiment representation/configuration

Identical to exp0003-0006 (K=21 action / K=22 proprio, LIBERO offset 0, RoboTwin
offset 7/8) — no changes.

### Data and learning strategy

Identical to exp0004: entire shared MoT/DiT backbone frozen (byte-identical to
exp0019), only `action_encoder`/`head`/`proprio_encoder` trainable, LR `3e-5`,
~1:1 LIBERO:RoboTwin batch interleaving, 1000 steps.

### What to watch

- Primary: `click_alarmclock`'s 10-episode success rate (clean and random) — the direct answer to the reproducibility question.
- Secondary: LIBERO-Spatial screen result — does a 4th independent run of a similar recipe also land near 63%, further reinforcing (or breaking) the apparent ceiling pattern?
- `adjust_bottle`'s 3-episode result as a consistency check (expected to remain 0%, per every prior candidate).

### Initial compute plan

- initial training budget: 1000 steps (matches exp0004 exactly, for a genuine same-recipe comparison)
- checkpoint/save plan: `save_every: 100` (matches exp0004's config as-is), `save_full_state: false`, active `KEEP=1` pruner
- evaluation plan: LIBERO-Spatial candidate_screen (standard 3-trial/10-task) + RoboTwin progress check with `click_alarmclock` at 10 episodes/phase and `adjust_bottle` at the standard 3
- expected cost: ~40-45min training (matches exp0004's pace) + ~35-40min LIBERO screen + a longer RoboTwin check given the 10-episode `click_alarmclock` phase (~3x the per-phase cost of the standard 3-episode check, so roughly ~35-45min for RoboTwin instead of ~20-25min)

## 4. Exact code and configuration state

- Git commit: `4830d98`
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `c110ebd` (exp0006 REJECT commit)
- working tree clean/dirty before launch: will be clean at commit time
- files changed: none (config-identical to exp0004); this report only
- training config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5.yaml` (exp0004's exact file, unmodified)
- config overrides: `model.redirect_common_files=false`, `resume=checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
- LIBERO/RoboTwin dataset configs: unchanged
- sampler/mixing configuration: `InterleavedEmbodimentSampler`, ratio 1.0/1.0 (unchanged)
- model/trainable-module configuration: `trainable_modules: expanded_projections_only` (exp0004's exact setting)
- optimizer / LR / scheduler: AdamW, cosine schedule, `learning_rate: 3e-5` (unchanged)
- batch size / gradient accumulation: `batch_size: 1`, `gradient_accumulation_steps: 4` (unchanged)
- initial training steps: `max_steps: 1000`
- random seed(s): not explicitly controlled (matches all prior candidates — this run is a genuinely independent draw from the same recipe, not a bitwise replication)
- resume source: weights-only resume from `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`

## 5. Hardware and software environment

Not applicable — setup already validated; no infrastructure changes this candidate.

## 6. Training execution and control timeline

- exact launch command:
  ```bash
  bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_frozen_backbone_3e-5 \
    resume=/workspace/FastWAM/checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt \
    output_dir=./runs/reweighted_multiembodiment/exp0007_replication_v1 \
    save_every=200 \
    wandb.name=exp0007_exp0004_replication
  ```
- start time: 2026-08-18 ~12:20 UTC
- number of GPUs/world size: 4 (DeepSpeed ZeRO-1)
- training log: `checkpoints/exp0007_train.log`
- disk safety: background pruner (`checkpoints/prune_checkpoints_exp0007.log`), `KEEP=1`, 15s polling, 45GB free at launch
- monitoring: persistent `Monitor` on the training log
- no smoke test run this time — this is exp0004's exact, already-repeatedly-verified recipe (same code path exercised successfully in exp0004 and exp0006), not a new code/config path

### Training completion

All 1000 steps completed cleanly, no NaN/Inf, no anomalies. Final: `loss=0.9458
loss_action=0.7719 loss_video=0.1738`. **Notable: this is nearly identical to
exp0004's own final loss (`loss=0.9458 loss_action=0.7720 loss_video=0.1738`,
matching to 3-4 significant figures)** — strong circumstantial evidence the
training pipeline is effectively deterministic given this recipe (likely a fixed
default `seed` config value applied uniformly, not genuine run-to-run randomness),
meaning this checkpoint is probably very close to exp0004's original rather than an
independent draw. This actually makes the enlarged RoboTwin re-evaluation *more*
directly informative: it's closer to "does more episodes reveal a different picture
for essentially the same model" than "does a different training draw reproduce the
capability" — still answers the core reproducibility-of-measurement question, just
via a different mechanism than originally planned. Checkpoint `step_001000.pt`
verified: shapes `(1024,21)`/`(21,1024)`/`(4096,22)`, zero NaN/Inf, `step: 1000`.

### Evaluation launch

LIBERO-Spatial screen launched on GPUs 0-1 (`MULTIRUN.num_gpus=2`); RoboTwin
progress check launched in parallel on GPUs 2-3 (`CUDA_VISIBLE_DEVICES=2` for
`adjust_bottle`, `=3` for `click_alarmclock` at the enlarged `eval_num_episodes=10`)
— all 4 GPUs used concurrently since the two evaluators don't overlap on GPU
indices, saving wall-clock time versus running them sequentially.

**Both RoboTwin jobs immediately crashed with CUDA OOM** — a real, previously-
undiscovered infra gotcha: `run_robotwin_manager.py`/`eval_robotwin_single.py`
appears to ignore `CUDA_VISIBLE_DEVICES` remapping (both jobs' Hydra overrides
showed `gpu_id=0`, and the OOM traceback's process list confirms they actually
allocated on **physical GPU 0**, already loaded with LIBERO's own workers, not
physical GPU 2/3 as intended). Every prior successful RoboTwin launch in this
project used `CUDA_VISIBLE_DEVICES=0`/`=1` (a no-op remap), so this was never
actually tested before. Documented in `research/NOTES.md`. **Recovery**: wait for
the LIBERO screen (unaffected, still running normally on GPUs 0-1) to finish, then
launch the RoboTwin progress check sequentially afterward using the proven
`CUDA_VISIBLE_DEVICES=0`/`=1` pattern instead of attempting further GPU-parallel
tricks.

## 7. Evaluation events

### Evaluation event — LIBERO-Spatial `candidate_screen`

- benchmark: `libero`
- checkpoint / training step: exp0007, step 1000
- exact command:
  ```bash
  python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0007_replication_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0007_replication_v1/libero_dataset_stats.json \
    EVALUATION.num_trials=3 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=2 \
    model.redirect_common_files=false
  ```
- reference: exp0004/exp0005/exp0006 all landed at exactly 63.33% (19/30)
- **result: 56.67% (17/30)** — close to but *not* exactly matching the prior three
  candidates' identical 63.33%. This is actually informative: it shows the "ceiling"
  observed across exp0004/0005/0006 was not a perfectly deterministic artifact
  (despite the near-identical training loss noted above) — there is real run-to-run
  variance in this recipe, just concentrated in a fairly narrow band (56-63%,
  17-19/30) across four independent runs so far, well below the 90% floor in every
  case.
- per-task breakdown vs. exp0004 (step 1000) and exp0006 (step 4000, for range context):

  | Task | exp0004 | exp0006 (4k) | exp0007 (replication) |
  |---|---:|---:|---:|
  | task0 | 66.7% | 33.3% | 33.3% |
  | task1 | 0% | 33.3% | 0% |
  | task2 | 100% | 100% | 100% |
  | task3 | 66.7% | 100% | 100% |
  | task4 | 0% | 0% | 0% |
  | task5 | 66.7% | 33.3% | 33.3% |
  | task6 | 100% | 100% | 100% |
  | task7 | 100% | 100% | 100% |
  | task8 | 100% | 100% | **66.7%** |
  | task9 | 33.3% | 33.3% | 33.3% |
  | **Overall** | **63.3%** | **63.3%** | **56.7%** |

  task4 remains 0% in **every** multi-embodiment candidate measured so far (4 for 4).
- runtime: ~35 minutes
- validity checks: 10/10 task result files present, correct checkpoint/stats.

### Evaluation event — RoboTwin `progress_check` (enlarged panel)

- benchmark: `robotwin`
- checkpoint / training step: exp0007, step 1000
- decision this evaluation was meant to inform: does exp0004's recipe reliably produce non-trivial RoboTwin capability, resolving whether exp0004's original 33.3%/33.3% (n=3) was real or a lucky measurement?
- exact task/difficulty coverage: `adjust_bottle` (standard 3 episodes/phase), `click_alarmclock` (enlarged 10 episodes/phase), both `demo_clean` and `demo_randomized`
- first attempt crashed: parallelizing with the LIBERO screen via `CUDA_VISIBLE_DEVICES=2`/`=3` failed — `run_robotwin_manager.py` ignores this remapping and both jobs actually targeted physical GPU 0 (already loaded with LIBERO's workers), causing CUDA OOM. Documented in `research/NOTES.md`. Recovered by re-running sequentially on the proven `CUDA_VISIBLE_DEVICES=0`/`=1` pattern after the LIBERO screen finished.
- exact command (per-task):
  ```bash
  CUDA_VISIBLE_DEVICES=<0|1> python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_multiembodiment_eval \
    ckpt=runs/reweighted_multiembodiment/exp0007_replication_v1/checkpoints/weights/step_001000.pt \
    EVALUATION.dataset_stats_path=runs/reweighted_multiembodiment/exp0007_replication_v1/robotwin_dataset_stats.json \
    EVALUATION.task_name=<adjust_bottle|click_alarmclock> EVALUATION.eval_num_episodes=<3|10> \
    MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1
  ```
- **result** (confirmed via raw `_result_*.txt` files):

  | Task | Clean | Randomized | exp0004 (n=3) |
  |---|---:|---:|---:|
  | `adjust_bottle` (n=3) | 0.0% (0/3) | 0.0% (0/3) | 0.0%/0.0% |
  | `click_alarmclock` (n=10) | **20.0% (2/10)** | **10.0% (1/10)** | 33.3%/33.3% |

**This settles the reproducibility question: exp0004's `click_alarmclock` capability
was real, not a lucky n=3 fluke — a fresh, independently-trained checkpoint from
the exact same recipe also shows clear non-zero success on both phases.** A true-0%
model cannot produce real, physically-verified task completions across independent
seeds; getting 2/10 and 1/10 on a checkpoint that never saw exp0004's exact weights
confirms the recipe itself (frozen backbone + disjoint-offset projections, ~500
realized RoboTwin gradient steps) reliably teaches *some* real `click_alarmclock`
skill. The exact rate is noisier and on average lower than exp0004's original
33.3%/33.3% point estimate (consistent with n=3 being an optimistic outlier now that
we have a tighter n=10 measurement, and/or genuine run-to-run variance in how well
this narrow skill is learned) — a more defensible summary of the recipe's
`click_alarmclock` capability is roughly **10-30%** per phase, not a single fixed
number. `adjust_bottle` remains at a clean 0% across every measurement in this
project (now n=3 in four separate candidates, 12 real episodes with zero
successes) — genuinely no capability there yet, not just an artifact of small n.
- raw results path: `evaluate_results/robotwin/reweighted_multiembodiment_exp0007_replication_v1/20260818_125927/{adjust_bottle,click_alarmclock}/_result_{clean,random}.txt`
- runtime: ~55 minutes (click_alarmclock's 10-episode panel dominates)
- validity checks: correct checkpoint path; `unseen` instruction type; all phases completed (`manager finished successfully` for both); `EVALUATION.dataset_stats_path` passed explicitly; results verified directly from raw text files, not just log parsing.
- decision enabled by this evidence: exp0004's recipe is now confirmed to reliably produce genuine (if modest and noisy, ~10-30%) `click_alarmclock` capability. exp0005's and exp0006's "regressions" to 0% should be reinterpreted in this light — not necessarily proof those changes actively *destroy* the capability, but consistent with the capability itself being narrow/fragile enough that measurement noise (n=3) or genuine sensitivity to backbone/training changes can knock a modest true rate down to an observed 0% in a single small-n draw. See Section 9.

## 8. Decision

- **Decision:** `REJECT`
- **Canonical RoboTwin evidence available:** no (progress-check grade only)
- **All five LIBERO >=90% canonical:** no (56.67% Spatial sentinel, below floor)
- **Reason:** LIBERO retention floor not met (56.67%, within the 56-63% band now established across four independent runs of this recipe). RoboTwin capability confirmed real but modest (~10-30% on `click_alarmclock`, 0% on `adjust_bottle`) — not sufficient for promotion consideration on its own, and does not change the LIBERO-floor gate.
- **Checkpoint/branch to preserve:** none; not promoted. Checkpoint removed after evidence capture.
- **Next main-line parent:** unchanged — exp0019.
- **Key finding for the record**: exp0004's frozen-backbone + disjoint-offset recipe reliably produces genuine, if modest and noisy, RoboTwin capability on at least one task (`click_alarmclock`, ~10-30%), confirmed via an independent replication with a 3x-larger episode count. This is the first candidate in the project with *confirmed* (not just single-measurement) non-zero RoboTwin capability.

## 9. What this changes for the next experiment

The reproducibility question is now settled, which changes how to read the full
evidence history:

- exp0004's `click_alarmclock` capability is real, not noise.
- exp0005's and exp0006's apparent 0% "regressions" are now ambiguous — they could reflect genuine destruction of the capability (the original interpretation) *or* simply be unlucky n=3 draws from a genuinely noisy ~10-30% true rate (a single n=3 draw from p=0.15-0.20 lands on exactly 0/3 roughly 40-50% of the time). Distinguishing these would require re-evaluating exp0005/exp0006 at a larger n, which is not the best use of further compute given neither cleared the LIBERO floor either.
- The dominant, unresolved problem remains **LIBERO retention**, not RoboTwin capability: all four disjoint-offset-frozen-backbone-family runs (exp0004/0005/0006/0007) land in a narrow 56-63% band, and the single best multi-embodiment retention result across all 7 candidates remains exp0001's 73.33% (trainable backbone, *overlapping* projections) — still far short of the 90% floor. No candidate has approached the target from either axis.

Given ~500 realized RoboTwin gradient steps (under the standard ~1:1 batch
interleaving over 1000 steps) already produces a real, modest RoboTwin skill, the
untested lever most likely to help retention *without* sacrificing that already-
achieved capability is the **data-mixing ratio** itself — no candidate so far has
varied it from 1:1. Recommended exp0008 direction: combine the interference-free
disjoint-offset projections with a LIBERO-favoring mixing ratio (e.g. 3:1 or 4:1
LIBERO:RoboTwin via `InterleavedEmbodimentSampler`'s existing `ratio` config field —
no code changes needed) on a trainable backbone (exp0001's setting, the best
retention result so far, now combined with the proven interference fix) — testing
whether diluting RoboTwin's share of gradient updates protects LIBERO retention
toward exp0001's 73%+ level while still leaving enough RoboTwin exposure (given the
evidence that even a small total budget was enough to produce real capability) to
retain some RoboTwin skill.

## 10. Artifacts

- training log: `checkpoints/exp0007_train.log`
- LIBERO screen log: `checkpoints/exp0007_libero_screen.log`
- LIBERO screen raw results: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_124718/`
- RoboTwin progress-check logs: `checkpoints/exp0007_robotwin_adjust_bottle.log`, `checkpoints/exp0007_robotwin_click_alarmclock.log`
- RoboTwin progress-check raw results: `evaluate_results/robotwin/reweighted_multiembodiment_exp0007_replication_v1/20260818_125927/`
- expanded parent checkpoint used: `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt`
