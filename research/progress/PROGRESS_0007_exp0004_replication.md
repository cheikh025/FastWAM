# PROGRESS_0007 — exp0004 recipe replication with a larger RoboTwin panel

- **Experiment ID:** 0007
- **Status:** `PLANNED`
- **Created:** 2026-08-18
- **Updated:** 2026-08-18
- **Parent experiment:** 0006_disjoint_offset_frozen_backbone_4k (rejected; this candidate directly answers the open question from its Section 9)
- **Parent checkpoint:** `checkpoints/exp0019_expanded_k21_disjoint/step_005000.pt` (same expanded checkpoint used for exp0003-0006 — reused)
- **Selected candidate checkpoint:** none yet
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Not yet run. This report records the candidate design before training launch.

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

- Git commit: pending (recorded after committing this report, before training launch)
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

Not yet launched.

## 7. Evaluation events

None yet.
