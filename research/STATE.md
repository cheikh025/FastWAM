# Current Research State

Status: `RESEARCH_LOOP_ACTIVE — exp0017 (full-backbone + LIBERO-Long-protected mixing) is the current best candidate, actively training past its 5th consecutive progress check. Full 50-task RoboTwin Clean mean has climbed unplateaued across every check: 12.6% (exp0014 frozen-backbone ceiling, for reference) -> 23.2% (cumulative step ~19,740) -> 32.8% (cumulative step ~30,740, latest). LIBERO-Spatial/Long stable at 94-96% throughout, comfortably above the 90% floor. See PROGRESS_0017 for full detail; this line supersedes the exp0011/exp0012 baseline-establishment status below (kept for history).`

## Parent checkpoint (current, since exp0011)

- checkpoint: **official FastWAM LIBERO release checkpoint**, `libero_uncond_2cam224.pt`
- source: `https://huggingface.co/yuanty/fastwam`
- local path: `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (symlinked to overlay-disk cache; downloaded, complete, 12,041,735,140 bytes)
- paired stats: `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- zero-init expanded (K=21/22) copy: `/home/claudeuser/local_cache/fastwam_release_expanded_zeroinit/step_000000.pt`
- code lineage: none — public release artifact, not sourced from `autoresearch/libero90-v1`
- research branch: `autoresearch/robotwin-multiembodiment-v1` (unchanged, active)

### Why the parent changed (full detail in `PROGRESS_0011`)

The project originally inherited `exp0019` (a heavily/narrowly fine-tuned research checkpoint) per the original `CLAUDE.md`. Investigating a user-reported step-0 anomaly (raw expanded checkpoint scoring 23.33% at zero training) found and fixed three real bugs: a dataset-stats mismatch, a self-introduced `embodiment_description` conditioning mechanism, and a checkpoint-expansion weight-initialization bug (`expand_checkpoint_for_multiembodiment.py` now zero-inits new `action_encoder`/`proprio_encoder` input columns, fixed in commit `631df95`). After all three fixes, `exp0019`-expanded still showed a large, reproducible LIBERO-Spatial-specific degradation (73-75% vs. ~97-98% native) that the release checkpoint did not share (~96-97% either way). A full-model output-equivalence test proved the expansion computation is mathematically exact (0.0 diff) for both checkpoints. The actual cause: `exp0019` has a narrow, **seed-sensitive** success margin specifically on Spatial (the suite its own final, narrowly-oversampled fine-tuning stage targeted) — re-running with `seed=100` instead of the default `seed=42` recovered ~98%, while the release checkpoint stayed ~97% regardless of seed. Given this intrinsic fragility, the user decided to switch parents to the release checkpoint, which is robust and reproducible.

## Standing goal (current, since exp0011 — see `CLAUDE.md`/`research/GOAL.md`)

- RoboTwin: **>=90% average success on the full 50-task Aloha-AgileX benchmark**, Clean and Randomized splits evaluated separately, both must clear 90%.
- LIBERO retention: **LIBERO-Spatial, Object, Goal, Long/LIBERO-10 all >=90%.**
- **LIBERO-90 is explicitly out of scope** — do not optimize for it or spend budget evaluating it unless directly relevant to a specific decision.

## Release-checkpoint LIBERO evidence — baseline established (exp0012)

| Suite | Expanded (K=21/22, zero-init), n=10 |
|---|---:|
| LIBERO-Spatial | 98.00% (also 96.00%/97.00%/97.00% across three earlier independent runs, two seeds — highly consistent) |
| LIBERO-Object | 99.00% |
| LIBERO-Goal | 98.00% |
| LIBERO-Long | 95.00% |
| **Overall** | **97.50%** |

Full combined run: `checkpoints/exp0012_release_expanded_4suites_n10.log`, `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260818_211301/`. Every suite comfortably clears the 90% floor with wide margin — this is the trustworthy pre-training baseline for the multi-embodiment work. Native (K=7, unexpanded) baseline was not completed (superseded by prioritizing the expanded-checkpoint numbers, which are what actually matters going forward); not blocking further work.

RoboTwin baseline check (`click_alarmclock`, small panel, n=3) launched immediately after (`exp0012_robotwin_baseline_click_alarmclock.log`) — first real policy-in-the-loop RoboTwin evaluation of this project's multi-embodiment checkpoint, expected near-0% since the release checkpoint has never seen RoboTwin data (pipeline sanity check + true pre-training reference point).

## RoboTwin evidence

Canonical protocol confirmed during original setup (`research/RUNBOOK.md`): 50-task Aloha-AgileX benchmark, Clean/Randomized tracked separately. No full canonical (50-task, both phases) evaluation has been run yet for any candidate. Cheap-panel progress-check evidence exists from `exp0013` (see below) — real RoboTwin capability emerged quickly then declined with further training; see `research/progress/PROGRESS_0013_release_parent_disjoint_offset_baseline.md` Section 12 for full analysis.

## Current accepted multi-embodiment checkpoint

**`exp0014`, cumulative training step 2000 from the release-checkpoint parent** (exp0013's step 1000 + exp0014's own 1000 frozen-backbone steps; `research/exp0014_release_parent_frozen_backbone/step_001000_cumulative.pt` on `cheikh025/ASR`) — current best validated checkpoint, superseding exp0013's step-1000 pick. Diagnostic candidate that froze the entire shared MoT backbone (`trainable_modules: expanded_projections_only`, only the newly-expanded action_encoder/head trainable) and resumed from exp0013's step-1000 checkpoint, to test whether backbone freezing stops the RoboTwin decline observed in exp0013 (see below). Cheap-panel evidence at this checkpoint:

| Benchmark | Result (cheap panel) |
|---|---:|
| LIBERO-Spatial (n=3) | 96.67% (29/30) |
| RoboTwin `click_alarmclock` Clean (n=5) | 80.0% |
| RoboTwin `turn_switch` Clean (n=5) | 80.0% |
| RoboTwin `press_stapler` Clean (n=5) | 100.0% |
| RoboTwin `open_laptop` Clean (n=5) | 40.0% |
| RoboTwin mean (4 tasks) | **75.0%** |

**Decisive result: freezing the backbone reversed exp0013's decline entirely** — RoboTwin 4-task mean rose from 30.0% (exp0013's declined step-1800 state on the same 4 tasks) to 75.0% by cumulative step 2000, and held stable at 75.0% through cumulative step 2600 (a genuine plateau, not a transient peak). Strongly supports shared-backbone LIBERO/RoboTwin gradient interference as the primary driver of exp0013's decline (see `research/progress/PROGRESS_0014_frozen_backbone_diagnostic.md`).

**Full 4-suite LIBERO confirmation at cumulative step 2600 (n=10/task) — real promotion blocker found**:

| Suite | Success |
|---|---:|
| LIBERO-Spatial | 99.0% |
| LIBERO-Object | 99.0% |
| LIBERO-Goal | 97.0% |
| **LIBERO-Long** | **87.0%** (below the 90% floor) |

Spatial/Object/Goal all comfortably clear 90%, but **LIBERO-Long is at 87.0%**, driven by two specific complex dual-object mug-placement tasks (`libero_10_4` 50%, `libero_10_6` 60%; every other Long task scored 80-100%). Note: promotion is a tradeoff judgment, not an automatic veto the moment one suite dips under 90% (user correction) -- but this specific case is a genuine regression either way, since Long was 95.0% at exp0012's pre-training baseline (not just below the 90% floor, but below the actual inherited reference). The real question for a promotion decision is whether the RoboTwin capability gained is worth this specific Long regression, once the full 50-task RoboTwin picture is in.

**Full 50-task RoboTwin Clean-only scans, two points 2000 training steps apart -- decisive finding**:

| Cumulative training step | Full-50-task Clean mean | n |
|---|---:|---:|
| 2600 | 12.6% | 10 |
| 4600 | 11.6% | 5 |

**Essentially no movement across 2000 additional frozen-backbone training steps, despite the curated-panel tasks (click_alarmclock, turn_switch) continuing to improve steadily over the same interval (reaching 80-100%).** Nearly the identical ~10-11 of 50 tasks show any nonzero success at both points. ~~This resolves the earlier "training-scale vs coverage" question decisively in favor of data coverage~~ **CORRECTED 2026-08-22: this "data coverage" attribution was wrong and has been independently disproven** -- see `research/NOTES.md` "CORRECTED (2026-08-22)". Re-verification against RoboTwin's own official per-task instruction templates found all 50 canonical tasks have real, roughly uniform training data (~550 episodes each, no zero-data tasks), and episode length is statistically identical between the successful and failing task sets (223.8 vs 220.3 frames mean). The ~12% ceiling is real, but its cause is still open -- most likely a representational-capacity/task-difficulty limitation under the frozen backbone, not missing/thin data. Full breakdown: `research/progress/PROGRESS_0014_frozen_backbone_diagnostic.md` Sections 10 and 14.

**Decision**: `STOP_TRAINING` this trajectory (exp0014). A follow-up attempt to precisely diagnose per-task RoboTwin data density via keyword matching against `meta/episodes.jsonl` proved unreliable at fine granularity and was retracted (see `research/NOTES.md`) -- but the coarse "4 tasks at zero" conclusion drawn from that same flawed method was *also* wrong, not just the fine-grained table, per the 2026-08-22 correction above.

**exp0015 (RoboTwin-heavy 1:3 sampling ratio, same frozen backbone) -- `REJECT`, second consecutive negative result**: full-50-task rescan (n=5) on the final checkpoint gave **11.6% mean, essentially identical to exp0014's own 11.6%** despite 3x RoboTwin gradient share per step. Two new tasks gained marginal (20%) success (`place_container_plate`, `place_object_basket`) while others lost it (`move_playingcard_away`, `put_bottles_dustbin`, `shake_bottle` all dropped to 0%) -- a wash, not real progress. LIBERO-Spatial stayed 100% throughout.

**Two consecutive negative results now rule out "more RoboTwin exposure" (steps or ratio) as the lever that moves the ~12% ceiling.** The likely bottleneck is structural: `expanded_projections_only`'s limited trainable capacity (only action_encoder/head) to represent 50 distinct bimanual behaviors from a frozen, LIBERO-tuned backbone. **Recommended next candidate**: try `trainable_modules: dit_with_backbone_low_lr` (two-parameter-group optimizer, full LR on projections + much lower LR on the shared backbone) -- untested under the current release-checkpoint lineage, trades a small controlled amount of LIBERO drift risk for more representational capacity.

Current best checkpoint: exp0014 cumulative step 4600 (`cheikh025/ASR:research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt`) -- strongest LIBERO-Spatial (100%) and curated-panel RoboTwin numbers, but the full-50-task mean (11.6%) is the honest metric against the actual project goal, still far below 90%.

### exp0017 (full backbone, LIBERO-Long-protected mixing, resumed from exp0014 cumulative-4600) -- CURRENT BEST CANDIDATE, actively training

Unfroze the full shared backbone (unlike exp0014/15/16's frozen/low-LR approaches, all of which plateaued 9.6-12.6% and are now superseded by this line) and added 3x oversampling of LIBERO-Long specifically, to test whether real backbone capacity can build genuine bimanual/handover coordination -- something a frozen backbone structurally cannot do -- without repeating exp0013's unprotected-mix collapse. Full detail and complete progress-check trail: `research/progress/PROGRESS_0017_backbone_low_lr_robotwin_heavy_longrun.md`.

**Full-50-task RoboTwin Clean mean, all measurements taken so far**:

| Cumulative step | Clean mean | Nonzero tasks | Bimanual tasks nonzero (of 10) |
|---:|---:|---:|---:|
| 19,740 | 23.2% | 23/50 | 3/10 |
| 30,740 | 32.8% | 36/50 | 8/10 |
| 40,740 | 42.4% | 41/50 | 9/10 |
| 50,740 (latest) | **42.8%** | 39/50 | 9/10 |

**First non-accelerating reading** at the latest check (+0.4 vs +9.6 twice in a row before it) -- not yet confirmed as a real plateau (substantial per-task churn underneath the flat aggregate, consistent with n=5 sampling noise; one flat check does not override the project's patience policy on its own). Milestone: `place_dual_shoes` finally broke through (0%->20%) at this check -- the single bimanual task that had never shown any success anywhere in this project's history. All 10 bimanual tasks have now shown nonzero success at some point, though not simultaneously. **Watch the next check closely** -- a second flat/negative reading would be real evidence of a plateau worth investigating rather than continuing to extend blindly.

**LIBERO retention, all 4 canonical suites**: 40,740 -- Spatial 96.0%, Object 100.0%, Goal 96.0%, Long 98.0% (overall 97.5%); 50,740 -- Spatial 96.0%, Object 98.0%, Goal 96.0%, Long 96.0% (overall 96.5%). Both essentially identical, comfortably clearing the 90% floor on every suite. LIBERO retention has never been at risk anywhere in this candidate's run.

**Operational note**: running the full-50-task RoboTwin scan and full 4-suite LIBERO check concurrently is **not reliably safe** -- worked cleanly at cumulative step 40,740 but crashed with `CUBLAS_STATUS_ALLOC_FAILED` at 50,740 under the identical pattern. Run them sequentially going forward.

**Decision at every check so far**: `CONTINUE_TRAINING`. Current best checkpoint: `runs/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont6_3e-5/2026-08-29_07-20-54/checkpoints/weights/step_040000.pt` (cumulative step 50,740; not yet durably backed up to `cheikh025/ASR` -- `HF_TOKEN` is currently unset in this environment, blocking upload; flagged to the user 2026-08-28, not yet resolved).

### Superseded: exp0016 (partial backbone plasticity, `dit_with_backbone_low_lr`)

Tests the recommended next lever above: shared backbone stays trainable but at `backbone_lr=3e-6` (10x below the `3e-5` projection LR), resumed from exp0014's cumulative-4600 checkpoint, standard 1:1 LIBERO:RoboTwin ratio (exp0015 already ruled out ratio as the lever).

**Phase 1 (1000 steps) result**: full-50-task RoboTwin Clean mean **9.6%** (n=5) -- flat/no improvement vs. exp0014/exp0015's 11.6%. LIBERO-Spatial held at 96.67%, no retention cost. Checkpoint backed up: `cheikh025/ASR:research/exp0016_backbone_low_lr/step_001000_cumulative.pt`.

**Reframe (key insight, changes how to read exp0014/15/16 phase-1 all together)**: `num_epochs=1` never actually binds -- `max_steps` does, and at this project's measured throughput (~0.0215 steps/s, `batch_size=2 x world_size=4 x grad_accum=4` = 32 samples/optimizer-step), RoboTwin's training pool alone is ~6.01M samples. exp0014 (cumulative 4600), exp0015 (~1600), and exp0016 phase 1 (1000) all covered well under 1% of even one RoboTwin epoch each. A flat full-50-task result at that scale is not strong evidence against a lever -- it's more likely nobody has tested any lever at a scale where a data-coverage effect could show up at all.

**Current action (per explicit user decision)**: running a much larger extension from the phase-1 checkpoint -- intermediate budget of **5000 steps** (~2.5 days wall-clock at current throughput) chosen as a step between the too-small 1-2k range and a full 20k+ commitment (~10 days), pending whether 5000 shows any RoboTwin movement. Mid-extension check at cumulative ~1500 (cont4, local step 500): **full 4-suite LIBERO = 98.33% overall (Spatial 96.67%, Object 100%, Goal 100%, Long 96.67%)** -- no retention damage from continued backbone plasticity. RoboTwin not yet re-checked at this later step; the full-50-task decisive rescan is planned for the end of the 5000-step run (cumulative step 6000). As of this note, training is in its `_cont5` phase, actively running.

**Also flagged, not yet root-caused**: GPU utilization during this multi-embodiment training rotates across the 4 GPUs (only 1-2 near-saturated at any instant) rather than staying synchronized near 100% -- likely a straggler effect from the `InterleavedEmbodimentSampler` giving different ranks LIBERO vs. RoboTwin batches of very different compute cost within the same synchronized DeepSpeed step (RoboTwin's 3cam/384px vs. LIBERO's 2cam/224px, plus `mot_checkpoint_mixed_attn: true` for this config vs. `false` for LIBERO-only). GPU memory is only ~65-67% utilized (not memory-bound), so this is a real lead for a future throughput investigation before committing to a 20k+ step run. Full detail: `research/NOTES.md`, `research/progress/PROGRESS_0016_backbone_low_lr.md`.

**Candidate idea recorded for exp0017 (not yet implemented)**: a RoboTwin-first curriculum -- front-load RoboTwin sampling ratio early in training (most backbone plasticity, freshest LIBERO retention margin), then rebalance toward LIBERO later for consolidation -- distinct from exp0015's flat-ratio test, which was itself undertrained. See `research/NOTES.md`.

No candidate has undergone the full canonical protocol (50 tasks x both Clean+Randomized phases, n=100/phase) yet -- current evidence remains screening-level.

### Superseded: exp0013 (full-backbone training on the release-checkpoint parent)

Cumulative step 1000 was the best point (LIBERO-Spatial 100%, RoboTwin click_alarmclock 80%/turn_switch 60%) before a real, multi-task-confirmed decline set in with further full-backbone training (2 of 4 tasks collapsed to 0% by cumulative step 1800). See `research/progress/PROGRESS_0013_release_parent_disjoint_offset_baseline.md` Section 12 for full evidence. Its step-1000 checkpoint remains durably archived (`research/exp0013_release_parent_disjoint_offset/step_001000.pt` on `cheikh025/ASR`) but is no longer the project's current-best pick.

## Superseded: exp0019-lineage history (parent abandoned in exp0011 — kept for record, not to be reused as evidence for the release-checkpoint parent)

All of the following used `exp0019` (now-abandoned parent) as the base checkpoint. None were promoted. The multi-embodiment implementation lessons (disjoint-offset padding, masked action loss, checkpoint expansion mechanics) remain valid and carry forward; the specific LIBERO-Spatial percentages below do **not** transfer to the release-checkpoint parent and should not be used as reference numbers going forward.

- `0000_parent_baseline` — `PROMOTE` (setup baseline for the now-abandoned exp0019 parent, not a research candidate).
- `0001_padded_multiembodiment_baseline` — `REJECT`. Full fine-tune, K=14 padded interface (overlapping columns), ~1:1 LIBERO:RoboTwin. LIBERO-Spatial 96.67%->73.33%, RoboTwin 0.0%.
- `0002_frozen_backbone_warmup` — `REJECT`. Backbone frozen, K=14 overlapping projections only trainable. LIBERO-Spatial 16.67% — refuted the backbone-interference-alone hypothesis (overlap in the *projection* columns dominates when nothing else can absorb it).
- `0003_disjoint_action_offset` — `REJECT`. Disjoint K=21/22 offset columns (LIBERO 0, RoboTwin 7/8), full trainable backbone. LIBERO-Spatial 50.00% — disjoint offsets alone insufficient.
- `0004_disjoint_offset_frozen_backbone` — `REJECT`. Disjoint offsets + frozen backbone. LIBERO-Spatial 63.33%; first non-zero RoboTwin result anywhere (`click_alarmclock` 33.3%/33.3%).
- `0005_disjoint_offset_backbone_low_lr` — `REJECT`. Disjoint offsets + backbone trainable at 10x-lower LR. LIBERO-Spatial 63.33% (unchanged); RoboTwin `click_alarmclock` randomized-phase capability lost entirely.
- `0006_disjoint_offset_frozen_backbone_4k` — `REJECT`. exp0004 recipe at 4000 steps. LIBERO-Spatial 63.33% again; RoboTwin `click_alarmclock` capability lost entirely at longer budget.
- `0007_exp0004_replication` — `REJECT`. Fresh exp0004 re-run, `click_alarmclock` at n=10. LIBERO-Spatial 56.67%; RoboTwin `click_alarmclock` 20.0%/10.0% (confirmed real, not n=3 luck).
- `0008_disjoint_offset_3to1_ratio` — `REJECT`. 3:1 LIBERO:RoboTwin ratio. LIBERO-Spatial 36.67% (worse than 1:1) — refuted more-rehearsal-helps. RoboTwin 0.0% on all measurements.
- `0009_libero_only_control` — `REJECT` (diagnostic). LIBERO-only training, zero RoboTwin exposure. LIBERO-Spatial 23.33% — **originally read as falsifying multi-embodiment interference as the primary retention mechanism; exp0011 later found the true cause was unrelated (checkpoint-expansion bug + exp0019-specific seed sensitivity), not an optimizer/LR effect as hypothesized at the time.**
- `0010_libero_only_low_lr` — planned, **never launched**, superseded by exp0011's findings before it ran.
- `0011_expansion_zeroinit_fix_and_parent_switch` — `DIAGNOSE`. Full investigation described above. Result: parent checkpoint switched from `exp0019` to the official release checkpoint; `CLAUDE.md`/`research/GOAL.md` updated. Full report: `research/progress/PROGRESS_0011_expansion_zeroinit_fix_and_parent_switch.md`.

## Current research notes (still valid, carries forward to the release-checkpoint parent)

- `ConcatLeftAlign` padding with `action_offset`/`state_offset` (disjoint columns, LIBERO at offset 0, RoboTwin at offset 7/8 within shared K=21/22) is the active multi-embodiment action/proprio interface — architecture-level design, independent of which checkpoint is expanded with it.
- `fastwam.utils.losses.masked_action_loss` implements the two-level per-channel masked loss — still correct and in use.
- `research/tools/expand_checkpoint_for_multiembodiment.py` **now defaults to zero-initializing new `action_encoder`/`proprio_encoder` input columns** (fixed in exp0011, commit `631df95`) — always re-expand from this fixed version; do not reuse any expanded checkpoint produced before this fix.
- **Always pass the checkpoint's own paired `dataset_stats.json`** (`pretrained_norm_stats` for training / `EVALUATION.dataset_stats_path` for eval) rather than letting the multi-embodiment LIBERO data config auto-compute stats from its own (differently-composed) subset.
- **Do not add `embodiment_description` or any other conditioning mechanism as a silent default** — if tested again, it must be an explicit, isolated, clearly-labeled candidate compared against a clean baseline, per `CLAUDE.md`'s original initial-baseline scope (padded action tensor only).
- **Prefer the existing eval pipeline (`run_libero_manager.py`/`eval_libero_single.py`) over hand-rolled comparison scripts** for any diagnostic check — see the exp0011 investigation and the standing feedback memory note for why.
- `Trainer.trainable_modules` config option (`"dit"` [default, full backbone trainable] | `"expanded_projections_only"` | `"dit_with_backbone_low_lr"`) remains available infrastructure.
- **Known open gap**: no downloadable LIBERO-90 lerobot-format training data exists in the public FastWAM dataset — moot for the current goal since LIBERO-90 is out of scope.
- **Known open gap**: RoboTwin's exact per-arm action-channel semantics (delta vs. absolute; per-arm layout) still not confirmed against RoboTwin's own env code — confirm before the first real RoboTwin training candidate.
- **Disk**: `/workspace` (persistent volume) and the container overlay disk (`/`, ephemeral but roomy) are both in active use — reproducible/re-downloadable caches (base Wan2.2 components, RoboTwin simulator/assets) live on the overlay disk via symlink from `checkpoints/`/`third_party/RoboTwin`; training data (`data/`, ~987GB) and this project's own checkpoints/research artifacts stay on the persistent volume. **`third_party/RoboTwin` is git-tracked — never replace it with a symlink to elsewhere; check `git status` before moving any tracked directory.** RoboTwin's own working-directory-relative checkpoint cache (`third_party/RoboTwin/checkpoints/Wan-AI/`, ~14GB) is load-bearing for RoboTwin evaluation (its policy subprocess runs with `cwd=third_party/RoboTwin`) but irrelevant to training (which never touches that directory) — do not delete it.
- Do not inspect/import `autoresearch/research-docs` as research history for experiment selection.
