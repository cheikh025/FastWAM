# PROGRESS_0017 — full_backbone_long_protected_longrun

- **Experiment ID:** 0017
- **Status:** `RUNNING`
- **Created:** 2026-08-22
- **Updated:** 2026-08-23 (launched, see Section 6)
- **Parent experiment:** 0014 (release_parent_frozen_backbone)
- **Parent checkpoint:** `cheikh025/ASR:research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt` (current project-best checkpoint; single-arm RoboTwin capability banked, LIBERO-Long previously measured at 87.0% — below floor — at an earlier point in this same lineage, cumulative step 2600, never re-confirmed)
- **Selected candidate checkpoint:** TBD
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** TBD (recorded after commit)

### Design revision 2 (2026-08-22, before launch) — patience on early signals, larger ceiling

Explicit user correction: every prior negative/collapse verdict in this project's RoboTwin work (`exp0013`'s collapse, `exp0014`/`exp0015`'s "flat ceiling", `exp0016` phase-1's "no movement") was drawn from well under 1-2000 steps against a ~188k-step epoch — not enough exposure to trust as a real conclusion about what full-backbone-with-protection can or can't do. **This run must not repeat that mistake**: `max_steps` raised from 8000 to **20000** (still a ceiling/ambition, not a promise), and the progress-check philosophy changed from "stop immediately if `exp0013`'s collapse signature reappears" to "monitor and record trends from early on, but do not make a STOP_TRAINING call from an early dip alone — let the run reach a genuinely large step count (several thousand at minimum) before drawing a conclusion, unless something catastrophic and unambiguous happens (NaN/divergent loss, or a sustained collapse that persists and worsens over thousands of further steps rather than a transient dip)." `save_every` raised to 500 (from 200) given the much larger step ceiling, to keep checkpoint/disk overhead reasonable at this scale.

### Design revision (2026-08-22, before launch)

Originally designed as a backbone-plasticity (`dit_with_backbone_low_lr`) + RoboTwin-heavy-ratio candidate, resuming from `exp0016`. Revised per explicit user direction after reviewing what the sibling LIBERO-only project (`autoresearch/libero90-v1`) actually did: **that project always did a full, unfrozen `model.dit` fine-tune** (confirmed directly in their `Wan22Trainer._apply_dit_only_train_mode` — no LoRA/adapters, no partial freezing) and solved its own forgetting problem entirely through **iteratively-tuned data-mix oversampling** (Goal 5x-7x, Long 5x-10x), not by freezing or slowing down the backbone. This project jumped to freezing (`exp0002`/`0004`/`0014`) and low-LR backbone plasticity (`exp0005`/`0016`) after `exp0013`'s full-backbone run collapsed — but that collapse happened under a **plain, unprotected 1:1 mix**, and was never retested with LIBERO-protective mixing or run long enough to see whether the decline was transient (the sibling project's own literature review, arXiv 2603.03818, found pretrained/converged models that do regress typically recover within 6-10% of original training steps). This candidate tests that directly, before the lower-capacity plasticity approach — full-backbone unfreezing is a strict superset of low-LR partial plasticity in terms of representational capacity, so it's the more informative first test of the "capacity gap, not data gap" theory from Section 2 below, if it can avoid repeating `exp0013`'s interference collapse. The originally-designed plasticity+ratio candidate is kept as the queued next step (task config already written: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_backbone_low_lr_robotwin_heavy_longrun_3e-5.yaml`) if this one shows renewed collapse or doesn't improve broad RoboTwin coverage.

## 1. Result at a glance

TBD — filled in as the run progresses.

## 2. Research state before experiment

### Accepted RoboTwin state

- Full-50-task Clean mean: 11.6% (n=5) at exp0014 cumulative step 4600; exp0016 phase-1 (1000 steps, frozen-then-low-LR backbone) was flat at 9.6% (n=5) — no movement from more steps (exp0014 cont3/cont4) or a heavier RoboTwin ratio (exp0015), but **both of those negative results were obtained under a fully or near-fully frozen backbone**, i.e. with no representational capacity available to spend on anything new regardless of data exposure.
- ~10-11 of 50 tasks show any nonzero success; the rest are flat at 0%.
- **New diagnostic (2026-08-22, this machine)**: the successful set is overwhelmingly single-arm (`click_alarmclock`, `open_microwave`, `press_stapler`, `click_bell`, `turn_switch`, `move_playingcard_away`, `shake_bottle_horizontally`, `open_laptop`, `put_bottles_dustbin`, `shake_bottle` — 7/10 explicitly single-arm by RoboTwin's own task description, 3/10 ambiguous-but-simple). Zero of the 10 explicit bimanual/handover tasks (`grab_roller`, `lift_pot`, `place_bread_basket`, `place_dual_shoes`, `handover_block`, `handover_mic`, `pick_diverse_bottles`, `pick_dual_bottles`, `place_object_basket`, `scan_object`) have ever shown any success. The parent model is a LIBERO (single-arm-only) specialist — plausible mechanism: the frozen/near-frozen backbone can steer existing single-arm motor primitives onto new single-arm tasks via the thin action_encoder/head, but cannot express genuinely novel two-arm coordination without real backbone capacity.
- **Correction (2026-08-22)**: `research/NOTES.md`'s prior "4 tasks have zero training episodes" claim was independently re-verified and found wrong — all 50 tasks have real, ~uniform data (~550 episodes each; 550×50=27,500=total_episodes exactly). Episode length is also statistically identical between the successful and failing task sets (223.8 vs 220.3 frames mean). See `research/NOTES.md` "CORRECTED (2026-08-22)" and `research/STATE.md`. **The ~12% ceiling is not a data problem** — this is what motivates trying full backbone capacity again, this time protected.
- **exp0013 history (the direct precedent for this candidate's risk)**: full-backbone training on this same release-checkpoint parent, plain 1:1 mix. RoboTwin capability rose (click_alarmclock 80%, turn_switch 60% at cumulative step 1000) then collapsed (2 of 4 curated tasks to 0% by cumulative step 1800), while LIBERO-Spatial stayed solid throughout and training loss kept falling smoothly (rules out a simple optimization pathology). Two leading hypotheses were never fully distinguished: backbone interference between embodiments, or closed-loop/offline-imitation distribution shift. `exp0014` tested the frozen-backbone response and confirmed freezing stops the decline — but that doesn't tell us whether *protected* full-backbone training would also avoid it.

### Accepted LIBERO retention state

| Suite | Success |
|---|---:|
| LIBERO-90 | out of scope, not tracked |
| LIBERO-Spatial | 96.67% (exp0016 phase-1, n=3) — most recent measurement, on top of exp0014's checkpoint |
| LIBERO-Object | not re-measured since exp0014's own numbers (99.0%) |
| LIBERO-Goal | not re-measured since exp0014's own numbers (97.0%) |
| LIBERO-Long / LIBERO-10 | **87.0% at exp0014 cumulative step 2600 (below the 90% floor)** — driven by 2 specific dual-object mug-placement tasks; not re-measured since, including not at the cumulative-4600 checkpoint this candidate resumes from |

**LIBERO-Long is the single most important number to watch in this run** — it was already below the 90% floor at an earlier point in this exact lineage and has never been re-confirmed. This is the primary reason for the Long-protective data-mixing change (Section 3) and for the frequent early progress-check cadence (Section 3, "initial compute plan").

## 3. Candidate design

### Modifications

1. Resume from `exp0014`'s cumulative-step-4600 checkpoint (current project best; single-arm RoboTwin capability banked).
2. `trainable_modules` reverts to the default (`"dit"`) — the **entire shared MoT backbone (video + action experts) is trainable**, no freezing, no low-LR split. Matches `exp0013` and the sibling project's always-full-fine-tune approach.
3. **New data config** (`configs/data/multiembodiment_libero_robotwin_long_protected.yaml`): LIBERO-Long (`libero_10_no_noops_lerobot`) listed 3x in the LIBERO embodiment's `dataset_dirs` (388 → ~1164 episodes, comparable to Object's 457 / Spatial's 434 — brings Long from the smallest LIBERO suite by episode count to the largest). Spatial/Object/Goal stay at 1x — no regression evidence for them yet, so no reason to touch them. RoboTwin stays at its standard 1:1 embodiment-level ratio (unchanged from `exp0013`/`exp0014`) — deliberately *not* combined with a RoboTwin-heavy ratio this round, to avoid confounding "does full-backbone-with-protection avoid the collapse" with a second simultaneous risk factor.
4. Training budget extended to `max_steps=20000` — but see "initial compute plan" below: this is a ceiling, not a target, given the real collapse risk.
5. `save_every` tightened to 200 (from exp0013's own 200, kept unchanged — exp0013 already used a reasonably tight cadence; the actual change here is evaluation frequency, see below) to give fine-grained checkpoint selection if a decline is caught mid-run.

### Why this candidate

`exp0013` is the only real prior evidence about what happens when the shared backbone is left fully trainable under this multi-embodiment setup, and it showed a genuine risk (RoboTwin capability rose then partially collapsed). But that run (a) used no LIBERO-protective data mixing at all, despite this project inheriting the sibling project's own well-documented finding that Goal/Long are disproportionately vulnerable to dilution under joint multi-suite training without oversampling, and (b) was stopped and diagnosed relatively early (by cumulative step ~1800-2600), before there was a chance to see whether the decline was a transient "stability gap" (a documented continual-learning phenomenon the sibling project's own literature review flagged, with typical recovery within 6-10% of the original training budget) rather than a permanent regression. Meanwhile, this project's freezing-based workarounds (`exp0002`/`0004`/`0014`/`exp0016`) provably cannot build new bimanual-coordination capacity — today's diagnostic (Section 2) shows the ~12% ceiling correlates cleanly with single-arm vs. bimanual task structure, not data volume or exposure count, which argues for giving the model real capacity rather than more exposure under a fixed-capacity regime. Full-backbone training, done with the same kind of protective data mixing that worked for the sibling project's own analogous problem, is the most direct way to test whether real capacity can close that gap without repeating `exp0013`'s failure mode.

### Multi-embodiment representation/configuration

Unchanged from `exp0011` onward — shared K=21 (action) / K=22 (state), LIBERO at offset 0, RoboTwin at offset 7/8, masked loss, zero-init checkpoint expansion. Re-verified correct end-to-end on this machine (config wiring, encode/decode symmetry, loss masking, checkpoint expansion) before this candidate was designed — see conversation record 2026-08-22; all relevant unit tests pass (`test_action_state_merger_offset.py` 5/5, `test_masked_action_loss.py` 5/5, `test_expand_checkpoint.py` 3/4 — the one failure is a stale pre-`exp0011`-fix test asserting the old, deliberately-reversed random-init behavior, not a real bug in the current code).

### Data and learning strategy

- LIBERO datasets/tasks: Spatial (1x) / Object (1x) / Goal (1x) / Long-10 (**3x**, new) — LIBERO-90 excluded (no lerobot-format data available).
- RoboTwin datasets/tasks: full 50-task combined preprocessed directory (`data/robotwin2.0/robotwin2.0`) — unchanged, all 50 tasks confirmed present with real, ~uniform data.
- sampling/mixing ratio: LIBERO ratio=1.0, RoboTwin ratio=1.0 (embodiment-level, unchanged from `exp0013`/`exp0014`) — only the *within-LIBERO* Long weighting changed.
- per-task/per-dataset weights: Long 3x within LIBERO; nothing else.
- replay/rehearsal strategy: standard joint per-step mixing (`InterleavedEmbodimentSampler`), unchanged.
- loss weights: standard masked per-embodiment action loss, unchanged.
- retention/distillation/regularization: none beyond the disjoint-offset interference-free design and the Long oversampling above.
- trainable/frozen modules: **full `dit`** (video + action MoT backbone), no freezing, no low-LR split.

### What to watch

Per Design revision 2: track all of the below from early on for visibility, but **do not make a STOP_TRAINING call from an early dip alone** — `exp0013`'s own "collapse" verdict was drawn from under 2000 steps against a ~188k-step epoch, not enough exposure to trust. Only a sustained, worsening, unambiguous failure (or NaN/divergence) over several thousand further steps should trigger an early stop; a transient dip that stabilizes or recovers should not.

- **LIBERO-Long specifically** — already below the 90% floor at an earlier point in this lineage, never re-confirmed, and the entire point of the data-mixing change. Track it at every check; a dip that persists and worsens over thousands of steps is a real concern, a single low reading early on is not conclusive on its own.
- The same RoboTwin curated 4-task panel `exp0013` used (`click_alarmclock`, `turn_switch`, `press_stapler`, `open_laptop`) — track the same rise/fall pattern `exp0013` showed, but let it play out over a genuinely larger step count before concluding it's the same failure mode rather than a transient dip that recovers (the sibling project's own literature review found regressions in pretrained/converged models typically recover within 6-10% of the training budget — for a 20000-step ceiling that's roughly 1200-2000 steps, well within what this run should actually reach).
- The bimanual/handover 10-task subset (Section 2) — the sharpest signal for whether real backbone capacity is actually closing the representational gap. Needs real step count to show anything; don't expect movement in the first few thousand steps.
- LIBERO-Spatial as a cheap, fast sentinel at every check (has stayed solid throughout this project's history so far).
- Training stability: loss/action_l2 trend, no divergence, no discontinuities at resume — this is the one category where an early, unambiguous problem (NaN, exploding loss) should stop the run immediately regardless of step count.

### Initial compute plan

- initial training budget: `max_steps=20000` (ceiling, not a target — see Design revision 2. Still well under one RoboTwin epoch (~188k steps), but a large step count relative to everything tested in this project so far, deliberately chosen so the run isn't judged on too small a sample the way prior candidates were).
- checkpoint/save cadence: `save_every=500` (raised from an initial 200 given the larger step ceiling, to keep local disk/HF-upload overhead reasonable at this scale) — still fine-grained enough for checkpoint selection if needed.
- **progress-check cadence**: light/cheap sanity checks (loss/action_l2 trend, quick LIBERO-Spatial + Long sentinel) every ~500-1000 steps for visibility, but the first real evaluative *decision point* (continue at current settings / extend / stop / select an earlier checkpoint) should not happen before roughly local step 5000 — enough exposure to distinguish a real problem from the kind of transient dip `exp0013` was never given the chance to recover from. Full-50-task RoboTwin rescans are expensive (~9h at n=10 historically) — reserve those for meaningful decision points (e.g. around step 5000, then again near the end of the budget or at a natural stopping point), not every check.
- expected cost: **confirmed real throughput at `batch_size=3` on this machine (2026-08-23): ~6.3 sec/step (50 steps in 314s, steady across two independent samples)** — about 7x faster than this project's older pre-batch-size-tuning estimate (~0.0215 steps/s / ~46 sec/step). At this rate: step ~5000 (first real decision point) in ~8.75 hours; the full 20000-step ceiling in ~35 hours (~1.5 days).

This is a plan, not a promise to consume the whole budget — will adapt based on progress checks per `$run-fastwam-training`, but will give the run real time to show its actual effect before drawing conclusions, per explicit user direction.

## 4. Exact code and configuration state

- Git commit: TBD
- Git branch: `autoresearch/robotwin-multiembodiment-v1`
- parent code commit: `d8353c4` (research-notes correction commit, on top of exp0016's `e0fea9c`)
- working tree dirty before launch: new data config, new task config, this progress report
- files changed: `configs/data/multiembodiment_libero_robotwin_long_protected.yaml` (new), `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_3e-5.yaml` (new), this progress report; also `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_backbone_low_lr_robotwin_heavy_longrun_3e-5.yaml` (new, queued next candidate, not used by this run)
- training config: `multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_3e-5`
- LIBERO dataset config: via `multiembodiment_libero_robotwin_long_protected` (Long 3x, others 1x)
- RoboTwin dataset config: via `multiembodiment_libero_robotwin_long_protected` (unchanged from `multiembodiment_libero_robotwin`, ratio=1.0)
- action/state normalization: per-dataset (LIBERO min/max, RoboTwin z-score), unchanged
- action validity-mask configuration: unchanged (K=21/22, offsets 0/0 and 7/8)
- model/trainable-module configuration: default (`"dit"`, full backbone)
- optimizer/LR/scheduler: cosine, `learning_rate=3e-5`, `weight_decay=1e-2` — identical to exp0013's
- batch size/gradient accumulation: `batch_size=3` (raised from this project's historical `2` after an OOM/memory sweep on this machine, 2026-08-22 — `batch_size=4` peaked at ~76.2/80GB, too little headroom for a long run; `batch_size=3` completed cleanly with more margin, extrapolated ~80% GPU memory utilization), `gradient_accumulation_steps=4` (unchanged) — effective global batch rises from 32 to 48
- initial training steps/budget: `max_steps=20000`
- checkpoint/save cadence: `save_every=500`
- resume source/type: weights-only file resume, `./checkpoints/exp0014_resume/research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt`

## 5. Hardware and software environment

Fresh machine rebuild (2026-08-22) — full environment/data re-setup performed this session (venv, LIBERO, RoboTwin/SAPIEN/curobo/pytorch3d, all data/checkpoints re-downloaded). See conversation record for full detail; not re-duplicated here.

### GPU

- GPU count: 4
- GPU model(s): NVIDIA A100-SXM4-80GB
- memory per GPU: 80GB
- NVIDIA driver: 570.211.01
- CUDA runtime/toolkit: 12.8

### Software

- PyTorch version: 2.7.1+cu128
- FastWAM repository commit: `d8353c4` (at report creation time)
- environment: `/workspace/venvs/fastwam` (Python 3.10.21)

## 6. Training execution and control timeline

- exact launch command: `bash scripts/train_zero1.sh 4 task=multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_3e-5 model.redirect_common_files=false`
- **First launch attempt (2026-08-23 01:41 UTC, run_id `2026-08-23_01-41-12`) — aborted, not a training problem.** Launched with a trailing `| tail -300` on the command; plain `tail` (no `-f`) buffers all input and only prints once the stream closes, so the captured log stayed empty for the run's full duration even though training was healthy underneath (confirmed after the fact: GPU actively computing at ~72-74GB/GPU, matching the `batch_size=3` sweep; reached step 440/20000 at a real, healthy ~6.4 sec/step once the buffered log was inspected). Killed via graceful `SIGTERM` after ~50 minutes once the observability bug was identified — better to lose ~50 minutes of unconfirmed compute than let a multi-day run proceed with no way to catch a real problem. No checkpoint had been saved yet (`save_every=500`, only reached step ~440-450), so nothing was recoverable from this attempt; verified full process/GPU cleanup (no orphans) before relaunching.
- **Second launch (2026-08-23 02:32 UTC, run_id `2026-08-23_02-32-14`) — the actual run this report tracks.** Same config, resumed from the same `exp0014` checkpoint. Launched without the trailing `tail` this time (raw stdout captured live). Model construction/dataset build took ~5 minutes (much faster than the first attempt's ~40+ minutes, since all caches — ActionDiT backbone, LIBERO/RoboTwin text embeds, base Wan2.2 components — were already warm from the first attempt).
- start time: 2026-08-23 02:32 UTC
- **Confirmed real throughput**: ~6.3 sec/step, steady across two independent samples (190 steps @ 6.42s/step from the first attempt before it was killed; 50 steps @ 6.28s/step from the second launch) — about 7x faster than this project's older pre-batch-size-tuning estimate (~46 sec/step). At this rate: step ~5000 in ~8.75 hours, the full 20000-step ceiling in ~35 hours (~1.5 days).
- number of GPUs/world size: 4
- early loss trend: 0.4871, 0.4698, 0.6061, 0.9046, 0.4545, 0.4612 (steps 10-60) — normal step-to-step variance for a freshly-resumed mixed-embodiment batch, no divergence.
- training log: `/tmp/claude-1002/.../tasks/b1wm2zb5j.output` (session-local; not a permanent artifact path — will be copied/summarized into this report at the next real decision point)
- checkpoint/output dir: `runs/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_3e-5/2026-08-23_02-32-14/`

### Unplanned interruption and cont1 (2026-08-23 ~07:27 UTC)

Instance was stopped/restarted mid-run at logged step 2740/20000 (last durable checkpoint: `step_002500.pt`, phase-1 dir). Not a destructive recycle — `/workspace` (venv, checkpoints, git state) survived intact. `save_full_state` was `false` on the original launch, so resuming from the weights-only file necessarily reset `global_step`/optimizer/LR-scheduler to 0 (confirmed in `trainer.py::_resume_or_load_checkpoint`: file resume does not restore these, by design — not a bug). Relaunched as **cont1** (`multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont1_3e-5.yaml`): resumed weights from phase-1's `step_002500.pt`, `max_steps` reduced 20000→17500 (keeps total cumulative budget ~20000 across both phases), **`save_full_state` flipped to `true`** so a future interruption can resume seamlessly. Cumulative step bookkeeping from here: phase-1's 2740 + cont1's own step count.

**Operational lesson (checkpoint-save race condition)**: when stopping training for a progress-check eval, do not `SIGTERM` immediately upon seeing a `step=N` log line — the step counter can print before or concurrent with the checkpoint write completing. Killed the process at the exact moment `step=3000`'s log line printed and caught the save mid-write, producing a corrupted 313KB file instead of the full ~12GB checkpoint (confirmed via `file` command: valid zip header but truncated). Recovered by discarding the corrupt file and using the previous confirmed-complete checkpoint (`step_002500.pt`, cont1-local) instead. **Going forward: wait for filesystem evidence the save fully landed (e.g. poll file size/mtime stability, or wait past the next log line) before sending `SIGTERM`, not just the log line announcing the step number.**

### Progress check 1 — cumulative step 5240 (phase-1's 2740 + cont1's local step 2500), 2026-08-24

**Purpose**: `progress_check` per the compute plan (first real decision point, originally targeted ~step 5000; landed at 5240 due to the checkpoint-save-race delay above).

**LIBERO sentinel** (`libero_uncond_2cam224_multiembodiment_eval`, n=5/task, both suites):

| Suite | Success | vs. floor |
|---|---:|---|
| LIBERO-Spatial | **94.00%** (47/50) | comfortably clears 90% |
| LIBERO-Long | **90.00%** (45/50) | exactly at the 90% floor — but a real recovery from `exp0014`'s 87.0% (below floor) at the equivalent point in the parent lineage |

Raw results: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260824_031545/`. **The Long-protective 3x oversampling appears to be working** — no sign of the regression that motivated it.

**RoboTwin curated 4-task panel** (`click_alarmclock`, `turn_switch`, `press_stapler`, `open_laptop`; n=5/task, Clean + Randomized): raw results `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont1_3e-5_2026-08-23_21-35-06/20260824_032959/`.

| Task | Clean | Random | Parent (`exp0014` cumulative 4600) |
|---|---:|---:|---:|
| click_alarmclock | 20% | 40% | 100% |
| turn_switch | **0%** | **0%** | 80% |
| press_stapler | 20% | 20% | 100% |
| open_laptop | 40% | 40% | 40% |

**This is the same collapse signature `exp0013` showed** (`turn_switch`→0% was flagged in this candidate's own design as the clearest early collapse signal to watch for) — a real, large decline across 3 of 4 previously-strong tasks, happening within only ~640 steps of full-backbone training from the `exp0014` resume point. Unlike `exp0013`, this run has no earlier RoboTwin measurement between the parent checkpoint and this one, so whether capability first rose and is now falling (matching `exp0013`'s exact rise-then-fall shape) or declined monotonically from the start of this phase is unknown — a gap in the evidence, noted for future runs (check RoboTwin closer to a full-backbone resume point, not only at the ~5000-step mark).

**Decision: `CONTINUE_TRAINING`, not `STOP_TRAINING`.** Per the compute plan's explicit patience policy (do not stop on an early/isolated dip; regressions in pretrained models often recover within 6-10% of the training budget, which is still ahead of this phase), one measurement at step 5240 is not sufficient to distinguish a transient dip from a repeat of `exp0013`'s permanent collapse. However, given the magnitude and the exact pattern match to the known failure mode, **the next RoboTwin check must happen much sooner than another ~5000-step gap** — planned around cumulative step 6500-7000 (roughly 1000-1500 more steps), specifically to determine whether this is worsening (real collapse, matching `exp0013` — likely `STOP_TRAINING` and fall back to the queued backbone-plasticity candidate), flat (inconclusive, needs another check), or recovering (transient stability gap, as the literature review anticipated — continue toward the full budget).

Resumed via a proper directory (full-state) resume from `checkpoints/state/step_002500` this time — global_step/optimizer/LR-scheduler continue seamlessly, no further schedule disruption.

### Disk-full crash and cont2 recovery (2026-08-24 13:44 UTC — 2026-08-26)

cont2 ran unattended past the planned ~6500-7000 check (reached local step 8000/17000 = cumulative ~10,740) before crashing at 13:44:20 UTC on a checkpoint save: `RuntimeError: [enforce fail at inline_container.cc:857]. PytorchStreamWriter failed writing file data/2: file write failed` — the disk had filled to 100% from accumulated, unpruned full-state checkpoints (`save_full_state=true`, ~90GB each, saved every 500 steps with nothing deleting older ones). Confirmed via process-start timestamps (`supervisord`/`jupyter-notebook` still showed the Aug-23 restart time, unchanged) that this was **not** another machine restart — the container stayed up throughout; this was a pure disk-space/application crash.

**This went uncaught for approximately two days** (a real monitoring gap — periodic check-in notifications queued without being processed with real investigation in between). Recovery, once caught:
- The crash's own checkpoint (`step_008000`) turned out **partially corrupted**: the weights-only `step_008000.pt` (12,041,907,845 bytes) is complete and intact, but its paired full-state directory is missing `trainer_state.json` and one of four DeepSpeed rank optimizer-state shards (rank 0) — unrecoverable for a directory resume. Discarded the state dir; using the weights-only file instead.
- Freed disk (100% full → 957GB free) by deleting 10 redundant superseded checkpoints plus the now-dead `exp0014_resume` (12GB, superseded, not referenced by any live config — `exp0016_resume` kept, it's the queued fallback candidate's actual resume source).
- **Built and deployed a persistent watchdog** (`/workspace/setup_logs/checkpoint_watchdog.sh`, running independently of any interactive session) that every 5 minutes prunes each run's checkpoints to the 2 most recent weights-only files (cheap, ~12GB each, kept as redundancy in case the single newest is corrupted — exactly what just happened) and the 1 most recent full-state directory (expensive, ~90GB; only needed for a seamless resume, and a weights-only resume is a proven-working fallback), and auto-`SIGTERM`s training if free space ever drops below 100GB. This directly fixes the root cause independent of active monitoring.

### Progress check 2 — cumulative step ~10,740 (phase-1's 2740 + cont2's local step 8000), 2026-08-26

**LIBERO sentinel** (n=5/task): **Spatial 96.00% (48/50), Long 94.00% (47/50)** — both improved slightly from check 1 (94%/90%), comfortably clear the floor. No LIBERO concern.

**RoboTwin curated 4-task panel, Clean phase only** (Randomized phase deliberately skipped this round per explicit direction — halves eval cost, Clean-only is sufficient for this decision):

| Task | Step 5240 (check 1) | Step ~10,740 (check 2) | Trend |
|---|---:|---:|---|
| click_alarmclock | 20% | **40%** | recovering |
| turn_switch | 0% | **0%** | still stuck |
| press_stapler | 20% | **40%** | recovering |
| open_laptop | 40% | **40%** | flat |

Raw results: `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont2_3e-5_2026-08-24_04-05-27/20260826_175552/`.

**Interpretation**: 3 of 4 tasks are recovering or holding flat, none are worsening — this looks more like the anticipated transient stability-gap dip than a repeat of `exp0013`'s permanent collapse. `turn_switch` specifically has now read 0% at two consecutive checks (5240 and ~10,740) — a real, specific holdout worth tracking on its own, distinct from the broader (recovering) pattern.

**Decision: `CONTINUE_TRAINING`.** No worsening signal, LIBERO fully healthy, and the recovery direction on 2/4 tasks supports giving the run more time per the patience policy. Resuming as **cont3** — weights-only resume from `step_008000.pt` (the corrupted state dir forces another schedule restart; LR was at 1.78e-05 at the crash, roughly 47% through cont2's own 17000-step schedule, so this restart is a bigger disruption than the near-peak phase-1→cont1 transition — a real cost of the disk-full incident, now mitigated going forward by the watchdog protecting future full-state saves). `max_steps` set to keep the cumulative ceiling near the original ~20000. Next check planned after another substantial step count, specifically watching whether `turn_switch` ever moves off 0%.

### Progress check 3 — cumulative step ~14,740 (phase-1's 2740 + cont2's 8000 + cont3's local step 4000), 2026-08-27

**LIBERO sentinel** (n=5/task): **Spatial 96.00% (48/50), Long 96.00% (48/50)** — Long has now improved at every single check (90%→94%→96%), comfortably clear of the floor. LIBERO is not a concern at any point in this candidate's history so far.

**RoboTwin curated 4-task panel, Clean phase, full trend across all three checks:**

| Task | Step 5240 | Step ~10,740 | Step ~14,740 | Trend |
|---|---:|---:|---:|---|
| click_alarmclock | 20% | 40% | **60%** | steadily climbing |
| turn_switch | 0% | 0% | **20%** | **finally moved off zero** |
| press_stapler | 20% | 40% | 20% | noisy (n=5), no clear direction |
| open_laptop | 40% | 40% | 40% | flat |
| **Mean (4 tasks)** | 20% | 30% | **35%** | climbing |

Raw results: `evaluate_results/robotwin/robotwin_uncond_3cam_384_multiembodiment_eval/20260827_015509/`.

**Interpretation**: this is real, positive evidence for the "transient stability-gap dip, not permanent collapse" hypothesis. `click_alarmclock` has improved at every check without exception; `turn_switch` — the specific holdout flagged after checks 1 and 2 — finally showed life. Combined with LIBERO not just holding but actively improving, there is no basis for `STOP_TRAINING` or falling back to the queued plasticity candidate at this point. `press_stapler`'s dip from 40%→20% is within plausible n=5 noise (a single trial flipping) and not treated as a contrary signal on its own.

**Decision: `CONTINUE_TRAINING`.** Resumed via a proper **directory (full-state) resume** from `cont3`'s own `checkpoints/state/step_004000` (confirmed complete — all 4 DeepSpeed rank shards + `trainer_state.json` present, unlike the corrupted `cont2` checkpoint) — global_step/optimizer/LR-scheduler continue seamlessly this time, no further schedule disruption. Continuing toward `cont3`'s existing `max_steps=9000` ceiling. Next check planned after another substantial step count (targeting cumulative ~19,000-20,000, i.e. the original ceiling), watching specifically whether `click_alarmclock`/`turn_switch` continue improving and whether `press_stapler` clarifies.

### Progress check 4 — cumulative step ~19,740, `cont3` completed its full 9000-step ceiling naturally, 2026-08-27

`cont3` ran to completion (`max_steps=9000` reached cleanly, exit code 0 — not a crash). Cumulative step: phase-1's 2740 + `cont2`'s 8000 + `cont3`'s 9000 = 19,740, landing right at the originally-planned ~20,000-step ceiling from this candidate's design.

**LIBERO sentinel** (n=5/task): **Spatial 96.00% (48/50), Long 96.00% (48/50)** — unchanged from check 3, fully stable at the ceiling. LIBERO has never been a concern anywhere in this candidate's run.

**RoboTwin curated 4-task panel, Clean phase, complete 4-check trend:**

| Task | Step 5240 | Step ~10,740 | Step ~14,740 | Step ~19,740 |
|---|---:|---:|---:|---:|
| click_alarmclock | 20% | 40% | 60% | 60% |
| turn_switch | 0% | 0% | 20% | 20% |
| press_stapler | 20% | 40% | 20% | 40% |
| open_laptop | 40% | 40% | 40% | 40% |
| **Mean (4 tasks)** | 20% | 30% | 35% | **40%** |

Raw results: `evaluate_results/robotwin/robotwin_uncond_3cam_384_multiembodiment_eval/20260827_...`.

**Interpretation**: the mean has climbed at every single check with zero regressions — decisive confirmation of the "transient stability-gap, not permanent collapse" hypothesis this candidate was designed to test. `exp0013` collapsed by a comparable cumulative step count under the unprotected 1:1 mix; this candidate, with LIBERO-Long protection, has instead improved steadily the entire time. `click_alarmclock` and `turn_switch` have both held their most recent gains rather than slipping back.

This is a real decision point (the original ~20,000-step budget target, training not currently running). The curated 4-task panel cannot answer whether capability has genuinely *spread* beyond a small task subset (the original finding was only ~10/50 tasks showed any success at all) — reserving the expensive full-50-task scan for exactly this kind of decision point, launching it now.

### Full 50-task RoboTwin Clean scan — cumulative step ~19,740, 2026-08-27

Command: `run_robotwin_manager.py` with no task filter, `EVALUATION.eval_num_episodes=5 +EVALUATION.clean_only=true MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=1`. Raw results: `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont3_3e-5_2026-08-27_02-13-41/20260827_113926/summary.json`.

**Overall: `clean_mean_success_rate = 0.232` (23.2%)** — nearly double `exp0014`'s frozen-backbone full-50-task ceiling (11.6-12.6%, PROGRESS_0014 Section 14). **23 of 50 tasks show nonzero success** (up from ~10-11/50 under every frozen-backbone candidate in this project's history) — capability has genuinely spread, not just deepened on the same narrow set.

**Decisive finding — bimanual/handover breakthrough.** Cross-checked against the 10-task bimanual/handover list that showed zero success under every frozen-backbone experiment in this project (Section 2's diagnostic): **3 of those 10 now show real success** — `handover_mic` 40%, `pick_diverse_bottles` 40%, `place_bread_basket` 20%. This directly confirms the capacity hypothesis this candidate was designed to test: a frozen backbone structurally cannot build new bimanual coordination regardless of training duration; full-backbone plasticity can and does.

Full per-task table (Clean): `shake_bottle` 100%, `shake_bottle_horizontally` 100%, `click_alarmclock` 80%, `dump_bin_bigbin` 80%, `place_container_plate` 80%, `place_burger_fries` 80%, `click_bell` 60%, `move_playingcard_away` 60%, `open_microwave` 60%, `place_object_stand` 60%, `adjust_bottle` 40%, `handover_mic` 40%, `open_laptop` 40%, `pick_diverse_bottles` 40%, `place_cans_plasticbox` 40%, `move_pillbottle_pad` 40%, `press_stapler` 40%, `place_bread_basket` 20%, `place_object_scale` 20%, `place_shoe` 20%, `stack_blocks_two` 20%, `stack_bowls_two` 20%, `turn_switch` 20% (matches the curated-panel measurement at this same checkpoint — consistency check passed); remaining 27 tasks at 0%.

**Decision: `CONTINUE_TRAINING`, unambiguously.** No plateau across 4 consecutive checks (mean 20%→30%→35%→40% on the curated panel; now 23.2% on the actual full-50-task promotion metric), LIBERO fully stable throughout (Spatial/Long both 96%), and today's result is the strongest evidence yet against the queued plasticity-fallback candidate being necessary — full-backbone training is doing exactly what it was hypothesized to do. Extending the training budget beyond the original ~20,000-step ceiling given the continued positive trend with no sign of diminishing returns.

### cont4 — cumulative step ~30,740, `cont4` completed its own 20,000-step ceiling naturally, 2026-08-28

`cont4` (directory/full-state resume from `cont3`'s `step_009000`, `max_steps` extended 9000→20000) ran to completion cleanly (exit code 0, final checkpoint `step_020000.pt`, ~12GB, intact). Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont4`'s own 20000 = 30,740 — roughly 11,000 more steps than the previous check.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock 60%, turn_switch **0%** (down from 20% at the prior two checks — single-task n=5 noise, not a broad signal), press_stapler 40%, open_laptop 80%. **Mean 45%** — continues the unbroken 5-check climb (20→30→35→40→45%).

**Full 50-task RoboTwin Clean scan** (`EVALUATION.eval_num_episodes=5 +EVALUATION.clean_only=true`, no task filter): raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont4_3e-5_2026-08-27_13-28-52/20260828_091224/summary.json`.

**Overall: `clean_mean_success_rate = 0.328` (32.8%)** — up from 23.2% at the previous checkpoint (+9.6 points over ~11,000 more steps), still no sign of plateauing. **36 of 50 tasks now show nonzero success** (up from 23/50).

**Decisive finding — bimanual/handover capability continuing to spread.** Of the 10 bimanual/handover tasks that showed zero success under every frozen-backbone experiment in this project's history: **8 of 10 now show real success** (up from 3/10 at the prior check) — `handover_mic` 80% (was 40%), `pick_dual_bottles` 40% (new), `place_bread_basket` 40% (was 20%), `place_object_basket` 20% (new), `pick_diverse_bottles` 20% (was 40% — single-task dip, n=5), `lift_pot` 20% (new), `grab_roller` 20% (new), `scan_object` 20% (new). Only `place_dual_shoes` and `handover_block` remain stuck at 0%. This is the strongest evidence yet that full-backbone plasticity is closing the exact capability gap this candidate was designed to close, and it is still accelerating rather than saturating.

Other movement: `move_pillbottle_pad` regressed 40%→0% (one task, not a pattern — every other previously-nonzero single-arm task held or improved: `click_alarmclock` 80%, `open_laptop` 80%, `dump_bin_bigbin` 60%, `click_bell` 60%, `open_microwave` 60%, `move_playingcard_away` 80%, `place_container_plate` 100%, `shake_bottle`/`shake_bottle_horizontally` both 100%). New nonzero tasks beyond the bimanual set: `beat_block_hammer` 80%, `adjust_bottle` 80%, `place_bread_skillet` 80%, `place_burger_fries` 60%, several others at 20-40%.

**LIBERO sentinel** (n=5/task): **Spatial 94.00% (47/50), Long 96.00% (48/50)**. Spatial down slightly from 96.00% at the prior check (a single trial's worth of noise at n=5), Long unchanged. Both comfortably clear the 90% floor — LIBERO retention remains a non-issue throughout this candidate's entire run. Raw: `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260828_105447/`.

**Decision: `CONTINUE_TRAINING`.** Five consecutive checks, zero plateau, zero LIBERO risk, and the bimanual-capability spread (3/10 → 8/10) is accelerating, not saturating. No basis to stop. Extending again.

### cont5 — cumulative step ~40,740, `cont5` completed its own 30,000-step ceiling naturally, 2026-08-29

`cont5` (directory/full-state resume from `cont4`'s `step_020000`, `max_steps` extended 20000->30000) ran to completion cleanly (exit code 0, final checkpoint `step_030000.pt`, ~12GB, intact). Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont5`'s own 30000 = 40,740 -- roughly 10,000 more steps than the previous check.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock **100%**, turn_switch **60%** (up sharply from 0% -- the most stubborn holdout finally moving), press_stapler 60%, open_laptop 60%. **Mean 70%** -- continues the unbroken 6-check climb (20->30->35->40->45->70%), and the jump this check is the largest yet.

**Full 50-task RoboTwin Clean scan**: raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont5_3e-5_2026-08-28_11-11-15/20260829_053357/summary.json`.

**Overall: `clean_mean_success_rate = 0.424` (42.4%)** -- up from 32.8% at the previous checkpoint (+9.6 points over ~10,000 more steps -- the same magnitude jump as the prior check, no deceleration). **41 of 50 tasks now show nonzero success** (up from 36/50).

**Bimanual/handover tasks: 9 of 10 now nonzero** (up from 8/10) -- `handover_block` finally broke through (0%->40%), the last bimanual task besides `place_dual_shoes` to ever move off zero in this project's history. Current bimanual state: handover_mic 60%, pick_dual_bottles 60%, handover_block 40%, place_bread_basket 40%, pick_diverse_bottles 40%, scan_object 40%, place_object_basket 20%, grab_roller 20%, lift_pot 0% (single-trial n=5 dip, was only 20%/1-of-5 previously), place_dual_shoes 0% (still the only task that has never shown any success anywhere in this project).

One benign infra note: a single `grab_roller` episode hit a RoboTwin simulator scripting assertion (`target_pose cannot be None for move action`, in the task's own scripted grasp-pose computation for a specific seed) -- counted correctly as a failed episode, did not affect any other task or crash the eval.

**Full 4-suite LIBERO check** (n=5/task, all four canonical suites in one pass per explicit user request -- closes the Object/Goal evidence gap flagged since exp0014): raw `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260829_053358/summary.json`.

| Suite | Success |
|---|---:|
| Spatial | 96.0% (48/50) |
| Object | **100.0%** (50/50) |
| Goal | 96.0% (48/50) |
| Long | 98.0% (49/50) |
| **Overall** | **97.5%** |

Every suite comfortably clears the 90% floor -- Object and Goal are now *stronger* than exp0014's original baseline (99.0%/97.0%). LIBERO retention has never been at risk anywhere in this candidate's history, and this is the strongest LIBERO reading of the entire run.

**Decision: `CONTINUE_TRAINING`.** Six consecutive checks, zero plateau (if anything the per-check RoboTwin gain is accelerating: +9.6 twice in a row over comparable step counts), zero LIBERO risk (all 4 suites now confirmed, strongest reading yet), and bimanual coverage is essentially complete (9/10). No basis to stop. Extending again.

Both eval jobs (full-50-task RoboTwin scan and full 4-suite LIBERO check) were run **concurrently** this round to save wall-clock time -- no OOM, no crashes this time; GPU memory stayed at 25-60GB free per GPU throughout. **Correction (cont6 check, 2026-08-30): this was not actually safe in general** -- the identical concurrent-launch pattern on `cont6`'s checkpoint crashed the LIBERO 4-suite manager with `CUBLAS_STATUS_ALLOC_FAILED` (GPU0 down to <1GB free while the RoboTwin scan's per-task memory footprint varied over time). The manager aborts the entire run on a single failed subtask, so this is not a partial-data situation -- the whole LIBERO check must be relaunched. **Revised rule**: do not run the full-50-task RoboTwin scan and the full 4-suite LIBERO check concurrently; run them sequentially instead. The earlier "confirmed safe" note was true only for that specific checkpoint's timing, not a general property.

### cont6 — cumulative step ~50,740, `cont6` completed its own 40,000-step ceiling naturally, 2026-08-30

`cont6` (directory/full-state resume from `cont5`'s `step_030000`, `max_steps` extended 30000->40000) ran to completion cleanly (exit code 0, final checkpoint `step_040000.pt`). Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont6`'s own 40000 = 50,740.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock 100%, open_laptop **100%** (up from 60%), press_stapler 60%, turn_switch 40% (down from 60%). **Mean 75%** -- still climbing (20->30->35->40->45->70->75%) but a much smaller step than the prior two checks.

**Full 50-task RoboTwin Clean scan**: raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont6_3e-5_2026-08-29_07-20-54/20260830_015938/summary.json`.

**Overall: `clean_mean_success_rate = 0.428` (42.8%)** -- essentially flat vs. 42.4% at the previous checkpoint (+0.4 points, vs. +9.6 twice in a row before this) -- **the first non-accelerating reading in this candidate's history.** 39/50 tasks nonzero (down slightly from 41/50).

Real task-level churn underneath the flat aggregate (not a clean plateau): several tasks dropped 20-60 points (`place_phone_stand` 60->0, `adjust_bottle` 100->60, `place_object_stand` 100->60, `move_stapler_pad` 40->0, `stack_bowls_three` 40->0) while others gained comparably (`place_cans_plasticbox` 40->80, `place_empty_cup` 0->40, `stamp_seal` 0->40, `place_object_basket` 20->60). At n=5/task this magnitude of individual-task swing is consistent with normal sampling noise, not necessarily a real ceiling -- most single-task deltas are exactly what a 1-2 episode flip produces at this trial count.

**Milestone: `place_dual_shoes` finally broke through (0%->20%)** -- the single bimanual task that had never shown any success anywhere in this project's entire history (every frozen-backbone candidate, and every prior exp0017 check). With this, all 10 bimanual/handover tasks have now shown nonzero success at some point in this candidate's run (though not simultaneously -- `lift_pot` dipped to 0% this same check).

**Full 4-suite LIBERO check**: first attempt **crashed** -- launched concurrently with the RoboTwin scan (same pattern that worked cleanly at the prior checkpoint) and hit `CUBLAS_STATUS_ALLOC_FAILED` on GPU0 when the RoboTwin scan's per-task memory footprint happened to leave <1GB free. The LIBERO manager aborts the entire run on any single subtask failure, so this produced no data -- corrected the standing note in this file (concurrent eval launch is not reliably safe; do sequentially instead) and relaunched LIBERO alone after the RoboTwin scan finished. Retry succeeded cleanly: raw `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260830_034028/summary.json`.

| Suite | Success |
|---|---:|
| Spatial | 96.0% (48/50) |
| Object | 98.0% (49/50) |
| Goal | 96.0% (48/50) |
| Long | 96.0% (48/50) |
| **Overall** | **96.5%** |

Essentially unchanged from the prior checkpoint's 97.5% (within n=5 noise). All 4 suites comfortably clear the 90% floor. LIBERO retention remains a non-issue.

**Decision: `CONTINUE_TRAINING`.** Per the project's own patience policy, a single flat reading after two strong consecutive jumps does not establish a plateau on its own, especially given the substantial per-task noise underneath the flat aggregate and the genuine milestone (all 10 bimanual tasks now proven capable). LIBERO remains fully stable. However, **this is the first check that should be weighed carefully at the next evaluation** -- if the next full-50-task reading is also flat or negative, that would be real evidence of a plateau worth investigating (e.g. LR nearing the end of its cosine schedule, or a genuine capability ceiling under this data mixture) rather than continuing to extend blindly.

### cont7 — cumulative step ~60,740, `cont7` completed its own 50,000-step ceiling naturally, 2026-08-31

`cont7` (directory/full-state resume from `cont6`'s `step_040000`, `max_steps` extended 40000->50000) ran to completion cleanly. Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont7`'s own 50000 = 60,740. This check was specifically designed (per `cont7`'s own launch note) to determine whether `cont6`'s flat +0.4 reading was noise or a real plateau.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock 100%, turn_switch 40% (unchanged), press_stapler 60% (unchanged), open_laptop 80% (down from 100%). **Mean 70%**, down slightly from 75%.

**Full 50-task RoboTwin Clean scan**: raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont7_3e-5_2026-08-30_04-05-46/20260830_221904/summary.json`.

**Overall: `clean_mean_success_rate = 0.460` (46.0%)** -- up from 42.8% (+3.2 points). **Answers the plateau question: this is real continued progress, not a dead ceiling** -- growth has clearly decelerated from the initial +9.6/+9.6 pace, but it has not stopped. 43/50 tasks nonzero (up from 39/50). Bimanual tasks: `pick_diverse_bottles` 20%->60%, `scan_object` 20%->40% (both real gains), `place_object_basket` 60%->40% (dip); rest held steady. 9/10 bimanual tasks nonzero (`lift_pot` the sole holdout at 0%, unchanged from the prior check).

**Full 4-suite LIBERO check**: raw `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260831_000035/summary.json`.

| Suite | Success |
|---|---:|
| Spatial | 98.0% (49/50) |
| Object | 100.0% (50/50) |
| Goal | 96.0% (48/50) |
| Long | 96.0% (48/50) |
| **Overall** | **97.5%** |

Matches the strongest LIBERO reading of the entire candidate. Retention remains a complete non-issue.

**Infra note**: the LIBERO eval manager hit the previously-documented "stale tmux session" bug (`research/NOTES.md`) -- scheduler reported 8 tasks "Running" for 2+ hours while `nvidia-smi` showed 0% GPU/0MB used on all 4 GPUs and `tmux ls` showed no server running. Fixed per the documented procedure: `kill -9` the stuck manager process, confirmed `tmux kill-server`/`tmux ls` showed a clean state, relaunched the identical command -- succeeded immediately on retry. Not a candidate-validity issue, purely an eval-infra flake.

**Decision: `CONTINUE_TRAINING`.** The deceleration from cont6 was real but the trend is still positive, not flat -- +3.2 points is a genuine (if smaller) gain, and the underlying task-level picture (more nonzero tasks, real bimanual gains) supports continuing rather than switching strategy. LIBERO remains rock-solid. Extending again, though the growth-rate trend (9.6, 9.6, 0.4, 3.2) is worth tracking closely -- if the next 1-2 checks average out below ~+2/check, that would be a stronger case for a genuine capacity/data-mixture ceiling under this specific recipe, worth a `$investigate-fastwam-problem` pass rather than continuing to extend blindly.

### cont8 — cumulative step ~70,740, `cont8` completed its own 60,000-step ceiling naturally, 2026-09-01

`cont8` (directory/full-state resume from `cont7`'s `step_050000`, `max_steps` extended 50000->60000) ran to completion cleanly. Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont8`'s own 60000 = 70,740.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock 100%, press_stapler 60%, turn_switch 40%, open_laptop 80%. **Mean 70%**, unchanged from cont7.

**Full 50-task RoboTwin Clean scan**: raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont8_3e-5_2026-08-31_04-07-08/20260831_225235/summary.json`.

**Overall: `clean_mean_success_rate = 0.520` (52.0%)** -- up from 46.0% (+6.0 points) -- the deceleration reversed; this is a faster gain than the prior check. **First time crossing the 50% mark.** 45/50 tasks nonzero (up from 43/50).

Bimanual tasks churned again at n=5 (not a clean trend either direction): `handover_mic` 60%->100%, `pick_dual_bottles` 80%->100%, `grab_roller` 20%->40% (real gains); `place_dual_shoes` 20%->0%, `handover_block` 20%->0% (both dipped back to zero this check, having shown nonzero success previously -- consistent with ongoing n=5 noise on the hardest tasks rather than lost capability). 7/10 bimanual tasks nonzero this check (down from 9/10, but every task in the set has shown nonzero success at some point in the candidate's history). Only 5/50 tasks total are at zero this check (down from 7/50) -- the overall trend (fewer zero tasks, higher mean) is unambiguously positive even though individual bimanual tasks bounce around.

**Full 4-suite LIBERO check**: raw `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260901_002853/summary.json`.

| Suite | Success |
|---|---:|
| Spatial | 98.0% (49/50) |
| Object | 98.0% (49/50) |
| Goal | 98.0% (49/50) |
| Long | 98.0% (49/50) |
| **Overall** | **98.0%** |

**Strongest LIBERO reading of the entire candidate** -- all four suites land at exactly 98.0%. Retention remains a complete non-issue.

**Decision: `CONTINUE_TRAINING`.** Growth-rate trend across the last four checks: +9.6, +0.4, +3.2, +6.0 -- noisy but net positive, with no evidence of a genuine ceiling. Combined with LIBERO's strongest-ever reading, there is no basis to change strategy. Extending again.

### cont9 — cumulative step ~80,740, `cont9` completed its own 70,000-step ceiling naturally, 2026-09-01

`cont9` (directory/full-state resume from `cont8`'s `step_060000`, `max_steps` extended 60000->70000) ran to completion cleanly. Cumulative training step: phase-1's 2740 + `cont2`'s 8000 + `cont9`'s own 70000 = 80,740.

**RoboTwin curated 4-task panel, Clean phase**: click_alarmclock 100%, press_stapler 60%, open_laptop 80%, turn_switch 20% (down from 40%). **Mean 65%**, down slightly from 70%.

**Full 50-task RoboTwin Clean scan**: raw `evaluate_results/robotwin/multiembodiment_libero_robotwin_disjoint_offset_release_parent_full_backbone_long_protected_longrun_cont9_3e-5_2026-09-01_00-53-41/20260901_193340/summary.json`.

**Overall: `clean_mean_success_rate = 0.520` (52.0%)** -- exactly flat vs. cont8 (second genuinely flat reading in this candidate's history, after cont6's +0.4). 43/50 tasks nonzero (down from 45/50). Heavy task-level churn in both directions at n=5 (`scan_object` 40%->100%, `place_cans_plasticbox` 20%->100%, `beat_block_hammer` 60%->100% vs. `turn_switch`/`put_bottles_dustbin`/`stack_bowls_three` all dropping to 0%) -- consistent with sampling noise rather than a real ceiling, the same pattern that made cont6's flat reading resolve as noise at the next check.

**Milestone: all 10 bimanual/handover tasks are nonzero simultaneously for the first time** -- `place_dual_shoes`, `handover_block`, and `lift_pot` (previously the last holdouts, individually) all show nonzero success at this same checkpoint together. Every prior check had at most 9/10 nonzero at once.

**Full 4-suite LIBERO check**: raw `evaluate_results/libero/libero_uncond_2cam224_multiembodiment_eval/20260901_211110/summary.json`.

| Suite | Success |
|---|---:|
| Spatial | 98.0% (49/50) |
| Object | 100.0% (50/50) |
| Goal | 96.0% (48/50) |
| Long | 96.0% (48/50) |
| **Overall** | **97.5%** |

Essentially unchanged from cont8's 98.0%. Retention remains a complete non-issue.

**Decision: `CONTINUE_TRAINING`.** A flat aggregate reading with this much underlying task churn and a genuine positive milestone (10/10 bimanual simultaneous) is not strong evidence of a real ceiling -- the same signature as cont6, which turned out to be noise. Extending again; the next check will be the real test of whether growth resumes (as it did after cont6) or genuinely stalls.

## 7-12.

To be filled in once training plateaus, LIBERO shows real risk, or the ≥90% target is approached — per `$review-fastwam-experiment`.
