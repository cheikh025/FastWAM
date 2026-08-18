# Current Research State

Status: `RESEARCH_LOOP_ACTIVE — PARENT CHECKPOINT SWITCHED (exp0011): exp0019 replaced by the official FastWAM LIBERO release checkpoint after exp0019 was found to be seed-sensitive/fragile specifically on LIBERO-Spatial (see PROGRESS_0011). Establishing a release-checkpoint baseline (exp0012) before the first real multi-embodiment training candidate on the new parent.`

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

## Release-checkpoint LIBERO evidence so far

| Suite | Native (K=7) | Expanded (K=21, zero-init) |
|---|---:|---:|
| LIBERO-Spatial | in progress (`exp0012`) | 96.00% / 97.00% / 97.00% (n=10, three runs, two seeds) |
| LIBERO-Object | in progress (`exp0012`) | not yet measured |
| LIBERO-Goal | in progress (`exp0012`) | not yet measured |
| LIBERO-Long | in progress (`exp0012`) | not yet measured |

`exp0012` (launched 2026-08-18, `checkpoints/exp0012_release_native_4suites_n10.log`): native release checkpoint across all four in-scope suites, n=10, in progress.

## RoboTwin evidence

Canonical protocol confirmed during original setup (`research/RUNBOOK.md`): 50-task Aloha-AgileX benchmark, Clean/Randomized tracked separately. Not yet measured for any release-checkpoint-derived candidate — no multi-embodiment training has happened on the new parent yet.

## Current accepted multi-embodiment checkpoint

None yet. The project is at the pre-training-candidate stage on the new (release-checkpoint) parent.

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
