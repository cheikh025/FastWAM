# PROGRESS_0015 — RoboTwin-heavy sampling ratio under frozen backbone

- **Experiment ID:** 0015
- **Status:** `REJECT` (no improvement over exp0014 on the full-50-task mean despite 3x RoboTwin gradient share -- a wash, not progress; see Section 6-7)
- **Created:** 2026-08-20
- **Updated:** 2026-08-20
- **Parent experiment:** 0014 (frozen-backbone diagnostic)
- **Parent checkpoint:** exp0014 cumulative step 4600 (`research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt` on `cheikh025/ASR`) -- current project best
- **Git branch:** `autoresearch/robotwin-multiembodiment-v1`
- **Git commit:** pending

## 1. Result at a glance

Tests whether a RoboTwin-heavier sampling ratio (1:3 LIBERO:RoboTwin, up from ~1:1) broadens RoboTwin capability beyond exp0014's ~10-task ceiling. exp0014's decisive full-50-task rescans (Section 14 of its report) showed the ~12% mean is flat across 2000 additional training steps under a 1:1 ratio, while the same small set of well-represented tasks kept improving -- this candidate gives RoboTwin 3x its previous share of gradient updates, now that LIBERO retention risk is essentially zero under the frozen-backbone regime (100% throughout exp0014).

## 2. Research state before experiment

See `research/progress/PROGRESS_0014_frozen_backbone_diagnostic.md` Sections 14-15 for full evidence. Summary:

| Cumulative step | Full-50-task RoboTwin Clean mean (n) | LIBERO-Spatial |
|---|---:|---:|
| 2600 | 12.6% (n=10) | 96.7% |
| 4600 | 11.6% (n=5) | 100.0% |

An attempted per-task data-density diagnostic via keyword matching proved unreliable and was retracted (`research/NOTES.md`) -- precise per-task RoboTwin episode counts cannot be reliably reconstructed from the preprocessed dataset (no ground-truth per-episode canonical-task label preserved).

## 3. Candidate design

### Modification

Single change from exp0014's config: new data config `multiembodiment_libero_robotwin_robotwin_heavy` (RoboTwin `ratio` 1.0 -> 3.0, LIBERO stays 1.0). `trainable_modules: expanded_projections_only` (frozen backbone) and everything else unchanged from exp0014.

### Why this candidate

Under exp0013's old full-backbone regime, a higher RoboTwin ratio was untested (exp0008 only tested the opposite direction, more-LIBERO, on the old exp0019 lineage). Under exp0014's frozen-backbone regime, LIBERO retention has proven essentially immune to whatever RoboTwin data does (backbone can't drift; disjoint action-channel offsets make cross-embodiment gradient contamination provably zero) -- so the original reason for a conservative 1:1 mix no longer applies. This tests the most direct, evidence-motivated lever for broadening RoboTwin capability: give RoboTwin a larger share of the gradient budget per step, since more *steps* alone (exp0014's cont3/cont4) did not broaden coverage.

### What to watch

- Full-50-task RoboTwin Clean mean vs. exp0014's flat 11.6-12.6% -- does 3x RoboTwin gradient share per step move tasks beyond the current ~10-task set, or does it just accelerate the same tasks further (as more steps alone did)?
- LIBERO-Spatial retention -- expected to stay near 100% given the frozen backbone, but confirm since ratio changes affect step composition, not just backbone plasticity.
- Curated-panel RoboTwin tasks (click_alarmclock, turn_switch) for continuity with exp0014's trend.

### Initial compute plan

Per standing feedback (start smaller, scale on evidence): `max_steps: 1000`, `save_every: 200`. Progress-check with the cheap panel around local step 400-600, then a full-50-task rescan (n=5) once meaningful training has accumulated, to directly compare against exp0014's flat trajectory.

## 4. Exact code and configuration state

- data config: `configs/data/multiembodiment_libero_robotwin_robotwin_heavy.yaml` (new)
- task config: `configs/task/multiembodiment_libero_robotwin_disjoint_offset_release_parent_frozen_backbone_robotwin_heavy_3e-5.yaml` (new)
- resume source: `runs/reweighted_multiembodiment/exp0014_release_parent_frozen_backbone_v1_cont4/checkpoints/weights/step_001000.pt` (local; = cumulative step 4600, durably backed up as `cheikh025/ASR:research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt`)

## 5-11. Pending

To be filled in as the run progresses.

## 5. Evaluation events

### Evaluation event — LIBERO-Spatial + RoboTwin `progress_check`, local step 600

- checkpoint: `runs/reweighted_multiembodiment/exp0015_robotwin_heavy_v1/checkpoints/weights/step_000600.pt`
- LIBERO-Spatial (n=3): **100.0% (30/30)** -- confirms retention holds even with the RoboTwin-heavier ratio (as expected, backbone frozen).
- RoboTwin Clean (n=5): click_alarmclock 100%, turn_switch 80%, open_laptop 40%, adjust_bottle 0% -- very similar to exp0014's own trajectory at comparable points; `adjust_bottle` (the persistent zero-task) shows no change yet.

No clear broadening signal at this early point (step 600/1000, effective RoboTwin exposure roughly comparable to exp0014's own early checks given the 3x ratio only partially offsets the shorter step count so far). Resuming training toward the full 1000-step budget; the decisive test is a full-50-task rescan once training completes, compared directly against exp0014's flat 11.6-12.6% baseline.

## 6. Full 50-task RoboTwin rescan (n=5) on the final checkpoint -- decisive negative result

Full rescan launched on the candidate's final checkpoint (local step 1000, cumulative training step ~5600 from the release-checkpoint parent). Command matches exp0014's Section 14 pattern (`EVALUATION.eval_num_episodes=5`, no task filter). Log: `checkpoints/exp0015_final_robotwin_full50_clean_n5.log`. Results: `evaluate_results/robotwin/reweighted_multiembodiment_exp0015_robotwin_heavy_v1_cont1/20260820_033957/summary.json`.

**Result: 11.6% mean Clean success -- essentially identical to exp0014's own final result (also 11.6% at cumulative step 4600).** LIBERO-Spatial 100% throughout (confirmed retention unaffected by the ratio change, as expected under frozen backbone).

9/50 tasks nonzero: `click_alarmclock`/`click_bell`/`open_microwave`/`press_stapler` (100%), `turn_switch` (60%), `open_laptop`/`shake_bottle_horizontally` (40%), `place_container_plate`/`place_object_basket` (20% -- **new**, never nonzero at any exp0014 checkpoint). But several previously-nonzero tasks dropped to 0% this time: `move_playingcard_away` (was 40% at exp0014 cumulative 4600), `put_bottles_dustbin` (was 20%), `shake_bottle` (was 20-30%), `place_a2b_right` (was 10-20%).

**Interpretation: the RoboTwin-heavy ratio (3x gradient share) did not broaden capability -- it's a wash.** Two new tasks gained marginal (20%, n=1/5) success while others lost it, net mean unchanged. This is a second consecutive negative result on the "give RoboTwin more of something" axis (first more *steps*, exp0014's cont3/cont4; now more *gradient share per step*, exp0015) -- both failed to move the ~12% full-benchmark ceiling. This suggests the bottleneck is not primarily about RoboTwin's total training exposure (steps x ratio), but something more structural: likely the limited capacity of `expanded_projections_only` (only the action_encoder/head are trainable) to represent 50 distinct bimanual manipulation behaviors from a frozen, LIBERO-tuned backbone representation.

## 7. Decision

**`REJECT`** this candidate -- no improvement over exp0014 on the metric that matters (full-50-task mean), despite 3x RoboTwin gradient share. Not worth further investment on this specific lever.

**Research direction for the next candidate**: reconsider backbone plasticity. Two consecutive negative results (more steps, more ratio) under `expanded_projections_only` argue that the frozen-backbone's limited trainable capacity -- not data exposure -- is now the likely bottleneck. The trainer's `dit_with_backbone_low_lr` mode (two-parameter-group optimizer: full LR on action_encoder/head, a much lower LR on the rest of the shared backbone) was specifically built as a middle ground between `expanded_projections_only` (current, capacity-limited) and `dit` (exp0013's setting, proven to interfere with LIBERO/decline over time) -- this was never tested under the current release-checkpoint parent lineage and is the natural next lever to try, trading a small, controlled amount of LIBERO drift risk for more representational capacity to learn broader RoboTwin behavior.

Current project-best checkpoint remains exp0014 cumulative step 4600 (unchanged by this candidate) -- durably backed up at `cheikh025/ASR:research/exp0014_release_parent_frozen_backbone/step_004600_cumulative.pt`. exp0015's final checkpoint is also preserved (`cheikh025/ASR:research/exp0015_robotwin_heavy/step_001000_cumulative.pt`) for reference but not selected as best.
