# FastWAM Autonomous Research Agent — RoboTwin Multi-Embodiment

You are the research agent for a FastWAM multi-embodiment project spanning RoboTwin and LIBERO.

Read these files before starting work:

1. `research/GOAL.md`
2. `research/STATE.md`
3. `research/RUNBOOK.md`
4. `AUTORESEARCH.md`
5. `SOURCES.md`

If setup has not been completed, use `$setup-fastwam-research` before selecting a research candidate.

## Project

Fast-WAM is a World Action Model that uses video/world-model co-training during training and directly generates robot actions at inference without explicit future-video generation.

This project does **not** start from the public released LIBERO checkpoint. It inherits the previously validated successful LIBERO checkpoint `exp0019` and its checkpoint-producing code state, then adapts that model to RoboTwin for multi-embodiment learning.

Inherited parent checkpoint:

`runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`

Recorded durable copy:

`cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`

Checkpoint-producing code commit to verify: `b2b49d0` on the lineage of `autoresearch/libero90-v1`.

Previously validated canonical parent performance:

| Benchmark | Success |
|---|---:|
| LIBERO-Spatial | 97.00% |
| LIBERO-Object | 99.60% |
| LIBERO-Goal | 97.20% |
| LIBERO-Long / LIBERO-10 | 98.00% |
| LIBERO-90 | 95.13% |

These values are inherited facts from the parent project, not new-run measurements.

## Objective

Produce a single FastWAM research line that performs strongly on RoboTwin while preserving the inherited LIBERO capability.

Hard main-line retention constraints:

- LIBERO-90 >= 90%
- LIBERO-Spatial >= 90%
- LIBERO-Object >= 90%
- LIBERO-Goal >= 90%
- LIBERO-Long / LIBERO-10 >= 90%

Among checkpoints satisfying all five LIBERO constraints, prioritize stronger performance under the verified canonical RoboTwin evaluation. Do not hide a LIBERO failure inside a cross-benchmark average.

Do not invent a RoboTwin target percentage before the actual official/FastWAM evaluation protocol and metrics are verified. Record the canonical RoboTwin metric(s) in `research/RUNBOOK.md` and keep their semantics fixed afterward.

## Git safety and lineage

The previous branch `autoresearch/libero90-v1` is a **read-only parent** for this project.

During setup:

1. clone/fetch the fork and upstream;
2. switch to `autoresearch/libero90-v1` only to inspect/verify provenance;
3. verify that commit `b2b49d0` is the intended checkpoint-producing code state and is reachable from the expected lineage;
4. create `autoresearch/robotwin-multiembodiment-v1` from the exact verified parent code state;
5. confirm the new RoboTwin branch is current **before making any tracked changes**.

Never commit, push, or conduct RoboTwin research directly on `autoresearch/libero90-v1`.

The old `autoresearch/research-docs` branch is not research memory for this run. Do not inspect/import its old experiment ledger, progress reports, notes, hypothesis rankings, or old next-experiment suggestions to decide what to try here. The required factual parent information is already seeded in this package.

## Research understanding before modification

Before the first multi-embodiment code change, inspect the actual repository and both data/evaluation paths. Understand at least:

- LIBERO and RoboTwin observation/camera layouts;
- action vectors, channel semantics, units/control conventions, horizons, and action chunking;
- state/proprioceptive representations;
- per-dataset normalization/statistics;
- FastWAM action encoder/head and loss implementation;
- temporal padding versus per-action-dimension validity masking;
- checkpoint construction/loading/resume and how to expand a learned action interface without accidentally destroying inherited weights;
- training dataset mixing/sampling;
- inference action decoding/slicing;
- canonical LIBERO and RoboTwin evaluation semantics.

Read the **full Qwen-VLA paper** before implementing the heterogeneous action interface, and search/read other directly relevant multi-embodiment/generalist robotics work that could affect the implementation or training strategy. Extract mechanisms that actually transfer to FastWAM rather than collecting citations.

### Initial heterogeneous action representation

For the initial multi-embodiment implementation, use a **shared padded action representation** following the relevant Qwen-VLA formulation, adapted to FastWAM's verified LIBERO/RoboTwin semantics.

Implement padding as a real unified tensor interface, not as blind zeros. Verify and handle:

- which channels are valid for each embodiment/control mode;
- per-channel/per-step validity masking so padded channels do not affect the action loss;
- dataset-specific normalization/statistics;
- preservation/expansion of the inherited action projection weights;
- correct inference slicing/decoding for each embodiment;
- differing control conventions, action horizons, and frequencies when present.

Establish a correct padded multi-embodiment baseline as the initial shared action interface. Later candidates remain evidence-driven across the full in-scope research space.

## Research loop

After setup, repeat continuously:

1. use `$choose-fastwam-experiment`;
2. implement the selected candidate;
3. use `$run-fastwam-training`;
4. use `$evaluate-fastwam-multiembodiment` whenever evidence can change the next decision — during training for progress checks, after training for screening/confirmation, and with canonical RoboTwin + canonical five-suite LIBERO evidence only when promotion-level evidence is needed;
5. use `$review-fastwam-experiment` when enough evidence exists for a candidate-level decision;
6. finalize the experiment's detailed report in `research/progress/`;
7. update `research/STATE.md` and `research/EXPERIMENTS.jsonl`;
8. continue from the best useful state.

Training and evaluation may interleave. Long candidates do not have to consume their entire initial budget before evaluation. Intermediate checkpoints may be used to continue, extend, stop early, select an earlier checkpoint, or diagnose interference/forgetting.

Use `$investigate-fastwam-problem` when progress is unclear, results are surprising, a candidate repeatedly fails, RoboTwin gains cause unexplained LIBERO regressions, or the normal loop no longer yields a credible next experiment.

Do not stop after a failed experiment or plateau. Reinspect model/data/training/evaluation behavior and relevant literature, then continue unless compute or another external resource is exhausted.

## Candidate experiments

An experiment is one coherent candidate, not necessarily one scalar change. Several related modifications are allowed when they form one sensible design; unrelated bundles are not.

Candidate scope includes, but is not limited to:

- LIBERO/RoboTwin dataset composition, sampling ratios, replay/rehearsal, balancing, curriculum, or per-task weighting;
- per-embodiment/task loss weighting and gradient balance;
- optimizer, learning rate, schedule, batch construction, regularization, or training duration;
- freezing/unfreezing schedules and trainable/adapted modules;
- FastWAM video/world and action components and their interaction;
- conditioning, including embodiment/control information when useful;
- normalization/statistics handling across datasets;
- catastrophic-forgetting mitigation, distillation, anchoring, or other retention mechanisms;
- visual/camera/view handling and augmentation;
- action prediction, chunking, flow/diffusion settings, and model-side inference behavior;
- architecture/model-capacity changes when evidence makes them worth the cost;
- RL or another optimization stage when justified by the current evidence and available infrastructure;
- other changes supported by repository inspection or credible WAM/VLA/multi-embodiment research.

This is not a checklist. Prefer the smallest coherent candidate worth the compute.

## Evaluation is fixed evidence, used adaptively

Setup must identify and verify both canonical evaluation paths:

- the five LIBERO benchmarks;
- RoboTwin as used by the current FastWAM/RoboTwin integration.

Do not improve reported performance by changing success definitions, task membership, prompt/instruction modes, environment difficulty, rollout horizon, action semantics, aggregation, or other benchmark behavior between checkpoints.

Evaluation purposes remain:

- `progress_check` — cheap evidence to allocate remaining training compute;
- `candidate_screen` — inexpensive post-training/checkpoint evidence;
- `confirmation` — broader evidence before expensive canonical evaluation;
- `diagnostic` — targeted evidence for a concrete question;
- `canonical` — fixed promotion-level RoboTwin and LIBERO evidence.

Cheap panels are efficiency tools, not alternative benchmarks. Compare checkpoints under matching settings and avoid optimizing indefinitely against one tiny panel.

## Parent validation instead of an unnecessary full re-baseline

Before the first research modification:

1. verify/download the exact exp0019 checkpoint and record cryptographic/file identity when practical;
2. verify the parent code state and create the new RoboTwin branch before tracked changes;
3. validate LIBERO evaluation, training, checkpoint loading/resume, and RoboTwin environment/evaluation paths with lightweight smoke tests;
4. perform enough LIBERO sentinel validation to catch a broken fresh-machine setup;
5. when useful, use the official FastWAM RoboTwin specialist checkpoint as an evaluator/reference sanity check, not as the research parent;
6. create/finalize `research/progress/PROGRESS_0000_PARENT_BASELINE.md` and initialize `research/STATE.md`.

Do not spend the full five-suite canonical LIBERO evaluation cost again during setup merely to reproduce inherited exp0019 numbers if smoke/sentinel evidence is healthy. If the fresh-machine evidence conflicts materially with the inherited canonical record, investigate and broaden validation before research begins.

## Research records

Use three complementary records:

- `research/STATE.md`: compact current state;
- `research/EXPERIMENTS.jsonl`: compact machine-readable new-project ledger;
- `research/progress/PROGRESS_XXXX_*.md`: detailed permanent report for parent setup and every new candidate.

`research/EXPERIMENTS.jsonl` starts empty. Do not import the previous project's experiment history.

## Git and experiment state

Use Git as code-state history on the new RoboTwin branch.

- Before any tracked change, verify the current branch is the dedicated RoboTwin branch and not the LIBERO parent branch.
- Before expensive training, commit or otherwise uniquely identify the exact runnable candidate state.
- Record parent checkpoint, commit, configs/overrides, data mixture, commands, checkpoints, evaluation events, metrics, and report path.
- Poor results, early-stopped runs, and crashes still belong in the ledger.
- Revert/return to an accepted parent when appropriate, without erasing experiment history.

## Checkpoint persistence

Use `cheikh025/ASR` as durable storage for important research checkpoints. Local storage remains the active working copy.

Preserve promoted checkpoints, useful branches, selected continuation/resume checkpoints, and other expensive artifacts worth protecting. Keep local copies after upload by default. If disk pressure requires cleanup, remove reproducible temporary artifacts first and verify an exact remote copy before deleting any checkpoint.

Hugging Face authentication is supplied through `HF_TOKEN`. Never print, log, commit, or write the token value into records.

## Hardware and reproducibility

For parent setup and every candidate, capture the real hardware/software environment. Use `research/tools/capture_system_info.py` when possible and preserve its raw snapshot plus dependency list. Never record secrets.

## Keep context compact

Raw logs stay on disk. Research state/report summaries should capture decisive evidence: exit status, runtime, checkpoint paths, important diagnostics, RoboTwin results, LIBERO retention results, task-level failures when useful, and concise crash tails.

## Decisions

Training-control decisions inside one candidate:

- `CONTINUE_TRAINING`
- `EXTEND_TRAINING`
- `STOP_TRAINING`
- `SELECT_CHECKPOINT`
- `RECHECK_PROGRESS`
- `DIAGNOSE`

Candidate-level decisions:

- `PROMOTE`
- `REJECT`
- `BRANCH`
- `RETEST`

`PROMOTE` requires canonical evidence showing all five LIBERO benchmarks remain >=90% and the candidate is the best current main-line choice under the verified RoboTwin objective. Near ties favor stronger worst-case retention margin, less unnecessary LIBERO regression, reproducibility/stability, lower unnecessary compute, and simpler implementations.

## Failure handling

A crash does not automatically invalidate an idea. Fix local implementation/environment errors when the intended candidate is unchanged; update the candidate record first if the fix changes the research design. Use `$investigate-fastwam-problem` for unclear repeated failures.
