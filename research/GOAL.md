# Research Goal: FastWAM Multi-Embodiment LIBERO + RoboTwin

## Inherited parent model

Start from the previously promoted LIBERO checkpoint:

`runs/reweighted_libero90_finetune/exp0019_spatial_weak_task_oversample/checkpoints/weights/step_005000.pt`

Recorded durable location:

`cheikh025/ASR:promoted/0019_spatial_weak_task_oversampling/step_005000.pt`

Checkpoint-producing code state to verify:

- fork: `https://github.com/cheikh025/FastWAM`
- parent lineage branch: `autoresearch/libero90-v1`
- exact commit: `b2b49d0`

The parent branch is read-only for this project. Create and work on `autoresearch/robotwin-multiembodiment-v1` from the exact verified parent code state.

## Inherited canonical LIBERO evidence

The parent checkpoint was previously validated canonically at 50 trials/task across all 130 LIBERO tasks:

| Suite | Parent success |
|---|---:|
| LIBERO-Spatial | 97.00% |
| LIBERO-Object | 99.60% |
| LIBERO-Goal | 97.20% |
| LIBERO-Long / LIBERO-10 | 98.00% |
| LIBERO-90 | 95.13% |

These values are reference facts inherited from the parent project. They are not new-run measurements and do not populate the new experiment ledger as if this project produced them.

## Objective

Adapt the same FastWAM research line for **RoboTwin multi-embodiment learning** while preserving LIBERO capability.

Hard promotion constraints:

- `LIBERO-Spatial success >= 90%`
- `LIBERO-Object success >= 90%`
- `LIBERO-Goal success >= 90%`
- `LIBERO-Long / LIBERO-10 success >= 90%`
- `LIBERO-90 success >= 90%`

Among checkpoints satisfying all five constraints, optimize the canonical RoboTwin performance defined and verified during setup.

Do not collapse RoboTwin and LIBERO into one average that can hide catastrophic forgetting.

## Initial multi-embodiment representation direction

Before implementation, read the full Qwen-VLA paper and other directly relevant multi-embodiment/generalist robotics work, then verify the actual FastWAM LIBERO/RoboTwin code and data semantics.

For the initial heterogeneous action interface, use a **shared padded action representation** following the relevant Qwen-VLA formulation, adapted correctly to FastWAM. The implementation must account for valid action channels, loss masking, dataset-specific normalization, checkpoint expansion/loading, control conventions, and per-embodiment inference decoding.

Use this as the initial shared action interface while the research loop explores the broader multi-embodiment learning problem.

## Ranking checkpoints

A checkpoint is eligible for the main line only after canonical evidence shows all five LIBERO measurements are >=90%.

Among eligible checkpoints:

1. prefer stronger canonical RoboTwin performance under the fixed metric/protocol recorded in `research/RUNBOOK.md`;
2. then prefer a larger minimum LIBERO margin above 90% and healthier RoboTwin task/difficulty balance;
3. then prefer smaller unnecessary regression from the inherited/accepted LIBERO performance;
4. then prefer more stable/reproducible evidence;
5. then prefer simpler, less fragile, and less expensive solutions.

If the canonical RoboTwin protocol reports multiple primary metrics rather than a single official aggregate, preserve them separately and define the project comparison rule in the runbook before optimizing candidates. Do not invent or change the rule after seeing candidate results.

## In-scope research

FastWAM combines world/video modeling with action prediction, so research may consider both WAM-specific and ordinary multi-task/continual-policy choices. Useful areas include data mixture/replay, task balance, loss weighting, optimization, freezing/unfreezing, retention methods, embodiment conditioning, normalization, camera/observation handling, action prediction and chunking, world/action interaction, robustness, architecture when justified, and later optimization/RL stages when supported by evidence.

Use current evidence, code, data, and relevant research to decide what is worth compute. Do not import the old LIBERO project's experiment-selection history as a search prior.
