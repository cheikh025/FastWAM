# Research Goal: FastWAM Multi-Embodiment LIBERO + RoboTwin

## Parent model

Start from the **official FastWAM LIBERO release checkpoint**:

`checkpoints/fastwam_release/libero_uncond_2cam224.pt` (from `https://huggingface.co/yuanty/fastwam`; paired stats `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`)

This project previously started from a research checkpoint (`exp0019`, from the `autoresearch/libero90-v1` lineage) but switched parents after finding `exp0019` was unusually sensitive to the checkpoint-expansion process specifically on LIBERO-Spatial — see `research/progress/PROGRESS_0011*.md` for the full investigation. The release checkpoint expands cleanly and reproducibly and is now the parent. `autoresearch/libero90-v1` remains read-only history, no longer the checkpoint source. All work happens on `autoresearch/robotwin-multiembodiment-v1`.

## Parent LIBERO evidence

Verified fresh on this machine so far (LIBERO-Spatial only, expanded K=21 checkpoint, n=10, three independent runs across two seeds): 96.00% / 97.00% / 97.00%. Native (unexpanded) Spatial and all of Object/Goal/Long are not yet measured for the release checkpoint — establish this baseline before the first real training candidate. Do not reuse `exp0019`'s old inherited numbers (Spatial 97.00%, Object 99.60%, Goal 97.20%, Long 98.00%, LIBERO-90 95.13%) as if they applied to the release checkpoint; they were specific to a different, now-abandoned parent.

## Objective

Adapt the release-checkpoint FastWAM model for **RoboTwin multi-embodiment learning**, targeting **>=90% average RoboTwin success on the full 50-task Aloha-AgileX benchmark** (Clean and Randomized splits evaluated separately, both >=90%), while preserving LIBERO capability.

Hard promotion constraints:

- `LIBERO-Spatial success >= 90%`
- `LIBERO-Object success >= 90%`
- `LIBERO-Goal success >= 90%`
- `LIBERO-Long / LIBERO-10 success >= 90%`

**LIBERO-90 is out of scope** — do not optimize for it or spend evaluation budget on it unless directly relevant to a specific decision.

Among checkpoints satisfying all four constraints, optimize the canonical RoboTwin performance defined and verified during setup.

Do not collapse RoboTwin and LIBERO into one average that can hide catastrophic forgetting.

## Initial multi-embodiment representation direction

Before implementation, read the full Qwen-VLA paper and other directly relevant multi-embodiment/generalist robotics work, then verify the actual FastWAM LIBERO/RoboTwin code and data semantics.

For the initial heterogeneous action interface, use a **shared padded action representation** following the relevant Qwen-VLA formulation, adapted correctly to FastWAM. The implementation must account for valid action channels, loss masking, dataset-specific normalization, checkpoint expansion/loading, control conventions, and per-embodiment inference decoding.

Use this as the initial shared action interface while the research loop explores the broader multi-embodiment learning problem.

## Ranking checkpoints

A checkpoint is eligible for the main line only after canonical evidence shows all four LIBERO measurements (Spatial, Object, Goal, Long) are >=90%.

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
