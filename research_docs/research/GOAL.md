# Research Goal: Extend FastWAM to LIBERO-90

## Starting model

Use the official released FastWAM LIBERO checkpoint:

`libero_uncond_2cam224.pt`

If the launch environment stores it under a different path, record the actual path in `research/STATE.md` and `research/RUNBOOK.md`.

Fast-WAM's public LIBERO results on the four standard suites are:

| Suite | Reported success |
|---|---:|
| LIBERO-Spatial | 98.2% |
| LIBERO-Object | 100.0% |
| LIBERO-Goal | 97.0% |
| LIBERO-Long | 95.2% |
| Four-suite average | 97.6% |

These values describe the published Fast-WAM result and show that the starting model family is already strong on the original LIBERO benchmark.

The first action in this project is still to run the released checkpoint locally and record the measured baseline on the exact evaluation setup used for research.

## Objective

Extend FastWAM so the same research line also performs strongly on **LIBERO-90**.

Primary target:

`LIBERO-90 success >= 90%`

Retention constraints:

- `LIBERO-Spatial success >= 90%`
- `LIBERO-Object success >= 90%`
- `LIBERO-Goal success >= 90%`
- `LIBERO-Long / LIBERO-10 success >= 90%`

The objective is not to maximize the average of all suites if doing so hides a failure in one original suite.

## Ranking checkpoints

A checkpoint is valid for the main line only when all four original suites remain at or above 90%.

Among valid checkpoints:

1. higher LIBERO-90 success is better;
2. then prefer a larger minimum margin above 90% across all five suites;
3. then prefer smaller unnecessary regression on the original suites;
4. then prefer more stable/reproducible evidence;
5. then prefer simpler, less fragile, and less expensive solutions.

A drop on an original suite is allowed while the suite remains >=90% if the candidate substantially improves the constrained objective.

## FastWAM context

Fast-WAM is a World Action Model built around video/world-model co-training plus action prediction. Its public design uses a pretrained video DiT backbone with an action expert; world/video modeling is used during training, while inference directly produces actions without explicit future-video generation.

Research may therefore consider both ordinary policy-learning choices and WAM-specific choices involving world/video representation learning, action prediction, their interaction, conditioning, training objectives, and trainable/frozen components.

Do not assume any particular intervention will work. Use the current model, code, data, metrics, and experiments to decide what to try.
