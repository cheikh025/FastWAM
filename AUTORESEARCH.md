# FastWAM AutoResearch Loop — RoboTwin Multi-Embodiment

This file is the operating loop for the autonomous research session.

## Start once

Run `$setup-fastwam-research`.

Setup should finish with:

- the FastWAM fork cloned and `upstream` configured;
- `autoresearch/libero90-v1` inspected only as the immutable parent lineage;
- exact parent code commit `b2b49d0` verified;
- new branch `autoresearch/robotwin-multiembodiment-v1` created from that exact code state and active before tracked changes;
- the exp0019 parent checkpoint verified/downloaded;
- the fresh AutoResearch files integrated into the new branch without importing old research history;
- LIBERO and RoboTwin environments/data/assets understood and verified;
- the full Qwen-VLA paper plus directly relevant multi-embodiment literature read before implementing the heterogeneous action interface;
- the initial shared padded-action implementation direction understood, including validity masking/normalization/checkpoint expansion/inference decoding;
- setup-only evaluation/training/reload smoke tests passed;
- canonical LIBERO and RoboTwin evaluation semantics fixed in `research/RUNBOOK.md`;
- an adaptive dual-benchmark evaluation plan documented;
- inherited exp0019 canonical LIBERO metrics recorded as parent evidence;
- fresh-machine sentinel/reference validation completed;
- `research/progress/PROGRESS_0000_PARENT_BASELINE.md` finalized;
- `research/STATE.md` initialized.

Do not begin optimization while still on the parent LIBERO branch, before checkpoint/environment validation, or before the action/observation semantics are understood.

## Continuous loop

```text
accepted multi-embodiment checkpoint + fresh experiment history
                         |
                         v
             $choose-fastwam-experiment
                         |
               create progress report
                         |
                         v
                 implement candidate
                         |
                         v
              $run-fastwam-training
                         |
               intermediate checkpoints
                         |
                         v
       $evaluate-fastwam-multiembodiment
          RoboTwin progress + LIBERO sentinels
                         |
              +----------+----------+
              |          |          |
           continue    extend     stop/select
           training   training     checkpoint
              |          |          |
              +----------+----------+
                         |
                   candidate screen
                         |
                  broader confirmation
                         |
      canonical RoboTwin + full LIBERO if earned
                         |
                         v
             $review-fastwam-experiment
                         |
               promote / reject /
                 branch / retest
                         |
                         v
             finalize progress report
                         |
                         v
                 update research state
                         |
                         +----------------> repeat
```

The diagram is illustrative, not mandatory. Short runs may train to completion without an intermediate check. Expensive runs may alternate training and evaluation several times.

When the next move is unclear, use `$investigate-fastwam-problem`, then return to the loop.

## Evaluation purposes

- `progress_check`: cheap RoboTwin progress and LIBERO-retention evidence from an intermediate checkpoint;
- `candidate_screen`: cheap post-training/checkpoint evidence;
- `confirmation`: broader evidence for RoboTwin generalization and LIBERO retention;
- `diagnostic`: targeted evidence for a specific behavior/failure;
- `canonical`: fixed promotion-level RoboTwin evaluation plus the full five-suite LIBERO protocol.

Setup smoke tests validate infrastructure; they are not research performance evidence.

## Main-line validity and success

A main-line candidate is valid only when canonical evaluation shows:

- LIBERO-90 >= 90%
- LIBERO-Spatial >= 90%
- LIBERO-Object >= 90%
- LIBERO-Goal >= 90%
- LIBERO-Long / LIBERO-10 >= 90%

Among valid checkpoints, improve the verified canonical RoboTwin objective as strongly as possible. Further experiments are worthwhile when compute remains and there is a credible route to stronger RoboTwin performance, better worst-case retention margin, improved stability, or a simpler/more efficient solution without violating the five LIBERO floors.
