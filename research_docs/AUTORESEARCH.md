# FastWAM AutoResearch Loop

This file is the operating loop for the autonomous research session.

## Start once

Run `$setup-fastwam-research`.

Setup should finish with:

- a verified FastWAM environment;
- the released LIBERO checkpoint identified;
- a dedicated research branch;
- `research/RUNBOOK.md` filled with the actual training/evaluation commands from this repository;
- setup-only evaluation/training/reload smoke tests passed;
- canonical evaluation fixed;
- a cheap progress-evaluation panel and broader candidate-evaluation plan documented;
- a measured five-suite baseline;
- `research/STATE.md` initialized;
- `research/progress/PROGRESS_0000_BASELINE.md` finalized with baseline hardware/software and evaluation details.

Do not start optimization before setup smoke tests pass and the baseline is recorded. Smoke tests are setup validation only and are not repeated for ordinary experiments.

## Continuous loop

```text
current accepted checkpoint + experiment history
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
      $evaluate-fastwam-on-libero
          cheap progress check
                    |
        +-----------+-----------+
        |           |           |
     continue     extend       stop/select
      training    training      checkpoint
        |           |           |
        +-----------+-----------+
                    |
             candidate screen
                    |
           broader confirmation
                    |
        canonical eval if earned
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

The diagram is illustrative, not a mandatory fixed sequence. Short runs may train to completion without an intermediate progress check. Long or expensive runs may alternate training and progress evaluation several times. Use evaluation when it can change the next action.

When the next move is not clear, use `$investigate-fastwam-problem`, then return to the loop.

## Evaluation purposes

- `progress_check`: cheap evidence from an intermediate checkpoint to control training compute;
- `candidate_screen`: cheap post-training or checkpoint-level evidence to decide whether broader evaluation is worthwhile;
- `confirmation`: broader evidence for generalization and retention;
- `diagnostic`: targeted evidence for a particular question;
- `canonical`: full fixed five-suite evidence required for promotion.

Do not confuse setup smoke tests with progress checks. Smoke tests validate infrastructure once. Progress checks are normal research evidence and may be used during candidate training.

## Success condition

The main target is reached when one canonically evaluated checkpoint simultaneously has:

- LIBERO-90 >= 90%
- LIBERO-Spatial >= 90%
- LIBERO-Object >= 90%
- LIBERO-Goal >= 90%
- LIBERO-Long / LIBERO-10 >= 90%

Reaching the threshold is a valid solution. Further experiments may continue when compute remains and there is a credible chance of improving LIBERO-90, the worst-suite margin, stability, or simplicity without breaking the constraints.
