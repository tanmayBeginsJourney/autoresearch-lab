# actbias
<!-- Current state only. Rewrite in place each session; history goes in journal.md. Keep under ~80 lines. -->

stage: reproduce   <!-- frame | literature | reproduce | gap | prereg | experiments | analysis | review | writing -->

## Problem
Does removing the positive mean of post-activation outputs ("activation bias") genuinely improve
late-stage training, and through which mechanism (conditioning, function-class restriction, or LR mismatch)?
Candidate project, not yet chosen; started from arXiv:2610.14500.

## Current understanding
- 2610.14500 card written; GPT-2 evidence weak (single run, ~0.01 nats).
- Teacher-student CPU repro (repro/teacher-student/REPORT.md, exploratory): ReLU shift lowers final loss
  5-6.5% and survives per-arm LR tuning and a function-preserving bias control; GELU shows no gain;
  "slower early" holds only for GELU. Effect is real but small at toy scale.

## Open questions
- Does the gain survive per-arm LR tuning and >=3 seeds?
- Is adaptive shifting just NormFormer-style FFN centering? How does it compare to FFN-LN?

## Next steps
- Literature: NormFormer, Raiko 2012, Schraudolph 1998, mean-only BN, centering + Adam.
- Confirmatory teacher-student: more seeds, large size, finer LR grid, decaying schedule, GELU control arm.

## Tried recently (last ~2 weeks)
