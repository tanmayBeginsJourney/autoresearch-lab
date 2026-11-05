# actbias
<!-- Current state only. Rewrite in place each session; history goes in journal.md. Keep under ~80 lines. -->

stage: literature   <!-- frame | literature | reproduce | gap | prereg | experiments | analysis | review | writing -->

## Problem
Does removing the positive mean of post-activation outputs ("activation bias") genuinely improve
late-stage training, and through which mechanism (conditioning, function-class restriction, or LR mismatch)?
Candidate project, not yet chosen; started from arXiv:2610.14500.

## Current understanding
- 2610.14500 read in full (card). Teacher-student evidence moderate; GPT-2 evidence weak (single run, ~0.01 nats).
- Missing-bias confound already addressed by the paper (App C.3).

## Open questions
- Does the gain survive per-arm LR tuning and >=3 seeds?
- Is adaptive shifting just NormFormer-style FFN centering? How does it compare to FFN-LN?

## Next steps
- Literature: NormFormer, Raiko 2012, Schraudolph 1998, mean-only BN, centering + Adam.
- CPU reproduction of the teacher-student with a per-arm LR sweep.

## Tried recently (last ~2 weeks)
