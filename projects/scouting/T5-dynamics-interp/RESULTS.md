# T5 kill-test result: RLGP (random-label gradient projection)

**Verdict: not supported; leaning kill.** The testbed was miscalibrated: no arm reached 95% val accuracy
within the 6000-step cap, so the preregistered grok-step ratio is undefined. On the only usable proxy
(val accuracy at the cap, paired seeds) RLGP does no better than plain AdamW and costs 2x the backward
passes. The preregistered bar was a 2.6x per-step speedup over AdamW (1.3x per gradient evaluation).
RLGP showed no lead in any seed, so it is unlikely to clear that bar.

## Numbers (p=59, 40% split, MLP d=64 h=128, AdamW lr 1e-3, wd 1, full batch, 6000 steps)

| Seed | AdamW val acc at cap | RLGP val acc at cap | Grokfast-EMA val acc at cap | Train acc (all arms) |
|---|---|---|---|---|
| 0 | 0.027 | 0.025 | 0.028 | 1.00 by step ~650-1050 |
| 1 | 0.018 | 0.016 | 0.014 | 1.00 |
| 2 | 0.053 | 0.033 | 0.014 | 1.00 |

Chance is 1/59 = 0.017. All arms memorize by about step 1000. Val accuracy then rises slowly from below
chance, the usual pre-grokking pattern. Over steps 850-5650 the RLGP val curve follows AdamW within
about 0.01 in every seed. The wd=3 and random-direction-control arms were not run (CPU budget, load
average 10-16 on 4 shared cores). Grokfast also failed to grok within the cap, which confirms the problem
is the step budget and learning rate, not one arm.

## What went wrong and what it implies
- Calibration error: lr 1e-3 with a 6000-step cap is too short for this MLP at a 40% split. Before
  running the arms I should have checked that the baseline groks.
- Likely mechanism for the null (not tested): once train loss is about 0, the real gradient is tiny. The
  random-label gradient is large and nearly orthogonal to it, so the positive projection removes almost
  nothing. Logging cos(g, r) during training would test this directly.

## What would settle it (cheap)
Rerun with lr 3e-3, a 50% split and a 20k-step cap, all 5 arms, 3 seeds, plus logging of cos(g, r).
That is about 1 slice-hour, or 30 CPU-minutes on an idle machine. Kill if RLGP's grok step is not
at least 2.6x shorter than tuned AdamW's.

## NeurIPS-scale version (only if the rerun passes)
RLGP as a label-agnostic memorization filter. Test it on noisy-label CIFAR/ImageNet subsets against
small-loss, ELR and AND-mask baselines, and on LM fine-tuning memorization (canary extraction) at matched
compute. Show mechanistically which circuits RLGP prevents from forming. Given this null, the expected
value is low. The scouting conclusion is that grokking-centred dynamics work is saturated. A NeurIPS-grade
original direction in this area probably needs a testbed beyond modular arithmetic.

Files: `IDEAS.md` (9 hypotheses and novelty audit), `PREREG.md`, `rlgp.py`. Raw JSON is in the session
scratchpad (not kept).
