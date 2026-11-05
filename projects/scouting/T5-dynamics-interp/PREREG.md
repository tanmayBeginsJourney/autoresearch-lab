# Preregistration: random-label gradient projection (RLGP), CPU kill-test

Written 2026-11-05 before any full run (only a 200-step timing smoke was run).

**Method.** At each full-batch step, compute the real gradient g and the gradient r of the same loss at the
same weights with the train labels randomly permuted (fresh permutation each step). Per parameter tensor,
remove the positive projection of g on r: g <- g - max(0, <g,r>/|r|^2) r. Feed g to AdamW.
Rationale: r is a label-agnostic "fit whatever labels" direction, so it estimates the memorization
direction at the current weights; removing it should leave the label-structured (generalizing) component.

**Testbed.** Modular addition p=59, 40% train split, 2-layer MLP on concatenated embeddings (d=64, h=128),
AdamW lr 1e-3, betas (0.9, 0.98), full batch, max 6000 steps. Seeds {0,1,2}, paired: same split, init and
label-permutation stream across arms. Grok step = first eval (every 50 steps) with val acc >= 0.95.

**Arms.** adamw wd=1 and wd=3 (baseline = the faster of the two per seed, i.e. tuned on val, favouring the
baseline); grokfast-EMA (alpha 0.98, lambda 2, wd=1); rlgp (wd=1); rlgp_rand control (project out a random
Gaussian direction, wd=1).

**Prediction (H1).** RLGP reaches grok in fewer *gradient evaluations* than the best AdamW arm (RLGP costs
2 evaluations per step): ratio best_adamw_steps / (2 x rlgp_steps) >= 1.3 in all 3 seeds, and RLGP steps <=
Grokfast steps in at least 2 of 3 seeds, and rlgp_rand does not show the same speedup (step ratio vs adamw
wd=1 < 1.2), and final val acc of RLGP >= 0.95.

**Kill.** Kill if the per-step speedup over best AdamW is < 1.3x in any seed, or if RLGP is slower than
Grokfast in 2 of 3 seeds, or if the random-direction control matches it. Weaker outcome: a per-step but not
per-evaluation win is reported as "not compute-positive" (kill for the paper goal).
