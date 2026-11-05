# Reproduction: SPD on TMS 5-2 (arXiv:2506.20790, Table 1)

**Verdict: partially reproduced.** Feature directions are recovered (MMCS 0.991-1.000) and the target
is reconstructed, but subcomponents split each feature over 2-8 pieces (ML2R 0.15-0.8 vs paper 0.993)
at 7.5-25% of the paper's training budget. One diagnostic with a faithfulness warmup (from the current
reference config, not the paper) recovered linear2 exactly. TMS 40-10 not yet run.

## Claim
SPD finds one rank-one subcomponent per feature on TMS 5-2: MMCS 1.000±0.000, ML2R 0.993±0.002 on W,
C = 20, 40k steps, batch 4096 (number of seeds behind ± not stated).

## Setup
- Target: 5 features, 2 hidden, tied weights, trained per App A.4.1; learned pentagon (column norms
  1.12-1.14, MSE 1.83e-3).
- SPD (`src/spd_min/`, 10 unit tests): C = 20 for linear1 and linear2 (as the torch-oracle reference config);
  per-subcomponent CI MLP 1-16-1 GELU; leaky hard sigmoids (lower-leaky for masks, upper-leaky in minimality);
  S = 1; all loss coefficients 1; p = 1; Adam lr 1e-3, cosine decay.
- Metrics on 20k fresh samples: MMCS, ML2R (paper Eq. 9-10); `fullcos_min` (stricter: worst feature's best
  cosine between a whole subcomponent matrix and the ground-truth rank-one matrix); alive = CI > 0.1 on any input.
- Differences from the paper: 3k x 1024 steps (sweep) and 10k (diagnostics) vs 40k x 4096; CPU fp32;
  both linear1 and linear2 decomposed (the paper text describes W only).

## Results
Sweep over the importance-minimality coefficient, seeds 0-2, 3k steps, linear1 (mean ± std):

| coeff | MMCS | ML2R | fullcos_min | alive l1/l2 (of 20) | stochastic recon | wall (s) |
|---|---|---|---|---|---|---|
| 1e-3 | .9945 ± .0028 | .170 ± .059 | .618 ± .098 | 17.3 / 19.0 | 7.5e-4 | 137 |
| 3e-3 (paper) | .9943 ± .0026 | .225 ± .043 | .620 ± .098 | 15.7 / 16.0 | 1.26e-3 | 108 |
| 1e-2 | .9952 ± .0023 | .216 ± .048 | .603 ± .084 | 11.7 / 10.3 | 2.71e-3 | 119 |

Unmasked recon MSE 2e-6 to 1.4e-5 and faithfulness ~1e-5 in all 9 runs; no divergence. A higher coefficient
reduces alive subcomponents at a reconstruction cost but does not remove splitting.

Diagnostics (coeff 3e-3, 10k steps, one seed each):

| Run | Outcome |
|---|---|
| batch 1024, seed 0 | 10 / 8 alive; 2 subcomponents per feature |
| batch 1024, seed 1 | 7 / 5 alive; ML2R 0.51 / 0.16 |
| batch 4096, seed 0 (847 s) | 10 / 9 alive; ML2R 0.27 / 0.77 |
| batch 1024 + 200-step faithfulness warmup | linear2: 5 alive, MMCS 1.000, ML2R 0.983, fullcos_min 0.991, one subcomponent per feature. linear1: 5 alive, ML2R 0.80, one feature off (fullcos_min 0.61) |

In the no-warmup seed-0 run, the largest linear1 subcomponent mixes features 3 and 4 (columns at cosine
-0.82) and smaller ones correct the residual; the separated solution has lower minimality loss, so this
looks like a local optimum.

Raw numbers: `results_tms52_3k.json`, `results_tms52_10k.json`. Wall-clock is inflated (node load 5-14).

## Implications
- The library is usable for exploratory work; claims that depend on exact one-to-one recovery need the
  paper's full budget first.
- Faithfulness warmup matters; check its role and the CI-MLP init against the torch-oracle code.

## Next
1. Full budget (40k x 4096), 3 seeds, with and without warmup, on an s24 slice.
2. If splitting persists, diff CI-MLP init and loss details against torch-oracle.
3. TMS 40-10 (C = 200, coeff 1e-4, p = 2).
