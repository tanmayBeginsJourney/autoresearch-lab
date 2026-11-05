# Mask attacks on SPD decompositions: is plain PGD over-optimistic? (EXPLORATORY, G2)

Status: exploratory CPU prototype, 2026-11-05. Generates hypotheses only; no claim is supported.

## Setup
- Decompositions (spd_min, imported from `src/spd_min`, it works):
  - TMS 40-10, C=80 per layer, faith warm-up 500 + 700 SPD steps (batch 1024, lr 2e-3), imp coeff
    1e-3 and 3e-3, seeds 0,1 (`train.py`, metrics in `runs/`). **Under-trained**: SPD paper uses
    40k steps; MMCS to true features only 0.73-0.76. A 2500-step run did not fit the CPU budget, and
    imp 1e-2 collapsed (all CI dead) at 500 steps.
  - TMS 5-2, C=20, 3000 steps, imp 1e-3 and 1e-2, seeds 0,1: decompositions trained earlier by
    `spd_min/train_tms.py` (read-only from `$LAB_ROOT/tmp/spd-toy/runs`).
- Threat model (VPD App. A.3.4): m = g + (1-g) r, r in [0,1]^(2C) over both layers, g = clamp(CI).
  **Per-input** sources (VPD's reported eval shares sources across the batch, a weaker adversary).
  Objective: per-input MSE between masked and target output. Inputs: 2048 draws from the training
  distribution, all-zero inputs dropped (1780 for 40-10, 496 for 5-2). Score: mean over inputs of the
  best value seen so far.
- Budget: model evaluations per input; a forward+backward counts as 1 (favours gradient methods
  ~3x in FLOPs). Checkpoints 10...640.
- Attacks (`attack.py`): (a) PGD from a uniform random start, sign steps {0.01,0.03,0.1,0.3} and
  Adam {0.01,0.05,0.2}; the baseline is the best of these 7 per budget (oracle-tuned on the eval
  data, so generous to PGD). (b) APGD (Croce & Hein 2020 schedule: eta0=1, momentum 0.75, halving
  checkpoints, restart from best), 1 and 4 restarts. (c) Frank-Wolfe on the box; greedy vertex
  search (corners r in {0,1}, gradient-ranked single flips + all-positive-gain flip). Also each of
  PGD-sign-0.3 and vertex search with 8 equal-budget restarts. (d) random uniform and random vertex.
- "Ensemble" = per-input max over all 16 attacks at 640 evals; a lower bound on the true worst case.

Useful fact: the output is ReLU of a multilinear function of the masks, so the per-input MSE is
convex along each coordinate and its maximum over the box is attained at a vertex.

## Results (ratio to best single-run PGD at the same budget; 20 / 160 / 640 evaluations)

| model | stoch MSE | PGD@20 | PGD@640 | ensemble | APGD-1 | APGD-4 | FW | vertex | PGD x8 restarts | vertex x8 | random vertex |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40-10 imp1e-3 s0 | 2.2e-3 | 1.76e-2 | 1.79e-2 | 2.29e-2 | 1.03/1.03/1.04 | 1.03/1.04/1.15 | .91/.94/.95 | .91/1.03/1.08 | 1.00/1.07/1.17 | .91/1.13/1.26 | .32/.39/.44 |
| 40-10 imp1e-3 s1 | 2.3e-3 | 1.69e-2 | 1.72e-2 | 2.11e-2 | 1.03/1.03/1.04 | 1.03/1.03/1.13 | .92/.95/.96 | .92/1.03/1.08 | 1.00/1.06/1.14 | .92/1.11/1.21 | .33/.40/.44 |
| 40-10 imp3e-3 s0 | 1.1e-2 | 2.20e-2 | 2.28e-2 | 2.83e-2 | 1.06/1.10/1.13 | 1.06/1.12/1.19 | .88/.93/.94 | .83/1.04/1.10 | 1.00/1.06/1.13 | .83/1.04/1.16 | .60/.61/.62 |
| 40-10 imp3e-3 s1 | 1.1e-2 | 2.48e-2 | 2.67e-2 | 3.61e-2 | 1.12/1.17/1.19 | 1.12/1.17/1.27 | .80/.85/.88 | .74/1.02/1.13 | 1.00/1.07/1.20 | .74/1.05/1.25 | .54/.52/.53 |
| 5-2 imp1e-2 s0 | 1.3e-2 | 1.07e-1 | 1.07e-1 | 1.90e-1 | .96/.96/.97 | .96/.97/1.25 | .85/.93/.94 | .87/1.12/1.12 | 1.00/1.17/1.46 | .87/1.40/1.76 | .42/.60/.75 |
| 5-2 imp1e-2 s1 | 1.1e-2 | 9.95e-2 | 1.02e-1 | 1.76e-1 | .95/.94/.94 | .95/.94/1.21 | .87/.89/.90 | .87/.97/.97 | 1.00/1.10/1.43 | .87/1.29/1.72 | .43/.61/.73 |
| 5-2 imp1e-3 s0 | 3.4e-3 | 5.68e-2 | 5.68e-2 | 7.49e-2 | 1.02/1.03/1.03 | 1.02/1.03/1.17 | .93/1.02/1.02 | .94/1.08/1.08 | 1.00/1.12/1.22 | .94/1.17/1.31 | .37/.56/.67 |
| 5-2 imp1e-3 s1 | 3.7e-3 | 6.57e-2 | 6.59e-2 | 9.66e-2 | .96/.95/.95 | .96/.95/1.25 | .87/.90/.91 | .89/1.15/1.15 | 1.00/1.14/1.36 | .89/1.29/1.45 | .35/.55/.68 |

Full curves (10...640), p90 values and per-attack fraction of the ensemble: `results/*.json`;
`python summarize.py` prints the table above.

Curve shape: single-run PGD saturates by 20-40 evaluations (PGD@20 is within 0-8% of PGD@640) at
a local maximum. Single-run PGD reaches 74-82% of the ensemble mean on 40-10 and 56-76% on 5-2.
Restarted methods keep climbing through 640 evaluations, so the true worst case is above the ensemble.

## Verdict
- Hypothesis "(b)/(c) beat (a) substantially at equal budget": **partly true, for a different reason
  than expected.** At VPD-like budgets (~20 evals), no method beats tuned PGD by more than 12%;
  FW and vertex search are 6-26% *worse* there. Single-run APGD gains 3-19% (40-10) and loses up
  to 6% (5-2).
- What matters is **restarts / diversity of starting points**: the per-input landscape has many local
  maxima. At 640 evals, vertex search with 8 restarts finds 1.16-1.26x (40-10) and 1.31-1.76x (5-2)
  the worst-case MSE of the best single-run PGD; PGD with 8 restarts gets 1.13-1.46x. Vertex x8
  beats PGD x8 in all 8 models at 640 (by 1.03-1.21x).
- So "PGD for N steps" understates the worst case mostly because it is a single local search;
  spending the same budget on restarts (vertex search ideally) gives 1.2-1.8x worse cases in these
  toys. That counts as a positive result for G2, but a modest one, and the "better optimizer" part
  (APGD, FW) is close to null.
- Random search is far weaker (0.3-0.75x): gradient masking does not seem to be an issue here.

## Caveats (why this is not yet evidence for a paper)
- MSE on TMS, not KL on an LM; the vertex property holds for this ReLU/MSE toy and need not hold in
  transformers. Under-trained 40-10 decompositions. One attack seed; input-level SEs not reported
  (n=496-1780 inputs, ratios consistent across 2 seeds x 2 imp levels x 2 models).
- The eval-count budget favours gradient methods 3x in FLOPs; at equal FLOPs vertex search (mostly
  forward-only candidate evals) would look relatively better.
- VPD's numbers (KL 0.83@20, 25@160) keep growing with steps; our single-run PGD saturates by 40 evals.
  That gap (LM vs toy, batch-shared vs per-input sources) is unexplained.

## Next steps (if G2 continues)
1. Repeat on a decomposition trained to convergence and on a 1-2 layer transformer with KL (GPU slice).
2. Budget-matched comparison in FLOPs; more restarts (16-64) to estimate how far the ensemble is from
   the true max; check whether the restart gain shrinks for VPD-trained (adversarially trained) PDs.
3. Test whether training with a restarted/vertex adversary improves ground-truth recovery (G2 + idea 3).
