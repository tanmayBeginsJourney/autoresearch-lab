# G4 prototype: activation function vs SPD decomposability (EXPLORATORY, 2026-11-05)

Status: exploratory CPU toy. Its job is to generate hypotheses; it supports no claim.
**Verdict: no consistent G4 ordering on sparsity or robustness.** One weak hint is worth a follow-up:
on the input matrix, ground-truth recovery runs ReLU > GELU > SiLU. That hint is confounded by
how long each target was trained (see Confounds).

## Setup
- Target task: `y = W_out act(W_in x + b) + b_out`, with 16 sparse features, x_i ~ U[-1,1] w.p. 0.08,
  label y_i = |x_i|, d_mlp = 16. Bilinear: `h = (W_in x + b1) * (W_in2 x + b2)`, which gives it 3 matrices vs 2.
- Loss matching: every activation trains at width 16 for 4000 steps. L* is the worst final validation
  loss (SiLU in every seed). Each model is then retrained with early stopping at the first eval
  (every 50 steps) where validation loss <= L*.
- SPD: `src/spd_min` (unit tests pass). C=40 per matrix, p=1, beta1=beta2=1, 200 faithfulness warmup
  steps, 1500 steps, lr 2e-3, batch 256, same SPD init seed in every arm. coeff_imp is swept over
  {1e-3, 3e-3}. The 1e-2 setting was run for seed 0 only (no bilinear) and collapsed to L0 ~ 0.
- Metrics on 2048 held-out inputs:
  - CI-L0 (CI > 0.1)
  - stochastic and adversarial recon MSE, both divided by the target's mean y^2. Adversarial means
    PGD on the mask randomness r in [0,1]: 20 sign steps, alpha 0.1.
  - alive subcomponents
  - ground-truth recovery: mean over features of max_c |cos(U_c V_c^T, W_in[:,i] e_i^T)|, input matrix
    only (for bilinear, the worse of its two input matrices)
- 3 seeds (0,1,2), paired across arms. CIs are 95% t-intervals, n=3.
- Run: `run.py`, `launch.sh` and `relaunch.sh`; table: `aggregate.py` writes `results/summary.md`.
  An external kill stopped the first wave before its bilinear arms. `relaunch.sh` reran only the
  missing arms with identical deterministic training (L* reproduced exactly).
- The x^2 target, planned as a check against target-function bias, was dropped for lack of time.

## Results (mean ± 95% CI, 3 seeds; full table in results/summary.md)
| coeff | act | task val | steps | L0/matrix | L0/feat (W_in) | stoch rel | adv rel | GT recov |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | relu | 0.0116 | 933 | 1.52±0.68 | 1.02±0.22 | 0.080±0.034 | 0.67±0.65 | 0.62±0.18 |
| 1e-3 | gelu | 0.0116 | 1530 | 1.93±0.20 | 1.35±0.34 | 0.063±0.014 | 0.46±0.06 | 0.49±0.06 |
| 1e-3 | silu | 0.0116 | 3970 | 1.73±1.4 | 1.35±0.15 | 0.069±0.028 | 0.95±0.93 | 0.42±0.04 |
| 1e-3 | bilinear | 0.0097 | 200 | 1.26±0.18 | 0.98±0.07 | 0.117±0.023 | 1.03±0.38 | 0.53±0.05 |
| 3e-3 | relu | 0.0116 | 933 | 0.65±0.08 | 0.50±0.12 | 0.164±0.040 | 1.45±1.3 | 0.61±0.27 |
| 3e-3 | gelu | 0.0116 | 1530 | 0.70±0.05 | 0.67±0.06 | 0.160±0.050 | 1.72±0.46 | 0.50±0.05 |
| 3e-3 | silu | 0.0116 | 3970 | 0.69±0.08 | 0.63±0.05 | 0.143±0.14 | 1.97±2.8 | 0.44±0.03 |
| 3e-3 | bilinear | 0.0097 | 200 | 0.54±0.03 | 0.38±0.05 | 0.301±0.068 | 1.48±0.27 | 0.42±0.07 |

Per-seed check of "max(ReLU, bilinear) < min(GELU, SiLU)":
- L0/matrix: holds in 2/3 seeds at both coefficients.
- Adversarial recon: holds in 0/3 seeds at 1e-3 and 1/3 at 3e-3.
- Stochastic recon: holds in 0/3 seeds at both coefficients.

## Reading
- Sparsity trades against reconstruction. No activation dominates.
  - Bilinear has the lowest L0 per matrix but about 2x worse stochastic recon at each coefficient.
    It sits at a different point on the L0/recon curve, not on a better curve.
  - ReLU vs GELU/SiLU: the L0 and recon differences fall inside the CIs. Fixed-coefficient
    comparisons cannot separate "sparser" from "a different trade-off". The right test is a
    Pareto frontier (L0 at matched stochastic recon), which this sweep is too coarse to give.
- No arm's decomposition is adversarially robust: adversarial recon error is 0.5-2x the output's mean
  square. This most likely reflects 1500 SPD steps, not the activations.
- GT recovery on W_in: ReLU > GELU > SiLU in 6/6 (seed, coeff) cells. One cell is near-tied
  (s2, 3e-3: 0.49/0.48/0.43). This is the only consistent ordering, and it rests on one task and
  one metric.

## Confounds (why none of this is evidence for or against G4)
1. **Training length.** Matching loss by early stopping trains each arm for a very different number
   of steps: SiLU ~3970, GELU ~1530, ReLU ~933, bilinear 200. Less-trained weights may be more
   column-aligned, which alone could produce the GT-recovery ordering. Fix: match loss by width,
   or report the decomposition along a training trajectory.
2. **Bilinear loss not matched.** Bilinear lands 16% below L* (0.0097 vs 0.0116) because it beats L*
   before the first eval it reaches it at. This breaks the 5% criterion. Bilinear also has 3
   decomposed matrices and 1.5x the parameters.
3. **Target-function bias.** |x| = relu(x) + relu(-x) is native to ReLU. x^2 would favour bilinear.
   This run does not test the planned x^2 check.
4. **Hyperparameter fairness.** One SPD configuration and two coefficients were tuned for nothing in
   particular. The comparison needs Pareto curves over coeff_imp, more SPD steps (adversarial recon
   shows they are undertrained), and C chosen per arm.
5. **Metric choice.** The GT metric assumes the per-feature column is the "true" component. A ReLU
   model can legitimately split a column into sign-specific pieces, and the metric would score that
   as worse.

## Next step (if pursued)
Match loss by width at a fixed step count. Run both |x| and x^2 targets. Sweep coeff_imp over 5
values with 5000 SPD steps and compare L0 at matched stochastic recon. Expected cost: ~1 CPU-hour
on an idle node. Kill G4 at toy scale if the frontiers overlap.
