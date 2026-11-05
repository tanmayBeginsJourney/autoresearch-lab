# Metric calibration on TMS 40-10 (EXPLORATORY, 2026-11-05)

Question (gaps.md G2, faithfulness idea 3): which faithfulness metric best predicts ground-truth
mechanism recovery? Exploratory, one target model, CPU, about 35 min. It generates hypotheses and
supports no claims.

## Setup
- Target: TMS 40-10 (`spd_min.tms.train_tms`, seed 0, 6k steps, batch 2048). 39 of 40 features are
  represented (one column has norm 0.02), so ground truth (GT) uses those 39.
- Decomposition: `spd_min` SPD on `linear1` only (C=50 unless noted; the layerwise loss is off
  because with one layer it equals the stochastic loss). linear2 is not decomposed. Not the paper setup.
- GT recovery (`run.py:ground_truth`): `gt_fullcos` = mean over features of the max full-matrix
  cosine to `W[:,j] e_j^T` (primary); `gt_frac_recovered` = distinct subcomponents with cosine > 0.9;
  `gt_onehot_acc` = one-hot input activates exactly the matched subcomponent (CI > 0.1);
  `gt_mmcs_col` (paper MMCS); `gt_spurious_alive`.
- Candidate metrics on 4096 held-out samples: stochastic-mask MSE (mean of 8 draws), PGD-adversarial
  sources at 10 and 40 steps (sign-gradient, step 0.1, random init), rounded masks (CI > 0.5),
  CI-masked, unmasked, CI L0, faithfulness.
- Population, n=34 (`results/*.json`):
  - natural SPD, n=8: 4000 steps, importance coefficient in {3e-4, 1e-3, 3e-3 x2 seeds, 1e-2, 3e-2},
    C=20, plus a 1000-step run.
  - natural SPD runs mixed after training, n=6: subcomponents of the two 3e-3 runs mixed by
    `U R, R^T V`, with t in {0.1, 0.3, 1.0}.
  - oracle_mixed, n=20: U,V start at the true mechanisms plus 10 near-zero spares, mixed by
    `R = exp(t A)` with t in {0, .05, .1, .2, .35, .5, .75, 1, 1.5, 3} and 2 seeds; then only the CI
    MLPs are trained (1500 steps, imp 3e-3). This is the only family with good decompositions.
  - Planned but not finished: 10k-step, C=100, 0- and 300-step runs, and second seeds. The CPU was
    shared and a 4000-step run took about 7 min, so I stopped the sweep.

## Results
Spearman rho with `gt_fullcos`, 95% bootstrap CI (2000 resamples). Metrics are losses, so negative
is good. L0 is not a loss.

| metric | all (n=34) | oracle_mixed (n=20) | natural+mixed (n=14) |
|---|---|---|---|
| stochastic recon | -0.90 [-0.97,-0.78] | -0.99 [-1.00,-0.93] | -0.77 [-0.96,-0.31] |
| PGD-10 recon | -0.91 [-0.97,-0.78] | -0.99 [-1.00,-0.95] | -0.86 [-0.96,-0.54] |
| PGD-40 recon | -0.92 [-0.97,-0.80] | -0.99 [-1.00,-0.93] | -0.85 [-0.96,-0.50] |
| rounded-mask recon | -0.94 [-0.98,-0.84] | -0.98 [-1.00,-0.90] | -0.77 [-0.95,-0.28] |
| CI-masked recon | -0.91 [-0.98,-0.78] | -0.99 [-1.00,-0.90] | -0.77 [-0.95,-0.28] |
| unmasked recon | -0.55 [-0.79,-0.20] | -0.04 [-0.48,+0.44] | -0.61 [-0.92,+0.05] |
| faithfulness | -0.60 [-0.82,-0.28] | -0.20 [-0.65,+0.31] | -0.61 [-0.92,+0.04] |
| CI L0 | +0.90 [+0.76,+0.98] | +0.99 [+0.92,+1.00] | +0.76 [+0.31,+0.96] |

Paired bootstrap of rho(metric) - rho(stochastic): every CI includes 0 or sits within 0.05 of it.
In natural+mixed, PGD-10 gives -0.10 [-0.40,+0.12]. Full tables are in `summary.json` and the
output of `analyze.py`.

## What this shows
1. All mask-based recon metrics (stochastic, PGD-10, PGD-40, rounded, CI-masked) predict GT
   recovery well, and on this population I cannot tell them apart. PGD-40 adds nothing over PGD-10.
   The hoped-for result ("metric X predicts recovery and the commonly reported Y does not") is
   **not supported**. The data neither rejects nor confirms it, because the population is too
   easy: the families separate cleanly on every mask metric.
2. Unmasked recon and faithfulness carry no information within oracle_mixed. This holds **by
   construction**, because mixing preserves `UV = W`. It confirms they are not recovery metrics, but
   it is not a finding.
3. One hint worth following: the natural run with importance coefficient 3e-4 (dense masks, L0 5.2,
   `gt_fullcos` 0.40, 0/39 features recovered) has lower stochastic and CI-masked recon than the
   exact ground-truth decomposition (0.00078 vs 0.00093). PGD-10/40 (0.0047 vs 0.0025) and rounded
   recon still rank the ground truth first. Stochastic recon may therefore reward under-sparsified
   decompositions when L0 varies, while adversarial recon does not. This rests on one decomposition.
4. Natural SPD never recovered a single feature here (`gt_frac_recovered` = 0 for all 14). The cause
   is probably 4000 steps with no faithfulness warmup. The paper reports recovery on TMS 40-10 with
   longer training. So the good end of the population is synthetic (oracle_mixed). Its CI functions
   were trained, but its U,V were never found by SPD.
5. L0 tracks recovery inside oracle_mixed (higher is better: mixed subcomponents lead CI to switch
   off). Across importance coefficients it is non-monotone, so it is unusable as a recovery metric.

## Caveats
One target and one seed of it. Only linear1 is decomposed. The 6 mixed natural runs are near-copies
of 2 parents (pseudo-replication). oracle_mixed has 2 seeds per mixing level. The bootstrap treats
decompositions as independent. Results are pooled across families. No natural run reached good recovery.

## Next step, if pursued (decision: does adversarial recon beat stochastic recon as a recovery metric?)
Build a population where the metrics can disagree. Run natural SPD to convergence (faith warmup,
more than 20k steps, decomposing both layers) over a dense importance-coefficient grid with 3
paired seeds, so that recovery varies *within* natural runs and L0 varies independently of recovery.
Add a test of point 3: decompositions with dense CI at matched recovery. About 30 CPU-h, or run
it on `s24`.

## Files
`run.py` (one decomposition, then scoring), `analyze.py` (Spearman, bootstrap, paired test),
`jobs.txt` and `jobs2.txt` (configs; jobs.txt was only partly run), `results/` and `summary.json`.
Target and logs are in the session scratchpad and are not kept.
