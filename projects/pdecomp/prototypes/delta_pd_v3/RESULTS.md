# Delta-PD v3: regimes where rotated SVD should fail (EXPLORATORY)

Status: exploratory, 3 seeds per regime, not preregistered; supports no claim. Code `delta_pd_v3.py`
(toys, methods, metrics; reuses v2's ICA/varimax/recovery and `src/spd_min`), `summarize.py`;
per-run json in `results/`, aggregates in `summary.json` / `summary.txt`, run list in `jobs.txt`.
Success criterion was written to `CRITERION.txt` before any run: Delta-PD must beat the best of
SVD+ICA / SVD+varimax (k = true, true+2) by >= 0.15 in recovery or specificity, ahead in all 3 seeds.

## Verdict
- **A (overcomplete, 6 changes in a rank-3 dW1): criterion met on recovery, by construction.**
  Hungarian cos Delta-PD 0.53 / 0.73 / 0.99 (seeds 0-2) vs best rotated 0.28 / 0.28 / 0.28.
  Specificity (restore - collateral) 0.33 / 0.47 / 0.92 vs 0.25, so not >= 0.15 in seed 0.
  Only 1 of 3 seeds recovers all 6 changes; the others merge or split some. Rotations are capped by
  construction: they live in the 3-dim column space (best possible per-piece cos 0.71, and only 3
  components exist for 6 changes), so this shows Delta-PD *can* escape the cap, not that it reliably does.
- **B (backdoor via ReLU conjunction): no win.** Removing the backdoor-matched components: Delta-PD
  restore 0.98-0.99, clean damage 0.01-0.02, benign collateral 0.01-0.02; SVD+ICA 1.00 / 0.00 / 0.00
  in all seeds. Varimax is worse (restore 0.55-0.85, clean damage up to 0.82). Backdoor piece cos:
  Delta-PD 0.84-0.92, ICA 0.82. Dormancy: Delta-PD's backdoor gates fire on 0% of clean samples
  (gate > 0.1) and are 1.00 on trigger probes when the trigger is in the decomposition data. Trained
  trigger-free, the backdoor gates stay at 0 on clean data, but fire on trigger probes in only some
  seeds (mean 0.51 ± 0.30), and 13 ± 2 components are dormant of which 11 ± 2 match the backdoor:
  a dormant-component detector has false positives and is not demonstrated. Any method's components
  can be screened the same way (ablate, measure clean output change); ICA's backdoor components also
  give clean damage 0.00, so dormancy is not unique to Delta-PD.
- **C (cross-layer, incl. a coordinated edit of a base neuron): no win.** Hungarian cos Delta-PD
  0.93 ± 0.02 vs ICA 1.00; restore 0.86 vs 0.99; collateral 0.00 vs 0.01. Cross-layer pairing:
  ICA by inner-activation correlation 1.00, Delta-PD by gate co-activation 0.89 ± 0.16.

## Constructions (f = W2 ReLU(W1 E x + b1), x_i ~ Bern(p) U[0,1], E and b1 fixed, exact pieces)
- A: 6 features in d=3 along icosahedron axes (|cos| = 0.447), p = 0.05. Spare neurons 6-11 have
  bias -0.5 and already write u_k; change k sets W1 row 6+k = e_k^T. dW1 = sum_k e_{6+k} e_k^T has
  rank 3, each change is separable by behaviour (fires only when feature k is present; the bias
  filters interference). TMS-style superposition, but of *changes*. dW2 = 0.
- B: E random orthogonal (removes the privileged input basis). Benign b1 (spare neuron 8 reads x2,
  writes u_b1), benign b2 (base neuron 4 also writes u_b2), backdoor (spare neuron 9, bias -1, reads
  x2 + x7, writes 2 u_t; cos(u_t, u_b1) = 0.5). Trigger x7 is also an ordinary feature.
- C: two spare-neuron adds (reads x1 / x3, writes u0 / u1) and one coordinated edit: base neuron 5
  also reads x6 (W1) and also writes u2 (W2); x6 -> u2 needs both. Pairwise cos(u_i) = 0.5.

## Methods and metrics
Delta-PD: spd_min with base_weights, C = 10 per changed matrix, coef 1e-3, 300 warmup + 4000 steps
(2x v2), reaches recon <= 5% of change MSE by step 500-1000; final CI-masked recon 0.4-2.9% of
change MSE. Scored with alive components (gate > 0.1) plus never-gated components of norm > 0.1;
alive-only scores are within 0.02 in A and B, lower in C (restore 0.80 vs 0.86) and in trigger-free B
(all-change cos 0.72 vs 0.81, restore 0.69 vs 0.86, because the dormant backdoor comps drop out). Full PD of W_ft (C = 2H). SVD, SVD+varimax, SVD+ICA (k capped at
min(d_out, d_in), i.e. 3 in A), rank-k LoRA refit (fails through dead spare neurons, as in v2: cos <=
0.2), random rank-one control (restore -0.24 to -0.04) and oracle revert (restore 1.00).
Restore = 1 - |f_rev - f_(ft minus change)| / |f_ft - f_(ft minus change)| on the change's probes;
collateral on other changes' probes; clean damage on 2048 trigger-free natural samples.

## Limits
Hand-set toys, 3 seeds (sd is not a CI), one coefficient, no tuning. A's win is guaranteed for the
baselines by the rank argument; Delta-PD's seed variance there (0.53-0.99) is the real open issue.
All 12 runs were rerun with the alive-only re-scoring; the rerun reproduces the first sweep exactly.
Next: more seeds and a coefficient sweep in A; a rotated-feature overcomplete case where neither
input nor output side is axis-aligned; check whether Delta-PD's A failures are optimisation (restarts).
