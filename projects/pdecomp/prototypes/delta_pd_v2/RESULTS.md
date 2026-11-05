# Delta-PD v2: exact planted deltas vs SVD, rotated SVD, LoRA, full PD (EXPLORATORY)

Status: exploratory, 3 seeds per cell, not preregistered; nothing here supports a claim. Code:
`delta_pd_v2.py` (toy, methods, metrics; uses `src/spd_min`), `summarize.py`; per-run json in
`results/`, aggregates in `summary.json`, run list in `jobs.txt`.

## Verdict
Delta-PD beats plain SVD of dW when two equal-norm changes share a matrix (Hungarian cos 0.99 vs
0.43, restore 0.98 vs 0.24, collateral 0.01 vs 0.26). The margin shrinks as the norms diverge and is
gone at ratio 3 (0.85 vs 0.89). **It does not beat rotated SVD.** ICA or varimax inside the top-k SVD
subspace recovers every planted change exactly (cos 1.00, restore 0.99-1.00, collateral <= 0.01) at
every norm ratio. With drift added they tie with Delta-PD or beat it slightly. So in this toy, the
v1 "win over SVD" comes from SVD's orthogonality constraint, and a linear rotation removes it. By
the kill criterion as the lab lead amended it, "Delta-PD separates changes better than SVD-based
methods" is dead here. What the data still favour Delta-PD on: it picks its own number of components
(0 alive on pure drift), and its gates are sparse per input, unlike SVD with a learned gate. Neither
point is a recovery or specificity win.

## Setup
- Toy f(x) = W2 ReLU(W1 x), 12 sparse features (p=0.1, U[-1,1]), 28 neurons. Base hand-set:
  neurons 2i, 2i+1 = ReLU(+-x_i), y_i = |x_i|; neurons 24-27 spare (zero).
- Fine-tune = base + hand-set rank-one pieces (exact ground truth, unlike v1 where Adam leaked ~50%):
  c1 add (W1: spare neuron 24 reads x0; W2: it writes y11), c2 and c3 in W2 (neurons for x3 and x5
  also write 0.6 u2 and 0.6 r u3, with cos(u2,u3) = 0.5: disjoint input supports, overlapping
  outputs), c4 reroute in W1 (neuron 14 reads x8 instead of x7). With r = 1 the right singular
  vectors of dW2 are (e6 +- e10)/sqrt2, which is maximal mixing.
- Arms: exact (r in {1, 1.5, 3}); exact+drift (dense noise, ||noise||_F = 0.5 ||pieces||_F per
  matrix); drift only (||noise||_F = 0.05 ||W_base||_F, change MSE 1.4-1.7e-4).
- Methods: Delta-PD (spd_min, base_weights=W_base, C=8 per matrix, 300-step faithfulness warmup,
  2000 steps, batch 256, lr 2e-3, p=1). Full PD of W_ft (C=40, same settings, coef 1e-3 only).
  SVD of dW (k = number of planted pieces per matrix: 2 and 3, an oracle choice). SVD+varimax (rotate
  input-side loadings). SVD+ICA (FastICA on inner activations V_k^T a over 4096 natural inputs).
  SVD+gate (top-k SVD directions frozen as Delta-PD subcomponents, only the CI gates trained).
  Rank-k LoRA distilled to f_ft (1500 steps), scored as SVD of BA and as raw b_i a_i^T.
  Controls: a random rank-one revert of matched norm, and the oracle revert of the planted piece.
- Metrics, per planted change: Hungarian |cos| of each planted rank-one piece to the method's
  pieces in the same matrix (PD: alive components only, CI max > 0.1); cluster cos (sum of all
  components whose best match is that piece, |cos| > 0.2). Revert the matched components from W_ft,
  then restore = 1 - |f_rev - f_base| / |f_ft - f_base| on the change's support probes; collateral
  = |f_rev - f_ft| / |f_ft - f_base| on the other changes' probes; untouched dmg on probes of
  features no change uses.

## Main table (mean ± sd over seeds 0-2, coef 1e-3; "eq" = the c2,c3 pair in W2)
| arm, r | method | Hung. cos eq | restore eq | collateral eq | Hung. cos all 4 | restore all 4 |
|---|---|---|---|---|---|---|
| exact, r=1 | Delta-PD | 0.99 ± 0.00 | 0.98 ± 0.01 | 0.01 ± 0.00 | 0.93 ± 0.01 | 0.93 ± 0.02 |
| | SVD | 0.43 ± 0.03 | 0.24 ± 0.03 | 0.26 ± 0.01 | 0.68 ± 0.02 | 0.62 ± 0.01 |
| | SVD+varimax | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| | SVD+ICA | 1.00 ± 0.00 | 0.99 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.00 |
| | LoRA (SVD of BA) | 0.47 ± 0.04 | 0.33 ± 0.01 | 0.23 ± 0.03 | 0.43 ± 0.06 | 0.33 ± 0.05 |
| | full PD | 0.39 ± 0.03 | -0.90 ± 0.24 | 0.03 ± 0.02 | 0.44 ± 0.01 | -0.56 ± 0.16 |
| | random rank-1 | - | 0.01 ± 0.02 | 0.17 ± 0.04 | - | -0.05 ± 0.04 |
| exact, r=1.5 | Delta-PD | 0.95 ± 0.04 | 0.98 ± 0.02 | 0.01 ± 0.01 | 0.91 ± 0.03 | 0.91 ± 0.02 |
| | SVD | 0.71 ± 0.02 | 0.47 ± 0.01 | 0.23 ± 0.02 | 0.83 ± 0.02 | 0.73 ± 0.01 |
| | SVD+varimax / ICA | 1.00 / 1.00 | 1.00 / 0.99 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 0.99 |
| exact, r=3 | Delta-PD | 0.85 ± 0.16 | 0.85 ± 0.20 | 0.05 ± 0.07 | 0.86 ± 0.09 | 0.86 ± 0.11 |
| | SVD | 0.89 ± 0.01 | 0.68 ± 0.01 | 0.15 ± 0.02 | 0.94 ± 0.01 | 0.84 ± 0.01 |
| | SVD+varimax / ICA | 1.00 / 1.00 | 1.00 / 0.99 | 0.00 / 0.01 | 1.00 / 1.00 | 1.00 / 0.99 |
| exact+drift, r=1 | Delta-PD | 0.85 ± 0.04 | 0.64 ± 0.07 | 0.08 ± 0.03 | 0.87 ± 0.05 | 0.72 ± 0.03 |
| | SVD | 0.38 ± 0.02 | 0.18 ± 0.10 | 0.17 ± 0.05 | 0.63 ± 0.03 | 0.47 ± 0.05 |
| | SVD+varimax | 0.93 ± 0.01 | 0.62 ± 0.00 | 0.06 ± 0.00 | 0.95 ± 0.00 | 0.71 ± 0.01 |
| | SVD+ICA | 0.93 ± 0.01 | 0.62 ± 0.00 | 0.04 ± 0.00 | 0.95 ± 0.00 | 0.71 ± 0.01 |
| | oracle revert | - | 0.58 ± 0.01 | 0.02 ± 0.01 | - | 0.64 ± 0.02 |

The oracle checks pass: restore 1.00 and collateral 0.00 in the exact arm. In exact+drift the oracle
restores only 0.58-0.64, because the drift also moves behaviour on these probes and is not reverted.
Untouched damage is 0.00 for every delta method in the exact arm and 0.02-0.08 with drift. With SVD
k set to oracle + 2 in exact+drift, varimax and ICA are unchanged (cos 0.94 / 0.93 eq, restore 0.62).
Coefficient sweep (exact, r=1): Delta-PD eq cos is 0.92 ± 0.08 at 3e-4, 0.99 ± 0.00 at 1e-3 and
0.92 ± 0.09 at 3e-3. So 1e-3 is the best of the three, and the other two each had a seed that
partly merged or split a change.

## Size, gate sparsity, cost (exact arm)
| method | alive comps (l1+l2) | active comps on x3=1 / x5=1 probe | l2 comps active on both | steps to recon <= 5% of change MSE |
|---|---|---|---|---|
| Delta-PD 1e-3 (r=1) | 6.3 ± 0.5 (5 planted pieces) | 1.0 / 1.0 | 0.0 | 500, 600, 700 (after 300 warmup) |
| Delta-PD 3e-4 / 3e-3 | 9.0 ± 0.8 / 5.7 ± 0.5 | 1.7 / 2.0, 1.0 / 1.3 | 0.7 / 0.3 | 400-500 / not reached (final 0.09) |
| full PD (C=40) | 49.7 ± 0.5 | 4.0 / 4.3 | 0.0 | not reached in 2000 (final 0.20) |
| SVD+gate (r=1) | 5 (fixed) | 2.7 / 2.7 | 2.7 | final recon 0.16 ± 0.12 of change MSE |

- Drift-only arm: Delta-PD has 0 alive components in all 3 seeds. Plain SVD returns a flat spectrum
  (0.115, 0.099, 0.086, ...) and has no way to say "no mechanism".
- Wall clock: Delta-PD 40-65 s, full PD 43 s, but sd is about 30 s on a node at load 20-25, so
  timing is not interpretable. SVD, varimax and ICA take under 1 s; the LoRA refit about 6 s.
- Full PD: its matched components merge the change with the base column (restore -0.90 means
  reverting them also deletes the base |x_j| computation). This repeats v1.
- LoRA cannot learn c1 through spare neuron 24 (ReLU(0) gives zero gradient through B = 0); it routes
  the change through other neurons instead (output recon 0.00 of change MSE, cos ~0 for c1). Matching
  the outputs does not identify the weights.

## Where Delta-PD does and does not beat the strengthened baselines
- Does: beats plain SVD and LoRA at r <= 1.5. Uses one gate per input on the changed inputs where
  SVD+gate needs 2.7, because mixed SVD directions cannot be gated apart. Has no free k. Is silent
  on pure drift.
- Does not: ties with or loses to SVD+ICA and SVD+varimax on recovery, restore and collateral in
  every arm. Matches plain SVD at r=3 and has the higher seed variance there. ICA separates c2 and c3
  because their input supports are disjoint, so their inner activations are independent sparse
  sources, the same signal the CI gates use. Varimax is also helped by the privileged neuron basis
  (v = e6, e10); ICA does not rely on that and gives the same result.
- Untested: settings where a linear rotation of the dW subspace should fail. These are more changes
  than rank(dW) (overcomplete), changes that share an input direction but are gated by context
  (backdoor conjunctions), one change spread across matrices, and real fine-tunes where the useful
  change is small next to drift. Those are the only places left for a Delta-PD advantage.

## Prior work to engage (named by the lab lead; not read or verified in this session)
WeightWatch (arXiv:2508.00161) uses SVD directions of dW for monitoring and steering, and
arXiv:2510.00553 reports that the top singular direction of RL deltas recovers >99% of gains. Both
make SVD-of-dW the incumbent. Together with this result, the bar for G1 is "beats rotated SVD", not
"beats SVD". Add cards to `lit/` before citing.

## Limits
One hand-set toy, 4 changes, oracle k for the SVD baselines, 3 seeds (sd is not a CI). The
coefficient and steps were not tuned beyond the 3-point sweep. A first sweep (3000 steps, 4
parallel workers) was killed after 3 min with no output, to add the lab lead's baselines; it
produced no results.
