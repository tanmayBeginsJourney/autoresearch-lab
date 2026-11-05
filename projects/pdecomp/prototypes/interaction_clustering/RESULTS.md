# G3 prototype: clustering subcomponents by ablation interaction vs co-activation

Status: EXPLORATORY (CPU toy, 2026-11-05). Generates hypotheses only; supports no claim.

## Setup (`toy.py`)
- Model `y = W2 relu(W1 x + b1)`, K=12 mechanisms. Mechanism k = {layer-1 piece `u_k e_k^T`,
  layer-2 piece `c_k u_k^T`}: 24 known rank-one subcomponents, true partition = 12 pairs.
  No SPD training: we use the ground-truth pieces to isolate the clustering step.
- Data: `x_k = Bern(0.05) * U(0,1)`; features paired (2m, 2m+1), the second copies the first's
  on/off with prob rho (marginals unchanged). rho in {0, 0.3, 0.6, 0.9} plus 1.0. Seeds 0, 1, 2
  (same seeds in every arm). N=20k.
- f = per-example MSE of masked vs full output (PD reconstruction loss).
- Regimes: `clean` (h=d_out=12, orthogonal), `outsup` (orthogonal hidden, d_out=6, so mechanisms'
  outputs overlap), `hidsup` (h=8, d_out=6, hidden superposition).
- Similarities: `coact_act` (Jaccard of activity > 0.05), `coact_ci` (Jaccard of causal CI:
  single-ablation loss > 1% of max), `int_full` (|E_x[f(-i,-j) - f(-i) - f(-j)]| at the all-on
  context), `int_banz` (Banzhaf interaction from least squares on 4000 examples x 32 random masks,
  q=0.5, per-example fixed effects), `comb` = sqrt(coact_ci * int_banz). Interactions normalized by
  min single effect, clipped to [0,1].
- Clustering: average linkage on 1-S. ARI vs truth with (i) oracle K=12, (ii) K by max silhouette,
  (iii) fixed merge threshold S > 0.5. Also margin = mean S(within mechanism) - mean S(pieces of the
  two co-occurring mechanisms in a pair). Numbers: `summary.json` (mean [min, max] over 3 seeds).

## Results
Threshold-free margin (within minus co-occurring pair), mean over 3 seeds:

| regime | method | rho=0 | 0.3 | 0.6 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| clean | coact_ci | 0.98 | 0.83 | 0.64 | 0.37 | 0.25 |
| clean | int_full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| clean | int_banz | 0.80 | 0.82 | 0.83 | 0.80 | 0.82 |
| outsup | coact_ci | 0.98 | 0.83 | 0.64 | 0.37 | 0.25 |
| outsup | int_full | 0.98 | 0.86 | 0.73 | 0.62 | 0.58 |
| outsup | int_banz | 0.79 | 0.77 | 0.76 | 0.68 | 0.67 |
| hidsup | all | <= 0.22 everywhere | | | | |

ARI, fixed threshold 0.5 (clean / outsup): coact_act and coact_ci 1.00 for rho <= 0.6, 0.47 at
rho 0.9 and 1.0 (every seed: pairs of mechanisms merged, 6 clusters). int_full 1.00 at all rho in
clean; in outsup 1.00, 1.00, 0.86, 0.86, 0.69. int_banz clean 1.00 everywhere; outsup 1.00, 0.89,
0.86, 0.67 [0.43, 1.00], 0.69. comb outsup 1.00, 1.00, 0.95, 0.79, 0.74.

ARI with oracle K or silhouette K: co-activation stays at 1.00 for all rho <= 0.9 in clean and
outsup (silhouette coact_act drops to 0.47 only at rho=1.0). Interaction methods are at or below
co-activation here (outsup int_banz 0.90-0.96 for rho >= 0.3). hidsup: every method ARI <= 0.25 mean.

## Reading
- Positive only in a narrow sense. Co-activation's separation margin shrinks steadily with rho
  (0.98 -> 0.37 at rho 0.9) while interaction's stays ~constant when outputs are orthogonal. When
  clusters are cut at a fixed similarity (as a merge rule like VPD's MDL must effectively do),
  co-activation merges co-occurring mechanisms at rho >= 0.9 and interaction does not.
- Not a clean win: with exact CI and the right K (oracle or silhouette), co-activation recovers the
  partition perfectly up to rho 0.9, because within-mechanism co-activation (~1.0) still exceeds
  cross-pair co-activation. The task's "(a) degrades in ARI" did not happen under these selectors.
- Interaction has its own confound: with overlapping outputs (outsup) the MSE cross-term
  2 e_i . e_j makes co-occurring but independent mechanisms interact, so int_full's margin falls
  0.98 -> 0.58. The Banzhaf estimator (random context) is more robust (0.79 -> 0.67) but noisier
  (worst seed ARI 0.43). The combination is the most stable in outsup under the threshold rule.
- hidsup is a toy failure, not a method result: with hidden superposition the "true" layer-2
  pieces are causally used by other features, so the planted partition is not a faithful
  mechanism partition and no similarity recovers it. A trained SPD would pick a different basis.
- The toy favors interaction: every planted mechanism is a series (AND-like) path, which has
  maximal interaction. A mechanism with parallel, rarely co-active pieces (OR-like) would have
  ~zero interaction and ~zero co-activation; neither method would group it. Untested.

## What would make this decisive (next steps)
1. Replace planted pieces with trained SPD subcomponents (src/spd_min) and learned CI, so CI noise
   and the subcomponent basis are realistic; check whether co-activation degrades in ARI then.
2. Use VPD's actual MDL merge criterion instead of the ad hoc 0.5 threshold.
3. Add OR-like mechanisms and output-overlap sweeps; use a loss without cross terms (e.g. per-output
   KL) to see if the outsup confound is a property of MSE.
4. Cost: int_full needs O(C^2) forward passes; Banzhaf regression needs O(C^2) regressors. Both need
   sparsity restrictions at real scale.
