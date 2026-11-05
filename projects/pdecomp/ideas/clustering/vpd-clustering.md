# VPD subcomponent clustering (arXiv:2610.20748, App. A.8): what it does

Source: /home/user/lab/cache/papers/2610.20748.txt, App. A.8 (l.3802-3960) read in full; Sec 7.5 limits
(l.2718-2750) and App. B.11 (l.4935-5040) read in full. Page refs are to that text file.

## Role in the paper
Clustering is post-hoc, after training. VPD says "subcomponents were usually interpretable even
without clustering, and therefore used clustering only rarely in our analyses" (l.282). Case-study
components (e.g. bracket closing, Sec 5.4) are clusters, but no clustering benchmark or ground-truth
recovery result is reported. SPD (2506.20790, l.91-125) left clustering implicit: toy clusters were
known by construction.

## Objective (MDL)
- n rank-one subcomponents, partition into k groups. CI binarised: g in {0,1}, threshold tau=0.01
  (rounds <= tau to 0, > tau to 1). A group's CI is the OR of its members' CIs.
- Co-activation counts over B sequences x T positions: s_ij = sum_{b,t} g_i g_j; s_i = s_ii.
- Cost: L_MDL = sum_i s_i (log2 k + alpha * r(theta_i)), r = rank summed over constituent matrices.
  log2 k term = bits to name the active groups per datapoint (dictionary cost dropped as one-time);
  alpha * r term = per-datapoint size of the matrices one must read. alpha trades # components vs complexity.
- Merge delta (i,j): (s_sum - s_i - s_j) log2((k-1)/k) + s_ij log2(k-1) - s_i log2 k - s_j log2 k
  + alpha (s_ij r(ij) - s_i r(i) - s_j r(j)), with r(ij) ~ r(i) + r(j).
- Co-activation enters only through s_ij (and s_i). Merging pays off when the merged group is
  active together often. Pair rule of thumb (A.8.3, s1=s2): merge if alpha < co(2|1)/(1-co(2|1)) * log2(k)/2.

## Algorithm
Start with singletons. Each iteration: compute delta-L for all pairs, rank ascending, sample rank J with
P ~ exp(-gamma J) (gamma=0.2; gamma->inf greedy), merge. Continue until one group remains; return the
partition at the iteration where marginal delta-L crosses 0. Stochastic to escape local minima; no
statement about running it several times or about stability across runs.

## Inputs
Only binary CI statistics from a harvested batch (same sequence position only) and ranks. No weights,
no ablation effects, no activations (a_{b,t,c} are not used in clustering).

## Failure modes and limits the authors admit (Sec 7.5, l.2718-2750)
1. Blind to cross-position correlations: only same-position co-activation. Example given: Q and K
   components of an induction head may never be active at the same position.
2. Not carefully tuned; alpha not swept; "the parameter components we report may not reflect the best
   possible grouping."
3. Post-hoc clustering makes the adversarial mask sampler stricter than needed (two ablatable rank-one
   pieces are ablated independently, a 2-D mask subspace, instead of as one rank-2 component).
   Suggested future work: integrate clustering into training to inform mask sampling.

## Not admitted but visible from the method (our reading, not the paper's claims)
- Correlated but functionally independent features merge (co-activation cannot separate them); the
  "dense" always-active subcomponents have large s_i and are pulled toward everything or nothing.
- The OR rule means a group is active whenever any member is; no check that members compute jointly.
- Greedy bottom-up merging with approximate r(ij) = r(i) + r(j); no validation that the clusters are
  jointly ablatable without loss.

## Related: VPD already measures an interaction proxy, but does not cluster on it (App. B.11.2, l.4972-5030)
I^l_{c,c'} = [sum_i |U_ic||U_ic'| / sum_i U_ic^2] * [sum_{b,t} |g a_c||g a_c'| / sum_{b,t} |g a_c|^2]
for MLP-up subcomponents: weight overlap at neurons times activation overlap. Asymmetric, normalised
so diagonal=1. The authors call it "crude", "very preliminary", not output-influence-based, not
ablation-based. Observation: large off-diagonal entries are mostly within components (diagonal
blocks) or from a few hub subcomponents (stripes). This is supportive context for G3, and also the
first thing a reviewer will cite against a novelty claim. It is a structural proxy, not a joint-
ablation measure.
