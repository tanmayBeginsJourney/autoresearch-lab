# Novelty audit: G3, cluster PD subcomponents by ablation interaction

Date 2026-11-05. Status: audit draft for `lit/novelty.md`; read depth is stated per item. "Not found" means
not found in the queries below, not proof of absence. Not checked: the goodfire-ai/param-decomp repo
and Goodfire talks (they may already do this; check before any claim).

## Queries run
researchctl lit-search (project pdecomp, ~24 queries): Shapley interaction index feature interactions;
Faith-SHAP; Shapley-Taylor; Integrated Hessians; Archipelago; neural interaction detection weights;
feature interactions in transformers attention heads; circuit discovery interaction terms ablation
nonadditive; multi-perturbation Shapley analysis lesion; epistasis map gene modules; MoEfication
co-activation clustering; Banzhaf interaction ablation neuron coalition; SPEX; synergy neuron coalitions
hedonic games; Neuron Shapley; clustering units ablation similarity modularity; attention head ablation
pairwise redundancy backup heads; hierarchical clustering Shapley interaction feature groups; genetic
interaction profile similarity; hydra effect; causal modularity ablation interaction clustering;
interaction grouping of transcoder features. WebSearch (4): PD + clustering + non-additive interaction
(no hit beyond SPD/VPD themselves); SAE/transcoder pairwise joint ablation (hits below); Bandyopadhyay
2008; Banzhaf interaction estimation. Full text read (targeted sections): VPD A.8/7.5/B.11, Hedonic
Neurons, Archipelago (Sec 4.2), Lange et al., Bandyopadhyay (via fetched full text), MoEfication
(Sec 3.2 methods). Abstract/snippet only: Shapley-Taylor, Faith-Shap, Integrated Hessians, NID, Neuron
Shapley, MSA (Keinan 2005; Fakhar 2022), SPEX, Conditional Co-Ablation, PIE, SAR, H-Sets (2604.22045).

## Closest three works
1. **Hedonic Neurons** (arXiv:2509.23684, skim). Pairwise Ablation Synergy phi_PAS(i,j) = -E_x[l_{-ij} -
   l_{-i} - l_{-j} + l], the second-order ablation interaction, approximated by a Hessian x E[a_i a_j];
   neurons partitioned into "coalitions" by a PAC-Top-Cover hedonic game. Claims coalitions have higher
   synergy than K-means/hierarchical baselines and 3-5x larger OOD drops under ablation.
   Difference from G3: neurons in single MLP layers of fine-tuned rerankers, scalar output, no parameter
   decomposition, no cross-matrix mechanisms, no ground-truth recovery metric; baselines cluster neuron
   weights/activations, not a CI co-activation MDL method. This is the main threat: the idea "cluster by
   pairwise ablation non-additivity" is already published for neurons.
2. **Archipelago / ArchDetect** (arXiv:2006.10965, skim). Pairwise interaction omega_ij = squared
   mixed second difference of f, in the context of other features Bernoulli(0.5) (approximated by two
   contexts), O(p^2) calls; overlapping pairs are merged into disjoint interaction sets. Evaluated by AUC
   on synthetic ground-truth interactions.
   Difference: input features of a prediction, not internal parameter subcomponents; same algorithmic
   pattern as G3 (pairwise non-additivity then merge). Our estimator is the Banzhaf pair index of this form.
3. **Bandyopadhyay et al. 2008** (PLoS CB 4:e1000065, doi:10.1371/journal.pcbi.1000065, skim of fetched
   text). Genetic-interaction S-scores (|S|) on gene pairs plus physical interaction data, agglomerative
   merging by a likelihood score into 91 protein-complex modules; profile-correlation did not help.
   Within-module links skew alleviating (19 vs 9 modules), between-module links skew aggravating
   (219 vs 94 of 313 relations). Difference: biology, not NNs; uses a physical-interaction prior.
   The sign pattern is the useful lesson (see Algorithm).
Honourable mentions: VPD B.11.2 (structural interaction proxy, not used to cluster, not ablation-based);
Lange, Rolnick, Kording 2022 (arXiv:2203.11815: clusters hidden units by pairwise similarity,
upstream vs downstream measures; finds little agreement between families; skim); Keinan MSA / Fakhar &
Hilgetag 2022 (lesion interaction via Shapley, no clustering of units into modules; abstract only);
MoEfication (arXiv:2110.01786: co-activation graph split of FFN neurons into experts, the
co-activation analogue of VPD's approach; Sec 3.2 read); Conditional Co-Ablation (2607.01940: pairwise
co-ablation to find backup heads; diagnostic, not clustering); PIE/FAP-Synergy (2604.16889: synergy-aware
reranking of transcoder features; selection, not grouping); 2606.09607 (co-activation clusters of heads
validated by ablation closure; validates clusters, does not cluster by interaction).
Shapley-Taylor, Faith-Shap, Integrated Hessians, NID, SPEX, Kernel Banzhaf provide indices and
estimators, not groupings of units.

## Verdict
- Principle novelty: **low**. Pairwise ablation non-additivity as an affinity for grouping units is
  published (Hedonic Neurons, ArchDetect, epistasis maps).
- Novelty for PD: **plausibly open** (no PD hit in 4 web queries and the lit-search set), but VPD B.11.2
  and SPD/VPD text show the authors consider interaction structure directly relevant.
- Expected venue value: an evaluation-and-tool contribution (does causal interaction recover cross-matrix
  mechanisms where co-activation fails; when does it fail), not a new method claim. Medium at best;
  rises only if it gives a decisive win on correlated-feature toys and then on VPD's LM clusters.

## Most defensible claim (if the toy results hold)
"In decompositions of toy models with correlated features, clustering rank-one subcomponents by
estimated pairwise ablation interaction recovers ground-truth cross-matrix mechanisms (ARI) where
CI co-activation MDL clustering (VPD A.8) degrades, using a probe estimator built on SPD/VPD masks."
Do not claim "new interaction-based clustering" or "mechanism discovery"; say which known idea is
applied (second-order Banzhaf interaction; ArchDetect-style pair merging). Interactions are
intervention-based, so the causal-claim rule is met only with controls: random-pair and matched-norm
baselines, and held-out-token estimates.

## Algorithm proposal

Important constraint found in the method text (SPD l.100, VPD App A.7): training masks are
m_c = g_c + (1 - g_c) r_c, r_c ~ U(0,1). Subcomponents with CI near 1 are never ablated in training
masks. So the existing masks only vary the low-CI components. A regression on those masks therefore
estimates interactions among components the model says are unimportant: useful as a faithfulness
diagnostic, useless for finding mechanisms. Mechanism structure needs a probe that ablates the active
set. Cost stays low because it is post-hoc on the frozen decomposition.

**Probe estimator (post-hoc, frozen decomposition).**
1. Take N held-out tokens (toys: N=4096; LM: single positions, see caveat). For each token the active set
   A(x) = {c : g_c(x) > tau}; inactive components fixed at their CI-mask value m_c = g_c.
2. Draw K coalition masks z in {0,1}^A with z_c ~ Bernoulli(1/2) i.i.d. (Banzhaf weighting, as ArchDetect
   and Kernel Banzhaf); ablated active components get m_c = 0, kept ones m_c = 1. One forward pass over
   N tokens per mask (the code already applies independent masks per position). K = 128-512.
3. Value v_x(z) = -||f(x|W'(m)) - f(x|W)||^2 in output (logit) space. Use output space, not KL: for
   additive logit contributions KL has a cross term, which would fake interaction. Report the KL
   variant as a robustness check.
4. Pair estimate with maximum sample reuse (same K masks for all pairs): sigma_ab(x) = E[v|z_a=1,z_b=1]
   - E[v|1,0] - E[v|0,1] + E[v|0,0]. Aggregate over tokens where a and b are both active:
   sigma_ab = sum_x sigma_ab(x)/n_ab with n_ab co-active token count; keep a pair only if n_ab >= n_min
   (this is where co-activation enters as a data-availability gate). Optional sparse alternative:
   order-2 Kernel-Banzhaf/FBII regression with lasso on pairs (SPEX-style).
5. Null and scale: z-score sigma_ab against interaction of random same-layer/other-layer pairs with the same
   marginal effects; add a multiplicative-null variant (genetic epistasis uses a multiplicative baseline)
   to damp hub subcomponents that interact with everything (VPD B.11).
   Cost: K forward passes x N tokens x (cheap per-pair arithmetic, O(|A|^2) per token). Toys with <= 200
   subcomponents: exact all-pairs ablation on a batch is also possible and is the ground-truth check.

**Sign matters (from the epistasis literature and simple algebra).** In a chain a->b (both needed,
AND) fidelity is complementary: sigma_ab > 0. In OR-redundancy (parallel copies) sigma_ab < 0. Same-matrix
rank-one pieces add in that layer's output, so their interaction is only through downstream nonlinearity.
Hence the proposed affinity is the signed complementary interaction max(sigma_ab, 0), not |sigma|; negative
values are reported separately as redundancy between mechanisms (cf. aggravating between modules).
Cluster by merging in decreasing affinity with the MDL stopping rule kept: replace the co-activation term
s_ij in delta-L (or gate merges by significant sigma) and compare three arms with the same seeds:
(a) VPD MDL on co-activation, (b) interaction only, (c) interaction-gated MDL. Metric: ARI vs
ground-truth mechanisms; loss when ablating whole clusters vs random clusters of equal rank; stability
across stochastic-merging runs. Toy: SPD residual-MLP with feature co-occurrence rho in {0, .5, .9, 1}
(see ideas/cross-field/ideas.md #1).

**Predicted failure modes (design the toys to expose them).**
- Same-matrix rank-2 mechanisms (e.g. 2-D rotation inside one matrix) have ~zero interaction and are
  split; co-activation or weight-geometry must supply that link. Expect (c) > (b) there.
- Hubs: always-active dense subcomponents interact with everything (normalise, or drop from clustering).
- LM: attention mixes positions, so per-position masks confound; interaction estimates are only clean in
  MLP-only / position-wise settings. This also means the LM-scale claim needs separate care.
- Estimator variance: MSR Banzhaf estimates are noisy; check ranking stability across K and token splits.
- Exploratory vs confirmatory: the toy sweep above is exploratory; any claim needs a preregistration.
