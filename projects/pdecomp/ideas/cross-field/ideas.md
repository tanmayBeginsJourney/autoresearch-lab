# Cross-field ideas for parameter decomposition (PD)

Scouted 2026-11-05, ~35 min. Novelty checks are searches (`researchctl lit-search`, web), not a full
audit. "No prior found" means no hit in the queries listed, not proof of novelty. Every top-3 idea
still needs a `lit/novelty.md` audit before any claim.

Facts used below (from our cards and VPD arXiv:2610.20748): VPD clusters subcomponents by **CI
co-activation** (binarised CI, tau=0.01, MDL-driven stochastic hierarchical merging, App. A.8) and lists
"integrating clustering into training" and multi-position clustering as future work (Sec. 7.5). VPD
draws S=1 stochastic mask per step (App. A.7). Full-data VPD seed agreement is mean max cosine
0.48 (U) / 0.51 (V) (quoted in the tPD card, arXiv:2607.13047). SPD toys need 30k-200k steps.

## Ranked top 3

### 1. Cluster subcomponents by ablation interaction, not co-activation (cooperative game theory + lesion neuroscience)
- **Idea.** Two subcomponents belong to one mechanism when their joint ablation is non-additive.
  Estimate the pairwise Banzhaf interaction index (Grabisch & Roubens 1999) of the masked reconstruction
  loss, I_ab = E_m[L(m: a,b off) - L(a off) - L(b off) + L(none off)], and cluster on |I|. This is
  multi-perturbation Shapley analysis (Keinan et al. 2004; Fakhar & Hilgetag, PLoS CB 2022,
  doi:10.1371/journal.pcbi.1010250) applied to PD subcomponents. Known technique, new place.
- **Change to PD.** SPD/VPD already sample random masks every step. Regressing per-example losses on
  mask products (the Banzhaf/KernelSHAP-style least-squares estimator) gives interaction estimates as a
  by-product, restricted to pairs that are co-active (sparse CI keeps this O(active^2) per token).
  Clustering then runs on interaction (or interaction x co-activation) instead of co-activation alone.
- **Why it should beat co-activation.** Co-activation merges functionally independent mechanisms that
  happen to fire together (correlated features) and cannot tell an AND-like two-layer path (strong
  interaction) from two additive contributions. Interaction is zero for additive, independent parts.
- **Benefit.** Faithfulness and identifiability of full components; addresses a named VPD open problem.
- **Closest work.** VPD App. A.8 (co-activation MDL clustering); Neuron Shapley / MSA (neuron-level
  attribution, not grouping); synergy-based neuron coalitions via hedonic games (arXiv:2509.23684,
  neurons, not PD); Lange, Rolnick & Kording 2022 (arXiv:2203.11815, clustering units by
  non-ablation similarity); co-activation-proposes/ablation-disposes circuit validation
  (arXiv:2606.09607, attention heads, validation only).
- **Novelty confidence: medium-high.** No PD work uses ablation interactions to cluster (3 web queries,
  1 lit-search). Risk: Goodfire may already be doing "clustering in training" internally.
- **CPU toy.** SPD's 2-layer and 3-layer residual MLPs (100 features, ground-truth mechanisms span
  layers) plus a new variant where feature pairs co-occur with probability rho in {0, 0.5, 0.9, 1.0}.
  Decompose with SPD (same seeds 0-4 in both arms), then cluster by (a) CI co-activation MDL as in VPD,
  (b) interaction, (c) both. Metric: adjusted Rand index against ground-truth mechanisms; masked loss
  when ablating whole clusters. Prediction: (a) and (b) tie at rho=0; (a) falls and (b) holds as rho -> 1.
  Kill if (b) does not beat (a) by >= 0.1 ARI at rho >= 0.9, or if the estimator needs > 4x the
  masks SPD already samples. Then test cross-position clusters on the 4-layer LM (VPD's open problem).

### 2. Closed-form block-coordinate updates for subcomponents (dictionary learning: K-SVD / MOD)
- **Idea.** With masks fixed, the per-matrix local objective sum_x || sum_c m_c(x) U_c V_c^T x - W x ||^2
  is linear in each U_c given V (and vice versa). K-SVD (Aharon et al. 2006) and MOD solve the
  analogous dictionary step in closed form. Alternate: CI/mask step (gradient, as now) and a
  least-squares sweep over (U_c, V_c) on the active set of each subcomponent.
- **Change to PD.** Interleave one closed-form sweep every k gradient steps, or use it as a warm start
  for the first few thousand steps; keep the end-to-end stochastic/adversarial losses for the final
  phase so mechanisms are still defined by output faithfulness.
- **Benefit.** Speed (steps and wall-clock to ground-truth recovery); fewer dead subcomponents if the
  sweep re-seeds unused ones from residual top singular vectors, as K-SVD does.
- **Closest work.** No PD paper uses alternating minimisation (2 web queries). Sum-of-sparse-rank-one
  data approximation by block coordinate descent (arXiv:1511.08842) is the nearest algorithm but for
  data, not weights with input-dependent gates. ASPD (arXiv:2609.37731) uses a local output loss but
  with gradients. Schnass (arXiv:1301.3375) gives K-SVD identifiability theory usable for idea 3.
- **Novelty confidence: medium.** Algorithmic novelty is modest (textbook ALS applied to PD); the
  contribution stands or falls on a large measured speedup.
- **CPU toy.** TMS 5-2 and 40-10 and the 1-layer residual MLP (compressed computation). Arms: SPD
  baseline vs SPD + ALS sweep, same seeds 0-9, same total FLOPs budget. Metrics: steps and wall-clock
  until mean max cosine to ground-truth subcomponents >= 0.99 and CI-L0 matches the true feature count;
  final masked loss. Kill if speedup < 2x at equal final quality on two of three toys.

### 3. Identifiability conditions for PD (sparse coding theory + persistence of excitation + statistical physics)
- **Idea.** Sparse dictionary learning is identifiable only when supports are diverse enough (Hillar &
  Sommer; spark / coherence conditions; Spielman-Wang-Wright). System identification needs inputs that
  excite every mode (persistence of excitation). PD should inherit both: a subcomponent is identifiable
  only if the data activate it without its co-activators often enough. VPD's 0.5 seed agreement says
  this bites at LM scale.
- **Change to PD.** (a) A cheap diagnostic computed from CI statistics (support-diversity / empirical
  coherence of co-activation patterns) that predicts which subcomponents are seed-stable. (b) A fix:
  reweight or select inputs to raise excitation of poorly identified subcomponents, the PD analogue of
  SAE reweighting (Cui et al., arXiv:2506.15963) and optimal experiment design.
- **Benefit.** Identifiability; tells users which components to trust; a phase diagram (sparsity x
  co-occurrence x C x minimality coefficient) in the style of teacher-student specialisation
  transitions (e.g. arXiv:2603.20010) gives hyperparameter rules.
- **Closest work.** SAE identifiability: arXiv:2506.15963, arXiv:2605.31245 (aRIP), arXiv:2512.02194,
  ACL 2026 long.99 (seed consistency). None treat weight-space PD with input-dependent gates.
- **Novelty confidence: medium.** The SAE side is crowded; the PD side is open (2 web queries; only a
  passing remark on non-uniqueness found). Positive-result risk: a phase diagram is descriptive unless
  the diagnostic or reweighting measurably improves seed agreement.
- **CPU toy.** TMS 40-10 and compressed computation with controlled co-occurrence between feature
  pairs (rho grid) and feature probability p grid. 10 seeds per cell. Measure seed agreement (Hungarian
  matched cosine, not max cosine) and ground-truth recovery. Test (a): does the diagnostic, fit on 5
  seeds, predict per-subcomponent stability on 5 held-out seeds (Spearman > 0.5)? Test (b): does
  reweighting raise matched agreement at rho = 0.9 by >= 0.1?

## All candidates by field

| Field | Transferable idea | Concrete change to PD | Benefit | Closest work found | Novelty |
|---|---|---|---|---|---|
| Cooperative game theory | Banzhaf/Shapley interaction indices | Cluster subcomponents by interaction from existing masks (top-1) | Faithful full components | MSA, Neuron Shapley, arXiv:2509.23684 | medium-high |
| Neuroscience: lesions | Multi-lesion inference; double dissociation | Same as above; double-dissociation test as an eval of components | Evaluation | Fakhar & Hilgetag 2022; Jonas & Kording 2017 | merged into #1 |
| Neuroscience: connectomics | Group-testing stimulation of ensembles (Draelos & Pearson, NeurIPS 2020) | See group testing row | Cheaper evaluation | same | see below |
| Dictionary learning | K-SVD / MOD closed-form updates, atom re-seeding | ALS sweeps on subcomponents (top-2) | Speed, fewer dead components | arXiv:1511.08842 | medium |
| Dictionary learning / ICA theory | Spark, coherence, support diversity | Identifiability diagnostic and reweighting (top-3) | Identifiability | arXiv:2506.15963, 2605.31245 | medium |
| Control / system ID | Persistence of excitation; optimal experiment design | Choose inputs that excite under-identified subcomponents | Identifiability, cheaper tPD-style runs | none for PD | merged into #3 |
| Statistical physics | Specialisation phase transitions (committee machines) | Phase diagram of PD recovery vs lambda, p, rho, C | Hyperparameter rules | arXiv:2603.20010 (theory, not PD) | merged into #3 |
| Group testing / compressed sensing | Pooled tests find k defectives in O(k log C) tests | Search for small low-CI subsets whose joint ablation breaks the output, using disjunct pooling designs instead of PGD steps; evaluator of faithfulness | Cheaper adversarial evaluation | No group testing in interpretability found (2 queries); NeuCEPT arXiv:2209.08448 (critical neurons) | high, but value unclear: binary pooled masks may be weaker than continuous PGD masks |
| Monte Carlo methods (not on list) | Antithetic / randomised QMC mask sampling | Replace S=1 iid uniform masks with RQMC or stratified masks | Lower gradient variance, faster convergence | arXiv:2410.08125 finds antithetic weak, RQMC better (smoothing, not gates) | medium; small expected gain |
| Robust PCA | Low-rank + sparse split with nuclear norm | Nuclear/L1 penalty on the Delta catch-all | Cleaner residual | tPD/VPD already have Delta components | low |
| Tensor decompositions | Sparse CP of a weight/attribution tensor | Warm-start subcomponents from CP of per-example Jacobian tensors; tensor-train coupling of components across layers | Speed, cross-layer components | Bilinear MLPs (arXiv:2410.08417) decompose weight tensors directly | low-medium |
| Dense-to-MoE conversion | MoEfication, CMoE (arXiv:2502.04416), ExpertWeaver, DOT-MoE (arXiv:2606.01666) | (a) Neuron-cluster MoE as a **baseline decomposition**: does PD beat it at matched active parameters? (b) PD subcomponents as rank-1 experts for conditional compute | (a) honest evaluation; (b) inference speed | Rank-1 expert MoE trained from scratch (arXiv:2604.14434); no PD-to-MoE found | (a) medium; (b) high but VPD recovers only 8-12% compute with rounded masks, so a positive result is unlikely |
| Pruning | Hard-concrete L0 gates, movement scores, lottery-ticket rewinding | CI parametrised as hard-concrete; movement scores to init CI | Small | SPD's hard-sigmoid CI is already close to hard-concrete | low |
| Causal discovery | DAG learning from interventions | Learn sparse causal graph among subcomponents across layers | Circuits | VPD already builds attribution graphs (Sec. 7.2) | low |
| Program synthesis (Tracr) | Compiled transformers with known algorithms, compressed for superposition | Benchmark PD on attention mechanisms with ground truth (TracrBench, InterpBench) | Evaluation of attention PD | No PD-on-Tracr found; SPD on a toy induction head exists (arXiv:2511.08854) | medium-high; eval contribution, not a method |

## What to do next
1. Read VPD App. A.8 and B.3 in full (clustering cost, seed-consistency metric) before committing to #1.
2. Run the #1 toy first: it reuses SPD's code and ground truth, costs CPU minutes per run, and answers a
   named open problem. Run #2 alongside if the SPD toy code is already set up; it shares the setup.
3. Ask the literature subagent for a novelty audit of #1 (queries: "interaction index" + "parameter
   decomposition"; Goodfire clustering blog posts; MSA citers since 2024).
