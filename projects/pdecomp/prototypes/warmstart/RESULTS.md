# Warm-starting SPD from a data-driven init (EXPLORATORY, 2026-11-05)

Exploratory CPU prototype. It does not support any claim. It is here to decide whether this direction gets more time.

## Question
Does a cheap, data-driven initialization of the subcomponents (V_c = dictionary atoms of the inputs
the matrix sees, U_c fitted by least squares so that sum U_c V_c^T ~ W on the data) let SPD reach a
correct decomposition in a fraction of the steps that random init needs?

## Setup (`spd_warmstart.py`, `summarize.py`; raw run jsons in the session scratchpad, not kept)
- Targets: TMS 5-2 and 40-10, x_hat = ReLU(W^T W x + b), feature prob 0.05, trained as in SPD App A.4.1
  (AdamW 5e-3, wd 0.01, 10k steps), seed 0. All 40 features of the 40-10 model are alive (|W_j| ~ 1.12).
- My own minimal SPD: the tied W is decomposed into C rank-one subcomponents. Each has a CI MLP
  (1->16 GELU->1) with leaky hard sigmoids. Loss = faithfulness + 2 stochastic-recon samples (with a single matrix,
  layerwise = full) + beta3 * importance minimality. Adam 1e-3, cosine schedule.
  The 5-2 run matches the paper (C=20, beta3=3e-3, p=1). **Deviations from the paper, made to fit the CPU budget:**
  4k steps (paper: 40k) at bs 2048 for 5-2. 3k steps at bs 512 with C=80 (C is not listed in the paper) for 40-10,
  beta3=1e-4, p=2. Not tuned. The PD seeds are 0,1,2 in every arm, against one target model.
- Inits: `random` (Gaussian; SPD-default-like). `svd` (m1 SVD terms + small random). `ica` (complete
  FastICA on 20k inputs: m2 atoms, U by LS, other slots small random). `kmeans` (spherical k-means with C atoms,
  U by LS, so atoms are duplicated). `oracle` (V=e_j, U=W_j: an upper bound for any init).
- Metrics: the paper's MMCS/ML2R. MMCS turned out to be too weak: random init reaches MMCS 0.997 on 5-2 within
  300 steps, because it only checks the U direction. So I also use a **strict GT cosine**:
  gt_min = min_j max_c cos_F(U_c V_c^T, W_j e_j^T), and "recovered" means gt_min > 0.95.
  CI L0/feat = (#subcomponents with g > 0.1) / (#active features). An ideal value is ~1.

## Results (mean ± sd over 3 seeds; steps to first / sustained gt_min > 0.95, evaluated every 100 steps)
| model | init | first rec step (n rec/3) | sustained rec step | gt_min@0 | final gt_min | final MMCS | final ML2R | final CI L0/feat | final stoch recon | faith |
|---|---|---|---|---|---|---|---|---|---|---|
| 40-10 | ica | 0 ± 0 (3/3) | 0 ± 0 (3/3) | 0.999 ± 0.00012 | 0.97 ± 0.0015 | 0.999 ± 9.1e-05 | 0.895 ± 0.00058 | 13.2 ± 0.23 | 0.000253 ± 3.5e-06 | 2.23e-06 ± 6.3e-08 |
| 40-10 | kmeans | - (0/3) | - (0/3) | 0.855 ± 0.022 | 0.786 ± 0.1 | 0.971 ± 0.0065 | 0.674 ± 0.04 | 12.8 ± 0.15 | 0.000261 ± 4e-06 | 5.45e-06 ± 7.1e-08 |
| 40-10 | oracle | 0 ± 0 (3/3) | 0 ± 0 (3/3) | 1 ± 0 | 0.97 ± 0.00085 | 0.999 ± 6.3e-05 | 0.895 ± 0.00066 | 13.4 ± 0.13 | 0.000256 ± 1.3e-06 | 2.13e-06 ± 1e-07 |
| 40-10 | random | - (0/3) | - (0/3) | 0.0682 ± 0.0075 | 0.169 ± 0.02 | 0.72 ± 0.016 | 0.214 ± 0.011 | 23.3 ± 1.2 | 0.000465 ± 1.8e-05 | 3.9e-05 ± 4.6e-06 |
| 40-10 | svd | - (0/3) | - (0/3) | 0.11 ± 0.0083 | 0.0791 ± 0.0012 | 0.616 ± 0.005 | 0.619 ± 0.0023 | 4.42 ± 0.0061 | 2.44e-05 ± 2.4e-06 | 2.09e-07 ± 4.6e-08 |
| 5-2 | ica | 0 ± 0 (3/3) | 0 ± 0 (3/3) | 1 ± 2.9e-05 | 0.996 ± 0.00083 | 1 ± 1.6e-05 | 0.994 ± 0.00033 | 0.948 ± 0.0018 | 0.000306 ± 9.2e-06 | 8.59e-08 ± 2.9e-09 |
| 5-2 | kmeans | 0 ± 0 (3/3) | 0 (1/3) | 0.992 ± 0.0086 | 0.95 ± 0.005 | 0.992 ± 0.0052 | 0.509 ± 0.042 | 2.08 ± 0.17 | 0.000222 ± 1.8e-05 | 1.72e-07 ± 1.1e-07 |
| 5-2 | oracle | 0 ± 0 (3/3) | 0 ± 0 (3/3) | 1 ± 0 | 0.989 ± 0.0039 | 0.999 ± 0.00038 | 0.993 ± 0.0004 | 0.972 ± 0.021 | 0.000316 ± 2.4e-06 | 8.93e-08 ± 9.1e-09 |
| 5-2 | random | - (0/3) | - (0/3) | 0.397 ± 0.058 | 0.748 ± 0.12 | 0.996 ± 0.002 | 0.382 ± 0.039 | 3.49 ± 0.16 | 0.000933 ± 0.0001 | 1.35e-05 ± 3.7e-06 |
| 5-2 | svd | - (0/3) | - (0/3) | 0.44 ± 0.015 | 0.465 ± 0.033 | 0.961 ± 0.031 | 0.753 ± 0.15 | 1.2 ± 0.075 | 0.00264 ± 0.00027 | 6.26e-06 ± 1.2e-06 |

40-10 trajectory, seed 0, at steps 0 / 600 / 1800 / 3000, given as gt_min and CI L0/feat:
ica 1.00/0.3, 0.96/17.4, 0.97/14.1, 0.97/13.4 | oracle 1.00/0.4, 0.97/17.2, 0.97/14.0, 0.97/13.3 |
kmeans 0.88/0.6, 0.73/20.5, 0.72/13.3, 0.69/12.7 | random 0.06/2.1, 0.15/34.1, 0.19/26.3, 0.19/24.5 |
svd 0.12/0.2, 0.08/4.7, 0.08/4.4, 0.08/4.4.

## Verdict
**There is no usable positive result.** "PD in a fraction of the steps" holds on TMS, but only trivially, so TMS tells us nothing about whether a warm start helps.
- A complete-ICA init puts every ground-truth subcomponent in place at step 0 (gt_min ~ 1.00 on both models, all seeds), and
  it stays there. On 5-2 it ends at the oracle's quality (CI L0/feat 0.95, ML2R 0.99). Random init reaches neither within 4k
  steps (final gt_min 0.75 ± 0.12; the paper uses 40k steps). Why it is trivial: the input to W in TMS *is* a sparse code in the feature
  basis, so ICA solves the decomposition before SPD starts. (Axis alignment is not the reason: ICA, k-means, SVD and Gaussian init are all
  rotation-equivariant, so rotating the inputs would give the same numbers.) In that sense the init is the answer, and SPD only has to keep it.
- On 40-10, even the oracle init (perfect U, V) ends with ~13 active subcomponents per active feature after 3k steps. ICA matches the oracle
  exactly, random ends at ~23. So at this budget the slow part is learning the CI gates and the minimality trade-off, and
  initializing U, V does not touch it. (Caveat: 40-10 ran at C=80, bs 512, 3k steps and untuned beta3, so these numbers are not converged SPD.)
- Overcomplete k-means (C atoms, duplicated) is worse than complete ICA. It splits each feature's norm across duplicates (ML2R 0.51
  on 5-2, 0.67 on 40-10), and on 40-10 it never reaches gt_min > 0.95. SPD does not merge the duplicates in this budget.
- SVD init is the worst for recovering the ground truth. It stays in a dense low-rank solution: on 40-10 CI L0/feat is 4.42, which means all 10 SVD terms are on for
  every input. That solution has the lowest stochastic recon, and the run never leaves it (gt_min 0.08). It is a useful negative: a
  low-rank weight-space init is a bad basin for PD.
- The paper's MMCS metric cannot tell these runs apart (random 5-2: 0.996). Any init study needs a stricter metric, such as the GT cosine used here.

Would it transfer? Real matrices (and the TMS-with-ID, residual-MLP and LM settings) see inputs with features in
superposition, and their mechanisms are not plain input atoms. There a data-driven V means an SAE-quality dictionary, which
costs a real training run itself. So whether the warm start saves anything is open and untested here. The cheapest decisive next test, if this
continues: the SPD residual-MLP toy (paper hyperparameters), comparing SAE-on-activations init plus a CI-gate warm start against random,
with matched seeds and steps-to-target as the metric. Kill the direction unless it gives a >= 3x step reduction there, net of SAE cost. ROI looks moderate to low:
on the evidence here the bottleneck is gate learning, not finding the subcomponents.

## Novelty notes
- Queries run: `researchctl lit-search` for "warm start parameter decomposition initialization",
  "stochastic parameter decomposition initialization sparse autoencoder", "SAE initialized weight
  decomposition rank-one components" and "activation-aware low-rank factorization initialization dictionary
  learning". Web searches: "parameter decomposition SPD initialization warm start ... sparse autoencoder or SVD"
  and "parameter decomposition initialization converge faster". I also grepped the cached PD papers
  (SPD, VPD, tPD, ASPD) for "initiali".
- Found: no paper initializes APD/SPD/VPD subcomponents from data (SAE/ICA/k-means) or from an SVD.
  SPD (2506.20790, Sec 3.4) attributes seed sensitivity on the residual toy models to "unfavorable random
  initialization, which we intend to explore in future work". This is the closest statement, and an open
  invitation. tPD (2607.13047) notes that decompositions from different inits differ. The GPT-2 SPD
  extension (2511.08854) uses SVD as a baseline to compare against, not as an init. VPD (2610.20748) only reports C
  at initialization.
- The ingredients are textbook. ICA/dictionary learning under sparse sources is identifiable (classical
  sparse coding and ICA). Activation-aware factorization of W from input statistics (U by least squares under
  E[aa^T]) is the idea behind ASVD/SVD-LLM-style compression. "SAE features as a weight basis" also appears in
  SAE-constructed low-rank adaptation (ACL 2026). Any write-up would have to say it is applying these known ideas.
- Coverage is thin: one lit-search tool (its results were mostly off-topic) and two web queries. This is not a novelty audit
  (`lit/novelty.md` standard).
