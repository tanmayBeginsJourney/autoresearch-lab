# Ideas: improving or certifying PD faithfulness
All ideas need a novelty audit (`lit/novelty.md`) before any claim. Search so far was abstract-level only.
Mask-space facts used (from VPD card): masks live in a box, adversary maximizes KL(target || replacement) by persistent PGD; robustness collapses past ~40 steps.
CPU toy substrate: SPD toy models with known ground truth (TMS, residual MLP; arXiv:2506.20790), or a 2-4 layer MLP with rank-one components; no GPU needed.

## 1. Mask-space bound propagation: per-input certificates and attack gap
- Idea: treat the ablation mask r in [0,1]^C as the perturbation set. For fixed x the first masked layer is affine in the mask, so IBP/CROWN-style bounds on later layers give a certified upper bound on worst-case output deviation (and KL) over the whole box, per input. Report certified-bound vs PGD-found value (the "attack gap") as a faithfulness diagnostic, and use the bound as a training regularizer (CROWN-IBP style).
- Closest prior: IBP 1810.12715, CROWN 1811.00866 (input perturbations); Certified Circuits 2602.22968 (certifies discovery, not behavior); VPD (empirical PGD only).
- Why it might work: box + small layers is the regime where bounds are tight; a bound that sandwiches PGD turns "KL 0.83 at 20 steps" into "worst-case KL is in [a, b]".
- CPU toy: 2-layer ReLU MLP decomposed into ~20 rank-one components; use auto_LiRPA or hand IBP; compare bound vs 1000-step PGD with restarts across decomposition quality. Success: bound within ~10x of PGD on toy, rank-orders decompositions the same as PGD does. Kill: bounds vacuous (>100x) even on 2 layers.
- Risks: bilinear mask-times-activation terms loosen bounds with depth; scaling to 10k components/token is unlikely, so a toy-plus-small-LM-layer claim only.
- Novelty confidence: medium (about 60%). Technique is known; applying it to PD masks was not found. Name it for what it is: standard bound propagation.

## 2. AutoAttack-style adversary for masks (APGD + FW vertex + black-box), and what it does to training cost
- Idea: replace persistent fixed-step PGD with a step-size-free APGD, Frank-Wolfe on the box (linear oracle returns a vertex, so iterates are binary masks and directly probe the rounded-mask regime where VPD recovers 8-12%), and Square-style random search as a gradient-masking check; evaluate with the ensemble worst case.
- Closest prior: AutoAttack 2003.01690, Square Attack, FW attacks 1811.10828, sparse-PGD 2405.05075; VPD's own sampler is described by its authors as primitive.
- Why it might work: ensembles reveal that single-attack numbers understate vulnerability; APGD is more step-efficient, so equal robustness may cost fewer inner steps (efficiency is the lab's top concern).
- CPU toy: on the toy decomposition, fix a budget of N gradient evaluations; compare worst-case KL found by PGD-persistent, APGD, FW, Square. Then train two decompositions with equal inner-step budgets and compare ground-truth recovery and held-out adversarial KL. Kill: PGD already within a few percent of the ensemble.
- Novelty confidence: medium-low (about 45%) as a paper; positive and cheap, best as the evaluation component of idea 3 or 5.

## 3. Ground-truth calibration: which adversarial strength predicts mechanism recovery?
- Idea: VPD says full robustness may not be desirable and the right target is unresolved. In toy models with known true mechanisms, sweep adversary strength (steps, mask-set constraint, L0 budget) during training and test which robustness measure best predicts ground-truth component recovery and downstream causal tests. Output: a validated robustness metric plus a recommended training adversary.
- Closest prior: SPD toy-model recovery (2506.20790); Makelov et al. (optimized interventions can mislead); VPD Sec 7.3-7.5 argument. No metric-validation study found.
- Why it might work: toys give labels that real LMs lack; a metric validated there is a defensible proxy.
- CPU toy: TMS and residual MLP from SPD, 5 seeds paired, adversary strength x 6 levels; measure max cosine to true features and over-/under-splitting. Kill: robustness is uncorrelated with recovery (still a publishable negative for the metric).
- Novelty confidence: medium (about 55%); depends on how many toys behave like LMs.

## 4. Minimum adversarial support: an L0 robustness radius for decompositions
- Idea: instead of one KL at a fixed PGD budget, report the smallest number k of subcomponents whose adversarial setting pushes KL over tau (sparse-PGD with top-k projection, minimum-norm style). Gives an interpretable radius, comparable across decompositions, and splits failures into main effects vs interactions (AND/OR gates; backup paths as in 2505.10039 and 2607.01940).
- Closest prior: sparse-PGD/sAA 2405.05075, min-norm attacks (FAB; 2607.19855); circuit completeness work above. Not found applied to PD.
- Why it might work: a radius is robust to the arbitrary choice of step count that makes VPD's table hard to compare; k small means a few CI-miscalibrated components carry the failure, which tells you what to fix.
- CPU toy: on the toy decomposition compute k(x) over inputs; test whether k tracks injected defects (merge two true mechanisms; duplicate one). Kill: k is 1 almost everywhere.
- Novelty confidence: medium (about 50%).

## 5. Illusion test for adversarial masks: on-manifold vs dormant-path failures
- Idea: ask whether the large adversarial KL comes from masks driving hidden activations off the data distribution (Makelov-style dormant pathways) or from genuine unfaithfulness. Add an on-manifold constraint (activation-norm/Mahalanobis budget vs clean activations) to the adversary and report both curves; decompositions robust on-manifold but not off-manifold are faithful for explanation purposes.
- Closest prior: Makelov et al. 2311.17030; divergent representations from interventions 2511.04638 (title only); VPD argues full robustness is not desirable (consistent with this).
- Why it might work: it operationalizes VPD's own caveat; training against the on-manifold adversary should keep the cheap part of robustness and drop the part that costs capacity.
- CPU toy: toy MLP; measure activation divergence of adversarial vs clean; compare KL under constrained vs unconstrained adversary across training. Kill: constraint barely changes KL (then failures are on-manifold and real).
- Novelty confidence: medium-low (about 40%); the idea is natural, so check that Goodfire's repo/blog has not done it.

## Recommendation
Combine 2 (strong attack ensemble) as the evaluation core, 3 (ground-truth calibration) as the scientific claim, and 1 as the stretch result. Total toy cost: CPU only, about a few days. Run `lit-citers` on 2610.20748 again later (returned 0 today) and search Goodfire's repo/blog for existing work on 2 and 5 before spending time.
