# Novelty audit: G4 "decomposability by design" (2026-11-05)

Question: does the MLP activation/form (ReLU, GELU, SwiGLU, bilinear/GLU, SoLU, TopK) change how faithfully a
trained model can be parameter-decomposed (SPD/VPD) at matched loss?
Evidence depth: full text grepped for PD terms in 2509.01874, 2602.05635, 2605.08891, 2610.19907, 2410.08417;
read in part: 2509.01874 (Sec 3-4), 2610.19907 (abstract, intro, setup), VPD (all activation mentions).
Not read: SoLU full text (page seen only via search summary), 2502.02470, BIMT, MoE papers (search summaries).
Cards written: arxiv-2509.01874, arxiv-2610.19907, arxiv-2602.05635, web-transformer-circuits-solu.

## Queries run
researchctl lit-search (--project pdecomp): "bilinear MLP interpretability"; "activation function sparse autoencoder
feature splitting"; "softmax linear units monosemanticity"; "modular training brain-inspired modular training";
"interpretable by design architecture weight sparse"; "activation function effect on parameter decomposition
faithfulness"; "measure of modularity of trained neural network weights". lit-citers of 2506.20790 (1 citer: Sparse
Model Diffing, EMNLP 2026), 2410.08417 (empty), 2610.20748 (empty): citer coverage is useless, re-check later.
lit-fetch: 2509.01874, 2602.05635, 2605.08891, 2610.19907, 2606.22172, 2502.01032; grep of VPD/SPD/APD texts.
WebSearch: "parameter decomposition SPD activation function architecture effect decomposability ReLU GELU bilinear";
"activation function effect on sparse autoencoder quality feature splitting GELU ReLU SwiGLU"; "Softmax Linear Units
Anthropic 2022 SoLU"; "training neural networks for modularity aids interpretability"; "mixture of experts
interpretability monosemantic experts decomposable by design 2026"; '"parameter decomposition" architecture choice
nonlinearity matched loss decomposable 2026'.
Result: no paper compares PD (APD/SPD/VPD) faithfulness across activation functions or architectures. The only hit
near the question is a LessWrong comment on the SPD post (nonlinearity after the weight may matter for the
superposition argument), which is a comment, not evidence.

## Closest three works and how G4 differs
1. Bilinear MLPs line: Pearce et al. 2410.08417 (+ 2406.03947, Sharkey 2305.03452 note), SQS 2509.01874,
   structural disentanglement 2602.05635, bilinear autoencoders 2605.08891, closed-form polynomials 2502.01032.
   Closest because they pick the MLP form so a weight-based method works at near-matched loss (SQS: GELU MNIST
   final loss 0.0680 vs SQS 0.0834, their numbers). Differences: readout is eigendecomposition of the interaction
   tensor, judged by eigenvector plots and unlearning on toys, with no mask-ablation or adversarial faithfulness; no
   activation sweep under PD; none cover attention or whole-model decomposition. 2509.01874 tunes its activation
   hyperparameters for eigenvector interpretability, so it is a design, not a controlled comparison.
2. Sparse Transformer 2610.19907 (TopK after ReLU, wide MLP): interpretable-by-design model with a loss-matched
   dense control (0.866 vs 0.605 AutoInterp) and a width-matched control. Differences: evaluates neurons,
   SAEs and transcoders, never PD (no hits for SPD/VPD/APD). Its protocol (loss-matched and width-matched controls)
   is the template G4 should copy, and its released TopK models are a free extra arm.
3. SoLU (Elhage et al. 2022) and modularity training (BIMT, Liu/Gan/Tegmark 2023; Golechha et al. 2409.15747, withdrawn
   v1, corrected in 2502.02470). Closest because they intervene on architecture or training to raise interpretability
   at roughly matched loss (SoLU: 35% to 60% of neurons with a clear hypothesis, per the authors' companion post;
   verify in full text). Differences: neuron-level human judgment (SoLU) and clusterability/circuit size (BIMT) are the
   outcomes; neither measures parameter-space decomposition faithfulness. SoLU's own caveat (hides features rather
   than removing superposition) is a confound G4 must check.
Also seen, weaker: SAE-side work varies the SAE activation (TopK, JumpReLU, Gated), not the host model's, so it
does not answer G4. MoE interpretability (2604.02178, 2604.14434, 2603.05805) studies experts as units, not
decomposition of weights. VPD (2610.20748) uses GELU throughout, lists nonlinear subcomponent interactions at MLP
neurons as future work (Sec B.11, preliminary) and speculates that interacting subcomponents are hard to ablate.
Toy SPD uses ReLU residual MLPs. Neither varies the activation.

## Verdict
Open as of 2026-11-05 for the exact question (PD faithfulness x activation, matched loss). Adjacent work is dense
enough (bilinear, SQS, ST, SoLU) that "architecture changes interpretability" is not novel; the novelty is the
measurement (PD faithfulness and adversarial-mask robustness) and the matched-loss protocol. Risk of being scooped
is moderate: Goodfire's "Intentional Design" pillar (seen via mirror only) and the Pearce/Dooms group are the likely
actors. A negative result (no effect after matching loss) would still be publishable as a controlled study.
Do not write "first" until `lit/novelty.md` is regenerated with a re-run of the PD queries in a month.

## Most defensible claim
"In controlled toy and small-LM settings, with loss, width and training budget matched, the MLP nonlinearity
changes SPD/VPD faithfulness measures (adversarial-mask KL, alive-subcomponent count, ground-truth recovery where
known) by X." Defend only the measurement and the controls; do not claim that a form is "more interpretable".
Needed controls: matched validation CE (tune width or steps, report both), equal parameters, same seeds across arms,
per-arch tuning budget for PD hyperparameters (same number of trials), same causal-importance-function capacity,
random-mask and matched-norm ablation baselines, and ground-truth feature recovery on the SPD toy so the result is not
only a proxy. Report the arm where decomposition fails.

## Predicted direction and mechanism (a prior, not evidence)
Competing mechanisms, so the sign is not obvious:
- ReLU: pre-activations below zero produce exactly zero output, so an ablation that moves a neuron further into its
  dead zone has no effect, and mask noise on inactive subcomponents is absorbed. Prediction: ReLU gives lower
  adversarial KL than GELU at equal CE for small mask perturbations. The adversary can still push a neuron across zero,
  so the gain should saturate with attack budget.
- GELU/SiLU/SwiGLU: nonzero slope for negative pre-activations (and gate leakage) means every subcomponent perturbs
  every neuron slightly; small contributions are spread over many subcomponents, which is what an adversarial mask
  exploits. Prediction: more alive subcomponents and higher adversarial KL than ReLU. SwiGLU's multiplicative gate may
  partly recover sharpness.
- Bilinear/GLU without a gate nonlinearity: no dead zone, but the output change from ablating a rank-1 piece is an
  exact low-degree polynomial in the mask, so causal importance is well defined and cross-terms are explicit.
  Prediction: best faithfulness at small mask budgets and best ground-truth recovery on the toy, but no free
  absorption of leakage, so robustness to large adversarial masks may not beat ReLU. This is the least certain arm.
- TopK/sparse activations: exactly K active units, so most subcomponents are inactive by construction; predict the
  sharpest alive-subcomponent count, with the confound that TopK changes the function class (check loss).
- SoLU: concentrates activation in few neurons; predict between ReLU and TopK, with SoLU's hidden-feature caveat.
Overall prior: ReLU/TopK > bilinear > SwiGLU > GELU on adversarial-mask robustness; bilinear first on toy recovery.
P(any effect at matched loss exceeds seed noise) about 0.5 on the SPD toy, lower in 4-layer LMs because
the SPD toy has a hand-built superposition structure. Kill early: if toy arms differ by less than seed spread across
3 paired seeds, stop and record the negative result.

## Next steps
1. CPU toy: SPD residual-MLP compressed-computation task with ReLU/GELU/SiLU/bilinear/TopK arms; use the L4-loss
   variant (2607.04800) to avoid the noisy-residual caveat of 2606.14673.
2. Read SoLU and 2502.02470 full text before citing; run the PD queries again in a month.
