# T3 scouting: editing, unlearning, safety interventions in weight space (2026-11-05)

Searches: `researchctl lit-search` (S2/arXiv/OpenAlex, 2023-), arXiv API abstracts. 15 queries total.
Verdict key: KILL = known or incremental; OPEN = no direct prior found in 2-4 searches.

| # | Hypothesis (failure targeted -> mechanism) | Closest prior found | Verdict |
|---|---|---|---|
| H1 | Cross-fact relearning (train on forget subset A, held-out B returns) happens because unlearning builds one *shared* inhibitor; per-example independent targets (random-label / per-sample RMU vectors) remove the shared gate. | Layered Unlearning (arXiv:2505.09500) states and tests the shared-inhibitor hypothesis; random-label unlearning is a standard baseline. | KILL (mechanism known; method = known baseline) |
| H2 | Re-initialise only the forget-specific weight subspace (generalised eigenvectors of forget vs retain key covariance), then repair: info-destroying unlearning at a fraction of UNDO's cost. | RepSelect (2606.17168, collapse top gradient PCs); Minor Components (2605.11685); Distillation Robustifies Unlearning (2506.06278). | KILL (too close) |
| H3 | Edits/unlearning survive *benign* fine-tuning if the delta lies in the null space of benign key covariance (AlphaEdit projection reused for persistence). | Gradient Immunity (2608.05045); null-space LoRA unlearning (2606.10989); AlphaEdit. | KILL (incremental) |
| H4 | Abliteration tripwire: make the refusal direction also carry a capability-critical constant offset, so projecting it out breaks the model. | Fool's Gold decoys (2608.17202) adjacent; affine/mean-preserving ablation (ACE) defeats it trivially. | KILL (adaptive attack is one line) |
| H5 | Relearning barrier from *rotated* ill-conditioning that Adam's diagonal preconditioner cannot fix. | Nothing direct found (2 searches). | KILL on threat model: attacker switches optimizer / reparameterises. |
| H6 | Closed-form ridge-readout adversary as a cheap differentiable proxy of the relearning attack (replace TAR's inner loop). | TAR (2408.00761), AntiDote (2509.08000), SEPS (EMNLP 2025); probe != full fine-tune attack. | KILL (weak proxy, incremental on TAR) |
| H7 | Kernel-decoupled unlearning: first-order held-out recovery after relearning on A is eta * sum_a <grad logp_b, grad logp_a>; penalise positive cross-forget tangent-kernel alignment during unlearning so relearning A cannot move B. | KAU (2610.31075) uses the tangent kernel for KG-embedding relearning under continued training (shared entity embeddings, closed-form response), not deep nets / adversarial subsets. Layered Unlearning attacks the same symptom empirically. SAM unlearning (2502.05374), One-Point Contraction (2507.07754, collapses forget reps -- the opposite design). | OPEN, moderate novelty (closest: KAU) -> tested |
| H8 | Symmetry-imbalanced (rescaled) neurons alter fine-tuning effective LR for safety neurons. | T2-symmetries scout territory; canonicalisation undoes it. | KILL |

## Preregistration for H7 (written before running `kd_unlearn.py`)
Setup: synthetic key-value memory. MLP 64-256-256-32 (ReLU) memorises N=600 random key->label facts
(independent, so a model never trained on B cannot learn B from A). Forget F=100 (A=50, B=50), retain R=500.
Oracle = same init trained on R only. Seeds {0,1,2}, shared across arms (paired).
Arms (150 full-batch Adam steps, lr 1e-3, retain CE on R in all):
NPO (beta=0.5); RMU-shared (layer-2 reps of F -> one fixed random vector, retain reps anchored);
RandLabel (per-example random wrong labels; the H1/LU-style decorrelated baseline);
KD = RandLabel + lambda * mean_{a!=b in F} relu(cos(g_a, g_b))^2, g_i = grad_theta log p(y_i|x_i), lambda=3.
Attack: Adam lr 1e-3 on A only (true labels), T=100 full-batch steps; also direct relearn on B, T=25.
Primary metric: excess held-out recovery = acc_B(attacked arm) - acc_B(attacked oracle).
Prediction (H7 survives iff all hold): KD's excess recovery <= 0.5 x the smallest excess recovery among
{NPO, RMU-shared, RandLabel}, in 3/3 seeds, with pre-attack F acc <= 0.10 and retain acc within 2 points of
RandLabel. Otherwise KILL. If no baseline shows excess recovery >= 0.10, the synthetic setting cannot test
H7 (report as uninformative, not as support).
