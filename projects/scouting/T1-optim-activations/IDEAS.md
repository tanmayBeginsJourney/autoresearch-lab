# T1 scouting: optimization dynamics x activation functions x parametrization
Scouting note, 2026-11-05. Starting point: our CPU repro of arXiv:2610.14500 (ReLU mean shift lowers final
teacher-student MSE 5-6.5% under Adam after per-arm LR tuning and a bias control; GELU: no gain).
Searches: `researchctl lit-search` (s2/arxiv/openalex) + web search. "Closest prior work" lists only papers
whose title/abstract we actually saw; none were read in full (scouting depth).

## Candidate hypotheses

1. **Optimizer-level centering for Adam.** Remove the input-mean component from each weight gradient
   (G - g_b mu^T) and give it to the bias; gets the shift benefit for every linear layer with no
   architecture change. Searches: "mean-normalized stochastic gradient", "optimizer centering weight bias
   affine", "function-space orthogonalization weight bias". Closest: SBO-AdamW, arXiv:2607.22927 (Jul 2026,
   exactly this for AdamW in transformers; reports bias drift); Wiesler et al. 2014 mean-normalized SGD;
   Natural Neural Networks (arXiv:1507.00210); K-FAC is invariant to centering (arXiv:1503.05671).
   **Killed: known.**
2. **Gradient centralization as centering.** Subtracting the fan-in mean of each row of the gradient
   = projecting out the all-ones input direction, i.e. centering when the activation mean is uniform.
   Closest: Gradient Centralization (Yong et al., ECCV 2020); Fuhl & Kasneci 2021. **Killed: known.**
3. **Analytic, weight-norm-tracking shift.** For homogeneous phi the mean scales with the pre-activation
   std (= ||w_i|| after LN), so a constant b is right only at init; use b_i = c ||w_i||. Closest:
   Normalization Propagation (Arpit et al. 2016, closed-form ReLU moments from weight norms); mean-only BN
   + weight norm (Salimans & Kingma 2016). Note: no non-linear positively homogeneous phi has zero Gaussian
   mean, so shifts must be scale-aware. **Killed: known idea.**
4. **Slope centering explains the GELU null.** GELU's mean is small but its mean slope is 0.5; the
   pathology could be the linear component, fixed by Raiko-style slope centering + linear shortcut.
   Closest: Raiko, Valpola, LeCun AISTATS 2012; Schraudolph 1998 (Tricks of the Trade); Vatanen et al. 2013.
   No Adam/transformer revisit found, but that would be a re-test of a 2012 idea. **Killed: incremental.**
5. **Mean bias causes outlier features / quantization pain; centering fixes it.** Closest: Averis,
   arXiv:2603.10444 (rank-one mean bias in FP4 LLM training, mean removal); He et al. arXiv:2405.19279
   (outlier features); massive-activation / residual-sink papers (arXiv:2508.03616, 2601.22966).
   **Killed: crowded and known.**
6. **New zero-mean activation function.** Closest: ZC-Swish arXiv:2604.19453; bipolar ReLU
   arXiv:1709.04054; ELU arXiv:1511.07289. **Killed: crowded, low ceiling.**
7. **Width/sign-descent theory.** Adam's sign-like update turns an input mean mu into coherent drift of the
   effective bias W2 mu at rate ~ eta*m*mu, so the gain should scale with width and mu^2/E[a^2]. Closest:
   LeCun et al. 1991 (input-mean eigenvalue), Efficient BackProp 1998; sign-descent views of Adam
   (arXiv:2502.00213). Mostly classical conditioning restated for Adam; analysis paper.
   **Killed: low novelty, analysis-only.**
8. **Centering x orthogonalized optimizers (Muon).** W2's gradient is G = sum_t delta_t a~_t^T + g mu^T
   (g = sum_t delta_t). Muon gives every singular direction of G unit weight, so the rank-one mean term g mu^T,
   whose raw singular value |g| ||mu|| carries mostly batch noise, gets a full-size step on the effective
   bias. Prediction: uncentered activations hurt Muon more than Adam, and centering (architectural shift,
   or projecting the input-mean direction out of G before Newton-Schulz) gives a larger gain under Muon.
   Searches: "Muon activation centering", "Muon nonzero mean activations bias", "gradient centralization
   Muon", "orthogonalized gradient centering reparametrization". Closest: SBO-AdamW (Adam only); NorMuon
   arXiv:2510.05491 (row-wise normalization after NS, no centering); arXiv:2606.03899 (rank-1 spiked model of
   Muon); arXiv:2601.13474 (Muon preconditioning vs activation rank). Nothing on centering x Muon found.
   **Survives (moderate novelty; risk: "SBO for Muon").**

## Chosen: idea 8. Preregistration (written before running; exploratory CPU kill-test)
Setup: the repro teacher-student, ReLU, (d, m, L) = (20, 80, 10), pre-LN residual MLP, no biases, unshifted
teacher, fresh N(0, I) batches of 128, MSE, 3000 steps, constant LR, paired seeds 0, 1, 2 (same teacher, init and
batches for every arm/LR). Arms: Adam-U, Adam-S (phi - 0.230), Muon-U, Muon-S, Muon-P (unshifted; batch-mean
direction of each W2 input projected out of W2's gradient before momentum/NS). Muon: momentum 0.95 Nesterov,
NS5 (Keller Jordan coefficients), update scaled by 0.2*sqrt(max(rows, cols)) so LR is Adam-comparable; applied
to W1 and W2. LR grid {1e-3, 3e-3, 1e-2} for every arm; each arm picks its LR on validation; report test.
- P1 (mechanism): relative gain of Muon-S over Muon-U >= 2x the relative gain of Adam-S over Adam-U,
  with Muon-S better in 3/3 seeds.
- P2 (method): best of {Muon-S, Muon-P} beats the strongest baseline among {Adam-U, Adam-S, Muon-U}
  (each at its own best LR) by >= 5% in test MSE, better in 3/3 seeds.
- Deviation (decided before seeing any result): the CPU was shared (load ~15 on 4 cores), so the
  45-run x 3000-step design would have taken ~24 min against a 12-min budget. It was killed without reading
  any output and replaced by 1000 steps with 2-point grids per optimizer: Adam {1e-3, 3e-3}, Muon
  {3e-3, 1e-2}. Predictions and thresholds unchanged.
- Verdict: promising only if P1 and P2 both pass; dead if both fail; otherwise "mechanism only" or
  "method only" with the stated caveat. Report every run, diverged included.
