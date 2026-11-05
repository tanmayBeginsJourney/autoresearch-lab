# T4 scouting: exploiting learned structure for efficient inference

Date 2026-11-05. CPU-only scouting (45 min budget). Every claim below was checked with the searches
listed; "no hit" means 2-4 searches returned nothing, not that the idea is new.

## Candidate hypotheses and novelty verdicts

| # | Hypothesis (structural insight -> win) | Closest prior work | Verdict |
|---|---|---|---|
| H1 | Sparsity in a rotated residual basis is much higher than in the neuron basis; rotation is free via SliceGPT computational invariance. | LaRoSA (arXiv:2507.01299): layerwise PCA rotations absorbed via SliceGPT, Q_l^T Q_{l+1} residual adapters (~1.06x FLOPs), Top-K; beats TEAL (LLaMA2-7B 50%: PPL 5.87 vs 6.80). | **Killed: known.** Only leftover: global (adapter-free) or kurtosis-max rotation = incremental. |
| H2 | Affine gauge: threshold \|x - mu\| instead of \|x\| and fold W mu into a bias. Applies to the down_proj and o_proj inputs, which LaRoSA cannot rotate (Hadamard product, GQA). | TEAL assumes zero-mean inputs; R-Sparse (arXiv:2504.19449) models the non-sparse part as bias/low-rank terms; WINA covers the diagonal-scaling gauge. No hit for centered thresholding + bias folding. | **Incremental** (centering is textbook). Tested as a cheap side check. |
| H3 | Per-KV-group rotation of the value space (W_V <- R^T W_V, W_O <- W_O R) is free even with GQA (LaRoSA says it is not), so the o_proj input can be rotated too. | LaRoSA Sec 3; SliceGPT. | **Incremental, small ceiling**: o_proj is ~8% of a LLaMA-3 layer's weights. |
| H4 | Conditional compute from parameter decomposition: VPD finds ~2% of rank-one subcomponents active per token; compute only those. | VPD (arXiv:2610.20748): rounded (hard) masks recover only 8-12% of pretraining compute; tPD's "7% FLOPs" is analytic, one seed. | **Killed by prior numbers**: hard-masked quality is far below TEAL at 50%. Revisit only if PD faithfulness is solved. |
| H5 | One overcomplete sparse dictionary encoder for the normalized input, amortized over every consumer (Q,K,V or gate/up), with W*D precomputed. | SAEs/transcoders; MoEfication. | **Killed by FLOP/byte argument**: the encoder is a dense m x d read (m >= 2d) per token, which costs more than TEAL saves on 2-3 consumers. SAE reconstruction at k~64 is also worse than 50% magnitude pruning. |
| H6 | Temporal delta sparsity: y_t = y_{t-1} + W(x_t - x_{t-1}) with a thresholded delta. | Delta Networks (RNNs, 2017). | **Killed**: adjacent-token residuals are not close after centering, and errors drift. |
| H7 | **Activation-sparse self-drafting.** Use the target model itself at high TEAL sparsity (70-90%) as the speculative drafter, verify dense, lossless. Structural reason it should beat standalone TEAL: the dense verify pass rewrites the KV cache, so sparsification error never compounds through the history; only the current token's computation is approximate. Zero extra memory, no training, and in memory-bound decode the draft cost scales with (1-s). | Self-spec drafters use layer skipping (Draft&Verify, SWIFT, LayerSkip), weight sparsity (Draft&Verify+Wanda ~1.27x; SD^2 is a separate distilled model), sparse attention/KV (SparseSpec arXiv:2512.01278, SpecAttn, SparseSpec-L arXiv:2607.27735, MagicDec, TriForce), or quantization (QSpec, CAS-Spec). Sirius corrects a sparse model with the full one but is not lossless speculation. Four searches found no TEAL/CATS-style activation-sparse self-drafter. | **Plausibly unclaimed, but a composition of two known techniques.** Best ROI for a decisive test -> kill test below. |
| H8 | Verify-informed hot sets: use the dense verify activations to pick a per-sequence neuron set for the next draft round (contiguous loads, no predictor). | GRIFFIN (prompt-based neuron selection), CoreInfer, SparseSpec's verify-reuse for attention. | **Incremental**: an add-on to H7 if H7 survives. |

## Pre-registration (written before running `kill_test.py`)

Model: SmolLM2-135M (LLaMA arch, SwiGLU, GQA 9/3, 30 layers), fp32 on CPU. Data: wikitext-2 raw test,
calibration windows disjoint from evaluation windows. Seeds 0,1,2 = three disjoint sets of random
evaluation windows (8 x 256 tokens per seed). The same windows are used in every arm.

Sparsifier (TEAL-style, uniform s): per-layer magnitude thresholds at the 4 linear-input sites
(attn-in, o-in, mlp-in, down-in), set from calibration quantiles. Cost = fraction of transformer
linear-weight bytes read (memory-bound decode); the LM head is read by every arm and reported separately.

Baseline (strongest one feasible on CPU): sublayer-skip self-drafter (SWIFT-style granularity:
attention and MLP sublayers separately), with skipped sublayers chosen greedily by smallest calibration
||update||/||residual|| per byte, at matched linear-byte cost.

Metric: greedy 1-step acceptance alpha = P(argmax draft == argmax dense | dense KV history).
Expected speedup (k=4, memory-bound, verify ~= 1 dense step): E[acc]/(k*c_total + 1), with
E[acc] = (1 - alpha^(k+1))/(1 - alpha) and c_total including the LM head.

- **P1 (primary):** at matched cost c = 0.30 (s = 0.70), alpha_TEAL - alpha_skip >= 0.10, mean over
  3 seeds, sign consistent in every seed. **Kill H7 if the gap is < 0.05** or if TEAL's best estimated
  speedup is not above the skip baseline's.
- **P2 (mechanism):** at s = 0.70, alpha with dense history exceeds alpha with sparse (self-generated)
  history by >= 0.05. If not, the "no compounding" argument is wrong.
- **P3 (H2 side check):** at s = 0.50, centered thresholding lowers standalone PPL vs TEAL by >= 2%
  relative in every seed; else H2 is dropped.
