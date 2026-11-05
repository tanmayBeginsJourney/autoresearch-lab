# Weight-space learning x parameter decomposition: literature map
<!-- Depth: title/abstract only unless marked. Built 2026-11-05 from researchctl lit-search (logged to
lit/queries.log) and web search. One line per paper: id | what it does | relevance to PD. -->

## PD core (cards in lit/cards)
- arxiv:2501.14926 | APD: sparse sum of parameter components, MDL framing | origin of PD; MDL premise.
- arxiv:2506.20790 | SPD: rank-one subcomponents, learned causal importance (CI), stochastic masks | toy baseline.
- arxiv:2610.20748 | VPD: adversarial masks, whole 67M LM, 8-12% PCR under rounded masks | SOTA; card read in full.
- arxiv:2607.13047 | tPD: decompose only a target behaviour; cross-seed component matching | closest to diffing/editing.
- arxiv:2609.37731 | ASPD: activation-grounded PD, single matrices up to 8B; "editable" via SAE metrics | no KE baselines.
- arxiv:2511.08854 | SPD applied to small transformer models | toy-transformer reference.
- LessWrong 2026-06-25 (Bushnaq, "Exploration: fine-tuning with parameter decomposition") | tunes one VPD mask scalar to remove German; beats rank-1/4 LoRA at <=32 tokens; "can't learn anything new" | only PD-editing evidence; no ROME/MEMIT baseline.

## Weight-space interpretability neighbours
- arxiv:2608.03913 | SWD: factor dense W into two sparse factors; circuits at <1% transcoder data; zero-data variant | cheap weight-native competitor.
- arxiv:2511.13653 | Weight-sparse transformers have interpretable circuits | sparsity by training, basis-dependent.
- arxiv:2510.05092 | Diff interpretation tuning: model describes its own weight diffs in language | amortized diffing, not PD.
- doi:10.18653/2026.findings-emnlp.679 | Circuit-tuning: prune-then-tune for sparse fine-tuning diffs (cites SPD/APD) | diffing by construction.
- doi:10.18653/2026.emnlp-main.2453 | Delta-Crosscoder for narrow fine-tuning diffs | activation-space diffing baseline.
- arxiv:2605.24577 | Polymorphism: seeds differ by near-Haar residual rotations; one Procrustes aligns SAEs | symmetry x interpretability; no PD.
- arxiv:2603.30002 | Tracking equivalent mechanistic interpretations across networks | cross-model mechanism identity.
- arxiv:2604.13694 | Weight patching for source-level localization | weight-level causal localization.
- arxiv:2209.10652 | Toy Models of Superposition | standard CPU testbed.

## Weight-space learning (neural functionals, metanetworks)
- arxiv:2301.12780 | DWSNets (Navon et al.): permutation-equivariant layers on MLP weights | amortized-PD backbone.
- doi:10.52202/075280-1085 | Permutation Equivariant Neural Functionals (Zhou et al.) | same.
- arxiv:2305.13546 | Neural Functional Transformers | same.
- arxiv:2402.05232 | Universal Neural Functionals (arbitrary architectures) | same.
- arxiv:2312.04501 | Graph Metanetworks | architecture-agnostic weight processing.
- arxiv:2410.04209 | Equivariant NFNs for transformers | attention symmetries handled.
- arxiv:2610.18471 | Graph metanetworks for transformer LMs | weight-space nets reaching LMs.
- arxiv:2410.04207 | GL-equivariant processing of LoRA weight spaces | LoRA GL(r) gauge, relevant to delta-PD.
- arxiv:2511.12601 | GMN autoencoders: merging via parameter canonicalization | canonical-basis merging, closest to "PD as canonical basis".
- arxiv:2602.01083 | Expressive power of permutation-equivariant weight-space nets | limits of amortization.
- arxiv:2610.21995 | Interfaces and modules: weight-space learning across architectures | same.
- arxiv:2603.10090 | Survey of weight space learning | entry point.
- doi:10.52202/068431-2763 | Model Zoos dataset | zoo for amortized PD.

## Weight generation
- arxiv:2402.13144 | p-diff: diffusion over trained parameters | generative weights.
- arxiv:2501.11587 | Recurrent diffusion for large-scale parameter generation | same, larger.
- arxiv:2609.32833 | Permutation-equivariant flow matching, alignment-free weight generation | symmetry-aware generation.
- arxiv:2601.05052 | DeepWeightFlow: re-basined flow matching for weights | same.
- arxiv:2506.07998 | Generative models of weights mostly memorize | skeptic reference.

## Parameter symmetries and teleportation
- doi:10.52202/068431-1213 | Symmetry teleportation (Zhao et al.) | move along symmetry orbits.
- arxiv:2305.13404 | Teleportation improves convergence and generalization | same.
- arxiv:2210.17216 | Symmetries -> conserved quantities of gradient flow | predicts how balanced trained gauges are.
- arxiv:2506.13018 | Survey: symmetry in NN parameter spaces | entry point.
- arxiv:2609.33527 | Discovering symmetries in parameter spaces | non-obvious symmetries.
- arxiv:2610.18085 | AdamW not equivariant to Q/K reciprocal rescaling; +3.78 PPL; norm-balancing fix | PD optimization is likely gauge-dependent too.
- arxiv:2606.00442 | Exploiting weight-space symmetries to approximate curvature | symmetry as computational tool.
- arxiv:2401.15024 | SliceGPT: computational invariance under residual rotations (RMSNorm folded) | rotation gauge in LMs.
- arxiv:2405.16406 | SpinQuant: learned rotations for quantization | same gauge, used for efficiency.
- arxiv:2606.31963 | Signed-permutation coordinate transport for RMSNorm transformers | residual symmetry group.

## Re-basin, mode connectivity
- arxiv:2209.04836 | Git Re-Basin: weight/activation matching modulo permutations | alignment baseline.
- arxiv:2110.06296 | Permutation invariance conjecture for LMC (Entezari et al.) | same.
- arxiv:1802.10026 / arxiv:1803.00885 | Nonlinear mode connectivity (Garipov; Draxler) | background.
- arxiv:1912.05671 | LMC and the lottery ticket hypothesis | background.
- arxiv:2211.08403 | REPAIR: renormalize after permuted interpolation | merge baseline.
- arxiv:2305.03053 | ZipIt: merge models from different tasks | feature-level merging.
- arxiv:2502.00264 | Rotation symmetry for transformer fusion | beyond-permutation alignment.
- arxiv:2606.23607 | LMC and merging at billion-parameter transformers | scale reference.
- arxiv:2606.04754 | LMC via neuron identifiability | identifiability angle.

## Merging
- arxiv:2212.04089 | Task arithmetic | baseline.
- doi:10.52202/075280-0310 | TIES-Merging (sign conflicts) | baseline.
- doi:10.1109/cvpr52734.2025.01742 | Task Singular Vectors: per-layer SVD, decorrelate across tasks | strongest low-rank merge baseline.
- arxiv:2410.19735 | KnOTS: SVD-aligned LoRA merging | low-rank alignment.
- arxiv:2502.04959 | Iso-C/Iso-CTS: isotropic common + task subspaces | baseline.
- arxiv:2505.23117 | Decom-Renorm-Merge | decomposition-first merging.
- arxiv:2512.22511 | Decompose task vectors into shared + specific parts | same.
- arxiv:2608.25354 | SAE-dictionary task codes for merging | sparse-dictionary merging; nobody uses PD.

## Editing (rank-one updates)
- doi:10.52202/068431-1262 | ROME: closed-form rank-one MLP update with C^-1 key | baseline.
- arxiv:2210.07229 | MEMIT: batched multi-layer edits | baseline.
- arxiv:2410.02355 | AlphaEdit: null-space-projected edits | strongest locality baseline.
- arxiv:2606.26783 | AlphaEdit reproducibility study | check before reusing numbers.
- doi:10.18653/v1/2025.blackboxnlp-1.11 | Side effects of rank-one editing | failure modes PD could explain.
- arxiv:2609.12116 | Rank-level locality beyond parameter support | locality metric.

## Weights -> generalization / compression
- doi:10.1038/s41467-021-24025-8 | WeightWatcher: heavy-tailed spectra predict quality | weight-statistics baseline.
- arxiv:1810.00113 | Margin distributions predict generalization gap | baseline.
- arxiv:2310.05918 | Grokking as compression (linear mapping number) | competes with PD-as-MDL.
- arxiv:2610.35112 | Rank compression tracks the grokking transition | same.
