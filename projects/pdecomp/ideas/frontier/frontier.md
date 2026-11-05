# Mech-interp frontier (Nov 2026) mapped to our constraints

Scan date 2026-11-05, 35 min budget. Constraints: 8 MIG slices (24-48 GB), 150 H200-h/month, one person
plus agents, ~6 months to NeurIPS 2027. Crowding grades are my judgement from the searches listed at
the end; novelty is unaudited until a `lit/novelty.md` exists for the chosen idea.

## Bottom line
| Rank | Project | Novelty (1-5) | P(positive) | Feasibility (1-5) | Score |
|---|---|---|---|---|---|
| 1 | Delta-PD: input-conditional PD of post-training weight diffs separates behaviours that static weight subspaces cannot | 4 | 0.65 | 4 | 10.4 |
| 2 | Decomposability by design: activation function / MLP form vs VPD faithfulness at matched loss | 4 | 0.45 | 4 | 7.2 |
| 3 | PD against ground truth: VPD vs SWD vs transcoders on weight-sparse testbeds with known units | 3 | 0.55 | 4 | 6.6 |

P(positive) is a prior, not evidence. #1 also absorbs the refusal and editing interests; #2 the
activation-function interest; the GPU-systems interest fits as the efficiency layer of #1 (item E below).

## Sources for "the frontier"
- Sharkey et al., Open Problems in MI (arXiv:2501.16496): decomposition lacks foundations (2.1.2d),
  intrinsically decomposable models (2.1.2e), validation via model organisms and benchmarks (2.1.4a-b),
  predicting capabilities from fine-tuning (3.3.2), monitoring/auditing (3.2).
- VPD (arXiv:2610.20748) section 7.3-7.5: how much adversarial robustness is enough; better PGD sampler;
  better CI function; multi-position clustering; subcomponent-restricted LoRA editing; gradient
  projection; "apply VPD to only a 'weight diff' obtained by finetuning" (one sentence, p.68).
- Nanda, "A Pragmatic Vision for Interpretability" (LessWrong, Dec 2025): judge methods on proxy tasks
  against simple baselines (prompting, probing, steering). Reviewers in 2027 will apply this test.
- 2026 weight-space work: weight-sparse transformers (arXiv:2511.13653; 2607.02964), Sparse Transformer
  testbed (2610.19907, Oct 2026), SWD (2608.03913, compares to VPD on Qwen2.5 0.5-3B), "weight-sparse
  circuits may be interpretable yet unfaithful" (Drori, LessWrong, Feb 2026), Goodfire's "Intentional
  Design" pillar and "Anatomy of Post-Training" (Jun 2026; seen only via a mirror, unverified).
- I found no 2026 successor agenda to Sharkey et al. and no 2026 Transformer Circuits open-problems post.

## Open problems, who works on them, crowding, fit
| # | Open problem | Main groups (verified items) | Crowding | Fit to us |
|---|---|---|---|---|
| P1 | Faithful decomposition basis (SAE limits, weight-native alternatives) | Goodfire (APD/SPD/VPD), Anthropic (CLTs, attribution graphs), OpenAI (2511.13653), SWD | High for whole-model scaling | Low: scaling race needs compute |
| P2 | How much faithfulness is enough; adversarial robustness of explanations | Goodfire (VPD 7.3), Drori 2026 | Medium | Medium: toy-scale, theory-heavy |
| P3 | Model diffing: what did fine-tuning/RL change | DIT (2510.05092), WeightWatch (2508.00161), Delta-Crosscoder (EMNLP 2026), Safety subspaces (2505.14185), RL sparse subnetworks (2507.17107) | Medium in activation space, low for PD | High: deltas are small and cheap |
| P4 | Intrinsically decomposable models | OpenAI weight-sparse, bilinear MLPs (2410.08417, 2602.05635), Goodfire Intentional Design | Medium, but none targets PD | High: small LMs on s24 |
| P5 | Validation against ground truth | Sparse Transformer testbed (2610.19907), InterpBench, Tracr | Low-medium | High: CPU/s24 |
| P6 | Attention / QK mechanisms | Anthropic (attribution graphs omit attention), 2602.04752, Goodfire (multi-position clustering) | Medium | Low: Goodfire named it |
| P7 | Downstream utility (monitoring, auditing, editing, unlearning) | Anthropic auditing games (2503.10965), DeepMind pragmatic agenda, refusal-direction line | High at >7B | Medium only via cheap proxies |

## Candidate projects

### A (rank 1). Delta-PD: behaviours hidden in post-training diffs are separable as input-conditional rank-1 subcomponents
- Hypothesis (positive): when a fine-tune installs several behaviours with comparable weight-diff
  norm, SPD/VPD on Delta W = W_ft - W_base isolates each behaviour into its own subcomponent set
  (ablation removes >= 80% of that behaviour, < 10% collateral), while top-k SVD of Delta W (the
  WeightWatch basis) isolates <= 50%. Mechanism: SVD orders directions by norm and is
  non-identifiable inside near-degenerate singular subspaces, so equal-norm behaviours get rotated
  together; PD's causal-importance function splits them by which inputs use them.
- Ties to 2505.14185 (ICLR 2026): it finds safety is not in linearly distinct static subspaces. Our
  claim is the input-conditional version: separable per input even when not as a fixed subspace.
- Decisive first experiment (CPU, ~1 day): 2-layer MLP or 1-layer attention-only model on a synthetic
  task; fine-tune with 3 backdoors (distinct triggers, Delta W norms matched within 2x) plus a benign
  shift. Run SPD on Delta W only; baselines top-k SVD of Delta W and LoRA-rank truncation. Metric:
  per-backdoor isolation = removal fraction minus collateral. 3 paired seeds. Kill if SVD comes within
  0.1 of SPD on 2 of 3 seeds.
- Scale-up: OLMo-2 1B base -> SFT -> DPO checkpoints (verify which are public), decompose only the
  diffs at a few layers on one 48 GB slice. Downstream test: remove refusal (or an implanted backdoor)
  by ablating delta subcomponents vs the refusal-direction ablation (Arditi et al., NeurIPS 2024) and
  WeightWatch steering, scored on capability retention. Delta rank is low, so cost should be well
  below whole-model VPD (to be measured, no number yet).
- Why not done: VPD lists it as one line of future work; searches for PD of fine-tuning diffs returned
  only SVD-based (WeightWatch), self-description (DIT), crosscoder and PEFT-compression work. One
  blog (bigsnarfdude.github.io, May 2026) ran VPD on a full SFT checkpoint, not on the diff, without
  ground truth. Risk: Goodfire or a MATS scholar does it first; mitigate by moving fast and owning
  the multi-behaviour ground-truth benchmark and the safety baselines.

### B (rank 2). Decomposability by design: which MLP form makes networks parameter-decomposable
- Hypothesis (positive): at matched validation CE, models with bilinear (GLU without nonlinearity)
  or ReLU MLPs give VPD decompositions with >= 2x lower adversarial KL at 160 PGD steps and fewer
  alive subcomponents than GELU/SwiGLU models. Reason: in a bilinear layer a rank-1 ablation changes
  the output by an exact low-degree polynomial term, so causal importance is sharper; smooth gates
  spread small contributions over many subcomponents, which is what the adversary exploits.
- Decisive first experiment (CPU): SPD's toy residual-MLP compressed-computation model trained with
  ReLU / GELU / SiLU / bilinear; measure recovery of the 100 ground-truth features and robustness to
  adversarial masks. Then 4-layer TinyStories-scale LMs per activation on s24, VPD each.
- Why not done: no hits for activation function or architecture vs parameter-decomposition
  faithfulness (queries below). Open Problems 2.1.2e names intrinsic decomposability generally;
  bilinear work studies eigendecomposition, not PD. Risk: no difference (a clean negative), or the
  effect is explained by matched-CE differences; needs matched compute and matched loss.

### C (rank 3). PD against ground truth on weight-sparse testbeds
- Hypothesis (positive): on models whose circuits are known by construction (Sparse Transformer
  testbed 2610.19907; weight-sparse models 2511.13653), VPD recovers ground-truth units with higher
  precision/recall than transcoders and SWD at matched replacement fidelity.
- First experiment (s24, <1 GPU-h): VPD on the released Sparse Transformer models; score subcomponent
  support against known nonzero weights.
- Why not done: 2610.19907 never mentions SPD/VPD (grep: 0 hits); SWD compares to VPD on fidelity,
  not ground truth. Risk: VPD loses; still publishable as a benchmark, but not a positive method result.

### Lower-ranked
- D. Certified faithfulness: interval bounds on output change over all ablations of inactive
  subcomponents (answers VPD 7.3). Novel, but bounds are likely vacuous beyond toys. Score ~2.7.
- E. VPD efficiency: cheaper adversaries (single-step attacks with random init, the fast adversarial
  training recipe), fused masked low-rank kernels. P(positive) high, novelty 2: a tool and a section of
  A, not a paper. Note SWD already claims < 1% of VPD's data at matched fidelity. Score ~6.
- F. Joint QK / multi-position PD: Goodfire named it; competes on their codebase. Score ~3.6.
- G. PD on auditing games / downstream at >7B: compute-infeasible for us. Skipped.

## Next step
Run A's CPU experiment first (cheapest decisive test, highest score). If it fails the kill criterion,
B's CPU toy is next. Neither needs H200. Do the novelty audit for A (citers of WeightWatch, DIT,
2505.14185; Goodfire and MATS 2026 posts) before any GPU spend.

## Searches run (2026-11-05)
researchctl lit-search: "open problems mechanistic interpretability"; "parameter decomposition
fine-tuning weight delta"; "stochastic parameter decomposition"; "activation function parameter
decomposition interpretability"; "bilinear MLP weight-based interpretability"; "weight-sparse
transformers interpretable circuits"; "parameter decomposition compiled transformer ground truth
tracr"; "model diffing fine-tuning weight space interpretability"; "refusal fine-tuning weight delta
rank-one"; "attention QK circuit decomposition interpretability 2026"; "interpretability model
editing parameter subcomponents unlearning"; "decomposability architecture parameter decomposition
faithfulness"; "backdoor fine-tuning weight difference singular vectors isolation"; "train models to
be decomposable interpretability regularizer weights". lit-citers arxiv:2501.16496 (2026 citers:
surveys and governance, no PD-on-diffs work).
Web: "mechanistic interpretability open problems agenda 2026"; "Goodfire interpretability research
2026 parameter decomposition next steps"; "Transformer Circuits thread 2026 open questions";
LessWrong 2026 agenda / pragmatic interpretability; "parameter decomposition fine-tuning OR RL weight
difference 2026"; "SPD toy model architecture activation function decomposability 2026"; Goodfire
2026 blog. Full text read: VPD 7.3-7.5; Open Problems 2.1.2e and 3.4; heads of 2608.03913,
2607.02964, 2610.19907, 2602.05635, 2510.05092, 2608.25390, 2505.14185.
