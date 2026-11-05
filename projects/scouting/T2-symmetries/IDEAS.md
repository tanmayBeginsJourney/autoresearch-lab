# T2 scouting: parameter-space symmetries and teleportation
<!-- 2026-11-05, ~45 min. Novelty checks are search-level (titles/abstracts, some full text), not audits. -->

Shared fact behind most ideas: the loss is invariant on symmetry orbits, but Adam (diagonal) and Muon
(orthogonal-equivariant, scale-sensitive) are not. Under Adam with eps=0 and no decay, a diagonal scale
teleport equals a per-parameter LR change, so scale teleports for Adam reduce to LR reparameterization.

| # | Hypothesis | Closest prior work found | Verdict |
|---|---|---|---|
| 1 | Eigenbasis teleportation: periodically rotate the shared residual basis (absorbed into weights) into the gradient-covariance eigenbasis, so plain Adam acts like SOAP at Adam's per-step cost | ARO (arXiv:2602.09006) Sec 6 derives ARO as "symmetry teleportation under rotational symmetry groups" and tests globally shared residual rotations (NanoGPT/NanoChat); SOAP (arXiv:2409.11321); AdaDiag++ (arXiv:2502.07488); EGOP-reparameterized Adam (arXiv:2502.01594) | Killed (taken) |
| 2 | Periodic norm balancing of scale-symmetric pairs (Q/K, V/O, gain/W_in, up/down) during AdamW training | SAGGS (arXiv:2610.18085: Q/K pairs, -0.37 to -0.78 PPL, checkpoint moment transport); Lau & Su symmetry-compatible optimizers (arXiv:2605.18106); "The Loss Does Not See the Basis, but Adam Does" (arXiv:2608.05136) | Killed (taken) |
| 3 | Canonical gauge for Adam fine-tuning of downloaded checkpoints (full FT or LoRA) | Reduces to per-neuron LR multipliers under Adam (see shared fact); LoRA gauge atlas (arXiv:2610.12728); Stiefel-AdamW (arXiv:2609.21039) | Killed (incremental: LR in disguise) |
| 4 | Optimizer-aware gauge fixing: teleport an Adam-pretrained checkpoint into a Muon-suited gauge to fix the Adam->Muon fine-tuning mismatch | "Can Muon fine-tune Adam-pretrained models?" (arXiv:2605.10468) reports the mismatch, fixes it with LoRA, no reparameterization; no gauge-based fix found in 3 searches | Novel -> tested, **killed on CPU** (see RESULTS.md) |
| 5 | Teleport trainer weights into a quantization-friendly gauge at each RL weight sync, to cut FP8/FP4 rollout-vs-trainer mismatch | FP8-RL (arXiv:2601.18150), Jet-RL (arXiv:2601.14243), QuRL (arXiv:2602.13953), AIS (arXiv:2605.13907) correct the mismatch with importance sampling; SmoothRot/SpinQuant do the gauge part for PTQ | Plausible but incremental (known PTQ applied inside RL); also, test 4 shows teleporting the trainer can hurt later optimization |
| 6 | Gauge-aligned delta compression of RL weight sync (send g.W_new - W_old) | SparseRL-Sync (arXiv:2605.07330) already ~100x lossless; orbit drift per sync is small | Killed (small headroom) |
| 7 | Symmetry canonicalization (perm + rotation + sign) to merge independently trained LMs | Rotation symmetry fusion (ICML 2025, arXiv:2502.00264), signed-permutation transport (arXiv:2606.31963), depth alignment (NeurIPS 2026 sub.), arXiv:2606.23607 | Killed (crowded) |
| 8 | Rotation teleport to make weights 2:4-prunable at zero function change | RotPruner; DenoiseRotator (arXiv:2505.23049, NeurIPS 2025) | Killed (taken) |
| 9 | Random-gauge copies as free diversity for soups/upcycled MoE experts (Adam non-equivariance gives different trajectories) | Nothing direct found (2 searches) | Unexplored; reduces to random per-neuron LR multipliers under Adam, weak |

Why 4 was picked: only candidate with no prior fix found, a concrete practical problem (most open
checkpoints are Adam-trained; Muon is spreading), and a non-trivial mechanism (Muon's response to a
diagonal gauge is not an LR rescaling).
