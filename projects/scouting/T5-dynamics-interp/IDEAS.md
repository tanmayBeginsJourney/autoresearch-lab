# T5 scouting: interpretability of learning dynamics, used to intervene

Searches: `researchctl lit-search` (s2/arxiv/openalex) plus one web search, 2026-11-05. Grokking-centred
work is saturated in 2025-26 (dozens of 2026 arXiv papers on prediction, rank, refits, spectra).
"Recalled" marks prior work I know but did not re-verify in this session.

| # | Hypothesis (positive payoff) | Closest prior work found | Verdict |
|---|---|---|---|
| 1 | Truncating the cumulative update W-W0 to its top-k singular directions (or resetting the tail) at the end of memorization speeds grokking. | arXiv:2610.35112 (rank compression follows grokking; imposing low rank does not speed it), arXiv:2606.04405 (low-rank decay), arXiv:2506.05718 (non-Euclidean regularizers and grokking) | Killed: known, and the literature predicts failure. |
| 2 | Critical slowing down (rising lag-1 autocorrelation / variance of update projections) forecasts phase transitions (grokking, induction heads) early. | zenodo 10.5281/zenodo.21481019 (prospective audit of classical early-warning signals for grokking), arXiv:2603.29805 (spectral early warnings) | Killed: done. |
| 3 | Update-statistic predictor (stable rank, IPR) of grokking time, used to stop runs early. | arXiv:2604.20923 (ILDR early detection), arXiv:2610.35112 (IPR not a specific early warning) | Killed. |
| 4 | Periodic train-only ridge refit of the readout reveals and accelerates latent generalization. | arXiv:2610.22593 ("train-only refit", "refit-first recovery") | Killed: done. |
| 5 | Keep only update directions on which data-disjoint replicas agree (consensus projection) to suppress memorization. | AND-mask (arXiv:2009.00329), gradient agreement filtering (arXiv:2412.18052), replica-overlap probe of grokking (arXiv:2609.25634), local SGD (recalled) | Killed: incremental. |
| 6 | Transplant a small proxy model's circuit subspace into a wide model to skip the plateau. | Embedding transfer for grokking (recalled, Xu et al. 2024), Net2Net/LiGO growth (recalled), arXiv:2512.05117 (universal weight subspace) | Killed: known. |
| 7 | Weight-space data attribution picks "circuit-seeding" examples; front-loading them speeds emergence. | arXiv:2601.21996 (mechanistic data attribution to LLM units); parity curricula (recalled, Cornacchia & Mossel 2023) | Killed: incremental, and modular arithmetic is too symmetric to test it. |
| 8 | Freeze/decay LR on "committed" singular directions of W-W0 to protect formed circuits while new ones form. | GPM-style projection for continual learning (recalled), Muon/Shampoo spectral preconditioning | Shelved: incremental vs GPM. |
| 9 | **Random-label gradient projection (RLGP).** The gradient at the current weights under randomly permuted labels estimates the label-agnostic memorization direction; removing its projection from the real gradient leaves the generalizing component, so circuits form sooner and noisy labels are fitted less. | Closest: AND-mask (masks gradients that disagree across examples), random-label prediction heads as a memorization probe (ICLR 2026, measurement only), OrthoGrad/perp-Grad (projects out the weight direction, arXiv:2607.16231 for noisy labels). 4 lit-searches + 1 web search found no projection on a random-label gradient. | Selected for the kill-test (`PREREG.md`). |

Why RLGP: it is a one-line, optimizer-agnostic intervention with a mechanistic rationale (separates
label-structured from label-agnostic update components) and a falsifiable payoff (fewer gradient
evaluations to generalize; less noisy-label memorization). Its cost (2x backward) sets a high bar.
