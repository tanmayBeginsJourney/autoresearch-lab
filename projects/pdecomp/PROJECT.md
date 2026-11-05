# pdecomp: weight-space interpretability via parameter decomposition
<!-- Current state only. Rewrite in place each session; history goes in journal.md. Keep under ~80 lines. -->

stage: literature

## Target
- **Primary: NeurIPS 2027**, full paper deadline ~mid-May 2027 (projected May 14-21; official CFP not out).
- Checkpoint: an ICLR 2027 workshop (non-archival, deadlines ~Feb 2027) for early external feedback.
- Not ICML 2027 (projected Jan 22, 2027): 11 weeks from a standing start is too short for the bar in
  `docs/standards.md`. Fallback: ICLR 2028 (~late Sep 2027).

| Milestone | Date | Gate |
|---|---|---|
| Literature + novelty audit done | 2026-11-27 | >= 12 cards, map.md, novelty.md for the chosen gap |
| Gap decision | 2026-12-04 | `## Decision` in gaps.md; kill if no gap clears the ROI bar -> fall back to actbias |
| Reproduction (VPD on a small LM or toy) + cost profile | 2026-12-23 | repro/vpd/REPORT.md with GPU-hours measured |
| H001 preregistered | 2027-01-08 | committed prereg |
| Workshop version | ~2027-02 | first supported claim |
| Results frozen | 2027-04-16 | standards: D-4 weeks |
| Submission | ~2027-05-14 | every box in docs/standards.md |

## Problem
Parameter decomposition (APD -> SPD -> VPD) explains networks as sparse sums of rank-one weight
subcomponents. It is the main weight-native alternative to SAEs/transcoders, but it has only been
shown on a 67M, 4-layer LM, and its faithfulness under the masks used for explanations is low.
What contribution would move this line forward, at our compute (8 MIG slices + 150 H200-h/month)?

## Current understanding
- SOTA is VPD (arXiv:2610.20748): adversarial (persistent PGD) masks + frequency minimality.
  82% pretraining compute recovered unmasked, 27% stochastic, 8-12% with rounded masks; KL to target
  0.83 at 20 PGD steps, 25 at 160. Authors name scale, adversarial robustness, CI-function design and
  multi-position clustering as open problems.
- SPD's per-matrix layerwise loss (cost ~ L x fwd/bwd) was already replaced in VPD by random matrix
  subsets in one pass. "Make SPD cheaper by fixing the layerwise loss" is not a gap.
- ASPD's "8B" claim (arXiv:2609.37731) is one attention matrix at one layer, with no faithfulness metrics.
- tPD (arXiv:2607.13047) decomposes only target behaviour; its "7% FLOPs" is an analytic, single-seed estimate.
- No GPU-hour numbers for VPD exist in any paper; our feasibility at 67M scale is unmeasured.

## Candidate gaps (to be analysed in gaps.md)
1. Parameter decomposition of fine-tuning/RL weight deltas (model diffing in weight space). Cheaper
   than whole-model PD; links to sparse RL updates, editing, refusal. Novelty unverified: adjacent work
   is rank-1 LoRA interpretability (arXiv:2511.06739), Delta-Crosscoder, transcoder adapters.
2. Faithfulness: what adversarial-robustness level makes a decomposition's explanations trustworthy,
   and better adversarial samplers (named open problem; small-scale, compute-light).
3. Compute efficiency of VPD (PGD inner loop, transformer CI function): competes with Goodfire's
   engineering on their own codebase; only worthwhile with an algorithmic idea.

## Open questions
- What does a VPD decomposition of the 67M model cost in GPU-hours on one 48 GB slice?
- Has anyone decomposed fine-tuning deltas with PD (check Goodfire's Silico editing work, tPD's Delta)?

## Next steps
- Snowball: refs/citers of SPD and VPD; read rank-1 LoRA, Delta-Crosscoder, transcoder adapters, CLTs.
- Write lit/map.md; gap analysis for the three candidates.
- When the cluster is back: run nano_param_decomp / param-decomp on a toy, then the 4-layer config, and measure cost.

## Tried recently (last ~2 weeks)
- 2026-11-05: read SPD, VPD in full; subagent cards for APD, tPD, ASPD. Efficiency-of-layerwise-loss idea killed (already solved in VPD).
