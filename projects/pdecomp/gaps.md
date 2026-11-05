# Gap analysis
<!-- Current state. Sources: ideas/{diff-deltas,faithfulness,cross-field,weight-space,frontier}/, lit/snowball.md. -->

Six candidates survived scouting (~30 raised across five independent scouts). Probabilities are our
estimates, not evidence. "Value per cost" assumes CPU toys first, then 1-3 slice-weeks.

| # | Gap | Evidence it is open | Closest work | P(positive) | Ceiling | Cost |
|---|---|---|---|---|---|---|
| G1 | **Delta-PD**: decompose fine-tuning / RL weight deltas, base model as the ablation reference | No paper runs APD/SPD/VPD on an existing delta. Ranked #1 by 3 of 5 scouts independently. | VPD Sec 7.5 (one-line future work); Bushnaq LW post (masks base components, does not decompose the delta); bigsnarfdude blog (SVD on delta, VPD on full model); TopKLoRA (retrains with its own adapter); Sparse Model Diffing (deltas are dense in the parameter basis: main risk) | 0.5 | High: mechanism-level model diffing for SFT/RL/safety fine-tunes; cheap relative to whole-model PD | CPU toy now; 1B base/SFT/DPO pairs on slices |
| G2 | Calibrate mask-attack scope/strength/family against ground truth: which evaluation separates correct from defective decompositions (incl. false positives on correct ones) | Calibration half: no prior work found (ideas/faithfulness/novelty-G2.md). Attack-ensemble half: obvious (APGD/restarts) and Goodfire says a stronger adversary is 'being looked into' | VPD App A.3.4 already uses Adam-based persistent PGD with a batch-shared mask, deliberately weaker than per-datapoint worst case (Sec 7.3); Angsten LW audits (restarts, step sweeps) | 0.5 | Medium: evaluation methodology | CPU toys |
| G3 | Cluster subcomponents by ablation interaction (post-hoc Bernoulli ablation probe on the frozen decomposition), not CI co-activation | Principle not novel (Hedonic Neurons arXiv:2509.23684 clusters neurons by ablation synergy; ArchDetect arXiv:2006.10965); no PD application found (ideas/clustering/novelty-G3.md) | VPD App A.8 MDL merge on co-activation; VPD App B.11.2 computes an interaction proxy but does not cluster on it | 0.45 | Medium: an evaluation-backed method improvement for PD | CPU toys (experiment running) |
| G4 | Decomposability by design: activation choice (bilinear/ReLU vs GELU/SwiGLU) changes PD faithfulness at matched loss | No prior work found | Bilinear MLPs (2410.08417) argue weight-based interp, not PD | 0.4 | Medium-high; connects to actbias and architecture design | CPU toys, then small LMs |
| G5 | Faster PD: data-driven warm start / closed-form K-SVD-style updates | No PD warm-start found | K-SVD (textbook); SAE init literature | 0.4 | Medium: only matters if >= 3x | CPU toys (prototype running) |
| G6 | PD under parameter symmetries (rescaled function-identical copies) | Not tested | Symmetry teleportation literature | 0.5 | Low: a section, not a paper | CPU toys |

Notes that change plans:
- Bhagat et al. (arXiv:2606.14673): the compressed-computation toy's advantage comes from a noisy
  residual mixing matrix, not superposed computation. Do not use it as ground truth without that caveat;
  their L4-loss variant (arXiv:2607.04800) is a cleaner testbed.
- VPD citer coverage in OpenAlex is near zero (too new). Re-run the snowball and the G1 citer check
  before the decision and again monthly.

## Pending
Decision after the CPU prototypes (delta_pd, warmstart) and the SPD reproduction report land.
Kill criteria for G1: Delta-PD does not beat SVD of DeltaW on per-change recovery, or costs more than
whole-model PD.
