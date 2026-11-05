# Gap analysis
<!-- Current state. Sources: ideas/{diff-deltas,faithfulness,cross-field,weight-space,frontier}/, lit/snowball.md. -->

Six candidates survived scouting (~30 raised across five independent scouts). Probabilities are our
estimates, not evidence. "Value per cost" assumes CPU toys first, then 1-3 slice-weeks.

| # | Gap | Evidence it is open | Closest work | P(positive) | Ceiling | Cost |
|---|---|---|---|---|---|---|
| G1 | **Delta-PD**: decompose fine-tuning / RL weight deltas, base model as the ablation reference | No paper runs APD/SPD/VPD on an existing delta. Ranked #1 by 3 of 5 scouts independently. | VPD Sec 7.5 (one-line future work); Bushnaq LW post (masks base components, does not decompose the delta); bigsnarfdude blog (SVD on delta, VPD on full model); TopKLoRA (retrains with its own adapter); Sparse Model Diffing (deltas are dense in the parameter basis: main risk) | 0.5 | High: mechanism-level model diffing for SFT/RL/safety fine-tunes; cheap relative to whole-model PD | CPU toy now; 1B base/SFT/DPO pairs on slices |
| G2 | Stronger mask adversary (AutoAttack-style: APGD, Frank-Wolfe vertices) + toy calibration of which robustness measure predicts mechanism recovery | VPD uses plain PGD; no ensemble or certified mask attack found | Adversarial circuit eval (2407.15166) attacks inputs, not masks; VPD Sec 7.5 asks for better samplers | 0.6 (attack will find more failures) | Medium: evaluation paper; positive "method" only if it also improves training | CPU toys |
| G3 | Cluster subcomponents by ablation interaction (non-additivity), not co-activation | VPD names clustering as open; no PD use of interaction scores found | Neuron-level interaction / Shapley work | 0.45 | Medium-high if it recovers multi-matrix mechanisms co-activation misses | CPU toys |
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
