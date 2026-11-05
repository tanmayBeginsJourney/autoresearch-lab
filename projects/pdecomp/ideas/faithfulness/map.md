# Faithfulness and adversarial robustness of explanations: literature map
Scope: what is known about evaluating explanation faithfulness, attacking explanations, and adversarial-ML tools for mask spaces.
Depth: web search + abstract/summary level (WebFetch summaries, not full text) except VPD/SPD, which were read earlier. IDs marked (V) were returned by `researchctl lit-search` or arXiv pages; (W) came from web results only.
Cards exist for 2407.15166, 2311.17030, 2602.22968 (abstract-level; upgrade to full before citing details).

## T1. Circuit faithfulness metrics and their fragility
Established: faithfulness is measured by ablating everything outside the explanation and comparing outputs; the number depends on the ablation scheme, task, input set and discovery method.
Open: no agreed worst-case or certified notion; completeness is underspecified (backup paths, gates).
- arxiv:2211.00593 (V) Wang et al., IOI circuit; completeness/minimality/faithfulness criteria.
- Causal scrubbing, Redwood (LessWrong post JvZhhzycHu2Yd57RN; no arXiv found) (W): behavior-preserving resampling ablation to test hypotheses.
- arxiv:2304.14997 (W) ACDC, automated circuit discovery.
- arxiv:2310.10348 (W) Syed et al., attribution patching / EAP.
- arxiv:2403.17806 (V) Hanna et al., Have Faith in Faithfulness; overlap is not faithfulness; EAP-IG.
- arxiv:2407.08734 (W) Miller et al., faithfulness metrics not robust to ablation-method choices (COLM 2024).
- arxiv:2505.10039 (W) Rethinking circuit completeness: AND/OR/ADDER gates (NeurIPS 2025).
- arxiv:2607.01940 (W) Conditional co-ablation: self-repair backups reappear only after primary ablation (incompleteness 0.75 -> 0.21 on IOI).
- arxiv:2510.00845 (W) Mech-interp as statistical estimation; circuit instability across data/hparams; recommends stability reporting.
- arxiv:2606.06267 (W) Many Circuits One Mechanism; many-to-one structure-to-function; source-level evaluation inflates faithfulness.

## T2. Adversarial / worst-case evaluation of explanations
Established: worst-case inputs break published circuits; SAE concept latents are attackable by input perturbations.
Open: nobody found optimizing the ABLATION (mask) adversarially for circuits or SAEs outside PD; no certificates over ablation sets.
- arxiv:2407.15166 (V) Adversarial Circuit Evaluation: random-search worst-case inputs, order-statistics percentile bound.
- arxiv:2505.16004 (W) Li et al., GCG-style attacks on SAE latents (Llama-3-8B, Gemma-2-9B); EACL 2026.
- arxiv:2604.18756 (W) Robustness of SAEs (SAE bottleneck changes adversarial dynamics); only related work read.
- arxiv:2406.01365 (V) Adversarial model manipulation of visual circuits (title only).
- arxiv:2610.20748 (V) VPD: persistent PGD masks; KL 0.83 at 20 steps, 25.3 at 160, 40.2 at 320; authors say full robustness may not be desirable.
- arxiv:2407.15549 (W) Latent Adversarial Training improves robustness (Casper et al. line; perturbs residual stream). Training-side analogue of VPD's persistent adversary; exact authorship unchecked.

## T3. Causal abstraction, interchange interventions, subspace illusions
Established: optimized interventions can produce right behavior through wrong mechanism.
Open: how to separate feature effect from dormant-path/off-manifold effect in a general way.
- arxiv:2303.02536 (V) DAS (Geiger et al.).
- arxiv:2311.17030 (V) Makelov et al., subspace patching illusion (ICLR 2024).
- arxiv:2401.12631 (V) Reply to Makelov et al. (not read).
- arxiv:2511.04638 (V) Divergent representations from causal interventions (title only; closest to the off-manifold issue).
- arxiv:2501.06164 (V) Model Alignment Search (title only).

## T4. SAE / transcoder faithfulness
Established: reconstruction/spliced-in loss plus downstream-task benchmarks; proxy gains do not reliably transfer.
Open: worst-case/ablation faithfulness of SAE features; no certificate.
- arxiv:2503.09532 (V) SAEBench.
- arxiv:2501.18823 (W) Skip transcoders beat SAEs.
- arxiv:2405.08366 (V) Principled SAE evaluations (title only).
- Weight-native decomposition (SPD arxiv:2506.20790, VPD) is the only family where ablation masks are part of the training objective.

## T5. MDL framing
- arxiv:2410.11179 (W) MDL-SAEs: sparsity is a partial proxy for description length; independent additivity. No MDL treatment of adversarial residual found.

## T6. Adversarial-ML tools applicable to mask boxes
Established: PGD, ensembles of step-free attacks, bound propagation, smoothing certificates are mature for input perturbations.
Open: applying them to hidden-mask perturbation sets of an explanation.
- arxiv:1706.06083 (V) Madry PGD.
- arxiv:2003.01690 (W) AutoAttack: APGD-CE/DLR (step-size-free), FAB, Square; ensemble reveals broken defenses.
- doi:10.1007/978-3-030-58592-1_29 (V) Square Attack (black-box, random search).
- arxiv:2607.19855 (V) Minimum-norm attack ensembles (2026; title only).
- arxiv:1810.12715 (V) IBP for verifiable training.
- arxiv:1811.00866 (V) CROWN; alpha/beta-CROWN (W: arxiv:2103.06624).
- arxiv:1902.02918 (V) Randomized smoothing (Cohen et al.).
- arxiv:1811.10828 (W) Frank-Wolfe attacks (Chen et al., AAAI-20): projection-free, LMO; arxiv:2512.10936 (W), 2102.07360 (V) structured FW attacks.
- arxiv:2405.05075 (W) Sparse-PGD and Sparse-AutoAttack (ICML 2024): l0 attacks via continuous mask relaxation + top-k projection.
- arxiv:2601.14519 (W) How worst-case are adversarial attacks: concentration-factor family between random and adversarial perturbations.
- arxiv:2602.22968 (V) Certified Circuits: randomized deletion smoothing over dataset edits (certifies discovery stability, not behavior).
- arxiv:2602.05184 (W) Scale-aware interpretability agenda; aspires to worst-case guarantees, no results.

## Synthesis
1. Faithfulness-by-ablation is the shared currency; its known failure modes are scheme dependence, backup paths and optimized-intervention illusions.
2. Adversarial evaluation of explanations exists for inputs (circuits, SAEs). Adversarial optimization of the ablation set exists only inside VPD, with a plain PGD adversary.
3. Certification in interpretability exists only for discovery stability (2602.22968). Nothing found certifies an explanation's output under an ablation set.
4. VPD's mask space is a box (per-subcomponent mask in [CI,1]), differentiable, with known structure (first-layer pre-activation is affine in the mask for fixed input). Verification and attack-ensemble tools fit it directly.
5. Search was not exhaustive (no full-text reading; citers of VPD returned 0 on OpenAlex, so post-VPD follow-ups are untested). Run novelty audit before any claim.
