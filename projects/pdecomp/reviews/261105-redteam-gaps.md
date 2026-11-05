# Red-team review: gap analysis (gaps.md G1-G4), 2026-11-05

Reviewer: fresh-context skeptic. I read gaps.md (HEAD ff9eb0e), PROJECT.md, journal.md,
lit/snowball.md, ideas/*/ (diff-deltas audit and design; faithfulness ideas, map and novelty-G2;
clustering novelty-G3 and vpd-clustering; decomposability novelty-G4 and theory; cross-field;
frontier; weight-space), the VPD card, repro/spd-toy/REPORT.md, prototypes/delta_pd (code,
RESULTS.md), prototypes/interaction_clustering (toy.py, summary.json) and prototypes/warmstart/RESULTS.md.
Commands run: `researchctl gate pdecomp` (missing: reproduction report, `## Decision`;
`novelty_audit: false`). `researchctl claims check pdecomp` returned 0 claims, as expected; I deleted the
lock and macros files it generated so the tree is unchanged. About 25 `lit-search` queries, deliberately
not logged to the project, plus about 15 web searches and fetches.
The novelty audits for G2-G4 and the G1/G3 prototypes were committed while this review was in progress.
This version takes them into account.

## Score for the gap analysis as a decision document: 4/10 (not ready to decide)

| Gap | Promise for a novel positive NeurIPS 2027 paper (8 slices) | Top objection |
|---|---|---|
| G1 Delta-PD | **5/10** | The SVD baselines are much stronger than assumed. WeightWatch (ICLR 2026) already monitors and steers with SVD directions of dW, input-conditionally. For RL deltas, rank-1 SVD recovers >99% of the gains (2510.00553). The prototype's win over SVD comes from planting two equal-coefficient changes that share an output, which makes SVD degenerate by construction. |
| G2 Calibration of mask attacks | **3/10** | The planted defects define what "correct" means, so the calibrated metric is only as general as the defect menu. Our SPD cannot yet produce the correct reference decompositions (repro: features split over 2-8 subcomponents). |
| G3 Interaction clustering | **3/10** | The principle is published (Hedonic Neurons, ArchDetect). The prototype's win holds only on ground-truth components in a toy with no superposition. With output superposition the MSE-based interaction degrades (ARI 0.69 at rho=1), and with hidden superposition every method fails (ARI <= 0.2). |
| G4 Activation vs decomposability | **3/10** | theory.md's own proofs make the main result predictable (a dead zone permits exact sparse faithfulness; GELU/SiLU leak). A reviewer will read "ReLU decomposes better than GELU" as a corollary of known activation-sparsity facts. The effect may also be small next to attention and superposition at LM scale. |

## Cross-cutting objections (ranked)

1. **Our SPD does not yet recover toy ground truth, so no prototype comparison can be read yet.**
   The repro is partial: MMCS is fine, but ML2R is 0.15-0.8 against the paper's 0.993, at 7.5-25% of the budget.
   Delta-PD, G3 and G4 prototypes all run 2.5k-3k-step SPD. "Method A beats B" from under-trained
   decompositions is confounded with optimisation state, and the warm-up finding shows that training
   details change the outcome. Fix: run the paper's budget (40k x 4096, 3 seeds, s24) before reading any
   prototype as evidence either way. The VPD GPU-hour cost is also still unmeasured, and both G1 Stage B and
   G4's LM stage depend on it.
2. **The P(positive) values and scout votes are not evidence.** The five scouts read the same VPD 7.5
   sentence and the same cards, so the votes are correlated. frontier.md's 10.4 is arbitrary arithmetic.
   Replace P with "the kill test, its cost, and the result that would kill it".
3. **Missing prior work a reviewer will cite** (the audits now cover Hedonic Neurons, ArchDetect, SoLU,
   AutoAttack, the Angsten audits, VPD B.11.2; the following are still absent):
   - G1: WeightWatch, Zhong & Raghunathan arXiv:2508.00161 (ICLR 2026). frontier.md cites it, but it is
     not in audit.md's closest-five table, and it is input-conditional, which undercuts design.md's "SVD is
     static" argument. Also: Cai et al. arXiv:2510.00553 (top singular subspace of the RL update recovers >99%
     of the gains, 8 LLMs x 7 RL algorithms); arXiv:2605.21468 (RELEX, rank-1 RLVR trajectories); He & Cao
     arXiv:2509.17866 (post-training is a near-uniform singular-value rescaling plus consistent rotations);
     arXiv:2602.15195 (100% LoRA-backdoor detection from weight spectra).
   - G2: Miller et al. arXiv:2407.08734; Li & Janson, Optimal Ablation (NeurIPS 2024); MIB arXiv:2504.13151;
     InterpBench (NeurIPS 2024 D&B). These are reusable ground-truth or faithfulness protocols.
   - G3: Csordas et al. arXiv:2010.02066; Hod et al. arXiv:2110.08058; Local Interaction Basis
     arXiv:2405.10928 (Goodfire/Apollo authors, interaction-sparse bases).
   - G4: "ReLU Strikes Back" arXiv:2310.04564 and ReLU^2 arXiv:2402.03804 (ReLU-family LMs at near-matched
     loss, exact activation sparsity); TopK LMs arXiv:2506.21468; weight-sparse transformers arXiv:2511.13653.
4. **Toy validity.** Bhagat et al. (2606.14673) undercut the compressed-computation/residual-MLP toys.
   G3's and G1's toys are hand-built one-hidden-layer MLPs with axis-aligned sparse features. A reviewer will
   ask whether any result survives a rotated feature basis, superposition, and a learned (not ground-truth)
   decomposition.
5. **Competitor overlap.** Goodfire has publicly named G1 (VPD 7.5), G2 ("a more powerful adversary" is
   being looked into, Sharkey on LW), G3 (clustering in training, 7.5) and something close to G4
   ("intentional design"). Novelty that rests on "not done yet" decays fast. Novelty that rests on the
   evaluation (ground-truth organisms, strong baselines) is more durable.

## G1: Delta-PD. Score 5/10

**Done or obvious?** Unpublished (my searches agree with audit.md), but VPD names it, and the method
is SPD with a non-zero ablation reference. Reviewers will call the method obvious, so novelty must come
from results that the baselines cannot match.

**Strongest reason to reject.** "An incremental application that the VPD authors already named. On real
fine-tunes, SVD of dW already detects and steers backdoors and unlearning (WeightWatch, input-conditional
via activation cosine). For RL updates, rank-1 SVD recovers nearly all of the gain (2510.00553). The only
win shown is a toy in which two equal-coefficient changes share an output, so dW has a degenerate singular
pair that SVD must rotate."

Objections and fixes:
1. *The RL motivation is contradicted.* design.md argues that RL deltas are where SVD fails (full-rank per
   2505.11711). 2510.00553 shows they are functionally rank-1, so SVD should succeed there. Fix: drop the
   claim, or test it directly in about 1 GPU-hour: revert-to-base with the top-k SVD of a GRPO Qwen-0.5B delta.
2. *The prototype's SVD failure is built in.* delta_pd.py plants ReLU(x3) and ReLU(x7) with identical
   coefficients and an overlapping output (y11). Equal norms give near-degenerate singular values, so
   SVD purity of 0.32 is expected. Fixes: sweep the coefficient ratio over {1, 1.5, 2, 4, 10} and report where SVD
   stops failing; add **SVD + ICA/varimax rotation within the top-k subspace** (a cheap, standard fix for
   degeneracy) and **SVD directions + learned CI gate** (U, V frozen). If either ties Delta-PD, the
   contribution is the gating, not PD.
3. *Axis-aligned features make trivial decompositions competitive.* A "columns of dW1" decomposition
   solves this toy. Fix: embed features through a random rotation.
4. *Selection on the evaluation set.* The matched component is the argmax on the same one-hot probes used to
   score it. Fix: select on held-out natural inputs.
5. *Surgicality is not better.* Delta-PD base damage is 0.43, against 0.46 for SVD and 0.14 for LoRA. The
   ablated component disturbs unchanged outputs about as much as SVD does. The claim of "surgical" edits
   therefore lacks support, even on the toy.
6. *Faithfulness is easy by construction.* Interpolating between two models in one basin is low-loss for
   almost any split of dW. Fix: compare against the same base-anchored reference with random rank-one
   splits plus learned CI. Report the share of KL in the residual R and the number of components needed for 90% of the KL
   on a real SFT delta. He & Cao suggest a dense rescaling-plus-rotation part that may sit in R.
7. *Ground truth on LMs.* Train each organism separately and then jointly. The separate task vectors then
   serve as weight-space ground truth and as an edit baseline (task-vector negation) that design.md lacks.

**What would make it compelling.** On at least 5 real organisms (at least one jointly trained with several
behaviours, at least one RL), Delta-PD separates behaviours that WeightWatch, rotated SVD, Delta-Crosscoder
and Diff Mining conflate. Ablating S_k reverts behaviour k with less capability loss than task-vector
negation. Plus one causally verified finding in a public post-training delta that SVD misses. If the
norm-ratio sweep shows SVD failing only at ratios near 1, G1 shrinks to a niche result. Decide on that
sweep, which takes CPU-minutes, before any GPU work.

## G2: Calibrating mask-attack evaluations against ground truth. Score 3/10

**Done or obvious?** The reframe in novelty-G2.md is correct: the attack-ensemble half is obvious or
already done (VPD's Adam-PGD, Angsten's restarts and transfer audits, FW on a box is close to sign-gradient
PGD). The calibration half appears open. The audit is right that evidence of absence is thin.

**Strongest reason to reject.** "The authors choose the defects (merged mechanisms, duplicated components,
lookup tables) and then show that some attack detects them. The calibration only transfers to defects
that were on the menu. The 'correct' reference decompositions come from a pipeline that does not
reproduce SPD's toy results. Real LMs give no way to check transfer, so this is a methodology note for a
single method used by one lab."

Fixes:
- Define the correct class by construction: hand-set exact decompositions (as in theory.md P1, ReLU neuron
  basis) rather than SPD outputs. That removes objection 1 for the positive class.
- Include defects that were not designed in advance: decompositions from under-trained SPD, wrong C, or
  wrong minimality coefficient, labelled by ground-truth recovery. This tests whether the metric generalises
  beyond the menu.
- Pre-specify the non-adversarial baselines (donor-aggregation KL, random-mask percentile bound per
  2407.15166). If they match the best attack in AUROC, that is the main finding, and a negative one.
- A transfer check that does not need ground truth: does the toy-selected metric predict seed agreement
  or edit success on the 67M VPD decomposition (released)?

**What would make it compelling.** A metric or adversary scope that separates correct from defective
decompositions with high AUROC across 3 or more toy families, including defects not designed in advance,
and also predicts an independent outcome on the released 67M decomposition. Otherwise this is a workshop
paper or an evaluation section inside G1.

## G3: Cluster subcomponents by ablation interaction. Score 3/10

**Done or obvious?** novelty-G3.md concedes that principle novelty is low (Hedonic Neurons 2509.23684,
ArchDetect 2006.10965, epistasis maps). The PD application looks open. VPD B.11.2 already computes an
interaction proxy.

**Strongest reason to reject.** "Known technique applied to new units, shown only on ground-truth
components in a toy where co-activation is broken by construction (paired features copied with
probability rho). The advantage disappears under superposition, and there is no LM-scale ground truth."

Specific problems:
1. *The value function still fakes interaction.* novelty-G3.md proposes v = -||f(m) - f||^2 in logit
   space, on the grounds that, unlike KL, it has no cross-term. It does: for additive output effects
   da and db, ||da + db||^2 - ||da||^2 - ||db||^2 = 2 da.db, so the interaction is the overlap of output
   effects. The prototype shows exactly this. In `outsup` (non-orthogonal output directions), int_full
   drops from 1.00 to 0.86 (rho 0.6) and 0.69 (rho 1.0), while it stays at 1.00 in `clean`. Fix: use
   ArchDetect's form, the norm of the vector mixed difference ||f(11) - f(10) - f(01) + f(00)||. This is
   zero for additive effects whatever the output geometry.
2. *Superposition breaks every method.* In `hidsup` all arms score ARI <= 0.2 (comb 0.14-0.19). The
   regime in which PD is supposed to matter is where G3 has nothing yet.
3. *The prototype clusters ground-truth subcomponents, not SPD outputs.* With a learned decomposition,
   splitting (see the repro) adds same-matrix fragments whose interaction is near zero, the failure mode the
   audit predicts. Fix: run on SPD outputs at full budget.
4. *The construction favours interaction.* Mechanisms are two-stage AND chains, which are maximally
   non-additive. Add OR/backup mechanisms and same-matrix rank-2 mechanisms, and report per mechanism type.
5. *A trivial baseline is missing.* Cluster by the Fisher- or output-weighted cosine of per-component
   effects. If it matches interaction clustering in `outsup`, the interaction is mostly output overlap.

**What would make it compelling.** On learned decompositions of superposed toys and Tracr/InterpBench
models, the mixed-difference interaction recovers cross-matrix or cross-position mechanisms where
co-activation MDL fails. On the 67M VPD decomposition, whole-cluster ablations are more selective than
co-activation clusters and matched random clusters. Without the superposed case this is not a NeurIPS
paper.

## G4: Activation choice vs decomposability. Score 3/10

**Done or obvious?** The exact PD measurement appears open (novelty-G4.md). theory.md is good work, but it
cuts both ways. P1-P3 show that a dead zone permits exact sparse faithfulness and that analytic activations
with a negative tail forbid it. With that in hand, "ReLU > GELU" is close to a corollary, and
activation-sparsity work (ReLU Strikes Back, ReLU^2) already establishes that ReLU-family models have exact
zeros at near-matched loss.

**Strongest reason to reject.** "The headline follows from a three-line argument about exact zeros.
Measured with PD's own objective under one hyperparameter setting, it is confounded with tuning. The
practical recommendation (use ReLU-family MLPs if you want PD) changes nothing for the SwiGLU models
people actually analyse. If bilinear wins, closed-form eigendecomposition makes PD unnecessary there."

Fixes:
- Turn theory into prediction. H1 (adversarial masks on dead GELU neurons sit at m z = z*) is sharp and
  cheap; confirming it on trained models moves G4 from "obvious" to "explained". Also predict and test the
  size of the effect (P4/P5 bound it linearly in the leakage). If the bound says the effect is small next to
  observed VPD KL (25 at 160 steps), report that as the main finding: activation is *not* why VPD is
  non-robust.
- Give each architecture its own equal PD tuning budget. Match loss by curve interpolation, not by early
  stopping at the worst architecture's loss: the current run.py early-stops, which leaves stronger
  architectures under-trained and confounds the comparison. Use 2 or more sizes and paired seeds.
- Include a non-PD measure (transcoder fidelity on the same models) to show whether the effect is
  PD-specific.
- A post-hoc intervention: swap GELU for ReLU in a trained model and briefly fine-tune, or anneal GELU
  sharpness. This tests causation within one model rather than across separately trained models.

**What would make it compelling.** A quantitative account of how much of VPD's adversarial failure on the
67M model is due to GELU leakage, checked by a ReLU-swapped fine-tuned copy, together with toy-level ground
truth. Either sign is informative. As a positive "architecture X is more decomposable" paper it is a
section, not a standalone paper.

## Recommended order (cheapest decisive information first)
1. SPD toy repro at the paper's budget (prerequisite for reading any prototype).
2. G1 norm-ratio sweep with rotated SVD, SVD+CI and rotated-basis arms (CPU-minutes), then the GRPO
   top-k SVD revert test (about 1 GPU-hour). Kill or reframe G1 on these results.
3. G3: switch to the mixed-difference value, rerun on `outsup` and `hidsup` with SPD-learned components.
   Kill G3 if `hidsup` stays below ARI 0.3 for all arms.
4. G4: run the H1 z* test and the leakage-bound estimate on the 67M VPD model before training any LM arms.
5. Add cards for WeightWatch, 2510.00553, 2509.17866 and ReLU Strikes Back. Write lit/novelty.md only for
   the chosen gap.
