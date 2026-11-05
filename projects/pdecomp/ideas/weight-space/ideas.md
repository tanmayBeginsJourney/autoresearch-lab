# Weight-space x PD: ranked ideas
<!-- Exploratory idea list, 2026-11-05. Novelty checks are search-based (titles/abstracts, ~45 queries
in lit/queries.log plus web); none is a full novelty audit. Paper ids in map.md. -->

## Ranking
1. **Delta-PD: decompose the base once, decompose fine-tunes as reweighted base components plus a few new ones.**
2. **Gauge audit of PD: is PD invariant to function-preserving reparameterizations, and does a canonical gauge improve it?**
3. **PD-component edits vs ROME/MEMIT/AlphaEdit on locality.**

## A shared observation
The ideal PD objective is equivariant under every linear symmetry of hidden spaces that NNs have
(neuron permutation, ReLU positive scaling, Q/K and V/O GL(d_head), residual rotation with RMSNorm
folded, RMSNorm gain vs next W_in): an invertible map sends a rank-one u v^T to a rank-one matrix, and
ablating it has the same effect on the function. So "use symmetries to make PD identifiable" is not
available as stated; the exact optimum does not depend on the gauge. What does depend on it is the
optimization: Frobenius faithfulness weighting, U/V initialization, AdamW (arxiv:2610.18085 shows AdamW
breaks Q/K rescaling equivariance) and the CI function's input scale. Idea 2 tests whether that matters.
It also kills "teleportation makes PD cheaper" as a separate idea: teleportation can only change
conditioning, which idea 2 already measures.

## 1. Delta-PD (fine-tuning and checkpoint diffs in component space)
- Method: freeze the base model's PD (U, V, CI function). Fit the fine-tuned weights W' = sum_c s_c u_c v_c^T
  + sum_k new rank-one components + Delta, with the usual PD losses, a sparsity prior on (s_c - 1) and on
  the number of new components. Output: which base mechanisms were up/down-weighted, which are new.
- Why it could work: warm starts reuse most of the decomposition; narrow fine-tunes are reported to touch
  few directions; the LessWrong VPD post shows a single mask scalar implements a "remove German" change.
- Closest work: tPD (target-only PD, Delta components), Delta-Crosscoder, diff interpretation tuning
  (arxiv:2510.05092), circuit-tuning diffs (EMNLP-F 2026.679), SWD's fine-tune replacement (arxiv:2608.03913),
  LoRA-GL weight-space nets (arxiv:2410.04207). None found that warm-starts PD from the base or reports
  component-level diffs. Goodfire is the obvious competitor (they own the codebase and the VPD target).
- CPU toy: SPD residual-MLP or TMS toy (ReLU). Fine-tune three variants with planted changes: (a) suppress
  feature k, (b) add a new input-output mapping, (c) both. Arms, same 5 seeds: Delta-PD; PD from scratch on
  W'; top-r SVD of Delta W; SPD on Delta W alone. Metrics: recovery of planted changes (max cosine to ground
  truth), false-positive changed components, steps to reach the from-scratch faithfulness.
- Kill if: Delta-PD does not beat SVD(Delta W) on planted-change recovery, or saves < 2x steps.
- Payoff if positive: a testable hypothesis at LM scale ("fine-tuning / RL mostly rescales existing
  mechanisms"), and two applications on the same machinery: conflict-aware merging (two tasks rescaling the
  same base component in opposite directions is an explicit, nameable conflict; compare TIES, TSV, Iso-C)
  and edit auditing. Matches candidate gap 1 in PROJECT.md.
- Novelty confidence: medium (~60%). Positive-result probability on toys: high; on the 67M LM: medium.

## 2. Gauge audit of PD, plus canonical gauge
- Hypothesis: PD outputs (ground-truth recovery, alive components, adversarial-mask KL, seed agreement) vary
  when the target is replaced by a function-identical teleported copy; norm-balancing each symmetry pair
  before PD (the balancing used in arxiv:2610.18085, or an SVD split for Q/K) reduces that variance and
  improves recovery.
- Why it could work: PD's optimizer and losses are not equivariant (see shared observation); RMSNorm gains
  in LMs are often far from 1, which is a scaling gauge between the norm and W_in. Need to check in the VPD
  code whether gains are folded before decomposition.
- Why it may fail: gradient flow conserves ||W_Q||^2 - ||W_K||^2 per pair and weight decay drives toward
  balance (arxiv:2210.17216), so trained models may already sit near the balanced gauge. Then the result is
  a short robustness note, not a paper; it still matters for PD of merged, edited or teleported models.
- Closest work: no PD paper found addressing reparameterization; polymorphism (arxiv:2605.24577) does it for
  SAEs across seeds; SAE seed-dependence (ICML 2026, arxiv:2606.12138) for activation dictionaries.
- CPU toy: ReLU residual-MLP toy (positive scaling) and a 1-layer attention toy (Q/K GL). Teleport with
  log-normal scales sigma in {0, 0.5, 1, 2}, verify outputs match to 1e-6, run SPD with paired seeds, map
  components back. Then repeat with canonical gauge. Also a zero-GPU check: measure the actual imbalance
  (per-head ||W_Q||/||W_K||, RMSNorm gain spread) in the 67M VPD target checkpoint.
- Kill if: at sigma = 2 recovery changes by less than the seed-to-seed spread (5 seeds).
- Novelty confidence: medium-high (~65%). Cost: < 1 CPU-day. Scope: a section of a larger paper, or a
  workshop note; combine with idea 1 (delta-PD of a merged/edited model needs a fixed gauge anyway).

## 3. PD-component edits vs rank-one knowledge editing
- Method: ROME writes Delta W = (v* - W k*)(C^-1 k*)^T / (k*^T C^-1 k*). PD gives each MLP matrix a set of
  rank-one components with CI gates. Edit a fact by re-solving only the output vectors u_c of the components
  with high CI on the subject (or add one component whose read direction is the union of their v_c).
  Locality then follows how selective v_c is, which PD's minimality pressure optimizes, instead of C^-1 k*.
- Why it could work: rank-one edits fail locality when keys overlap in superposition
  (doi:10.18653/v1/2025.blackboxnlp-1.11); PD components are selected for sparse causal importance.
- Why it may fail: ROME/AlphaEdit are close to optimal for a linear associative memory with Gaussian keys;
  an advantage must come from superposed or nonlinear keys. PD cannot yet decompose GPT-J/Llama, so the LM
  test is limited to one MLP layer (ASPD-style targeted PD) of GPT-2-XL at most. KE is a crowded field.
- Closest work: ASPD's "causally editable" claim (SAE metrics, no KE baselines); the LessWrong VPD post
  (removal only, LoRA baselines only); tPD (unlearning framing). No PD vs ROME/MEMIT/AlphaEdit comparison found.
- CPU toy: 2-layer MLP memory trained on synthetic subject->attribute facts, subjects built from shared
  features so keys overlap. Arms: ROME, MEMIT, AlphaEdit, PD edit; 10/100/1000 edits; efficacy, paraphrase,
  neighbourhood accuracy on subjects sharing features. Kill if PD edit is not >= AlphaEdit on neighbourhood
  accuracy at matched efficacy.
- Novelty confidence: medium-high (~65%). Positive-result probability: low-medium (~30%).

## Considered and ranked lower
4. PD components as a cross-seed alignment basis (beyond git re-basin). Between independently trained
   models the relation is not a network symmetry, so component matching yields no function-preserving map
   for merging; activation Procrustes (arxiv:2605.24577) already aligns residual bases from one batch.
   Kept as a measurement: fraction of PD mechanisms shared across seeds (universality in weight space).
5. Amortized PD with an equivariant weight-space network trained on a toy zoo. Feasible and likely positive
   on toys (close to supervised learning), but there is no zoo of LMs, so it does not transfer to the case
   that matters. Novelty high, value low. The useful kernel (reuse a decomposition) is idea 1.
6. PD description length as a generalization predictor (grokking, memorization vs generalization). Tests
   APD's MDL premise on CPU (modular addition), but compression measures of grokking already exist
   (arxiv:2310.05918, arxiv:2610.35112). Medium-low novelty.
- Killed: "teleport to make PD cheaper" (shared observation); weight-statistics (heavy-tail spectra) of PD
  components (no hypothesis linking them to mechanism quality).

## Next step if adopted
Run the zero-GPU imbalance check (idea 2) and the delta-PD toy (idea 1) together: both need the same SPD
toy harness. Full novelty audit (lit/novelty.md) for idea 1 before any confirmatory run.
