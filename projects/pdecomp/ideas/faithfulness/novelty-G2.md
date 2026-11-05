# Novelty audit: G2 (stronger mask adversary + calibration on toys)
Date 2026-11-05. Claim audited: AutoAttack-style ensemble (APGD, Frank-Wolfe vertex, Square/random search, restarts) for
evaluating and training parameter decompositions, plus calibration of robustness metrics against ground-truth mechanism
recovery on toys. Evidence base: full text of VPD (cached text, Sec 7.3, 7.5, A.3.4, B.4, Table 4), Adversarial Circuit
Evaluation, SAE robustness (2505.16004), AutoAttack key sections; web fetches of the param-decomp repo, two LessWrong
audits and arXiv 2602.16823 (the web items are summaries from the fetch tool, depth `skim`; re-read before citing numbers).

## Verdict
- Attack-ensemble half: **partially done, and the rest is obvious.** Adversarial masks in the training loss and a PGD
  evaluation metric are VPD's core; restarts, step sweeps, scope variants and a transfer metric were already done by an outside
  audit; Goodfire says a stronger adversary is in progress. APGD, FW and Square on masks specifically were not found.
- Calibration half: **novel as far as found** (no search hit, and the audit's author states he has no ground truth), but
  thin evidence of absence (search was abstract-level plus targeted fetches; VPD has 0 OpenAlex citers).
- Overall: do not frame as "stronger attack". Frame as a metric-validation study on toys; the ensemble is the instrument.

## Queries run
researchctl lit-search (project pdecomp, all logged in lit/queries.log): "adversarial mask attack parameter decomposition";
"AutoAttack ensemble ablation mask faithfulness interpretability"; "worst-case ablation circuit faithfulness gradient attack
on mask"; "stochastic parameter decomposition adversarial ablation". All four returned off-topic adversarial-ML hits, no
interpretability mask-attack paper. lit-citers arxiv:2610.20748: 0 (OpenAlex). lit-fetch: 2407.15166, 2505.16004, 2003.01690.
Web: "parameter decomposition adversarial ablation masks stronger attack faithfulness VPD" (found both LessWrong audits);
"LessWrong parameter decomposition adversary persistent PGD / adversarial sampler improved coverage Goodfire Sharkey";
"arxiv 2026 worst-case ablation masks faithfulness ... Frank-Wolfe OR APGD OR AutoAttack ... circuit OR decomposition"
(found only 2602.16823, 2407.15166, 2407.08734). GitHub goodfire-ai/param-decomp: README, issues page, PR search "adversarial".
Not done: Google Scholar-style forward search on AutoAttack for interpretability, arXiv listing sweep for Oct-Nov 2026,
Alignment Forum search, repo code reading (only titles of PRs seen).

## Closest three, with exact differences
1. **VPD (2610.20748), A.3.4 / 7.3 / 7.5.** Already has: adversarial reconstruction loss in training (persistent PGD,
   Adam beta1 0.5, n_adv=3, per-batch-per-position sources, delta-components also attacked); evaluation PGD with fresh sources
   shared across the batch, 20 to 320 steps (KL 0.83 to 40.2); heuristic n_adv*lr ~ 2; asks for "improve adversarial sampling
   and mask parameterizations" and names a batch-subspace idea. Note: its adversary is Adam-based gradient ascent, not plain
   sign-PGD (our earlier cards said "plain PGD"). Key conceptual point we must respect: VPD argues the unconstrained
   worst case is too strict (superposition interference noise can be sign-selected), so it deliberately restricts the sampler
   to one shared mask per batch to force exploitation of systematic defects. A stronger per-datapoint ensemble would
   measure the wrong thing. Differences from G2: no step-size-free optimizer, no FW/vertex, no black-box check, no restart
   protocol in VPD, no ground-truth calibration; authors explicitly leave "how much robustness is necessary" open (7.3).
2. **Angsten LessWrong audits (2026-09-30, 2026-10-20) and Sharkey's comment (2026-10-17).** Evaluation adversary with 5
   random starts x 16 batches, steps 20 to 320, shared-per-batch masks; transfer ratio to unseen batches (78% at 20 steps to
   ~95% at 160-320), shuffled-source control, sign analysis, training-adversary diagnosis (per-position sharing never finds a
   defect in 1,000 batches; per-document or whole-batch sharing does). Proposes pooling gradients, starting from other
   inputs' labels, and donor-aggregation KL as a metric. Differences: one released 67M decomposition, not a toy; no attack
   family comparison; no ground-truth labels (stated); optimizer unnamed. Sharkey: coverage of ablation space is the
   issue and "a more powerful adversary" is being looked into. This is the main scoop risk and also a baseline we must beat
   (donor aggregation is search-free).
3. **Adversarial Circuit Evaluation (2407.15166), read in full.** Worst case over ~1e6 sampled (clean, corrupt) input pairs with
   a fixed circuit and fixed resample ablation; random sampling only, no gradient; order-statistic percentile bound
   (binomial/Chernoff/Hoeffding); future work is to plug the metric into circuit discovery. Differences: perturbs inputs
   through a discrete choice, never optimizes the ablation; no attack ensemble; no ground truth. Its percentile bound is
   reusable for random-mask baselines.
Also read: SAE robustness (2505.16004): GCG-style one-token input edits, single attack family, no ensemble, no training use,
no calibration to ground-truth concepts. AutoAttack (2003.01690): ensemble is APGD-CE, APGD-T, FAB-T, Square; Frank-Wolfe
is not in it (our "AutoAttack-style incl. FW" is our own extension, cite Chen et al. 1811.10828 separately); margin/DLR
components do not apply to KL objectives. Neighbour not read in full: Formal Mechanistic Interpretability (2602.16823,
verification, "robust patching under continuous perturbations", abstract only).
Goodfire repo (titles only): persistent PGD, scopes, sigmoid source parameterization, grad clipping (closed unmerged), warmup;
no APGD/FW/random-search item visible in 33 matching closed PRs; open issue #918 is JAX/torch adversary parity.

## Most defensible novel claim
"On toy decompositions with known mechanisms, which adversary (scope, steps, budget, attack family) and which robustness
number best separate correct from defective decompositions, and how does that ranking change as the attack gets stronger?"
Concrete, falsifiable design that the sources motivate: defects with known labels (merged mechanisms, duplicated component,
VPD's own Sec 7.3 per-datapoint lookup-table decomposition as a guaranteed negative); toys with superposition noise, where
VPD says the unconstrained worst case wrongly penalizes valid decompositions; report AUROC of defect detection versus
attack strength, plus the false-positive rate on correct decompositions. Include the search-free donor-aggregation KL and
random-mask percentile bounds as non-adversarial baselines. A negative or non-monotone result (stronger attack, worse
discrimination) is itself the finding and supports VPD's shared-mask design.
Secondary, only if exploratory results support it: attack ensemble as a coverage check (does best-of-ensemble exceed tuned
Adam-PGD at equal gradient evaluations on the toy?). Training-side claims (APGD cheaper at equal robustness) have no
evidence yet; keep as hypotheses.

## What a reviewer will call obvious
- "Use APGD/restarts instead of PGD": textbook since AutoAttack 2020; Angsten already used restarts and step sweeps; Adam
  with momentum already approximates step-size-free behavior. Frank-Wolfe on a box with a linear oracle returns sign(grad)
  vertices, i.e. a signed-gradient step to a corner, nearly l_inf PGD. Name it as known.
- "Worst-case estimates go up with a stronger attack": expected; only interesting if tied to ground truth.
- "Ensemble worst case understates vulnerability, so train against it": VPD notes this cost problem; compute-matched
  evidence needed, and per-datapoint worst case is the wrong target per VPD A.3.4.
- "Toys are not LMs": answer with the transfer step (one small-LM layer, or Angsten-style transfer ratio) only if toys pass.
- Overlap with input-space adversarial circuit/SAE evaluation: ours perturbs the ablation set, theirs the inputs; do not
  oversell the difference as a method contribution.

## Actions
Re-run lit-citers on VPD and re-check Goodfire repo/LessWrong comments monthly (scoop risk is live). Re-read both LessWrong
posts on the page before quoting numbers. Update gaps.md G2 text: closest work now includes the Angsten audits; "no ensemble
or certified mask attack found" is still true but "VPD uses plain PGD" is not (Adam-based).
