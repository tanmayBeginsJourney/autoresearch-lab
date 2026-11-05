# Submission standards

A paper leaves this lab only when every box below is checked. Each item is checkable; "good enough"
is not a criterion. Targets are NeurIPS/ICML/ICLR main track; reviewers score quality, clarity,
significance and originality, and these items map to those four.

## Research (quality, significance, originality)
- [ ] One-sentence contribution that a skeptical reviewer would accept as new and worth knowing.
- [ ] `lit/novelty.md` lists the queries run, the 3 closest papers, and the concrete difference from
      each; the red-team's own novelty search found nothing closer.
- [ ] The claims the work builds on have been reproduced (`repro/*/REPORT.md`).
- [ ] Every headline claim is `supported` in `claims.lock.json`: preregistered, >= 3 seeds (more
      when the effect is under ~2 seed-SDs), paired where possible, Holm-corrected, no excluded runs
      left unexplained.
- [ ] Baselines are the strongest available and tuned with the same budget as ours. For
      interpretability methods: current SOTA of the same family plus the strongest method of a
      different family (e.g. transcoders/CLTs vs parameter decomposition) at matched sparsity or compute.
- [ ] Controls rule out each alternative listed in the prereg (random-direction and matched-norm
      ablations for causal claims; parameter- and FLOP-matched baselines for method claims).
- [ ] Generality: the main result holds on >= 2 model sizes or families, or the paper states the
      limitation in the abstract.
- [ ] Faithfulness of any explanation is measured quantitatively, not shown by examples alone.
- [ ] Compute is reported (GPU type, GPU-hours per result) and the main result reproduces from the
      released code and configs on a fresh checkout.
- [ ] Negative results and failed variants that a reader would otherwise try are reported (appendix).
- [ ] Red-team subagent scores >= 7/10 in two consecutive rounds, every objection answered in
      `reviews/`. At least one human outside the lab has read the draft before submission.

## Writing (clarity)
- [ ] Figure 1 conveys the main result without the text.
- [ ] The abstract states the result with its number and the setting it was measured in.
- [ ] Every number in the paper comes from `\claimval` macros or generated tables.
- [ ] `researchctl lint paper/ --bib paper/refs.bib` is clean; all BibTeX came from `lit-cite`.
- [ ] Known techniques are called by their names; nothing ordinary is branded as new.
- [ ] Limitations are specific (scale, models, compute), not ritual.
- [ ] Within the page limit without shrinking fonts or margins; the appendix is optional reading.
- [ ] A reader who knows the field can reproduce the method from the paper alone.

## Process deadlines (relative to the submission deadline D)
- D-8 weeks: all confirmatory experiments preregistered.
- D-4 weeks: results frozen; only writing, figures and red-team fixes after this.
- D-2 weeks: complete draft; red-team round 1; external reader.
- D-3 days: final lint, anonymization, checklist, code release staged.
