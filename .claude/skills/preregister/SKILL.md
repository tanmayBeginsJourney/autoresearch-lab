---
name: preregister
description: Write a falsifiable preregistration (hypothesis, prediction, controls, analysis plan, budget) before confirmatory experiments. Use when a gap is chosen or exploratory results suggest a hypothesis worth testing.
---

# Preregistration

`researchctl prereg P` creates `projects/P/prereg/H###.yaml` with every required field.
Fill it in, commit it, and only then submit `kind: confirmatory` runs. Once committed, the hook
freezes the file. If the plan changes, write `H###b` and record the reason in PROJECT.md.

Field guidance:
- **hypothesis**: falsifiable and specific. Weak: "centering helps". Testable: "subtracting the
  running mean of GELU outputs lowers final validation loss of 2-layer, 50M-param transformers on
  OpenWebText at 2B tokens, because it removes a constant component the next layer otherwise has to
  cancel."
- **prediction / falsification**: direction plus a size threshold you would care about (the smallest
  effect worth reporting), not just "p < 0.05".
- **baselines**: the strongest fair ones, each tuned with the same budget as our method.
- **controls**: one per alternative explanation. Examples: random-direction ablation for a feature
  ablation, matched-norm noise for a steering vector, parameter- and FLOP-matched baselines for an
  architecture change, the same LR sweep for every activation function.
- **seeds**: >= 3. Use the same list in every arm. Use more when the exploratory variance says the
  effect is small relative to seed noise. Estimate this from exploratory runs: n ~ 8 (sd/delta)^2
  per arm for a paired design gives roughly 80% power.
- **analysis**: test (paired t over seeds unless argued otherwise), metric, step at which it is read,
  the decision rule, and how multiple claims are corrected (the claims checker applies Holm).
- **budget**: GPU-hours and calendar days. Exceeding it means stop and reassess, not keep going.
- **decision_value**: what we do after each outcome. If every outcome leads to the same action,
  skip the experiment.
