---
name: analyze
description: Analyze experimental results rigorously - statistics over seeds, confounders, alternative explanations, the claims ledger, and figures. Use after confirmatory runs are collected, or when interpreting any result.
---

# Analysis

## Claims ledger: `projects/P/claims.yaml`
```yaml
claims:
  - id: C1
    statement: Centered GELU lowers final validation loss by >=0.02 nats in 50M-param transformers.
    kind: performance | causal | correlational | descriptive
    hypothesis: H001            # omit for exploratory claims (they can never be 'supported')
    metric: val_loss
    goal: min
    treatment: {experiment: 261106-actbias-1a2b3c, where: {activation: gelu_centered}}
    control:   {experiment: 261106-actbias-1a2b3c, where: {activation: gelu}}
    paired_by: seed
```
`researchctl claims check P` recomputes every statistic from the records and applies Holm across
the confirmatory claims. It assigns supported / inconclusive / contradicted / exploratory / invalid,
lints the wording, and writes `claims.lock.json` and `paper/numbers.tex`. Rewrite or drop claims it
does not support. Never argue with the checker in prose.

## Before believing a result
- Effect size vs seed noise, not just the p-value. Is the CI width small relative to the effect?
- Alternatives: for each competing explanation in the prereg, which control rules it out? If none
  does, the claim must be weakened.
- Confounders specific to ML: tuning budget asymmetry, parameter/FLOP mismatch, LR-schedule
  interaction, data order, eval-format effects, tokenization, contamination, early stopping on test.
- Robustness: does the sign hold across the other grid points, seeds and a second dataset or model?
  If only one setting works, say so.
- Mechanistic claims: show necessity (ablation hurts) and sufficiency (patching restores),
  compare against random-direction and mean-ablation baselines, and check the effect is specific
  to the target behavior.

## Figures (scripts in `projects/P/analysis/`, outputs in `paper/figures/`, vector PDF)
- One message per figure. The caption states the takeaway and the setup (n seeds, CI type).
- Show uncertainty: CIs or individual seeds as points; never a bare mean.
- Use colorblind-safe palettes (Okabe-Ito); label axes with units; no chart junk. Use shared y-axes
  when comparing panels.
- Prefer showing the raw evidence (per-seed curves) next to the summary.
- Figures are generated from records by a script. Never hand-edit numbers or images.
