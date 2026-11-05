---
name: literature-review
description: Deep literature review for a research project - full-text reading, skeptical paper cards, citation-graph snowballing, reproduction targets, novelty audit. Use when starting or extending a project's related-work understanding, or before claiming anything is new.
---

# Literature review

Aim: understand the field well enough to say what is known, what is believed but weakly supported,
and what is open. Reading abstracts is triage, not review.

## 1. Map the territory (cheap, broad)
- Write 6-12 queries: the problem stated several ways, method names, the phenomenon, and adjacent
  fields' vocabulary (optimization, statistics, neuroscience, signal processing, compilers). Run each
  with `researchctl lit-search "q" --project P --year 2019-`. Every query is logged for the novelty audit.
- Snowball from the 3-5 most relevant hits: `lit-refs ID` (backward) and `lit-citers ID` (forward,
  which finds the follow-ups and rebuttals). Repeat once from the best new finds.
- Also search: GitHub (implementations, issues where claims fail to reproduce), OpenReview reviews of
  the key papers (reviewers often name the weaknesses), and blog posts from the authors' labs.
- Triage into: core (must read in full), context (skim), irrelevant. Expect 30-80 candidates and
  8-15 core papers.

## 2. Read the core papers in full
`researchctl lit-fetch arxiv:ID` returns a text file and a section index. Read the method,
experiments, and appendix (where the caveats live). For many papers, delegate in parallel to
subagents (model: sonnet). Each subagent reads one paper and writes one card, returning only a
three-line summary. You read the cards, not the papers.

## 3. Paper card: `projects/P/lit/cards/<id>.yaml`
```yaml
id: arxiv:2610.14500
title: ...
authors: [...]
year: 2026
venue: arXiv | ICLR 2026 | ...
code: https://github.com/... | null
depth: abstract | skim | full | reproduced
question: the problem as the authors frame it
hypothesis: their core hypothesis or mechanism story, in one or two sentences
journey: how they got there - earlier versions (arXiv v1 vs v2 diff), ablations, failed variants they mention
claims:
  - text: one claim, quoted or tightly paraphrased
    evidence: which table/figure, models, data, scale, seeds, baselines
    strength: strong | moderate | weak
    concerns: missing baselines, single seed, tuned-vs-untuned comparisons, small effect, test-set tuning
setup: models, datasets, scale, compute
skeptic: what would make the main claim wrong; conditions never tested; confounders; whether results
  would hold at other scales or architectures
relevance: what this means for our problem; which claim our work would depend on
links: [ids of related cards]
```
Be specific. "Results may not generalize" is useless; "only tested at 125M params with a fixed
LR, so the gain could be an LR-sensitivity artifact" is useful.

## 4. Synthesize: `projects/P/lit/map.md`
- Threads of work and how they relate; who disagrees with whom and on what evidence.
- What is established (multiple independent confirmations), believed (one paper, weak evidence),
  and open.
- Reproduction targets: claims our direction depends on that are weakly supported. Pass them to
  the reproduce skill.
- Write it so the user can read it in five minutes: a table plus short paragraphs.

## 5. Novelty audit: `projects/P/lit/novelty.md` (required before any "novel/first" wording)
- The candidate contribution in one sentence.
- Queries run (point to `queries.log`), including adjacent-field terms.
- The 3 closest prior works and the concrete difference from each.
- Whether the idea exists as folklore: textbook, blog post, workshop paper, an old non-deep-learning
  method. If it does, the contribution is the evidence or the analysis, not the idea. Say so.

Done when: `researchctl gate P` shows the card and full-read thresholds are met, map.md exists, and
every claim the project depends on has been checked against its evidence.
