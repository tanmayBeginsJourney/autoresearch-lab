---
name: write-paper
description: Write an ML conference paper (NeurIPS/ICML/ICLR) from a project's supported claims - narrative, structure, figures, verified citations, honest framing, and the pre-submission checks. Use when drafting or revising a paper or a substantial report.
---

# Writing the paper

Sources: Gopen & Swan, "The Science of Scientific Writing" (1990); Lipton & Steinhardt, "Troubling
Trends in Machine Learning Scholarship" (arXiv:1807.03341); Nanda, "Highly Opinionated Advice on How
to Write ML Papers" (2025). Read the originals if a rule below seems arbitrary.

## Preconditions
- `researchctl claims check P` shows every claim the paper makes as `supported`, or explicitly
  scoped as exploratory or descriptive.
- The red-team review is answered (`projects/P/reviews/`).
- `lit/novelty.md` exists if anything is called new.

## Narrative first
Before writing prose, write `paper/outline.md`:
1. The contribution in one sentence a reviewer would repeat to the area chair.
2. 1-3 claims (ledger ids) that make up that contribution.
3. For each claim, the single most convincing piece of evidence, usually a figure.
4. What a skeptical reader will object to, and where the paper answers it.
If step 1 does not fit in a sentence, there is no paper yet; go back to analysis.

## Structure and effort
Readers see the title, abstract, Figure 1 and the introduction; most stop there. Spend effort
roughly equally on (a) abstract + title, (b) introduction, (c) figures, (d) everything else.
- Abstract: context (1 sentence), gap, what we do, the main result with its number, why it matters.
- Introduction: the problem, why existing work does not settle it (specific papers), our approach,
  contributions as a short list tied to evidence, and one sentence each on limitations.
- Related work: organized by idea, not paper-by-paper; say how each thread differs from ours.
- Experiments: the claim first, then the setup, then the result. Every number comes from
  `\claimval{...}` or a generated table.
- Limitations: real ones (scale tested, models, compute), not ritual hedges.

## Sentences (Gopen & Swan)
- Put old information at the start of a sentence and new information at the end (the stress position).
- Keep subject and verb close; put the action in the verb ("we measured", not "a measurement was
  performed").
- One paragraph, one point; the first sentence says what the paragraph shows.
- Use one term per concept, defined once. Do not vary words for elegance.

## Honest framing (Lipton & Steinhardt)
- Separate explanation from speculation; label speculation.
- Attribute gains to their real sources. If tuning, scale or a known trick explains part of the gain,
  run the ablation and say so.
- Use mathematical notation only when it makes things clearer; no notation for show.
- Use plain words for known ideas. "We subtract the mean activation", not "we introduce
  Bias-Neutralized Activation Dynamics". Name a method only if people will need to refer to it.
- Hedge exactly as much as the evidence demands; no more, no less.

## Density: more information, fewer words
- Replace adjectives with numbers ("2.1x faster at batch 64 on an H200", not "much faster").
- Cut throat-clearing ("It is important to note that", "In this section we").
- Prefer one figure that answers a question over two that each answer half of it.
- `researchctl lint paper/ --bib paper/refs.bib` flags hype, stock phrases, literal numbers,
  placeholders and missing citation keys. Fix every hit or justify it.

## Figures
Figure 1 should convey the main result without the text. Show uncertainty and individual seeds
where readable. Use vector PDFs and colorblind-safe palettes, and write self-contained captions
that say what to look at.

## Citations
Get every BibTeX entry from `researchctl lit-cite <arxiv id | doi>`; never write BibTeX from memory.
Cite a claim only after reading the paper's evidence for it (its card). Unverifiable: leave
`[CITATION NEEDED]` and tell the user.

## Venue mechanics
Fetch the official LaTeX template for the target venue and year when drafting starts; do not edit its
style files. Check the page limit, anonymization, checklist (NeurIPS) and supplementary rules.
Release code and configs; the experiment records already contain what reviewers ask for.
