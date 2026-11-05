# Writing style for everything in this repo

Applies to notes, PROJECT.md, reports, commit messages and papers.

- Lead with the result or decision. Context second.
- Numbers instead of adjectives. Units always.
- Short sentences, active voice, concrete subjects ("the ablation", not "this").
- One idea per paragraph; tables for comparisons; bullets only for genuinely parallel items.
- Name known techniques by their known names. No invented branding for ordinary methods.
- State uncertainty once, precisely (CI, n, what was not tested). No stacked hedges.
- No stock phrases: delve, crucial, pivotal, leverage, seamless, robust (as filler), "it is worth
  noting", "in the realm of", "a testament to". No hype: groundbreaking, remarkable, dramatic.
- At most one em dash per paragraph; prefer commas, colons or a new sentence.
- Commit messages: `<area>: <what changed>` in the imperative, under 72 characters; a body only when
  the why is not obvious.
- `researchctl lint PATH` checks most of these mechanically.
