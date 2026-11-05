---
name: gap-analysis
description: Turn a literature map into candidate research gaps and decide which one is worth the compute, with an honest expected-value analysis. Use after literature review and reproduction, before writing a preregistration.
---

# Gap analysis

Most gaps are gaps because they are unimportant, already solved somewhere you did not look,
or too expensive. Your job is to find the rare one that is none of these.

For each candidate (generate 5-10, write them all down), add a section to `projects/P/gaps.md`:

- **Gap**: one sentence. What is unknown or not working?
- **Evidence it is open**: queries run, closest work, why that work does not settle it.
- **Why it matters**: who changes what they do if we answer it (practitioners, theorists,
  interpretability researchers). Write "nobody" if that is the honest answer.
- **Textbook check**: is the answer obvious to someone in optimization, statistics or signal
  processing? Is the "new idea" a known technique under a new name?
- **Outcome distribution**: plausible results with rough probabilities. For each, is it publishable,
  is it useful, what do we do next? A good gap has value even if the hypothesis fails (a clean
  negative result or a mechanism explanation).
- **Smallest decisive experiment**: model scale, data, seeds, GPU-hours on slices, calendar days.
- **Scale risk**: does the effect plausibly survive scale-up? What evidence at small scale would
  justify a request for H200 time?
- **Value per cost**: a rough score and a sentence defending it.

## Decision (required section, `## Decision`)
Pick one gap (rarely two). Say why it beats the runner-up and what result would make us abandon it
(the kill criterion). Ambition matters: prefer the gap that would change how people think or work
if it pans out, provided the first decisive experiment fits in about a week of slice time.
Incremental leaderboard gains on a single benchmark do not qualify.
