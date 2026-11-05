---
name: reproduce
description: Reproduce a prior paper's key claim at the smallest decisive scale before building on it. Use after literature review identifies claims the project depends on, or whenever a baseline number is needed.
---

# Reproduce before you build

Many published gains shrink or vanish under fair comparison. Reproduce the claims your project
depends on, and only those, at the smallest scale that still tests them.

1. Choose the claim: the one most load-bearing for the gap, and preferably the weakest-evidenced.
   Write down the number you expect from the paper (table/figure reference).
2. Prefer the authors' code at a pinned commit. Note any divergence from the paper (hyperparameters,
   data processing, eval details). These differences are findings in their own right.
3. Scale down deliberately: smaller model or fewer steps, keeping the comparison's structure. State
   why the downscaled version still tests the claim.
4. Run with 3 seeds, same budget for baseline and method. Tune the baseline as hard as the method
   (an untuned baseline is the most common source of fake gains).
5. Write `projects/P/repro/<paper>/REPORT.md`:
   - claim, paper's number, our number with CI, setup diff
   - verdict: reproduced / partially (smaller effect, holds only under X) / not reproduced
   - what it implies for our gap
6. Update the paper card (`depth: reproduced`, and add concerns).

A failed reproduction is valuable. Check it carefully, then record it, and consider whether it is
the more interesting research direction.
