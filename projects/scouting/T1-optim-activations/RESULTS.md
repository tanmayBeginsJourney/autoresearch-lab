# T1 kill-test result: centering x Muon (idea 8 in IDEAS.md)
Exploratory CPU test, 2026-11-05. Rerun: `python muon_center.py results.jsonl AU,AS,MU,MS,MP 1e-3,3e-3
3e-3,1e-2 0,1,2 1000 3` and then `python analyze.py results.jsonl` (30 runs, 137 s wall on a shared CPU).

**Verdict: dead.** P1 and P2 both fail. Centering does not help Muon. Projecting the input mean out of
the gradient makes Muon worse. Plain Muon is the strongest arm.

## Numbers (test MSE x1e-3, 1000 steps, ReLU (20, 80, 10), paired seeds 0-2, no divergences)
| arm | LR grid: mean test | LR picked on val | per-seed test at that LR |
|---|---|---|---|
| Adam-U | 1e-3: 24.11, 3e-3: 20.58 | 3e-3 | 18.94, 21.98, 20.82 |
| Adam-S | 1e-3: 22.96, 3e-3: 20.03 | 3e-3 | 18.66, 20.46, 20.98 |
| Muon-U | 3e-3: 16.21, 1e-2: 25.27 | 3e-3 | 14.85, 16.84, 16.94 |
| Muon-S | 3e-3: 16.29, 1e-2: 24.73 | 3e-3 | 15.16, 16.71, 17.01 |
| Muon-P | 3e-3: 17.31, 1e-2: 27.96 | 3e-3 | 15.67, 17.70, 18.55 |

- Adam S vs U: -2.5% (S better in 2/3 seeds). This is weaker than the 5000-step repro (-6.5%, 5/5).
- Muon S vs U: +0.6% (1/3), per-seed +2.1 / -0.8 / +0.4%. **P1 (>= 2x the Adam gain, 3/3): FAIL.**
- Muon P vs U: +6.7% worse (0/3).
- Best method arm (Muon-S) vs the strongest baseline (Muon-U): +0.6% (1/3). **P2 (<= -5%, 3/3): FAIL.**

## Interpretation (hypothesis-generating only)
The prediction had the wrong sign. Under Muon the mean-shift effect is about zero, not amplified. This fits
the view that orthogonalization already removes the gradient anisotropy that the activation mean creates,
much as K-FAC is invariant to centering (arXiv:1503.05671). Muon beat Adam by ~20% here, so the
centering gain seen under Adam may be smaller than the gain from switching optimizer. That is a reason not to
build a centering method paper on Adam-only evidence.

## Caveats
- 3 seeds and 1000 steps (cut from the preregistered 3000; see the deviation in IDEAS.md).
- 2-point LR grids. Muon picked 3e-3, the lower edge of its grid, so Muon tuning is incomplete.
  Adam picked 3e-3, the upper edge of its grid.
- One width/depth, one toy task. Muon-P has no bias that absorbs the projected component, unlike SBO-AdamW.
- CPU time: 2.3 min for this run, plus ~10.5 min on the killed full design, whose output was never read.
  That is over the 12-min budget because the machine was shared.

## What a NeurIPS-scale version would need (not recommended)
The only surviving claim is the analysis question "do orthogonalized or Kronecker optimizers (Muon,
SOAP) subsume activation centering?" Testing it means nanoGPT-scale runs (124M, ~3 slice-days) with per-arm
LR sweeps over Adam / Muon / SOAP x {U, S}. The user ruled out analysis-only papers, and the method half is
dead, so the T1 area does not yield a positive, original method from this line. Note for actbias: any
centering paper must include a Muon baseline, because this test suggests Muon alone erases the gain.
