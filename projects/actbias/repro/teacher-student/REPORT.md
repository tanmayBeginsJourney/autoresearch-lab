# Reproduction: arXiv:2610.14500 Sec 2.1 teacher-student, with per-arm LR tuning
Exploratory (not preregistered), CPU, 2026-11-05.

**Verdict.** ReLU: shifted reaches lower final loss, the effect is small (-5% to -6.5%) and survives per-arm
LR tuning and a function-preserving bias control. GELU: no final-loss advantage. "Slower early" reproduces
for GELU only.

Rerun: `python ts.py relu.json relu U,S,UB 3e-4,1e-3,3e-3,1e-2,3e-2 0,1,2,3,4 5000`;
`python ts.py gelu.json gelu U,S 1e-3,3e-3,1e-2 0,1,2,3,4 5000`; `python analyze.py relu.json gelu.json`.

## Setup
- (d, m, L) = (20, 80, 10); pre-LN residual MLP, no linear biases, LN without affine; init U(+-1/sqrt(fan_in)); teacher unshifted.
- Arms: U unshifted; S phi - b (ReLU b = 0.230, GELU b = 0.115); UB unshifted + trainable per-layer output
  bias initialised to -b W2 1 (same function as S at init).
- Adam, constant LR, batch 128, fresh x ~ N(0, I) each step, 5000 steps, MSE. Seeds 0-4 paired (same
  teacher, init, batches across arms and LRs). LR chosen per arm on a validation set; numbers on a separate test set.
- 95% paired-t CIs (df = 4). MSE values x1e-3 unless noted. LR, batch, steps, input distribution, LN affine
  and schedule are our choices (unstated in the paper).

## ReLU (75 runs, none diverged)
Mean final test MSE by LR:

| arm | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|
| U | 18.37 | 13.66 | 11.40 | 23.57 | 306.3 |
| S | 17.81 | 12.98 | 10.65 | 23.53 | 109.4 |
| UB | 18.24 | 13.41 | 11.41 | 23.24 | 41.1 |

All arms select 3e-3 (inside the grid).

| LR | S - U [95% CI] | relative | S better |
|---|---|---|---|
| 3e-4 | -0.55 [-0.76, -0.35] | -3.0% | 5/5 |
| 1e-3 | -0.69 [-1.03, -0.35] | -4.9% | 5/5 |
| 3e-3 (best, all arms) | -0.75 [-1.29, -0.20] | -6.5% | 5/5 |
| 1e-2 | -0.04 [-1.26, +1.18] | 0.0% | 3/5 |

Bias control: S - UB at best LR -0.76 [-1.43, -0.09]; at 3e-4 -0.42 [-0.55, -0.29]; at 1e-3 -0.44
[-0.67, -0.20]. UB - U at best LR +0.01 [-0.80, +0.82].

Early dynamics at LR 1e-3 (raw MSE U / S / UB; S - U x1e-3): step 0 1.152 / 0.956 / 0.956, -195 [-358, -33];
step 10 0.382 / 0.399 / 0.337, +16 [-20, +52]; step 20 0.219 / 0.225 / 0.200, +7 [-8, +22]; step 50
-9 [-16, -2]; step 100 -12 [-17, -8]; step 5000 -0.7 [-1.0, -0.3]. At 3e-3, S leads from step 10.
Loss still decreasing at step 5000.

## GELU (30 runs, U and S, LRs 1e-3 / 3e-3 / 1e-2)
Mean final test MSE S 2.56 / 3.11 / 7.36, U 2.57 / 3.06 / 7.22. S - U: -0.00 [-0.03, +0.02] (4/5);
+0.04 [-0.04, +0.13] (1/5); +0.14 [-0.36, +0.64] (3/5). Both select 1e-3, the grid edge (tuning incomplete).
Early S - U at 1e-3: step 10 +11.2 [+0.5, +21.9]; step 20 +8.9 [+5.3, +12.6]; from step 300 about 0.

## Claims
| claim | paper | ours | verdict |
|---|---|---|---|
| ReLU: shifted lower final loss at same LR | Fig 2, App C.2 | -0.69 [-1.03, -0.35] (-4.9%), 5/5 | reproduced, small |
| GELU: same | Fig 2 | -0.00 [-0.03, +0.02] | not reproduced |
| Shifted slower early | Fig 2 | ReLU no; GELU yes (steps 10-20) | partial |
| Survives per-arm LR tuning | not tested | ReLU yes at best LR, gone at 1e-2; GELU no effect | partial (ReLU) |
| Not a missing offset | App C.3 | S - UB = -0.76 [-1.43, -0.09], UB ~ U | consistent (ReLU) |

## Caveats
5 seeds, one size, one batch size, constant LR, 5000 steps (not converged); half-decade LR grid; GELU grid
truncated and no bias-control arm; the step-0 offset confounds early-speed comparisons. A confirmatory
version needs more seeds, the large size, a finer LR grid, a decaying schedule and GELU with the control arm.
