# Pre-registration: SPD full-budget reproduction on TMS 5-2 (repro/B-spd-fullbudget)

Written 2026-11-05, before any run in this directory.

## Claim under test
SPD (arXiv:2506.20790) Table 1, TMS 5-2, C = 20: MMCS 1.000 ± 0.000, ML2R 0.993 ± 0.002.
Budget: 40k steps, batch 4096, coeff-imp 3e-3, p = 1, Adam lr 1e-3 cosine.

## Runs (fixed before running)
- Library `projects/pdecomp/src/spd_min` unmodified, CLI `train_tms.py`.
- Seeds 0, 1, 2 (same list in both arms). Target TMS: seed 0, shared by all runs.
- Arm A: no warmup. Arm B: `--faith-warmup 200` (AdamW lr 1e-2, wd 0.1 on U, V; from the reference
  repo's current config, not the paper).
- 1 thread per process, at most 2 processes at once. Wall time recorded.

## Metrics (20k fresh samples, eval seed 1234)
MMCS, ML2R (Eq. 9-10) on linear1 (columns of W) and linear2 (rows of W^T); fullcos_min; alive
subcomponents (CI > 0.1 on any input); one-hot check (each feature alone activates exactly one
subcomponent, and it is the geometrically matched one); recon (stochastic, CI-masked, unmasked).

## Pass threshold (per run)
A run passes if, on both linear1 and linear2: MMCS >= 0.99, ML2R >= 0.95, and exactly 5 alive
subcomponents, one per feature (one-hot check true).

## Verdict rule (per arm)
- reproduced: >= 2/3 seeds pass.
- partially: MMCS >= 0.99 on both matrices in >= 2/3 seeds but < 2/3 seeds pass.
- not reproduced: otherwise.

## Predictions
- Arm A (no warmup): splitting persists at full budget in >= 1 of 3 seeds; I expect "partially"
  (P ~ 0.6), based on container A's 10k-step batch-4096 run (ML2R 0.27 / 0.77, 10 / 9 alive).
- Arm B (warmup): "reproduced" with P ~ 0.6; linear1 is the likelier failure (container A's
  10k-step warmup run had one linear1 feature off).
- Warmup effect: mean ML2R (both matrices) higher in Arm B than Arm A in >= 2 of 3 paired seeds.

## If splitting persists
Diagnose against paper App. A and the official code (goodfire-ai/spd): CI-MLP init, leaky hard
sigmoid, loss normalisation (sum vs mean over subcomponents / layers / outputs), minimality term,
which matrices are decomposed. Diagnostics: <= 10k steps, seeds 0-2, labelled exploratory.

## Optional: TMS 40-10
C = 200, coeff 1e-4, p = 2, at whatever fraction of 40k steps fits ~3 CPU-hours. Paper: MMCS
0.996-1.000-ish, ML2R ~0.99 (exact numbers to be read from Table 1). Reported with its budget
fraction; no pass/fail at a reduced budget beyond "consistent / inconsistent".
