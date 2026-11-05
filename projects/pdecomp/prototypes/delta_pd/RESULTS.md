# Delta-PD prototype: results (EXPLORATORY)

Status: exploratory hypothesis generation, 3 seeds, one hyperparameter setting, not preregistered.
Nothing here supports a claim. Code: `delta_pd.py` (minimal SPD written from 2506.20790 Sec 2/App A),
`summarize.py`; raw numbers: `results_coef0.001_seed{0,1,2}.json`.

## Result
Delta-PD separates the two planted changes causally where SVD and LoRA do not: ablating the
Delta-PD component matched to one changed feature removes 70% of that feature's behaviour change
and 1% of the other's; for SVD (rank 2) and LoRA (rank 2) the best single component is shared by
both features (57% / 53% removed from each). H1 holds only partly (2-3 components per change, not
one), H2 holds on cost per step (C=10 vs 40) but full PD was not given a fair convergence budget,
H3 holds on causal specificity and purity but not on input-detection AUROC (all methods ~0.96-0.98).

## Setup
- Model: f(x) = W2 ReLU(W1 x + b1) + b2, 20 sparse features (p=0.05, U[-1,1]), 40 neurons.
- Base task y = |x| (6000 Adam steps, MSE 3e-5 to 5e-5). Fine-tune (all params, 3000 AdamW steps,
  lr 1e-3): add ReLU(x3) to y10,y11 and ReLU(x7) to y11,y12 (overlapping output 11). Change MSE
  1.5e-3 to 1.8e-3 before, 2e-5 to 4e-5 after.
- Delta-PD: W = W_base (frozen) + sum_c m_c U_c V_c^T, C=10 per matrix, target W_ft. Full PD: same
  code with offset 0, C=40 per matrix. Both: faithfulness + stochastic + layerwise recon, p=1
  minimality coef 1e-3, leaky hard sigmoids, d_gate=16, Adam 2e-3, batch 512, 2500 steps.
  Biases fixed to fine-tuned values (bias deltas have norm 0.06-0.09, not decomposed).
- Baselines: truncated SVD of dW_l (k=2 and k=4 per matrix); rank-2 LoRA per matrix distilled to
  f_ft outputs (3000 steps), components = SVD of B A.
- Metrics, per changed feature j in {3,7}, using the component with highest response on positive
  single-feature probes: purity (share of that component's probe response on feature j); AUROC of
  its score for label x_j>0 on natural sparse inputs; removed = |f_all - f_ablate_c| / |f_ft - f_base|
  on probe x=e_j; collateral = same on the other changed feature; cos between ablation effect and
  true change; base damage = ablation effect on outputs other than 10-12 relative to base output.
  Scores: gates for PD, |inner activation| x |U| for SVD/LoRA.

## Table (mean ± sd over seeds 0,1,2)
| method | purity | AUROC x_j>0 | removed | collateral | cos(abl, change) | base damage | recon MSE / change MSE |
|---|---|---|---|---|---|---|---|
| Delta-PD (C=10) | 1.00 ± 0.00 | 0.97 ± 0.01 | 0.70 ± 0.04 | 0.01 ± 0.00 | 0.90 ± 0.02 | 0.43 ± 0.05 | 0.01 ± 0.00 |
| full PD (C=40) | 1.00 ± 0.00 | 0.98 ± 0.01 | 1.15 ± 0.09 | 0.01 ± 0.00 | 0.75 ± 0.08 | 1.04 ± 0.03 | 0.01 ± 0.00 |
| SVD dW k=2 | 0.32 ± 0.04 | 0.96 ± 0.01 | 0.57 ± 0.06 | 0.57 ± 0.06 | 0.69 ± 0.09 | 0.46 ± 0.06 | 0.23 ± 0.03 |
| SVD dW k=4 | 0.32 ± 0.04 | 0.96 ± 0.00 | 0.57 ± 0.06 | 0.57 ± 0.06 | 0.69 ± 0.09 | 0.46 ± 0.06 | 0.13 ± 0.03 |
| LoRA r=2 | 0.40 ± 0.06 | 0.97 ± 0.00 | 0.53 ± 0.05 | 0.53 ± 0.05 | 0.84 ± 0.03 | 0.14 ± 0.10 | 0.02 ± 0.00 |

AUROC for "output changed at all" (summed scores): Delta-PD 0.83-0.88, SVD k=2 0.80-0.84,
LoRA 0.81-0.82, full PD 0.68-0.71.

| | Delta-PD | full PD |
|---|---|---|
| alive subcomponents (gate max > 0.1, W1+W2) | 5-6 | 50-55 |
| alive responding only to features 3,7 (probes) | 4-5 of 5-6 | 4-6 of 50-55 |
| wall time, 2500 steps (shared 4-core node, load ~10) | 66-72 s | 197-199 s |
| masked-matmul cost per step (prop. to C) | 1x | 4x |
| stochastic recon MSE at step 2000 (seed 0) | 7.4e-5 | 2.6e-4 |

## Reading
- H1 (exact recovery): partial. Each changed feature gets one pure component (purity 1.00, collateral
  0.01), but it carries only 70% of the change; the rest sits in 1-2 further components, split
  across W1 and W2. Base damage 0.43 means the delta components also move unchanged outputs: the
  fine-tune did not implement the change as a clean new write. 50-57% of dW1 energy lies outside
  the changed input columns (Adam spreads updates), so the "planted" mechanism is not exactly what
  the weights contain. The toy's ground truth is the cause to fix before blaming the method.
- H2 (cheaper): per-step cost is 4x lower by construction (C=10 vs 40) and wall time 2.8x lower.
  Not shown: steps-to-equal-quality, because full PD got the same 2500 steps, C=40 is below the
  paper's C/true-count ratio, and only one coefficient was run. Full PD needs ~50 alive components
  to describe the base model, and its matched components merge the new write with the base |x_j|
  computation (removed 1.15, base damage 1.04): it cannot isolate the change without a base
  comparison. This is the stronger argument for Delta-PD, and it is qualitative.
- H3 (beats SVD/LoRA): yes on causal separation (collateral 0.01 vs 0.57 / 0.53; purity 1.00 vs
  0.32 / 0.40); the baselines' top component reads both changed features because the planted
  writes overlap on output 11. No on detection: AUROC for x_j>0 is 0.96-0.98 for every method, so
  "which inputs are affected" is easy here; "which mechanism" is where the methods differ.

## Failures and limits
- First run (coef 1e-2, C_F=80, 8000 steps, seed 0): minimality too strong; Delta-PD kept 1 alive
  component and missed feature 7, full PD reconstruction failed (recon/change MSE 3.3). Discarded
  after changing the coefficient, which is a tuning step on the same seed.
- A coef 1e-4 run and a 4-run batch timed out under node load (no results).
- One toy, one change pattern, one coefficient, 3 seeds; sd over 3 seeds is not a CI.
- Novelty not audited: fine-tuning-delta decompositions (task vectors, LoRA analyses, model diffing
  with crosscoders) exist; check `lit/novelty.md` before calling this new.

## Next experiment
1. Make the planted change exact: fine-tune with plain SGD or only on W2 (or plant dW by hand) so the
   ground truth is known rank-one pieces; then H1 is testable as yes/no.
2. Fair H2: full PD vs Delta-PD at matched final recon MSE, C/true-count ratio as in the paper,
   coef sweep {1e-4, 3e-4, 1e-3} for both, report steps and FLOPs to threshold.
3. Add a crosscoder / model-diffing baseline and a "full PD on W_ft minus full PD on W_base"
   matching baseline; add a random-direction ablation control for the causal metrics.
4. Harder change: new feature added, and changes that reuse base neurons in superposition (h < n).
