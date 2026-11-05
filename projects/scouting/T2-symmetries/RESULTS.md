# T2 kill-test result: optimizer-aware gauge fixing (idea 4)
Preregistration: `PREREG.md` (written before the run). Code: `gauge_muon.py` (preregistered),
`ablate_canon.py` (exploratory, after). CPU: 182 s + ~150 s, 1 thread.

## Verdict: killed
The balanced gauge made Muon fine-tuning worse on every seed (+10.6% B-test loss, prediction was
>= 5% better). Muon is very gauge-sensitive (random teleport +174%), but the native AdamW gauge was
the best of the gauges tried.

## Numbers (B-test = normalized MSE on the fine-tuning task; 3 paired seeds; LR tuned per arm on validation)
| Arm | B-test mean | vs muon_native, per seed | A-test (forgetting) |
|---|---|---|---|
| adam_native | 0.0159 | -1.4, -6.5, -12.9 % | 0.749 |
| muon_native | 0.0171 | 0 | 0.752 |
| muon_canon (fold gains + balance neurons) | 0.0189 | +11.0, +10.3, +10.5 % | 0.750 |
| muon_random (log-normal sigma=1 neuron scales) | 0.0467 | +188, +160, +174 % | 0.721 |

Exploratory split of canon (B-test, seeds 0/1/2): native 0.0163/0.0189/0.0161; fold gains only
0.0166/0.0196/0.0164 (+2 to +4%); balance only 0.0178/0.0207/0.0175 (+9%). Balancing does most harm.

## Interpretation and limits
- AdamW with weight decay already leaves neurons near balance (sd of log(|W_d col|/|W_u row|) = 0.10-0.33)
  and the residual imbalance seems to carry useful per-neuron step-size information; forcing exact
  balance discards it. Consistent with SAGGS's restart finding (arXiv:2610.18085) that any gauge
  move away from the trained one costs loss later.
- The toy shows only a small Adam->Muon gap (6.9%), so it does not reproduce the mismatch reported in
  arXiv:2605.10468 at LM scale; a different canonical gauge (e.g. spectral, learned on a probe task)
  was not tested. Toy: 3-block ReLU MLP, d=64, teacher-student regression.
- Not tested: real LM checkpoints, whose RMSNorm gains are far less uniform than here (0.5-1.2).

## NeurIPS-scale version (if revived)
Reframe from "balance" to "gauge as a fine-tuning hyperparameter that transfers": (1) zero-GPU audit of
gain/neuron/head imbalance in Llama/Qwen/OLMo checkpoints; (2) learn per-neuron/per-head gauge on a
small probe task with Muon, test transfer to other tasks at 0.5-1B scale (8 slices, ~200 slice-h);
(3) win condition: closes >= 50% of the Adam-vs-Muon full-FT gap of arXiv:2605.10468 at zero inference
cost. Current CPU evidence argues against simple canonical gauges, so expected value is low.
