# pd_toys: PD benchmark toys with known mechanisms

Target models whose ground-truth mechanisms are known as rank-one pieces per weight matrix, plus
recovery metrics. CPU only. Tests: `cd projects/pdecomp/src && python -m pd_toys.test_pd_toys`
(11 tests, ~13 s on one thread; pytest also works).

## Objects

- `Toy`: `model` (nn.Module), `params` (names of the matrices to decompose; biases stay fixed),
  `sample(batch, gen)`, `target(x)`, `mechanisms`, `info`. `toy.forward(x, weights)` runs the model
  with replaced matrices; `toy.residual()` gives ||W - sum of GT pieces|| / ||W|| (0 for exact toys).
- `Mechanism`: `factors[param] = (u, v)` (piece u v^T), `features` (inputs that use it),
  `active(x)` (batch mask), `alive`, `tags`.

## Toys

| Constructor | Model | Ground truth | Cost |
|---|---|---|---|
| `make_tms(5, 2)`, `make_tms(40, 10)`, `identity=True` | TMS, trained tied (AdamW 5e-3, 10k steps, p = 0.05), then untied | feature j: column j of W1 + row j of W2; identity matrix as e_k e_k^T tagged `basis_ambiguous` | 40-10: 18.6 s, all 40 features alive (seed 0) |
| `make_l4cc()` | 100 -> 50 ReLU -> 100, no bias, L4 loss, p = 0.02, U(-1, 1) (arXiv:2607.04800) | hand-set three-scalar ansatz W_in = a M^T + b(1 - M^T), W_out = c pinv(W_in), biregular K = 5 code with overlap-minimising edge swaps; feature j: column j of W_in + row j of W_out; `info["M"]` = codewords | 26 s (20k swaps, 400 Adam steps for a, b, c) |
| `make_crosslayer(F, rho=...)` | y = W3 ReLU(W2 ReLU(W1 x) + b2), hand-set | mechanism k computes y_k = ReLU(abs(x_k) - theta) with one rank-one piece in each of W1, W2, W3; disjoint, permuted neurons | < 0.1 s |
| `make_planted(F, n_spare, changes, drift)` | cross-layer base + fine-tuned copy | `deltas`: add (3 pieces), delete (3 negated pieces), edit (reroute input, W1 only), plus Gaussian `noise` (drift arm); ft = base + deltas + noise exactly | < 0.1 s |

Notes on the ground truth:
- TMS 40-10 and the L4 toy are superposed: ablating one GT mechanism also perturbs other features'
  outputs (`ref_off` in the ablation check is nonzero).
- The L4 toy is the paper's ansatz, not a trained network. The paper reports this ansatz family
  reaches ~1.12x the L4 loss of its trained network (Table 1); it does not print fitted (a, b, c).
  Our fit (seed 0): a = 0.377, b = -0.0027, c = 1.576, L4 loss 4.66e-5 vs 1.00e-3 for the naive
  half-the-features solution. A trained L4 network's mechanism matches this only approximately
  (per-entry encoder values and decoder residual are unexplained in the paper). The 1000-d embedded
  variant is not implemented.
- `rho` is the probability that a feature copies its group's shared on/off draw: marginal activity
  stays p and the within-group indicator correlation is rho^2.
- In the cross-layer toy every piece of a mechanism is necessary (ablating the W1 piece alone silences
  the mechanism), so the ablation check passes a component that holds only one of the three pieces;
  `match_mechanisms` scores it low. Use both.

## Metrics (`pd_toys.metrics`)

- `match_pieces(gt (G,...), found (C,...), threshold=0.9)`: Hungarian matching on cosine of the
  flattened pieces; precision over alive found pieces, recall over GT; signed by default.
- `match_mechanisms(gt, found: list[dict[param, matrix]], weights)`: whole-mechanism matching across
  matrices (each matrix block scaled by 1/||W_param||). `match_per_matrix(gt, {param: (C, o, i)})`.
- `ablation_recovery(toy, gt, found, assignment)`: subtract the matched component and the GT
  mechanism from the weights; `err_on` (does the component's ablation reproduce the mechanism's on
  inputs that use it), `dmg_off` vs `ref_off` (selectivity), `pass_rate`. For a planted fine-tune
  pass `toy=pl.ft, gt=pl.deltas`.
- `pieces_from_factors(U, V)` converts (C, d_out), (C, d_in) factors to pieces.
