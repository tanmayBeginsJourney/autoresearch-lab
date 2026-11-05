# Activation function and decomposability (G4, links G2)

Result: the property that matters is not smoothness or the Lipschitz constant but the **negative-tail
leakage** ell_phi = -inf_{z<0} phi(z): ReLU 0, GELU 0.170 (at z* = -0.752), SiLU 0.278 (at z* = -1.278).
A dead zone (phi = 0 on z <= 0) makes exact sparse faithfulness possible; any real-analytic phi with
a nonzero tail makes it impossible (proven, P1-P3). The adversarial excess over ReLU is linear in
ell_phi in output norm and at most quadratic in KL (proven, P4-P5). Adversarial masks on "dead" GELU/SiLU
neurons should sit at m_i z_i = z*, an interior point, which is a sharp, cheap test (H1).

Labels: **[P]** proven here (short proof given), **[H]** heuristic or numerical, **[N]** numbers from the
sanity script (random Gaussian weights, numpy, < 1 min; not a trained model).

## Setup
y = W2 phi(p), p = W1 x (+ b1, biases unmasked). Neuron basis: W1 row i is a rank-one subcomponent with
mask m_i, W2 column w2_i has mask n_i; masked output y(m,n) = sum_i n_i w2_i phi(m_i z_i). Feature basis
on W1 (sparse x = sum_j s_j e_j): column j with mask m_j. Faithful at x means y(m,n) = y(1,1) for all
masks in the box [g,1]. "Inactive" set I = subcomponents with g = 0. Identity used throughout **[P]**:
phi = ReLU + eps_phi with eps_phi(z) = phi(-|z|) for GELU and SiLU (since z*Phi(z) - z = -z*Phi(-z), same for
sigmoid). eps is even, in [-ell, 0], and checked numerically to 1e-15 **[N]**.

## (a) Existence of exactly faithful sparse decompositions
**P1 [P] (dead zone suffices).** If phi(z) = 0 for z <= 0 (ReLU, JumpReLU, shifted ReLU), the neuron basis
with g_i = 1[z_i > 0] on both W1 row i and W2 column i is exactly faithful at every x: for z_i <= 0,
phi(m z_i) = 0 for all m in [0,1]. L0(x) = 2 x (#active neurons); adversarial loss is exactly 0.
In the feature basis on W1, absent features (s_j = 0) have zero effect for *every* phi, so the W1 side
is phi-independent; phi acts through the hidden side and through partial masks on active units.

**P2 [P] (neuron basis, necessary and sufficient).** With phi(0) = 0, g_i = 0 is exactly faithful iff
w2_i = 0 or phi = 0 on the segment between 0 and z_i. Proof: set all other inactive n to 0 and this
n_i = 1; constancy forces w2_i phi(m z_i) to be constant in m, and it equals 0 at m = 0.
Corollary: GELU and SiLU vanish only at 0, so exact faithfulness in the neuron basis forces g = 1 on
every neuron with z_i != 0 and w2_i != 0. Exact L0 = full width, for every input.

**P3 [P] (any rank-one basis on W2).** Let phi be real-analytic with phi(s)/s -> 1 (s -> +inf) and
-> 0 (s -> -inf) (GELU, SiLU), W1 rows nonzero and pairwise non-parallel. If a W2 subcomponent q_b
has q_b . phi(W1 x + b1) = 0 on any nonempty open set of inputs, then q_b = 0.
Proof: analyticity extends the identity to all x; along x = t r, dividing by t and t -> inf gives
sum_i q_bi (w_i . r)_+ = 0 for all r; crossing the hyperplane w_i . r = 0 changes the gradient by
q_bi w_i, so q_bi = 0. The same argument applies within any coordinate subspace of active features
with continuous magnitudes (rows restricted to those columns). So for generic dense W1, no W2
subcomponent of a GELU/SiLU layer can be exactly inactive on any continuous family of inputs.
P3 fails for ReLU (q supported on neurons dead on the set works). It also fails for finite input
sets (binary features, n_inputs <= width): a dual basis is then exact for any phi, so toys must use
continuous feature magnitudes or they cannot separate activations.

**Bilinear [P].** h = (Wx) . (Vx) is zero only where a factor is zero, so neuron-basis L0 is dense
like GELU. But h(x) = sum_{j<=k} s_j s_k r_jk with pair vectors r_jk; if the active pair vectors are
linearly independent (needs #pairs <= width), W2 components dual to them are exactly faithful with
L0_W2 = #active pairs <= k(k+1)/2 and W/V feature components L0 = 2k. The masked output is a polynomial,
multilinear in the masks, so the worst case over the box is attained at a vertex (closed form).

**TopK [H].** Selection is not elementwise. With g in {0,1} it behaves like P1 (L0 = K exactly). With
g < 1 on an active unit, its masked pre-activation can drop below z_(K+1) and swap in a new unit: a
jump of size about |z_(K+1)| ||w2||. Freezing the selection on the clean pass removes this.

## (b) Worst case over the mask box
**P4 [P] (exact separable worst case, neuron basis, actives fixed at 1).** Let lo_i, hi_i be the min
and max of {0} union phi([z_i, 0]). For any unit u,
  sup_box u . dy = sum_{i in I} [ (u.w2_i)_+ (hi_i - phi(z_i)) + (u.w2_i)_- (phi(z_i) - lo_i) ],
since each neuron's reachable value n phi(m z) fills [lo_i, hi_i]. Consequences:
- Per neuron Lambda_i = max(hi_i - phi(z_i), phi(z_i) - lo_i) <= ell_phi, so
  max_i ||w2_i|| Lambda_i <= sup ||dy|| <= ell_phi sum_{i in I} ||w2_i||. ReLU: both sides 0.
- For z_i <= z*, lo_i = -ell, hi_i = 0, and the sign-averaged gain is exactly (ell/2)|u.w2_i|,
  independent of z_i. Deeply "dead" neurons do not get safer: the adversary rescales them to z*.
  GELU at z = -3 contributes -0.004 at m = 1 but -0.170 at m = 0.25 (42x).
- The Lipschitz constants (GELU 1.129, SiLU 1.100, ReLU 1) are nearly equal and give a vacuous
  ||dy|| <= L ||W2|| ||dp||; they do not order the activations. ell_phi does.

**P5 [P] (comparison to ReLU with the same weights, any masks, any basis).**
dy_phi = dy_ReLU + [W2(n) eps(p(m)) - W2 eps(p)]. In the neuron basis each bracket term lies in
[-ell, ell] x w2_i, so sup||dy_phi|| <= sup||dy_ReLU|| + ell_phi sum_i ||w2_i||. For a linear readout to
logits, Hoeffding's lemma gives KL(p || p') <= (max dlogit - min dlogit)^2 / 8, hence the leakage
part of adversarial KL is <= (1/2) max_v ||W_U,v||^2 (ell_phi sum_i ||w2_i||)^2: quadratic in ell.

**[H] Isotropic estimate.** For w2_i spread over d dims, E|u.w2_i| = ||w2_i|| sqrt(2/(pi d)), so the
worst case grows like |I|/sqrt(d) while i.i.d. U(0,1) masks grow like sqrt(|I|). Ratio adv/stoch
~ sqrt(|I|/d). **[N]** Gaussian weights, z ~ N(0,1), ||dy|| (adversary = alternating maximisation of P4):

| d, h | GELU stoch / adv | SiLU stoch / adv | SiLU_beta=4 stoch / adv | ReLU |
|---|---|---|---|---|
| 64, 256 | 0.50 / 0.99 | 0.82 / 1.52 | 0.16 / 0.43 | 0 / 0 |
| 64, 1024 | 0.50 / 1.44 | 0.86 / 2.11 | 0.18 / 0.66 | 0 / 0 |
| 256, 1024 | 1.00 / 2.00 | 1.71 / 3.07 | 0.35 / 0.88 | 0 / 0 |

adv/stoch is 1.8-2.9 and rises with h/d. Sharper activations (SiLU_beta(z) = z sigma(beta z), leakage
ell/beta) shrink the leak but raise adv/stoch (2.5-3.7): the minimum sits at m = z*/(beta z), which
uniform masks rarely sample. Stochastic-only training (SPD) should therefore underfit the leak most
for nearly-ReLU activations.

## (c) Predicted ordering (matched task loss, width, seeds; continuous sparse features)
| | ReLU / JumpReLU | TopK | Bilinear | GELU | SiLU |
|---|---|---|---|---|---|
| Exact-faithful L0, hidden side | #active [P] | K [P] | #pairs if no pair superposition, else dense [P] | full width [P] | full width [P] |
| L0 at tolerance delta (full ablation) | #active | K | as left | lower than SiLU | higher: needs z < -6.5 for abs(phi) < 0.01 vs GELU z < -2.7 [N] |
| Leakage term in adversarial KL | 0 [P] | 0 if g in {0,1}; rank swaps otherwise [H] | 0 from absent features; worst case at vertices [P] | ~ell^2 [H] | ~2.7x GELU [H] |

Predicted adversarial-KL ordering [H]: ReLU <= TopK < bilinear < GELU < SiLU, with KL(SiLU)/KL(GELU)
~ (0.278/0.170)^2 = 2.7 and ||dy|| ratio 1.5-1.6 [N] in the leakage-dominated regime. Bilinear is the
least certain entry: it depends on whether PD finds the pair basis within its C budget.
Predicted L0 ordering: TopK <= ReLU < bilinear < GELU <= SiLU. At a fixed adversarial budget, GELU/SiLU
pay in L0 by keeping the leakiest neurons on; the L0-vs-adversarial-KL frontier of ReLU should
dominate both.

Leakage share grows in sparse regimes **[N]**: fraction of a neuron's output power from z < 0, for
z ~ N(mu,1): GELU 1.9% (mu=0), 17% (mu=-1), 64% (mu=-2); SiLU 6%, 50%, 94%.

## (d) Making a trained GELU/SiLU model decomposable without retraining
1. **Dead-zone edit with mean correction** (clamping plus mean ablation, both textbook). Replace phi by
   phi_c = phi . 1[z > 0] and fold W2 E_x[eps_neg(p)] into the output bias. P1 then holds exactly for
   the edited model. Cost [N]: residual power after mean correction 1.1% (GELU) / 3.5% (SiLU) per neuron at
   mu = 0, 5.8% / 11.5% at mu = -1. Swapping to plain ReLU (also changes the positive side) costs
   0.6% / 1.9% at mu = 0 after mean correction. Go/no-go: the edit's dCE on validation.
2. **Leakage as an explicit unmasked residual** (same idea as VPD's Delta-component). Masked forward
   = W2(n) phi_c(p(m)) + W2 eps_neg(p_clean). Exact at m = 1, leak-proof under any mask. Honest only if
   mean-ablating the residual costs little CE; report its output share.
3. **Freeze gates on the clean pass** (the frozen-nonlinearity trick from linearised attribution):
   phi(m z) . 1[z_clean > theta]. theta must be 0: any theta < 0 leaves (theta, 0) reachable by
   rescaling a neuron with z < theta, which is exactly the P4 exploit.
4. Not a fix: restricting the adversary to vertices or rounded masks. For GELU the worst case is interior.
5. Cheap fine-tune fallback (retraining, so outside the brief): anneal beta in phi_beta, or relufication
   (Mirzadeh et al., arXiv:2310.04564, ReLU Strikes Back, ICLR 2024).

## Testable predictions
- **H1** (sharpest, no retraining): in PGD masks against a GELU (SiLU) target, g~0 W1-row
  subcomponents on neurons with z_i < z* have m_i z_i clustered at -0.75 (-1.28); for ReLU targets the
  adversarial masks sit at vertices. Expected: > 50% of such masks within +-0.3 of z* (vs 0.6/|z_i|
  under uniform masks). Falsified if they cluster at 0 or 1.
- **H2**: in the beta family, the leakage part of adversarial KL scales as beta^-2 (slope -2 +- 0.5 on
  log-log over beta in {1,2,4,8,16}); stochastic loss scales faster than adversarial (adv/stoch rises).
- **H3**: at matched L0, adversarial KL ReLU << GELU < SiLU, SiLU/GELU in [1.8, 4]; at matched
  adversarial KL, GELU L0 >= 1.5x ReLU.
- **H4**: edit 1 or 2 applied to the GELU target before PD removes >= 50% of its adversarial KL at
  < 2% dCE. Applied to VPD's GELU LM, this measures how much of its KL 25-40 at 160-320 PGD steps
  (arXiv:2610.20748, Table 4) is tail leakage. Unknown share; no prior estimate.

## Cleanest toy
One hidden layer, n = 100 sparse features with continuous magnitudes s_j ~ U(0,1) at p = 0.02
(P3 requires continuous magnitudes), target y = ReLU-style per-feature outputs (the residual-MLP toy of
arXiv:2506.20790 with L4 loss, per gaps.md caveat), width h in {50, 200}, d = 100. Train with ReLU,
JumpReLU, TopK, GELU, SiLU, SiLU_beta (beta in {1,2,4,8,16}), bilinear; same 5 seeds; stop at matched
validation loss. Step 1 needs no PD: compute P4 exactly in the neuron basis (closed form) and check H1/H2
with a 100-step PGD on neuron masks. Step 2: SPD/VPD on each model, report L0 and PGD KL at
{20, 160, 320} steps. CPU minutes for step 1, CPU hours for step 2.

## Prior work and novelty
No paper found that relates activation choice to PD faithfulness (gaps.md G4). A web search
(2026-11-05) surfaced "Through the Looking Glass: Directly Reading and Writing Transformers"
(arXiv:2609.10210), which reportedly attributes poor weight-based readings in GELU models to the same
non-monotone region below -0.752. Not read yet: card it before claiming novelty for H1.
Bilinear MLPs for weight-based interpretability: arXiv:2410.08417.
