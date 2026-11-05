# Design: does ablation-based CI on clean data find backdoors that activation criteria miss?

Status: proposal, exploratory. Nothing here is evidence. Read with audit.md.

## Is "dormant on clean data in a decomposed basis" novel?
Only partly. Dormancy-then-prune is fine-pruning (2018); dormancy in a dW basis with clean calibration
is WeightWatch (SVD basis, activation range, detection). The new part is the *criterion*: PD's CI is
"can this component be ablated on this input without changing the output", learned per input. It
differs from activation magnitude when a component's input direction is active on clean data but its
write has no downstream effect there (gated off later by a nonlinearity or attention). So the one
testable claim is:

> H: for backdoors whose trigger features are common on clean data (conjunctive, common-token, or
> semantic triggers) and share neurons with benign features, clean-only dW-PD removes the backdoor
> with less collateral than fine-pruning, ANP, WeightWatch-style SVD removal, whitened-SVD
> truncation, and clean-data distillation. For rare-token triggers all methods tie.

Prediction of the null: whitened-SVD truncation (keep the part of dW with clean activation energy)
ties PD everywhere, because in one linear layer "never causally important on clean data" mostly means
"never activated on clean data". PD can only win through cross-layer nonlinear gating.

## Does it plausibly beat fine-pruning / ANP under superposition?
- Fine-pruning: yes, plausibly, in the pruning-aware regime. When the backdoor update lives in neurons
  that clean inputs use, no neuron is dormant, but the backdoor's rank-one piece of dW can still be
  silent on clean data. This is a mechanical advantage of any dW basis, so SVD-based removal gets it
  too; it is not specific to PD.
- ANP: unclear. ANP's adversarial masks on clean data exploit "backdoored models collapse to the target
  under perturbation"; VPD's PGD masks are the same tool in the component basis. Run ANP-on-components
  as our own strongest arm rather than claim to beat it.
- Against WeightWatch / whitened SVD: only for triggers active on clean data (the MTBA cell).
- Failure case for every clean-only method: a backdoor that reuses a clean-used direction (adaptive
  attacker trains the trigger to share a rank-one piece with a benign behaviour). Measure the cost.

## Stage 0: CPU toy, kill test (1-2 days, extends src/pd_toys)
Model: 2-layer MLP (and 1-layer attention model) on sparse features, 3 paired seeds, as in delta_pd_v2.
Fine-tune = benign shift + one backdoor, five arms:
1. rare trigger (a feature with p ~ 0 on clean data) -> target output (control; all should tie);
2. conjunctive trigger: features A and B are each common (p = 0.1) but never co-occur on clean data;
   A AND B -> target (the regime where PD should win);
3. pruning-aware: backdoor written into neurons that clean data uses;
4. superposed: features > neurons, backdoor in a direction sharing neurons with 2+ benign features;
5. adaptive: backdoor shares a rank-one piece with a benign change (expected to break everything).
Methods: clean-only Delta-PD (gates on clean data; ablate components with max clean CI < tau),
Delta-PD + PGD masks (ANP-style), fine-pruning (+/- fine-tune), ANP, CLP, plain SVD of dW with
WeightWatch range removal, whitened SVD truncation (SVD of dW Sigma_clean^{1/2}, keep top-k by energy),
SVD+ICA (G1 lesson), clean-data distillation of f_ft into base + rank-r LoRA, revert-to-base (floor).
Metrics: ASR after removal; benign-shift retention = 1 - |f_rem - f_ft|/|f_ft - f_base| on clean
probes; false-positive removal on a model with no backdoor (drift-only and benign-only fine-tunes).
Kill criterion (fixed now): if in arms 2-4 the best linear baseline (whitened SVD, WeightWatch removal
or SVD+ICA) or clean distillation is within 0.1 of Delta-PD on (ASR reduction - collateral) in 2 of
3 seeds, drop the backdoor direction. Precondition: SPD reproduces at full budget (PROJECT.md).

## Stage 1: small LMs with ground truth (only if Stage 0 survives; ~1-2 slice-weeks, s24/s48)
- Organisms: Qwen2.5-0.5B/1.5B, full FT and LoRA, our own triggers: rare token, the MTBA-style
  multi-trigger, a conjunctive "year AND topic" trigger, a semantic trigger, a pruning-aware variant,
  and one benign-only control fine-tune. Plus TopKLoRA's released gemma-2-2b organisms if available.
- Decompose dW only in layers where revert-to-base of that layer cuts ASR (cheap per-layer scan).
- Clean set: instruction data incl. harmful-but-refused prompts, so refusal pieces count as used.
- Baselines: as Stage 0 plus CROW and BackdoorLLM pruning (DefenseBox), 2610.00685 null-space projection.
- Metrics: ASR (BackdoorLLM judge), clean task skill (the fine-tune's own task, not just PPL), MMLU,
  false-positive damage on the benign-only fine-tune. 3 seeds, same seed list in every arm.
- First throughput run: measure PD GPU-hours per layer on s48 before any sweep.

## Stage 2: benchmark (only if Stage 1 shows a win on conjunctive/semantic triggers)
BackdoorLLM LLaMA-2-7B-Chat released LoRA adapters, refusal and jailbreak tasks. Targets: jailbreak
avg ASR below CleanGen's 42.2 without a reference model, refusal avg ASR <= CROW's 4.7 with PPL within
0.1 of the undefended model; MTBA where WeightWatch flags 27%. 7B frozen in bf16 (~13.5 GB) plus PD
on a few layers' LoRA dW fits one 48 GB slice; H200 not needed.

## Expected value
Most likely outcome (~0.6, our estimate): whitened SVD or clean distillation ties PD on removal, and
the honest result is a negative section of the G1 paper ("for backdoor removal, the clean-only split
of dW is linear-easy"). Positive outcome (~0.2): PD wins on conjunctive/superposed triggers in toys
and small LMs; that is a workshop paper or a strong G1 section, and a NeurIPS paper only with the
jailbreak cell on BackdoorLLM. Cost of finding out: Stage 0 is CPU and ~2 days of agent time.
