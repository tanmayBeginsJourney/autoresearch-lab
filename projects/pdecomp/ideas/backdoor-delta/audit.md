# Novelty and feasibility audit: clean-only Delta-PD for backdoor removal

Idea: decompose dW = W_ft - W_base into rank-one components with per-input causal-importance (CI)
gates trained on clean data only; components that carry weight but have CI ~ 0 on all clean data are
backdoor candidates; ablate them without knowing the trigger.

**Verdict: partially done; the method is an untested combination of known parts, and in most regimes
cheaper baselines should match it.** No paper or post we found applies APD/SPD/VPD (or any learned
rank-one CI decomposition) to backdoors (searches below, 2026-11-05). But each half is published:
"remove what is dormant on clean data" is fine-pruning (Liu et al. 2018), and "dormant on clean data
in a decomposed basis of dW" is WeightWatch (arXiv:2508.00161: SVD of dW, clean-calibrated activation
range, detection only). What remains is narrow: whether an *ablation-based, nonlinear* CI criterion
finds backdoor pieces that *activation-energy* criteria miss. That only happens when trigger features
are common on clean data but cause nothing there (conjunctive or common-token triggers, superposed in
shared neurons). WeightWatch's one weak cell, BackdoorLLM LoRA MTBA (27% of triggered prompts flagged,
because '2024'/'OpenAI' occur in calibration data), is direct evidence that this regime exists.
As a standalone NeurIPS paper: P(positive, survives review) ~0.2 by our estimate (not evidence).
As a test case inside G1 (Delta-PD): worth a 1-2 day CPU kill test.

## Queries run
`researchctl lit-search --project pdecomp` (21 queries): backdoor removal low-rank decomposition
dormant components; fine-pruning dormant neurons; superposition neurons pruning fails; weight
difference backdoored fine-tune detection singular; LoRA adapter detection weight spectra; parameter
decomposition backdoor; SAE backdoor localization LLM; crosscoder sleeper agent model diffing;
data-free clean-only pruning LLM; rank-one subspace removal; BackdoorLLM; BackdoorBench; BEEAR; CROW;
simulate-and-eliminate; neural polarizer; TrojAI weight meta-classifier; backdoor task vectors;
neuron entanglement; orthogonal subspace LoRA unknown trigger; low-rank SVD purification clean data.
Web: PD + backdoor removal + causal importance; NEEDLE; feature/dictionary pruning for backdoors;
Goodfire/Apollo SPD/VPD + backdoor/trojan/sleeper; CS-ADS; LLM Forensics SAE. arXiv abstracts for
all ids below; full text grepped for BackdoorLLM, CROW, 2610.00685; WeightWatch HTML read.

## Closest works and exact differences
| Work | What it does | Difference from clean-only dW-PD |
|---|---|---|
| **WeightWatch** 2508.00161 (ICLR 2026) | Top-20 SVD directions of dW (O_proj, W_down); clean cosine range; flags 100% of most BackdoorLLM/FFT/PPO triggers at FPR < 1%; MTBA LoRA 27% | Same idea in a fixed orthogonal basis with an activation criterion; detection, no removal. PD differs in basis (learned, non-orthogonal) and criterion (ablation effect, not activation magnitude). Must be the first baseline, in removal form (project out flagged / clean-silent directions). |
| **Fine-pruning** 1805.12185; **ANP** 2110.14430; **RNP** 2305.14876; **CLP** 2208.03111 (data-free); **Wanda pruning** in BackdoorLLM | Prune neurons/channels dormant on clean data (FP), sensitive to adversarial masks (ANP), exposed by unlearn-recover (RNP), or with high channel Lipschitz constant (CLP) | All in the neuron/channel basis of the full model. Fine-pruning's own pruning-aware attack (clean and backdoor on the same neurons) is the superposition case where a rank-one basis of dW could help. ANP's adversarial masks on clean data ~ VPD's PGD masks: "ANP in the component basis" is the natural variant and must be an arm. |
| **Null-space projection** 2610.00685 | Synthetic backdoored variants -> shared backdoor direction -> project LoRA dW onto its null space; Mistral-7B sentiment avg ASR 100 -> 10.4 (CROW 34.4, Fine-Pruning 41.1) | Also edits dW without retraining, same threat model. Finds the direction from synthetic attacks, not from clean dormancy. Current removal number to beat on LoRA backdoors. Companion 2603.13461 does the same for full models. |
| **TopKLoRA** 2610.32866 | Interpretable-by-design sparse LoRA; finds and ablates a necessary-and-sufficient sleeper-agent circuit (gemma-2-2b, Qwen2.5-1.5B/7B) | Training-time: replaces the fine-tune; circuit search uses triggered data. Post-hoc dW-PD is the analogue for adapters someone else trained. Their organisms are a ready ground-truth set. |
| **SAE / crosscoder backdoor work**: 2606.07963 (shared backdoor SAE latents, 4-32B), 2605.07324 (Diff-SAE > crosscoder on '2024' SQL backdoor), 2608.30403 (feature roles), Delta-Crosscoder 2603.04426 | Activation-space features that shift between clean and backdoored models, mostly found *with triggered inputs*; steering/ablation shows causality | Need triggered prompts or contrast pairs; not tied to weights. A clean-only weight method is a different threat model; still, 2606.07963's SAE-feature classifier outperforms weight-diffing baselines on detection. |

Other defenses checked (not closer): I-BAU 2110.03735, NAD 2101.05930, Neural Cleanse (S&P 2019,
trigger inversion), BEEAR 2406.17092 (embedding-space adversarial removal; RLHF backdoor ASR >95% ->
<1%), CROW 2411.12768, CleanGen (reference-model decoding), Simulate-and-Eliminate (AAAI 2025), DUP 2508.01647, NEEDLE
2610.00348 (needs trigger), LoRA-as-Oracle 2601.11207, QES 2610.00663 (training-time quarantine),
unlearning transfer 2606.03785, dummy-backdoor / backdoor-collapse (EMNLP/ACL 2026, abstracts only),
CS-ADS (ACL 2026). Detection from weights is saturated on standard attacks: linear weight
classifiers win TrojAI-style competitions (2411.03445); LoRA spectral detector 100% (2602.15195);
PEFTGuard (S&P 2025). Anthropic sleeper agents 2401.05566 and the 2024 defection-probe note use
probes on activations with contrast prompts; not a removal method (blog, not re-read here).

## What a reviewer will say
1. "Fine-pruning in a different basis." True for the criterion; the paper must show the basis and the
   ablation criterion matter, on cases built to break neuron and SVD bases.
2. "Training on clean data only leaves the clean-irrelevant part of dW unconstrained." CI~0 on clean
   data selects *everything* in dW that clean data does not use: backdoor, drift, and benign skills
   missing from the clean set. For removal, PD's separation inside that residual is irrelevant; what
   matters is the split dW = (clean-used) + (rest). Cheap linear splits exist: activation-whitened SVD
   truncation of dW on clean calibration data, and clean-data distillation of f_ft into a new LoRA.
   PD wins only if its nonlinear CI draws the split better than these.
3. "Any clean-only method fails when the backdoor reuses a clean-used direction" (adaptive attack).
   Unavoidable; must be measured, not hidden.

## SOTA to beat, and where (our compute: 8 slices, 24-48 GB)
- **BackdoorLLM** (LLaMA-2-7B-Chat LoRA; released adapters give dW directly). Targeted refusal avg
  ASR: none 84.1, Wanda pruning 27.2 (PPL 11.9 vs 7.7), CROW 4.7, CleanGen 0.09 (uses a reference
  model). **Jailbreak backdoor avg ASR: none 96.1, best CleanGen 42.2, CROW 85.6, pruning 85.7** (Table
  7). Jailbreak removal is the open cell. Sentiment steering (Mistral-7B, from 2610.00685): ours-to-beat
  10.4 avg ASR (Vaccine 15.5, CROW 34.4, Fine-Pruning 41.1). WeightWatch detection on MTBA: 27%.
  Feasible: 7B bf16 = ~13.5 GB frozen; PD only on LoRA dW of a few layers fits one 48 GB slice. VPD
  GPU-hours are still unmeasured (PROJECT.md), so cost is a risk.
- **Small-model ground truth**: TopKLoRA organisms (gemma-2-2b, Qwen2.5-1.5B), own Qwen2.5-0.5B/1.5B
  organisms with conjunctive and pruning-aware triggers. This is where the claim is decided.
- **BackdoorBench** (CIFAR-10/GTSRB, PreAct-ResNet18): defenses are near-saturated on standard
  attacks; not worth our time unless an adaptive-attack table needs it. **TrojAI**: detection-focused
  and won by weight meta-classifiers; skip.

## Residual risk in this audit
Abstract-level for most 2026 defenses; CleanGen, SANDE, BEEAR numbers are from abstracts or
BackdoorLLM. Fine-pruning's pruning-aware attack is from a secondary summary (verify). VPD is too new
for citer checks; re-search "parameter decomposition backdoor" monthly.
