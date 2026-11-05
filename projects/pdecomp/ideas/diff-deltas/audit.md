# Novelty audit: parameter decomposition of fine-tuning / RL weight deltas

Verdict: **partially done (method unpublished, idea already named).** As of 2026-11-05, no paper or
post we could find runs APD/SPD/VPD, or any rank-one causal-importance decomposition, post hoc on
an existing weight delta dW = W_ft - W_base. But VPD itself lists exactly this as future work (VPD
arXiv:2610.20748, Sec. 7.5: "one could apply VPD to only a 'weight diff' obtained by finetuning"),
and the neighbouring cells are all filled: training-time sparse/rank-1 deltas (TopKLoRA, rank-1 LoRA,
circuit-tuning), post-hoc activation-space diffing (crosscoders, Delta-Crosscoder, transcoder
adapters), and post-hoc weight-space diffing by SVD or learned introspection (blog, DIT). The method
is an obvious next step with real race risk from Goodfire. A paper therefore needs its novelty in the
results: ground-truth recovery that the baselines miss, and coverage (attention, RL deltas) they lack.

## Queries run (2026-11-05)
`researchctl lit-search --project pdecomp`: "parameter decomposition fine-tuning weight delta";
"stochastic parameter decomposition model diffing"; "rank-1 LoRA interpretability fine-tuning";
"crosscoder model diffing base chat"; "delta crosscoder"; "transcoder adapters fine-tuning";
"sparse model diffing dynamic circuits"; "weight diff interpretability fine-tuning rank-one
components"; "RL fine-tunes small subnetworks".
`lit-citers`: APD 2501.14926 (4 citers), SPD 2506.20790 (1 citer: Sparse Model Diffing), VPD
2610.20748 (0 citers in OpenAlex; too new to be indexed, so this check is weak).
Web: "parameter decomposition" + fine-tuning/model diffing/SPD; "Sparse Model Diffing via Dynamic
Circuits"; TopKLoRA; "decompose weight delta ... rank-one causal importance"; "interpreting weight
differences"; Goodfire Silico editing; "model diffing weight space decomposition"; EM rank-1 LoRA;
LessWrong SPD + finetuning diff/LoRA; Delta-Crosscoder; RL small subnetworks.
Full text grepped: VPD (fine-tun/diff/delta), tPD card, TopKLoRA, rank-1 LoRA, transcoder adapters,
Diff Mining, Sparse Model Diffing, MTH, depth profile, DFC tool-use, SPD-on-transformers.

## Closest five works and exact differences

| Work | What it does | Difference from dW-PD |
|---|---|---|
| **TopKLoRA** (arXiv:2610.32866) | Trains a LoRA with ReLU + top-k gated bottleneck; each latent = (condition row of A, steering column of B); circuit discovery on sleeper-agent organisms in gemma-2-2b, Qwen2.5-1.5B/7B; circuits up to 10x smaller than dense LoRA. | Training-time: it replaces the fine-tune with a new, interpretable-by-design one. It cannot explain a released full-FT or RL model; the explained model is not the model of interest. Units are input-gated rank-1 updates, which is close to what dW-PD would find post hoc. Strongest baseline and a natural cross-check. |
| **Transcoder adapters** (arXiv:2602.20904) | Learns a sparse transcoder added to frozen base MLPs to reproduce the fine-tuned model's MLP outputs (Qwen2.5-Math-7B vs R1-Distill-Qwen-7B); recovers 50-90% of accuracy gains. | Activation-space replacement of the MLP *computation difference*, not a decomposition of dW; not constrained to sum to dW. Explicitly MLP-only: "changes to non-MLP parameters, most notably attention ... remain unexplained" (Sec. 7). dW-PD covers attention matrices natively. |
| **Rank-1 LoRA reasoning** (arXiv:2511.06739) and **EM rank-1 adapters** (arXiv:2506.11618) | Constrain the fine-tune to rank-1 per matrix and read the adapter scalars as neurons; rank-1 recovers 73-90% of full-FT gains on Qwen-2.5-32B-Instruct. Steering with LoRA directions was "inconclusive" (2511.06739 Sec. 4). | Training-time constraint, again a different model from the full fine-tune. One fixed direction per matrix, no input-conditional importance, no sum-to-dW faithfulness. Evidence that real deltas have few important rank-1 pieces (supports dW-PD). |
| **Crosscoder diffing** (Lindsey et al. 2024; Minder et al., NeurIPS 2025 "Overcoming sparsity artifacts"; **Delta-Crosscoder** arXiv:2603.04426, EMNLP 2026; DFC tool-use arXiv:2606.26474) | Shared/exclusive activation dictionaries for base and fine-tuned models; Delta-Crosscoder recovers causally validated latents on 10/10 narrow organisms vs 4-6/10 for prior crosscoders. | Activation space, one site per dictionary, reconstruction-trained; latents are not tied to which weights changed. No weight-level attribution or weight-level edit. These are the main empirical baselines, and the 10-organism suite is a ready ground-truth benchmark. |
| **VPD Sec. 7.5** (arXiv:2610.20748) + **bigsnarfdude blog** (2026-05-06, informal) + **DIT** (arXiv:2510.05092, ICLR 2026) | VPD proposes PD of a fine-tuning diff and projecting gradients onto subcomponents, untested. The blog runs SVD on talkie-13b SFT-base and IT-SFT diffs, but runs VPD on the full SFT checkpoint, "not the diff". DIT trains an introspection LoRA so a model verbalises its own weight diff. | Nobody has run it. SVD gives per-matrix orthogonal directions with no input-conditional importance and no cross-layer linking; DIT gives text, not components or edits. |

## Other works checked (not closer)
- Sparse Model Diffing via Dynamic Circuits (Li et al., Findings EMNLP 2026): prune-then-tune
  "circuit-tuning" that makes updates sparse by construction; cites APD/SPD only as related work.
  Training-time, node-level (heads/MLPs), not rank-1.
- Diff Mining (arXiv:2608.26462): logit-difference token sets + NMF; output-only. Useful
  black-box baseline and organisms (false facts, injected biases).
- RL fine-tunes small subnetworks (Mukherjee et al., arXiv:2505.11711, NeurIPS 2025; the id
  2507.17107 given in the task returned 404): RL updates touch 5-30% of parameters yet are nearly
  full-rank per matrix. Multiple Ticket Hypothesis (arXiv:2602.01599): random 1% masks match full
  RLVR, masks overlap <= 0.005 Jaccard. Both say RL deltas are small in effect but not low-rank in
  the SVD sense: a direct test case where SVD should fail and causal-importance PD might not.
- Depth profile of fine-tuning (arXiv:2604.17177): representational change concentrates in
  output-proximal layers across 240 runs; motivates decomposing only the layers where dW matters.
- tPD (arXiv:2607.13047): its Delta is a full-rank residual W - sum U_i V_i^T inside one model, not
  a fine-tuning delta. Its targeted-loss design is reusable here.
- Goodfire Silico / "Exploration: fine-tuning with parameter decomposition" (LessWrong): tunes the
  scalar on one VPD subcomponent of the 67M model to suppress German. This is PD *for* fine-tuning,
  the reverse direction; it does not decompose a delta.
- Persona features (arXiv:2506.19823): SAE-based diffing of EM models, activation space.
- Task-vector merging work (e.g. SVD-based shared bases for deltas) targets merging, not mechanism.

## Residual risk in this audit
- VPD has no indexed citers yet; a preprint from Nov 2026 could be missed. Re-run
  `lit-citers arxiv:2610.20748` and an arXiv keyword search ("weight diff" + "parameter
  decomposition") before the gap decision, and check Goodfire's blog and GitHub.
- Cards are not yet written for the five closest works; do that before `lit/novelty.md`.
