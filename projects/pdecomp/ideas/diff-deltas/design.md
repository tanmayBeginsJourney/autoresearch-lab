# Design: base-anchored parameter decomposition of weight deltas (dW-PD)

Status: idea, not preregistered. Novelty: see `audit.md` (method unpublished, named in VPD Sec. 7.5).

## Method
For each decomposed matrix, write the fine-tuned weights as the base plus rank-one subcomponents
plus a residual:

  W_ft = W_base + sum_c m_c(x) u_c v_c^T + m_R (dW - sum_c u_c v_c^T)

W_base is fixed and never masked. Masks m_c come from a VPD-style causal-importance (CI) function
reading the fine-tuned model's inputs to each layer, with stochastic plus adversarial (PGD) mask
sampling, importance minimality and frequency minimality as in VPD. The residual R plays the role of
VPD's Delta-component and tPD's catch-all: ablatable on every input, L2-penalised. This is SPD/VPD with a
non-zero ablation reference (the base model) and a changed target. It is not a new mechanism and the
writeup should say so; the contribution is whether it recovers what fine-tuning changed better than
the alternatives.

Ablating every component gives the base model exactly; keeping all gives the fine-tuned model exactly.
A component is "what fine-tuning added to this matrix that is used on this input".

## Why it should give positive results
1. **The ablation reference is a working network.** VPD ablates toward zero weights, which is far
   off-distribution; its faithfulness drops from 82% compute recovered unmasked to 8-12% with rounded
   masks (VPD, see `lit/cards/arxiv-2610.20748.yaml`). Here every mask interpolates between two
   functional models. Fine-tunes from one pretrained init are known to stay in one loss basin
   (Neyshabur et al. 2020, "What is being transferred in transfer learning?"), so masked models should
   stay low-loss. Prediction: adversarial-mask KL to target is far lower than whole-model VPD at the
   same PGD budget.
2. **Few components should suffice.** Rank-1 LoRA recovers 73-90% of reasoning gains on a 32B model
   (arXiv:2511.06739); TopKLoRA circuits are small (arXiv:2610.32866). The number of components needed
   scales with what fine-tuning changed, not with what the base model computes.
3. **Cheaper.** Only matrices with large functional dW need decomposing (change concentrates in late
   layers, arXiv:2604.17177), with C in the tens to hundreds per matrix rather than thousands. The CI
   function is the main cost and shrinks with C. Cost must be measured, not assumed: no GPU-hour number
   exists for VPD (PROJECT.md).
4. **A built-in win over SVD of dW.** SVD returns orthogonal directions ordered by norm. Two changes in
   the same matrix with similar norms come out as rotated mixtures, and RL deltas are nearly full-rank
   and small (arXiv:2505.11711), so the top singular vectors need not be the functional changes. CI-based
   PD separates changes that are used on different inputs (the same argument SPD makes against SVD for
   whole models) and links one change across matrices.
5. **Ground truth exists.** Planted toy changes, and LLM organisms with known objectives
   (Delta-Crosscoder's 10 organisms; TopKLoRA's 15 backdoored gemma-2-2b adapters on HF; Diff Mining's
   false-fact and injected-bias organisms) give a recovery score instead of anecdotes.

Main risks: (a) functionally important change is spread across many tiny full-rank pieces and lands in
R (measure R's share of the KL; kill if it dominates); (b) superposition in dW makes components
polysemantic, as in whole-model PD; (c) Goodfire publishes first; (d) masked interpolation may be
faithful yet uninformative (components equal to "the whole fine-tune at layer L").

## Experiment 1 (CPU, toy, the single best experiment)
**Planted fine-tuning changes with ground truth.** Base: a small residual-MLP or 2-layer attention
model trained on a synthetic sparse-feature task (e.g. the SPD compressed-computation toy, or a
key-value lookup transformer). Fine-tune the *full parameters* by gradient descent (so dW is realistic,
not planted rank-1) on data that induces five known changes at once:
1. add: new input feature -> new output;
2. edit: feature i's output target changed;
3. delete: feature j suppressed (unlearning);
4. backdoor: conjunction of features A and B triggers output z;
5. two changes in the *same* matrix with matched norm (the case designed to break SVD);
plus a nuisance arm: continued training on base data with noise (diffuse drift, no behaviour change).

Methods (same seeds, 5 seeds per arm): dW-PD (ours); top-k SVD of dW per matrix; activation crosscoder
and Delta-Crosscoder-style diff dictionary at each layer; rank-r LoRA and TopKLoRA refit on the
fine-tuning data; whole-model SPD/VPD on W_ft with components then matched to base (cost baseline).

Metrics:
- **Recovery**: for each planted change k, the smallest component set S_k whose ablation (revert to
  base) removes behaviour k. Report |S_k|, and a selectivity matrix: change in behaviour j != k when
  ablating S_k. Score = mean over k of (effect on k) - max over j != k of (effect on j).
- **Faithfulness**: KL to W_ft under stochastic and adversarial masks; KL on held-out base data.
- **Nuisance rejection**: fraction of components (or of dW norm) assigned to R or to unused
  components in the drift arm.
- **Cost**: wall-clock and FLOPs to reach the recovery threshold, dW-PD vs whole-model PD.
- Controls: random-direction and matched-norm rank-one ablations for every reported S_k.

Predictions (to preregister): dW-PD recovers >= 4 of 5 changes with selectivity > 0.8 in >= 4 of 5
seeds; SVD fails change 5; crosscoders cannot localise to weights by construction, and their latents
are judged on the same behavioural ablation via steering. Kill if dW-PD does not beat SVD of dW on the
recovery score, or costs more than whole-model PD. Budget: CPU or one s24 slice, under 1 day of
wall-clock.

## Experiment 2 (GPU, s24/s48, small LMs)
Stage A, known objectives. Base Qwen2.5-0.5B or Qwen3-0.6B. Train our own organisms by full fine-tune
and by LoRA: EM (bad medical advice), synthetic false facts, a trigger backdoor, and one model with all
three at once (tests disentangling, compare to Diff Mining's NMF clusters). Decompose dW of the layers
holding most functional change (found by per-layer revert-to-base ablation, a cheap first pass).
Score per-objective recovery and selectivity as in Experiment 1; also: does an LLM agent given top
components' activating examples name the objective (Delta-Crosscoder / Diff Mining protocol)?
Edit test: removing S_k reverts objective k while preserving the other two and base capability
(MMLU-style held-out, base-data KL), compared against TopKLoRA's surgical ablation.

Stage B, real post-training. Decompose RL and SFT deltas where SVD is expected to struggle:
(i) our own GRPO run on GSM8K from Qwen2.5-0.5B-Instruct (sparse, near-full-rank delta per
arXiv:2505.11711); (ii) staged public checkpoints, e.g. OLMo-2 1B base/SFT/DPO/RLVR (verify the exact
releases), decomposing each stage's delta separately; (iii) Qwen3-0.6B-Base vs Qwen3-0.6B. Question:
how many components explain most of the KL between stages, and do RL components look like upweighting
existing base directions (project onto a base-model decomposition if one exists, VPD Sec. 7.5).

Baselines throughout: SVD of dW (per matrix, rank-matched), crosscoder and Delta-Crosscoder, transcoder
adapters (MLP only, which exposes the attention coverage gap), rank-1 LoRA and TopKLoRA refits,
Diff Mining (black-box), random and matched-norm rank-one controls.

Compute: smoke on s24 first (<= 30 min). VPD cost at 67M is still unmeasured; the reproduction milestone
(PROJECT.md, 2026-12-23) must produce that number before Stage B is sized. H200 only if Stage A works
at 0.5B and the scaling question is the actual question.

## Decision this informs
If Experiment 1 passes, dW-PD becomes candidate gap 1 in `gaps.md` with a fast path (toy result by
mid-December, Stage A by the January prereg). If it fails on recovery or cost, kill it and record the
negative result in the journal.
