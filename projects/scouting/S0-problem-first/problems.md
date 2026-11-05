# S0: problem-first scouting (2026-11-05)

Method: start from problems in the user's interest areas, not from methods. For each problem I check
whether current methods fail, whether an idea (not scale) is the bottleneck, and how crowded the
ICLR 2027 submission wave is. Crowding probes used the OpenReview search API
(`api2.openreview.net/notes/search`, group `ICLR.cc/2027/Conference`). Hit counts there are fuzzy
full-text matches, so I judged crowding by how many of the top titles are on-topic. All P(positive)
values are my priors, not evidence. Nothing below has a `lit/novelty.md` yet.

## Bottom line
| Rank | Problem | Method idea | P(pos) | Crowding (ICLR 27) | Cheapest decisive test |
|---|---|---|---|---|---|
| 1 | No interpretability or editing tools for hybrid recurrent LMs | Memory lens + closed-form rank-1 edits on the recurrent state | 0.45 | Low | s24, 1-2 days |
| 2 | White-box auditing of fine-tunes loses to black-box (AuditBench) | Delta-guided trigger search: directions that are rare under base data but strongly drive Delta W | 0.45 | Medium | s24, 2-3 days |
| 3 | Subliminal trait transfer in distillation | Test whether transfer comes from a shared parameter basis (Adam) or a shared function: function-preserving rotations of the student | 0.25 | Medium (on the problem); idea unchecked | CPU, hours |

Killed as crowded or scale-bound: harmful fine-tuning defences, emergent misalignment, unlearning
robustness, knowledge editing, model merging, task-vector transfer, steering-vector variants, loss
spikes, FP4/FP8 training (details in "Lower-ranked").

---

## 1. Hybrid recurrent LMs have no interpretability, steering or editing toolkit
- **Problem.** Production models are moving to hybrids that mix attention with linear-recurrent layers:
  Gated DeltaNet (arXiv:2412.06464) in Qwen3-Next and Kimi Linear, Mamba-2 in Nemotron-H, Falcon-H1
  and Granite-4.0-H (model lineups from public releases; verify before citing). Logit lens, patching,
  ROME-style editing and steering all assume a transformer residual stream. A hybrid stores most of
  its context in a matrix-valued recurrent state that none of these tools reads.
- **Evidence it matters.** Practitioners who adopt hybrids for long-context cost lose their tools for
  safety and debugging. A web search found no interpretability, steering or editing work on
  Nemotron-H or Qwen3-Next. The only tooling found was `hybridlens` on PyPI, whose core functions are
  unimplemented.
- **Why current methods fail or are missing.** Prior work is Mamba-1 weight editing (ROME on Mamba,
  arXiv:2404.03646), scalar steering of SSM activations (arXiv:2602.22719), ICL mechanisms in hybrids
  (arXiv:2510.23006) and "Residual Streams Read, Recurrent States Remember" (arXiv:2609.32102, unread).
  None of them reads or writes the recurrent state as a memory.
- **Crowding.** In ICLR 2027 the hybrid and linear-attention titles are all about efficiency
  (FReS, LeapQuant, RunningTensor). "From Recurrent Computation to Attention Memory in Hybrid
  Language Models" is the only one that may overlap; read it first. Low.
- **Method idea (original as far as checked).** A delta-rule state is an associative memory written by
  rank-1 outer products, S_t = S_{t-1}(I - b k k^T) + b v k^T; Mamba-2's SSD state has the same
  outer-product form with scalar decay. Two tools follow from that.
  (a) *Memory lens*: query S with the model's own key projections of candidate tokens or entities, then
  decode the retrieved values through later layers. This shows what the model holds about an entity
  without running the model forward.
  (b) *State edits*: apply a closed-form rank-1 write (or a key-projected erase) to the cached state so
  an in-context fact is inserted, overridden or removed without re-running prefill. Edits are
  per-session, reversible and O(d^2) in cost. Example uses: scrub a prompt injection from a cached
  long context; resolve conflicts between context and parametric knowledge.
- **Cheapest decisive test (s24).** Use Falcon-H1-0.5B/1.5B or an fla-hub GatedDeltaNet-1.3B
  (check availability and licence). Prefill documents that contain N counterfactual facts, then apply
  (i) a lens readout of each fact at each layer and (ii) a state edit of one fact.
  Comparisons: re-prefill with edited text (oracle), activation steering, ROME weight edits, and a
  random-direction write with matched norm.
  Metrics: edit success, specificity on the other N-1 facts, fluency, and wall-clock time against
  re-prefill. Run 3 paired seeds over document sets.
  Kill if the lens beats a linear probe of the state at no N, or if edits do not beat steering on
  specificity.
- **P(positive).** 0.55 that the lens works (the layer is built to be an associative memory); 0.4 that
  edits beat steering at matched specificity. Combined about 0.45 for a paper-level result.
- **Against "incremental".** A reviewer will call it "ROME on the state". Answer: the object is new.
  It is an inference-time fast-weight memory written by the model itself, which transformers do not
  have. The scientific claims are about what that memory contains and whether a write propagates
  through the later attention layers of a hybrid; neither question exists for transformers. The risk
  is that the state is too entangled (superposed keys) for clean reads at 1B scale. The lens result
  measures that directly, and it is publishable either way as a characterisation.

## 2. White-box auditing of fine-tunes loses to black-box tools
- **Problem.** Given a fine-tuned open-weight model (and usually its public base), find behaviours that
  only appear under a hidden trigger or context, with no training data.
- **Evidence it matters.** AuditBench (Anthropic Fellows, Mar 2026: 56 Llama-3.3-70B models with 14
  hidden behaviours, plus a Qwen3-32B replication suite) finds that scaffolded black-box tools beat
  every white-box tool (SAEs, logit lens, steering, activation oracles). Context-dependent behaviours
  (data poisoning, secret loyalty) go unelicited. AuditBench did **not** test model diffing against
  the base model. Hugging Face hosts many derivative fine-tunes, and no one audits them.
- **Why current methods fail.**
  - Activation-difference methods only read what the probe text triggers, so a trigger absent from the
    probe text leaves no trace.
  - WeightWatch (arXiv:2508.00161) ranks directions by norm, and a small-norm, input-gated backdoor
    ranks low.
  - Weight-space LoRA backdoor detection (arXiv:2602.15195) classifies adapters as backdoored or not
    but does not recover the trigger.
  - Activation-matched fine-tuning (arXiv:2609.00351) scores prompts it is given and still needs
    candidate prompts.
  - An adaptive attack on diffing audits exists ("Read It and It's Gone", ICLR 27 submission).
- **Crowding.** Medium. ICLR 27 has 7 "model diffing" titles and several hidden-behaviour detection
  papers, but no title I saw recovers triggers from the weight delta.
- **Method idea.** A behaviour hidden behind a trigger needs Delta W to respond strongly on trigger
  inputs and stay quiet on normal text. So search for input directions h that maximise
  h^T dW^T dW h relative to h^T Sigma_base h, where Sigma_base is the base-model activation covariance
  on generic text. That is a generalised eigenproblem per early or mid layer, solved in closed form
  from weights plus one covariance pass.
  Map the top directions back to tokens (embedding or lens) and use them to seed a discrete prompt
  search that maximises KL(ft || base). The search can feed an auditing agent as a tool.
  The lab's Delta-PD (G1) can be an optional refinement here; it is not the core idea.
- **Cheapest decisive test (s24).** Qwen2.5-0.5B/1.5B. Plant 5 context-triggered behaviours with LoRA
  and full fine-tuning (lexical, semantic and format triggers), mixed with a benign SFT delta.
  Metric: trigger recall in the top-20 candidates, and KL lift on the elicited prompts.
  Comparisons: random or generic prompts, activation-difference lens, diff amplification, WeightWatch
  SVD, activation-matched fine-tuning, and GCG with random init. Run 3 paired seeds.
  Next stage: the weights of the AuditBench Qwen3-32B suite, analysed per layer on a 48 GB slice or
  on CPU (no generation needed). Generation-based validation of the elicited prompts needs H200 or a
  4-bit model.
- **P(positive).** 0.45: high for lexical triggers, low for semantic ones.
- **Against "incremental".** The closest methods detect but do not recover, or recover only from
  supplied prompts. Ours turns the base model into a whitening reference, and AuditBench did not
  include that tool family. Risks: (i) it is too close to what an auditing team at Anthropic or AISI
  may already run internally; (ii) semantic triggers occupy diffuse directions. If only lexical
  triggers work, the result is a workshop paper.

## 3. Subliminal learning: shared function or shared optimizer basis?
- **Problem.** A teacher passes hidden traits to a student through unrelated data (number sequences),
  but only when the two share an initialisation (Cloud et al. 2025). Distillation pipelines cannot
  screen for this.
- **Evidence it matters / crowding.** 20 ICLR 27 titles on it. Defences already exist: trait-direction
  corridor regularisation (arXiv:2609.01091: malicious transfer 29.6% -> 6.5%), multi-teacher
  consensus (arXiv:2607.23394), "On Mitigation of Subliminal Learning" (EMNLP 2026 Findings).
  The problem is crowded; the specific question below appears open but is **unchecked**.
- **Method idea.** Apply a function-preserving random rotation of the residual basis to the student
  before distillation (QuaRot/SliceGPT-style invariance), so the student computes the teacher's
  function but in a different parameter basis.
  SGD is rotation-equivariant, so under SGD the rotation cannot change transfer. Adam is not
  equivariant ("The Loss Does Not See the Basis, but Adam Does", ICLR 27).
  Prediction if transfer rides on a shared privileged basis: transfer drops under Adam with the
  rotation, and stays unchanged under SGD. If transfer is unchanged in both cases, the mechanism is
  functional.
  Either outcome is a mechanism result. The positive outcome also gives a one-line defence:
  "rotate, then distill".
- **Cheapest decisive test (CPU, hours).** Reproduce the MNIST auxiliary-logit experiment from Cloud et
  al. Run a 2x2 design (rotation or not, Adam or SGD) over 5 paired seeds, then Qwen2.5-0.5B on the
  animal-preference setting on s24.
- **P(positive).** 0.25 (a defence that works). Ceiling medium-high because the claim is crisp.
- **Against "incremental".** A rotation is a textbook invariance, not a new mechanism; the
  contribution is the controlled experiment that separates basis from function. Kill if
  "Subliminal Learning Through Mechanism Reuse" or "Regularization Amplifies Subliminal Learning"
  (ICLR 27) already run rotations.

---

## Lower-ranked problems (checked, not recommended)
4. **Mechanistic anomaly detection** (flag inputs processed by an unusual mechanism). Matters for
   catching backdoors and deception at run time. Few explicit ICLR 27 titles, but functional
   attribution (arXiv:2604.18970, DER 0.93 vs 0.83 on BackdoorBench) and activation-matched
   fine-tuning cover the base-delta residual idea. Idea: score each input by the share of its output
   logit carried by Delta W, against a trusted distribution. P 0.35; overlaps #2. Merge into #2 as its
   runtime mode.
5. **Hidden capability elicitation without demonstrations** (sandbagging, password-locked models).
   Prior work: MELBO (arXiv:2606.29604), noise injection (Tice et al. 2024), the Elicitation Game
   (arXiv:2502.02180), and ICLR 27 attention-head interventions. No base model is available in the
   realistic setting, so weight-delta tricks do not apply. P 0.3, no clear new idea.
6. **Collateral damage of edits** (steering, abliteration, adapter merges break other capabilities).
   Idea: a precomputed per-layer low-rank "capability Fisher atlas" that predicts per-capability
   damage from Delta W before deployment. P 0.6, but this is Fisher/EWC applied to a new use; an ICLR
   27 evaluation suite ("Does Steering Break Your Model?") already exists. Safe fallback, low
   ceiling.
7. **Interventions push states off-distribution** (patching and steering artefacts). ICLR 27 already
   has "What Can Activation-Patching Evidence Establish?", "When Activation Patching Admits Many
   Faithful Circuits" and "Sparse-Dictionary Causal Claims Are Not Robust...". Theory-crowded.
8. **Loss spikes / training instability.** "Understanding and Mitigating AdamW Loss Spikes", "Spiking,
   Fast and Slow" and MSign (stable-rank optimizer) are all ICLR 27 submissions. Showing a fix
   convincingly needs scale. Not for us.
9. **Fine-tune transfer across base versions / task vectors.** Crowded: Cayley re-basing, bilinear
   factor alignment, ReFract and curvature compensation are all ICLR 27 submissions.
10. **Hyperparameter transfer for fine-tuning / LoRA.** ICLR 27 has 17 titles, mostly on pretraining;
    the PEFT angle is possibly open but unchecked. P 0.4, medium ceiling.

Crowded areas, with representative ICLR 27 titles:
- harmful fine-tuning defences: dozens of titles, e.g. "Gradient Immunity", AIRP, "Open-Weight LLM
  Fine-Tuning Defenses are Susceptible to Simple Attacks"
- emergent misalignment: AlignRadar, "Narrow Neurons Mediate Broad EM", a longitudinal EM-defence
  benchmark
- unlearning versus relearning attacks: CINDER, a gradient-entropy certificate
- multi-hop knowledge editing: StructKE and others
- model merging
- steering-vector variants: "Activation Steering Is a Data Problem", "First-Order Steering"

## Honest caveats
- Every novelty statement rests on title-level OpenReview probes plus `researchctl lit-search`. Full
  novelty audits (`lit/novelty.md`) are required before any GPU spend.
- #1 depends on a small open hybrid with an accessible recurrent state in a usable library (fla,
  mamba-ssm). Verify this first; it takes about 1 hour.
- #2 overlaps the lab's pdecomp G1 in data (weight deltas) but not in the question asked. If the user
  wants a clean break from pdecomp, #1 is the cleaner choice.

## Sources and queries
- ICLR 27 OpenReview probes: steering vectors; harmful fine-tuning defense; emergent misalignment;
  unlearning relearning; knowledge editing multi-hop; fine-tuning transfer; model merging; loss spike;
  backdoor detection weights; model diffing; task vector transfer; training instability; loss of
  plasticity; model growth; hyperparameter transfer; evaluation awareness; data attribution; weight
  decay; over-refusal; steering reliability; mechanistic anomaly detection (0 hits); subliminal
  learning; trigger inversion; sandbagging; password-locked; model fingerprinting; activation patching;
  weight space learning; diff amplification (0); anomaly detection backdoor; auditing fine-tuned;
  state-space interpretability; diffusion LM interpretability; Adam basis rotation; Mamba/hybrid
  interpretability; weight difference audit; AuditBench; hybrid steering; recurrent state editing.
- lit-search: the six novelty probes for #1-#5, plus three probes on state editing and state tuning.
- Abstracts read: arXiv:2602.15195, 2609.01091, 2604.18970, 2609.00351; AuditBench blog
  (alignment.anthropic.com/2026/auditbench).
- Web: 2026 interpretability agendas (found no 2026 lab-level agenda; "Interpretability Can Be
  Actionable", ICML 2026, arXiv:2605.11161); hybrid interpretability (arXiv:2602.22719, 2404.03646,
  2510.23006, hybridlens).
