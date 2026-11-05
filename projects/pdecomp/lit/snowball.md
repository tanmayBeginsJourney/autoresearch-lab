# Snowball: parameter decomposition (PD) line, 2026-11-05

Method: `lit-citers`/`lit-refs` on SPD, APD, VPD (OpenAlex), 17 `lit-search` queries (logged in
queries.log), web search (Goodfire, LessWrong, Alignment Forum, ACL Anthology), full-text reads.
Caveat: OpenAlex coverage of this line is nearly empty. `lit-refs` returned 0 references for SPD and
VPD; `lit-citers` returned 0 for VPD and 1 for SPD. Forward snowballing therefore rests on web search and
keyword search, and will miss citers. Re-run `lit-citers` in 4 weeks, and check Google Scholar by hand.

Depth key: card = full card in cards/; read = text read, no card; abs = abstract/snippet only; web = web page only.

## Core (PD or direct weight-space decomposition of trained models)
- arxiv:2501.14926, 2025, APD (Braun et al.): origin of the line. card (existing).
- arxiv:2506.20790, 2025, SPD (Bushnaq et al.): rank-one subcomponents, stochastic masks. card (existing).
- arxiv:2610.20748, 2026, VPD "Interpreting Language Model Parameters" (Bushnaq et al.): SOTA; 67M LM. card (existing).
- arxiv:2607.13047, 2026, tPD (Vigouroux, Sharkey): targeted PD, analytic cost estimate. card (existing).
- arxiv:2609.37731, 2026, ASPD (Cao et al.): local per-matrix read/write PD; one-matrix "8B". card (existing).
- arxiv:2511.08854, 2025, Decomposition of Small Transformer Models (Christensen, Riggs Smith; NeurIPS 2025 MI workshop): first SPD on transformers (attention-based CI function, partial losses). card (new).
- arxiv:2504.00194, 2025, L3D "Identifying Sparsely Active Circuits Through Local Loss Landscape Decomposition" (Chrisman, Bushnaq, Sharkey): gradient-space sibling of APD/SPD, Tucker (higher-rank) subnetworks. text fetched (218k chars), NOT yet read; read before the novelty audit.
- lesswrong:ieoWstubDQWLrMnhH, 2026-06-25, "Exploration: fine-tuning with parameter decomposition" (Bushnaq, Goodfire): trains scalar masks on VPD subcomponents; closest published item to "PD for fine-tuning". card (new, blog).
- web:bigsnarfdude-svd-vpd, 2026-05-06, "SVD and VPD: Two Methods for Reading What Training Wrote Into a Model's Weights": SVD of SFT-base weight diffs plus a VPD run on a 13B checkpoint; unreviewed blog. card (new, blog).

## Core for gap 1 (weight deltas / model diffing)
- doi:10.18653/2026.findings-emnlp.679, 2026, Sparse Model Diffing via Dynamic Circuits (Li et al., Findings of EMNLP 2026): argues fine-tuning deltas are dense; trains with circuit-restricted updates. Cites APD and SPD in one sentence only; no PD. card (new).
- doi:10.18653/2026.emnlp-main.2453, 2026, Delta-Crosscoder (Kassem et al.): activation-space diffing in narrow fine-tuning. abs. Needed for the novelty audit; card next.
- doi:10.18653/v1/2025.emnlp-main.1641, 2025, "Reviving Your MNEME" (Kassem et al.): sparse (SAE) model diffing to predict side effects of unlearning/fine-tuning. abs. Context-to-core.
- arxiv:2511.06739, rank-1 LoRA interpretability: listed in PROJECT.md as adjacent; not re-found by this snowball. Unread.

## Core for gap 2 / toy-model validity (PD benchmarks)
- arxiv:2606.14673, 2026, "Compressed Computation is (probably) not Computation in Superposition" (Bhagat et al.): the APD/SPD compressed-computation toy model gets its advantage from a noisy-residual mixing matrix. card (new). Undercuts a headline APD/SPD toy benchmark.
- arxiv:2607.04800, 2026, "Compressed Computation under L4 Loss is likely Computation in Superposition" (Ferreira da Silva, Heimersheim): replacement toy model with a reverse-engineered ground truth; authors say it will be a PD test bed. card (new).

## Core for bilinear / weight-based interp
- arxiv:2410.08417, 2024 (ICLR 2025), Bilinear MLPs enable weight-based mechanistic interpretability (Pearce, Dooms, Rigg). card (new; partial read, see card).

## Context (skim or cite)
- arxiv:2406.03947, 2024, Weight-based Decomposition: A Case for Bilinear MLPs (ICML 2024 MI workshop): precursor of 2410.08417; abs.
- arxiv:2605.08891, 2026, Bilinear autoencoders find interpretable manifolds (Dooms, Gauderis, Wiggins): bilinear tensor decomposition with learned dictionaries; the 2410.08417 limitations section proposes this direction. abs. Read if the bilinear route is pursued.
- arxiv:2502.01032, 2025, Converting MLPs into Polynomials in Closed Form (Belrose, Rigg): abs.
- arxiv:2412.00944, 2024, Bilinear Convolution Decomposition for Causal RL Interpretability (Oozeer, Erisken, Rigg): abs.
- arxiv:2509.01874, 2025, Preserving Bilinear Weight Spectra with a Signed and Shrunk Quadratic Activation: abs. Low priority.
- arxiv:2602.05635, 2026, Structural Disentanglement in Bilinear MLPs via Architectural Inductive Bias: abs. Low priority, unknown authors.
- arxiv:2507.09816, 2025, Compressed Computation: Dense Circuits in a Toy Model of the Universal-AND Problem (Newgas): abs.
- arxiv:2511.13653, 2025, Weight-sparse transformers have interpretable circuits (OpenAI): alternative "train for weight sparsity" route; abs from search snippet only. Cited by 2026.findings-emnlp.679 as Gao et al. 2025.
- Goodfire Silico (private beta, 2026-04/05; press: MIT Tech Review, Bereska critique): PD-backed editing demo (German removal). web, press only; the underlying content is the LessWrong post above.
- Goodfire VPD explainer and research page (goodfire.ai/research/interpreting-lm-parameters, /vpd-explainer): web, duplicates arxiv:2610.20748.
- Gibson 2025/2026 (computation in superposition constructions), cited in 2607.04800; not located, unverified. Hanni et al. 2024 (random sparse binary codes for CiS): cited, unread.
- arxiv:2608.27540, 2026, Towards a mathematical theory of superposition (Ivanitskiy et al.): abs; Ivanitskiy is a VPD author.
- arxiv:2603.24653, 2026, From Weights to Concepts: data-free CLIP interp via SVD: abs; SVD-of-weights baseline family, vision.
- doi:10.52202/085713-4836, 2025, Mixture of Decoders faithful dense layer decomposition (Oldfield et al.): abs; faithful MLP decomposition via sparse layers, an alternative to transcoders. Context for faithfulness comparison.

## Irrelevant (name collisions or off-topic)
- arxiv:2511.02003, 2025, Bulk-boundary decomposition of neural networks (Lee, Lee, Yi): field-theory view of training dynamics; no relation to PD. Verified only from a mirror's abstract/search snippet.
- arxiv:2604.27408 (cancer models via parameter decomposition), federated parameter-decomposition aggregation, PV-model parameter decomposition, Alzheimer sCCA: name collisions.
- Silico / ping-pong superposition queries: no arXiv paper found for "ping pong superposition"; 0 relevant hits. Do not cite.
- LoRA/PEFT/sparse fine-tuning papers returned by diffing queries (e.g. 2505.24037, 2502.14816, 2508.02668): engineering, not interpretability.
- Generic mech-interp surveys (2607.07316, 10.1016/j.cosrev.2026.101011, tjsai.41-4_an40-d): context at best; no new method.
- doi:10.1145/3774904.3792324 (LLM unlearning, cites APD) and iasdr.2025.601 (steering interfaces, cites APD): cite APD in passing.

## Findings that change the picture
1. No published paper decomposes fine-tuning/RL deltas with PD. The VPD paper and the Bushnaq post leave it open; the one weight-diff attempt (bigsnarfdude) runs SVD on the diff and VPD on the final weights, not on the diff. Gap 1 is not scooped as of 2026-11-05, but unreviewed blogs and the citer list are incomplete. A claim of novelty still needs lit/novelty.md.
2. The Bushnaq fine-tuning post is the nearest neighbor for any "edit via PD components" idea: 1 scalar on 1 subcomponent matches rank-4 LoRA data efficiency (4 vs 32 tokens) at 67M, but German tokens were used to build the decomposition, so the token count is not clean.
3. The compressed-computation toy model, the main toy benchmark for APD/SPD, is probably not computation in superposition (Bhagat et al.). Any faithfulness claim that rests on it needs the L4 model (2607.04800) as ground truth instead.
4. Li et al. state that fine-tuning deltas are dense (U = P). If true for our checkpoints, delta-PD sparsity must come from the subcomponent/causal-importance structure, not from the delta itself. Check this on a small fine-tune before gap analysis.
5. Nothing found on PD efficiency or scaling beyond VPD, tPD and ASPD. No independent replication of VPD found.
