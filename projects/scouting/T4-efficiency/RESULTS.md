# T4 kill test results: H7 activation-sparse self-drafting (exploratory, CPU)

Setup and preregistered predictions: `IDEAS.md`. Code: `kill_test.py`. Raw numbers: `results.json`
(seeds 0-1), plus seed 2 from the run log (the process was OOM-killed, exit 137, after seed 2's
TEAL arms and before its skip arms). The run was 6 x 256-token wikitext-2 windows per seed, SmolLM2-135M in fp32.
Dense PPL was 26.7 / 23.4 (seeds 0/1). Sanity check: the dense-history path with no sparsity reproduces the dense argmax (alpha = 1.000).

## Numbers (alpha = greedy 1-step acceptance vs dense; c = fraction of linear-weight bytes)

| Arm | c | alpha seed 0 / 1 / 2 | standalone PPL (s0/s1/s2) |
|---|---|---|---|
| TEAL s=0.5, dense history | 0.50 | 0.744 / 0.731 / 0.710 | - |
| TEAL s=0.7, dense history | 0.30 | 0.337 / 0.338 / 0.341 | - |
| TEAL s=0.8 / 0.9, dense history | 0.2 / 0.1 | 0.13-0.15 / 0.03-0.05 | - |
| TEAL s=0.5, sparse history | 0.50 | 0.618 / 0.596 / 0.570 | 50.4 / 43.5 / 54.6 |
| TEAL s=0.7, sparse history | 0.30 | 0.137 / 0.129 / 0.083 | 1307 / 1467 / 2830 |
| Centered TEAL s=0.5, sparse history | 0.50 | 0.451 / 0.411 / 0.413 | 108.8 / 112.4 / 117.5 |
| Centered TEAL s=0.7, dense history | 0.30 | 0.235 / 0.253 / 0.252 | - |
| Sublayer skip (best of 2 orderings) | 0.49 | 0.176 / 0.167 / n/a | - |
| Sublayer skip (best of 2 orderings) | 0.28-0.29 | 0.032 / 0.023 / n/a | - |

## Verdict against the preregistration
- **P1 passes, by a wide margin:** at c ~= 0.3, alpha_TEAL - alpha_skip = 0.31 (seeds 0,1; seed 2 has no skip arm).
  The same holds at c ~= 0.5 (0.74 vs 0.17). At equal bytes, activation sparsity is a far better drafter
  than skipping sublayers. Caveat: my skip baseline uses a heuristic selection, not SWIFT's optimized search.
- **P2 passes:** at s=0.7, dense-history alpha is 0.34 vs sparse-history alpha 0.08-0.14. Rewriting the KV cache
  with dense values roughly triples acceptance, which supports the "no compounding" argument.
- **P3 fails (H2 dropped):** centering makes things worse (PPL roughly doubles vs plain TEAL). Mean-shifted
  thresholding keeps the wrong entries; I did not investigate further.
- **But the practical claim fails at this scale.** With the preregistered k=4, every arm is a slowdown
  (best: 0.98x with a 7B-like LM-head share). Post hoc, picking the best k (exploratory, not preregistered):
  TEAL s=0.5 reaches 1.15x (7B-like head) / 1.08x (135M head), and s=0.7 reaches 1.01x. A
  135M model is too fragile under uniform 4-site sparsity (standalone PPL 2x at 50%) for alpha to reach the
  ~0.8 at c=0.3 that a 1.4-1.5x speedup needs.

**Overall: no positive result yet.** The relative claims (vs layer skipping; dense vs sparse
history) are confirmed. The absolute speedup is negative at 135M, and this test cannot settle it for 7B+.

## Limits of this test
Uniform sparsity per site (TEAL allocates greedily per layer, which is stronger). One small model. Greedy
acceptance only (no sampling). The speedup formula assumes iid acceptance, memory-bound decode, and
verify ~= 1 dense step. Attention KV reads are not counted. Seed 2 is missing its skip arms.

## What would make it a NeurIPS/MLSys paper (and the risk)
The decisive cheap step is one s24 slice: alpha(s) for LLaMA-3-8B / Qwen3-8B with real TEAL greedy
allocation, dense history, k in 1..6. Kill if alpha < 0.75 at s=0.6. If it holds:
- Kernel: sparse-GEMV drafter and a dense batched verifier that share weights and the KV cache. No extra memory, so it
  is attractive for edge / single-GPU serving. Compare against EAGLE-3, SWIFT, LayerSkip and the
  SparseSpec family, measuring wall-clock at batch 1-8.
- Novel parts: (a) dense-history drafting as the reason activation sparsity tolerates far higher s than
  standalone use (P2, here 3x alpha); (b) verify-informed per-sequence hot sets (H8) for contiguous loads;
  (c) stacking with sparse-KV drafting (SparseSpec) to cut both the weight and the KV bytes.
- Risk: reviewers may see it as TEAL + self-speculation. EAGLE-style trained heads reach acceptance
  lengths of 3-6 and may simply dominate. Expected value: moderate (MLSys-plausible, NeurIPS main unlikely).
