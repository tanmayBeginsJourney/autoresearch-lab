# T3 results: H7 kernel-decoupled unlearning (2026-11-05)

Script: `kd_unlearn.py` (prereg in `IDEAS.md`). Raw output was in the session scratchpad (not kept).
Seeds run: seed 0 only. Seeds 1-2 were cut by the CPU budget: 6 arms with double-backward took about 530 s
per seed with two runs sharing the cores. The prediction needed 3/3 seeds, so failing on seed 0 already
decides it.

Seed 0, attack = 100 Adam steps on A; excess = acc_B(arm) - acc_B(oracle). Oracle acc_B after the attack = 0.00.

| arm | F acc before | R acc before | cross-forget kernel | excess B recovery | R acc after attack |
|---|---|---|---|---|---|
| NPO | 0.17 | 1.00 | 0.0064 | 0.26 | 0.74 |
| RMU-shared | 0.22 | 1.00 | 0.0508 | 0.44 | 0.89 |
| RandLabel | 0.00 | 1.00 | 0.0021 | 0.08 | 0.88 |
| KD (lambda=3, prereg) | 0.00 | 1.00 | 0.0020 | 0.08 | 0.88 |
| KD (lambda=30, exploratory) | 0.00 | 1.00 | 0.0014 | 0.10 | 0.89 |
| NPO+KD (lambda 3 / 30) | 0.19 / 0.16 | 1.00 | 0.0020 / 0.0007 | 0.16 / 0.18 | 0.79 / 0.85 |
| RMU+KD (lambda 3 / 30) | 0.57 / 0.72 | 1.00 | 0.0052 / 0.0015 | 0.42 / 0.58 | 0.99 / 0.97 |

Verdict: KILL. The prereg threshold was KD excess <= 0.04 (half of RandLabel's 0.08). KD got 0.08.
- The penalty did its job (the RMU kernel fell 30x), but held-out recovery did not drop (0.44 to 0.42/0.58),
  and the penalty fought forgetting (F acc before the attack rose to 0.57-0.72). So the first-order
  tangent-kernel coupling does not drive recovery after 100 Adam steps.
- Side observation (one seed, exploratory): a shared target causes the most held-out recovery
  (RMU 0.44 > NPO 0.26 > per-example random labels 0.08). This replicates the shared-inhibitor picture of
  Layered Unlearning (arXiv:2505.09500), so it is not new.

## Area assessment and the version on <=8B models
Relearning-robust unlearning is saturated (2026: RepSelect, Minor Components, Margin Cliff, KAU, Layered
Unlearning, UNDO, TAR, AntiDote, SAM-unlearning). Abliteration defences already have decoy approaches
(Fool's Gold). None of H1-H8 is both new and supported. Recommendation: do not open a project in T3 on
these hypotheses. If it is revisited, the cheapest new question is the end-of-training (non-linearised)
attacker response, i.e. KAU-style closed-form anticipation extended past first order. On <=8B models that
means TOFU/WMDP on Llama-3.2-1B to Llama-3.1-8B, comparing against RepSelect and Layered Unlearning, at
about 2-4 slice-days. Run it only after a synthetic positive result, which we do not have.
