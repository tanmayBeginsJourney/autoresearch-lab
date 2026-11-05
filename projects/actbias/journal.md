# actbias journal
<!-- Append-only. One entry per ML decision or result: date, what, evidence (exp ids), consequence. -->
- 2026-11-05: Read arXiv:2610.14500 in full; card written. Flagged: no per-arm LR tuning, single-run GPT-2 result, NormFormer FFN-LN uncited.
- 2026-11-05: Teacher-student repro (exploratory, 5 paired seeds, CPU): ReLU shift lowers final loss -5% to -6.5% (5/5), survives per-arm LR tuning and the bias control; GELU shows no final-loss gain; slower-early only for GELU. See repro/teacher-student/REPORT.md.
- 2026-11-05: T1 tournament: optimizer-level centering for Adam already exists (SBO-AdamW arXiv:2607.22927; also Wiesler 2014, K-FAC centering). Centering under Muon: +0.6% (no gain); plain Muon beats Adam by ~20% on the teacher-student. Any centering paper needs a Muon baseline; the actbias direction looks like analysis-only now.
