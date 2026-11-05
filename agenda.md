# Research agenda
<!-- Current state of candidate directions. Rewrite in place; decisions go in the chosen project's journal. -->

Status (2026-11-05): direction 3 chosen -> `projects/pdecomp` (target NeurIPS 2027). Direction 1 (`projects/actbias`) is the fallback. Compute: 4x 24 GB + 4x 48 GB MIG slices (rtx), 150 H200 GPU-h/month.

| # | Direction | Evidence so far | Cost of a decisive first experiment | Ceiling | Crowding |
|---|---|---|---|---|---|
| 1 | Activation centering (from arXiv:2610.14500) | Card in `projects/actbias`. Missing-bias confound already handled by the paper (App C.3). Weak spots: no per-arm LR tuning; GPT-2 gain ~0.01 nats from apparently one run; adaptive shift = centering half of NormFormer's FFN-LN (uncited). | Teacher-student with per-arm LR sweep: CPU, hours. nanoGPT-scale check: a few slice-days. | Careful analysis paper; top venue only with a general mechanism (centering x LayerNorm x Adam). | Low-moderate |
| 2 | Single-model capture of ensemble gains under fixed data (Slowrun regime) | Kim et al. 2025 (arXiv:2509.14786): averaging independent pretraining runs fails; distillation works. Baron et al. 2026 (arXiv:2610.37842): probability averaging beats weight averaging for snapshots. q0 (arXiv:2606.03938, Q Labs + Princeton): 12.9x data efficiency with 1.8B-param populations. PAPA/WASH exist for image classification. | Competitive results need 1B+ models x tens of epochs: hundreds of slice-hours per run. | High, but the field is leaderboard-driven by well-resourced groups. | High |
| 3 | Weight-space interpretability via parameter decomposition (chosen) | See `projects/pdecomp`: SOTA is VPD (arXiv:2610.20748, 67M LM). The layerwise-loss cost is already solved; open problems are faithfulness/adversarial robustness, scale, CI function, clustering. | Cost profile of VPD on a 4-layer LM: one slice-day (unmeasured). | High if we find an angle Goodfire is not racing on (candidate: decomposing fine-tuning deltas). | Moderate, fast-moving |
| 4 | Compiled circuits inside pretrained LMs (qwenFS-style) | Tracr-Injection (ACL Findings 2025) exists; qwenFS is engineering, not evaluated science. | Small. | Niche. | Low |
| - | Refusal-direction editing | Many 2026 papers on abliteration variants. | Small. | Low marginal value. | Very high |

## Next research step per direction
1. CPU reproduction of the teacher-student result with per-arm LR tuning (`projects/actbias`).
2. Not recommended at our compute unless a small-scale question appears.
3. In progress: see `projects/pdecomp/PROJECT.md`.
