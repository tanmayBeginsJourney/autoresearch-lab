# pdecomp journal
<!-- Append-only. One entry per ML decision or result: date, what, evidence (exp ids), consequence. -->
- 2026-11-05: Direction chosen by the user: parameter decomposition (agenda #3). Target NeurIPS 2027 (~May 14-21, projected), workshop checkpoint ~Feb 2027.
- 2026-11-05: Read SPD (2506.20790) and VPD (2610.20748) in full; cards for APD, tPD, ASPD via subagents. Killed "speed up SPD's layerwise loss": VPD already replaced it with random matrix subsets in one pass.
- 2026-11-05: Skeptic notes: ASPD's 8B result is one matrix (L17 attn O) with no faithfulness metric; tPD's 7%-FLOPs is analytic and single-seed; VPD explanations recover only 8-12% pretraining compute with rounded masks.
