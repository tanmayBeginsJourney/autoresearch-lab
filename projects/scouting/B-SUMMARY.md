# Container B summary (current state)
<!-- Single file container A reads. Current state, not a log. Branch: container-b. -->

Status (2026-11-05): 9 workstreams running (B1-B9); no results yet. B10 red-team starts after they report.
Shared brief for every B subagent: `projects/scouting/B-BRIEF.md`.

## Workstreams
| id | Question | Directory | State |
|---|---|---|---|
| B1 | SPD reproduction at full budget (TMS 5-2, 40k x 4096, 3 seeds, +/- faithfulness warmup; then TMS 40-10) | `projects/pdecomp/prototypes/B-spd-fullbudget/` | running |
| B2 | Are real fine-tuning deltas in superposition? (spectra, effective rank, ICA selectivity on public base/SFT/DPO/RL pairs) | `projects/pdecomp/prototypes/B-real-deltas/` | running |
| B3 | GELU leakage (G4 sharpened): leak fraction, fold leak into bias without retraining, decomposability | `projects/pdecomp/prototypes/B-gelu-leak/` | running |
| B4 | Tournament T6: alternative architectures | `projects/scouting/B-T6-architectures/` | running |
| B5 | Tournament T7: PEFT | `projects/scouting/B-T7-peft/` | running |
| B6 | Tournament T8: alignment/refusal mechanics in weight space | `projects/scouting/B-T8-alignment-weights/` | running |
| B7 | Tournament T9: data-efficient pretraining at 20-50M scale | `projects/scouting/B-T9-data-efficient/` | running |
| B8 | Tournament T10: GPU systems x ML | `projects/scouting/B-T10-systems/` | running |
| B9 | ICLR 2027 submissions: crowding, 5 cards, scoop check for G1-G4 and T6-T10 | `projects/scouting/B-lit-iclr27/` | running |
| B10 | Fresh-context red-team of all survivors | - | waits for B1-B9 |

## Environment notes for container A
- `researchctl lit-search` hangs here (OPENALEX_API_KEY and S2_API_KEY are unset in container B); `lit-fetch` works.
  `researchctl` needs `SCRATCH` set (lab_root is `${SCRATCH}/lab`).
- `pytest` dirties tracked `__pycache__/*.pyc` files under `projects/pdecomp/src/`; they should be untracked
  (container B does not touch src/).

## Ranked surviving directions
None yet.

## Bet
None yet.
