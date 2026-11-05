# Container B: shared brief for every B-workstream subagent

Read this whole file first. It is binding.

## Goal
The user wants an original, paper-worthy research direction with a positive result for NeurIPS 2027
(main track, deadline ~mid-May 2027). A small extension of someone else's paper, or an analysis-only
study, does not count. Compute later: 8 GPU slices (24-48 GB) plus 150 H200-hours per month. Now: CPU only.

## Repo and environment
- Repo: `/home/user/autoresearch-lab`, branch `container-b` (already checked out). Container A works
  on the same GitHub repo in parallel on another branch.
- **Do not run any git command that changes state** (no add, commit, checkout, stash, pull, push). The lead
  commits and pushes for you.
- Shell setup for every Bash call:
  `export SCRATCH=$HOME LAB_ROOT=$HOME/lab RESEARCHCTL_REPO=/home/user/autoresearch-lab PATH=$HOME/.local/bin:$PATH HF_HOME=$HOME/lab/cache/hf`
- Python: `$LAB_ROOT/venvs/research/bin/python` (torch 2.11 CPU, transformers 5.22, scikit-learn, scipy, pytest).
- No GPU, no Slurm in this container: `researchctl` cluster commands (submit, wait, inspect) do not work here.
- Throwaway files: `$LAB_ROOT/tmp/<your-workstream>/`. Large downloads: `$HF_HOME`; delete them when done.

## Shared machine: 4 CPU cores, 15 GB RAM, ~30 GB free disk, for ~10 agents at once
- `torch.set_num_threads(1)` in every script (and `OMP_NUM_THREADS=1`). At most one heavy process per agent.
- Stay under ~1.5 GB RSS per agent unless your prompt says otherwise. Never load two >0.5B models in fp32 at once.
- Keep each CPU experiment short; run long jobs in the background with `nohup ... &` and a log file, and
  check them with `tail`, not by printing whole logs.

## Literature tools
- `researchctl lit-fetch arxiv:ID` works: full text plus a section index, saved under `$LAB_ROOT/cache/papers/`.
  Read by section (sed -n on line ranges), never the whole text.
- `researchctl lit-search "q"`, `lit-citers`, `lit-refs`, `lit-cite` may hang (API keys are not set in
  this container). Always wrap in `timeout 90`. Do **not** pass `--project pdecomp` (it appends to a shared
  log file). Record your own queries in your output files instead.
- Fallback: WebSearch and WebFetch (load them with ToolSearch, query `select:WebSearch,WebFetch`).
- Never print API keys or other secrets.

## Integrity rules (from CLAUDE.md; binding)
- Literature and novelty search before any claim. A novelty verdict lists the queries run, the closest
  prior work (with arXiv ids), and the concrete difference.
- **Pre-register before running.** Write `PREREG.md` (prediction, success threshold, the strongest
  baseline, the analysis, the seeds) in your directory *before* the experiment that tests it.
- Paired seeds, at least 3, the same seed list in every arm.
- Exploratory results generate hypotheses; nothing here is a confirmed claim. Say "exploratory".
- Report negative results and failed seeds honestly. If a result looks surprisingly good, look for the bug first.
- Name known techniques by their known names. Correlation or probing is not mechanism.
- Writing style (`docs/style.md`): lead with the result, numbers not adjectives, short sentences, no hype.

## Where you may write
- Only your own directory (given in your prompt) and, if your prompt says so, new card files in
  `projects/pdecomp/lit/cards/` (YAML, same format as `projects/pdecomp/lit/cards/arxiv-2508.00161.yaml`;
  never overwrite an existing card).
- Do NOT edit CLAUDE.md, agenda.md, any PROJECT.md, gaps.md, journal.md, src/, or other agents' directories.
- If writing a `.md` file is blocked, write the same content as `.txt`, or return it in your final message.

## Project context (read these as needed, by section)
- `CLAUDE.md`, `docs/standards.md`, `agenda.md`.
- `projects/pdecomp/PROJECT.md`, `gaps.md`, `journal.md`: the current project (weight-space interpretability via
  parameter decomposition: APD -> SPD -> VPD arXiv:2610.20748). Gaps G1 Delta-PD (decompose fine-tuning deltas),
  G2 mask-attack calibration, G3 interaction clustering, G4 activation choice vs decomposability.
- `projects/pdecomp/reviews/261105-redteam-gaps.md`: red-team scored the gaps 3-5/10.
- `projects/pdecomp/prototypes/*/RESULTS.md`: exploratory CPU prototypes. Key one: Delta-PD v2, rotated SVD
  (ICA/varimax in the top-k subspace of Delta W) ties Delta-PD whenever changes are linearly separable.
- `projects/actbias/repro/teacher-student/REPORT.md`: activation-centering reproduction.
- `projects/scouting/S0-problem-first/problems.md`: container A's problem-first scout.
- Libraries: `projects/pdecomp/src/spd_min` (SPD; CLI `train_tms.py`), `projects/pdecomp/src/pd_toys`
  (toys with ground truth). 21 tests pass.

## Already being done by container A (do not duplicate)
Delta-PD v3 toy (superposed, gated, cross-layer changes); backdoor-in-delta novelty audit; idea tournaments
T1 (optimization x activations), T2 (symmetries / teleportation), T3 (editing / unlearning), T4
(learned-structure efficiency), T5 (training-dynamics interpretability); S0 problem-first scout.

## Tournament protocol (B4-B8)
1. Generate >= 6 original ideas in your area. For each: one sentence, the mechanism for why it could work,
   and what a positive result would look like.
2. Novelty check each (>= 3 queries per idea; arXiv, Semantic Scholar/OpenAlex via WebSearch, 2025-2026 work).
   Verdict: open / partly done / done, with the closest paper ids.
3. Score the survivors: novelty, P(positive), paper ceiling, cost at our compute, crowding. Kill most.
4. For the best 1-2: write `PREREG.md` (prediction, threshold, strongest baseline, seeds), then run a CPU
   kill test (<= ~2 CPU-hours total, 1 thread, >= 3 paired seeds). The kill test must be able to fail.
5. Verdict: survives / killed, with the number against the threshold. If it survives, a 2-week GPU plan
   (slice-hours, which experiments, what decision each informs).

## Final message to the lead (<= 350 words)
Verdict first, then key numbers vs thresholds, the closest prior work, the top risk, and the list of files
you wrote. The lead integrates this into `projects/scouting/B-SUMMARY.md`.
