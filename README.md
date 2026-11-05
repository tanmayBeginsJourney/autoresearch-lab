# autoresearch-lab

Infrastructure for running an autonomous ML research agent (Claude Code) on a university Slurm cluster,
with the quality controls enforced in code rather than in prompts.

- `researchctl`: CLI the agent uses for everything cluster-related. Compact JSON in and out; gates on
  submission (preregistration, clean git, duplicates, cost caps, H200 approval); job status, waiting,
  inspection, collection; claim statistics; literature search, citation graph, verified BibTeX.
- `labkit`: in-job harness (seeding, manifest, metrics, checkpoints, preemption, summaries).
- `.claude/`: guard hooks, research-stage skills, and a red-team reviewer subagent.
- `projects/<slug>/`: the research record. Literature cards, reproductions, gap analysis,
  preregistrations, experiment records, claims ledger, paper.

Start with [docs/architecture.md](docs/architecture.md), then [docs/operations.md](docs/operations.md).
The agent's rules are in [CLAUDE.md](CLAUDE.md).

```
researchctl gpu-status                      # free slices, queue pressure
researchctl submit projects/p/configs/x.yaml --dry-run
researchctl wait <exp>; researchctl collect <exp>
researchctl claims check <project>
```

Tests: `uv run --extra dev pytest` (Slurm is faked; no cluster needed).
