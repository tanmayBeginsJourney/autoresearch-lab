"""Stage gates: artifacts that must exist before a project may run confirmatory experiments."""
from __future__ import annotations

from pathlib import Path

from .core import Lab, load_yaml

CARD_FIELDS = ("id", "title", "year", "depth", "question", "claims", "skeptic", "relevance")
DEPTHS = ("abstract", "skim", "full", "reproduced")


def card_problems(path: Path) -> list[str]:
    c = load_yaml(path) or {}
    errs = [f"{path.name}: missing {f}" for f in CARD_FIELDS if not c.get(f)]
    if c.get("depth") and c["depth"] not in DEPTHS:
        errs.append(f"{path.name}: depth must be one of {DEPTHS}")
    return errs


def status(lab: Lab, pdir: Path) -> dict:
    cards = sorted((pdir / "lit" / "cards").glob("*.yaml"))
    depths = [(load_yaml(c) or {}).get("depth") for c in cards]
    gap = pdir / "gaps.md"
    repro = [p for p in (pdir / "repro").glob("*/REPORT.md")] if (pdir / "repro").is_dir() else []
    return {
        "cards": len(cards),
        "full_read": sum(d in ("full", "reproduced") for d in depths),
        "card_errors": [e for c in cards for e in card_problems(c)][:10],
        "reproductions": len(repro),
        "gap_decision": gap.is_file() and "## Decision" in gap.read_text(),
        "novelty_audit": (pdir / "lit" / "novelty.md").is_file(),
        "preregs": sorted(p.stem for p in (pdir / "prereg").glob("*.yaml")),
    }


def missing_for_confirmatory(lab: Lab, pdir: Path) -> list[str]:
    g = lab.policy["gates"]
    s = status(lab, pdir)
    miss = []
    if s["cards"] < g["min_paper_cards"]:
        miss.append(f"literature: {s['cards']}/{g['min_paper_cards']} paper cards")
    if s["full_read"] < g["min_full_read_cards"]:
        miss.append(f"literature: {s['full_read']}/{g['min_full_read_cards']} cards read in full")
    if s["card_errors"]:
        miss.append(f"literature: malformed cards ({s['card_errors'][0]} ...)")
    if g["require_reproduction"] and not s["reproductions"]:
        miss.append("no reproduction report (repro/<paper>/REPORT.md)")
    if g["require_gap_decision"] and not s["gap_decision"]:
        miss.append("gaps.md lacks a '## Decision' section")
    return miss
