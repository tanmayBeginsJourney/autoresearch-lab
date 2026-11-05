"""Claims ledger: every claim in a paper maps to logged runs and a pre-specified statistical test.

`researchctl claims check P` recomputes all statistics from experiment records, applies Holm correction
across confirmatory claims, assigns verdicts, lints wording, and writes:
  projects/P/claims.lock.json   machine-checked verdicts
  projects/P/paper/numbers.tex  macros; papers cite numbers only via \\claimval{C1.diff}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import stats
from .core import Lab, load_yaml, now, sha256_file

KINDS = ("performance", "causal", "correlational", "descriptive")

LINT = [
    (r"\b(novel|first to|for the first time|new (method|approach|mechanism))\b", "novelty",
     "novelty wording requires lit/novelty.md (search log + closest prior work)"),
    (r"\bsignificant(ly)?\b", "significance", "say 'significant' only for a supported claim; report the test"),
    (r"\b(always|universally|in general|all models|any model)\b", "scope", "scope exceeds the evidence; name the models/settings tested"),
    (r"\b(groundbreaking|revolutionary|remarkable|unprecedented|dramatic(ally)?|paradigm|breakthrough)\b", "hype", "delete hype"),
    (r"\b(causes?|drives?|mechanism|because of|leads? to|responsible for)\b", "causal",
     "causal wording needs kind: causal with an intervention-based design"),
]


def _select(rec: dict, where: dict, metric: str) -> tuple[dict[int, float], list[str]]:
    """(seed -> final metric for completed matching tasks, list of excluded tasks with their status)."""
    res = rec.get("results") or {}
    by_i = {t["i"]: t for t in res.get("tasks", [])}
    out, excluded = {}, []
    for t in rec["tasks"]:
        if any(t["point"].get(k) != v for k, v in (where or {}).items()):
            continue
        r = by_i.get(t["i"], {})
        if r.get("status") == "completed" and metric in r.get("final", {}):
            out[t["seed"]] = float(r["final"][metric])
        else:
            excluded.append(f"seed {t['seed']}: {r.get('status', 'missing')}")
    return out, excluded


def _lint(statement: str, kind: str, has_novelty: bool) -> list[str]:
    msgs = []
    for rx, tag, msg in LINT:
        if not re.search(rx, statement, re.I):
            continue
        if tag == "novelty" and has_novelty:
            continue
        if tag == "causal" and kind == "causal":
            continue
        msgs.append(f"{tag}:{msg}")
    return msgs


def check(lab: Lab, project: str) -> dict:
    pdir = lab.project_dir(project)
    ledger = load_yaml(pdir / "claims.yaml") or {}
    claims = ledger.get("claims") or []
    has_novelty = (pdir / "lit" / "novelty.md").is_file()
    min_seeds = lab.policy["min_seeds_confirmatory"]
    results, pvals = {}, {}
    for c in claims:
        cid = c.get("id")
        problems = [f"missing {k}" for k in ("id", "statement", "kind", "metric", "goal", "treatment") if not c.get(k)]
        if c.get("kind") and c["kind"] not in KINDS:
            problems.append(f"kind must be one of {KINDS}")
        if problems:
            results[cid or "?"] = {"verdict": "invalid", "problems": problems}
            continue
        recs = {}
        for arm in ("treatment", "control"):
            if c.get(arm):
                recs[arm] = load_yaml(lab.find_record(c[arm]["experiment"]))
        confirmatory = bool(c.get("hypothesis"))
        for arm, rec in recs.items():
            if confirmatory:
                if rec["kind"] != "confirmatory" or rec.get("hypothesis") != c["hypothesis"]:
                    problems.append(f"{arm}: {rec['exp_id']} is not a confirmatory run of {c['hypothesis']}")
                pre = pdir / "prereg" / f"{c['hypothesis']}.yaml"
                if rec.get("prereg_sha256") and pre.exists() and sha256_file(pre) != rec["prereg_sha256"]:
                    problems.append(f"prereg {c['hypothesis']} was edited after {rec['exp_id']} ran")
            if not rec.get("results"):
                problems.append(f"{arm}: {rec['exp_id']} has no collected results (researchctl collect)")
        t, ex_t = _select(recs["treatment"], c["treatment"].get("where"), c["metric"]) if "treatment" in recs else ({}, [])
        entry: dict = {"statement": c["statement"], "kind": c["kind"], "confirmatory": confirmatory}
        excluded = {"treatment": ex_t}
        if "control" in recs:
            k, excluded["control"] = _select(recs["control"], c["control"].get("where"), c["metric"])
            paired = c.get("paired_by") == "seed"
            if paired:
                seeds = sorted(set(t) & set(k))
                tv, kv = [t[s] for s in seeds], [k[s] for s in seeds]
            else:
                tv, kv = list(t.values()), list(k.values())
            n = min(len(tv), len(kv))
            if n < min_seeds:
                problems.append(f"only {n} usable seeds per arm (< {min_seeds})")
            if n >= 2:
                cmp = stats.compare(tv, kv, paired=paired)
                entry.update(treatment=stats.describe(tv), control=stats.describe(kv), comparison=cmp)
                if confirmatory and not problems and not cmp.get("insufficient"):
                    pvals[cid] = cmp["p"]
        else:
            entry.update(treatment=stats.describe(list(t.values())))
        entry["problems"] = problems
        entry["lint"] = _lint(c["statement"], c["kind"], has_novelty)
        if any(excluded.values()):  # dropped seeds are a result, not noise: they must be reported
            entry["excluded"] = excluded
            entry["lint"].append(f"excluded: {sum(map(len, excluded.values()))} runs failed/diverged; report them with the claim")
        results[cid] = entry

    adj = stats.holm(pvals, lab.policy["alpha"]) if pvals else {}
    for c in claims:
        cid = c.get("id")
        e = results.get(cid)
        if not e or e.get("verdict") == "invalid":
            continue
        cmp = e.get("comparison")
        sign = -1 if c["goal"] == "min" else 1
        if e["problems"]:
            v = "invalid"
        elif not cmp:
            v = "descriptive"
        elif not e["confirmatory"]:
            v = "exploratory"
        else:
            e["p_holm"] = adj.get(cid)
            lo, hi = cmp["ci"]
            if adj.get(cid, 1) <= lab.policy["alpha"] and sign * lo > 0 and sign * hi > 0:
                v = "supported"
            elif sign * hi < 0 and sign * lo < 0:
                v = "contradicted"
            else:
                v = "inconclusive"
        e["verdict"] = v
        if v == "supported":  # 'significant' is allowed only for supported claims
            e["lint"] = [m for m in e["lint"] if not m.startswith("significance")]

    lock = {"project": project, "checked_at": now(), "alpha": lab.policy["alpha"],
            "n_confirmatory_tests": len(pvals), "claims": results}
    (pdir / "claims.lock.json").write_text(json.dumps(lock, indent=1, default=str))
    _write_macros(pdir, results)
    return {"project": project, "verdicts": {k: v.get("verdict") for k, v in results.items()},
            "problems": {k: v["problems"] for k, v in results.items() if v.get("problems")} or None,
            "lint": {k: v["lint"] for k, v in results.items() if v.get("lint")} or None,
            "wrote": ["claims.lock.json", "paper/numbers.tex"]}


def _fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.2e}" if x != 0 and abs(x) < 10 ** -nd else f"{x:.{nd}f}"
    return str(x)


def _write_macros(pdir: Path, results: dict) -> None:
    lines = ["% Generated by `researchctl claims check`. Do not edit. Use \\claimval{C1.diff} in text.",
             "\\makeatletter", "\\providecommand{\\claimval}[1]{\\@ifundefined{claim@#1}{\\textbf{??#1??}}{\\@nameuse{claim@#1}}}"]
    for cid, e in results.items():
        vals = {"verdict": e.get("verdict")}
        if "comparison" in e:
            c = e["comparison"]
            vals.update(diff=_fmt(c.get("diff")), cilo=_fmt(c.get("ci", [None])[0]), cihi=_fmt(c.get("ci", [None, None])[1]),
                        p=_fmt(c.get("p")), pholm=_fmt(e.get("p_holm")), d=_fmt(c.get("cohen_d"), 2),
                        n=str(c.get("n")))
        for arm in ("treatment", "control"):
            if arm in e and "mean" in e[arm]:
                vals[f"{arm}mean"] = _fmt(e[arm]["mean"])
        for k, v in vals.items():
            lines.append(f"\\@namedef{{claim@{cid}.{k}}}{{{v}}}")
    lines.append("\\makeatother")
    out = pdir / "paper" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# ---------- prose lint for papers and notes ----------

AI_TELLS = r"\b(delve|crucial|pivotal|intricate|tapestry|underscore[sd]?|showcase[sd]?|leverag(e|es|ing)|seamless(ly)?|robust(ly)? (and|yet)|it is worth noting|notably,|in the realm of|a testament to|game[- ]chang\w*|holistic)\b"


def lint_prose(paths: list[Path], bib: Path | None = None) -> dict:
    issues = []
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text())) if bib and bib.exists() else None
    for p in paths:
        files = [p] if p.is_file() else [*p.rglob("*.tex"), *p.rglob("*.md")]
        for f in files:
            if f.name == "numbers.tex" or "generated" in f.parts:
                continue
            for ln, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if line.lstrip().startswith("%"):
                    continue
                for rx, tag, msg in LINT[2:4]:
                    if re.search(rx, line, re.I):
                        issues.append(f"{f.name}:{ln} {tag}: {msg}")
                if re.search(AI_TELLS, line, re.I):
                    issues.append(f"{f.name}:{ln} wording: replace stock phrase ({re.search(AI_TELLS, line, re.I).group(0)})")
                if line.count("—") + line.count("---") > 1:
                    issues.append(f"{f.name}:{ln} wording: more than one em dash in a line")
                if re.search(r"CITATION NEEDED|\?\?|TODO|XXX", line):
                    issues.append(f"{f.name}:{ln} unresolved placeholder")
                if f.suffix == ".tex" and re.search(r"(?<![\\{\w.])\d+\.\d{2,}(?![\w}])", line) and "\\claimval" not in line:
                    issues.append(f"{f.name}:{ln} literal number: take results from \\claimval or generated tables")
                if keys is not None:
                    for grp in re.findall(r"\\cite\w*\{([^}]+)\}", line):
                        for k in (s.strip() for s in grp.split(",")):
                            if k not in keys:
                                issues.append(f"{f.name}:{ln} citation key not in bib: {k}")
    return {"issues": issues[:80], "total": len(issues)}
