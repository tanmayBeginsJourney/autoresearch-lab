"""Literature tools: search, citation graph, verified BibTeX, full-text fetch with a section index.

Outputs are compact so the agent can triage many papers cheaply, then read full text by section.
Optional: S2_API_KEY in the environment raises Semantic Scholar rate limits.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .core import Fail, Lab, now

UA = "autoresearch-lab/0.1 (research tooling)"
S2 = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,venue,citationCount,externalIds,authors"


def _get(url: str, headers: dict | None = None, timeout: int = 30, tries: int = 4) -> bytes:
    h = {"User-Agent": UA, **(headers or {})}
    if "semanticscholar" in url and os.environ.get("S2_API_KEY"):
        h["x-api-key"] = os.environ["S2_API_KEY"]
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(2 ** (i + 1))
                continue
            raise Fail(f"HTTP {e.code} from {urllib.parse.urlparse(url).netloc}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise Fail(f"network error from {urllib.parse.urlparse(url).netloc}: {getattr(e, 'reason', e)}")
    raise Fail("unreachable")


def _norm_id(pid: str) -> str:
    pid = pid.strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", pid):
        return "arXiv:" + re.sub(r"v\d+$", "", pid)
    if pid.lower().startswith("arxiv:"):
        return "arXiv:" + re.sub(r"v\d+$", "", pid.split(":", 1)[1])
    if pid.startswith("10."):
        return f"DOI:{pid}"
    return pid


def _row(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {"id": ("arxiv:" + ext["ArXiv"]) if ext.get("ArXiv") else ("doi:" + ext["DOI"]) if ext.get("DOI") else p.get("paperId"),
            "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue") or None,
            "cites": p.get("citationCount"), "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:3])}


def search_s2(q: str, limit: int, year: str | None) -> list[dict]:
    url = f"{S2}/paper/search?query={urllib.parse.quote(q)}&limit={limit}&fields={FIELDS}"
    if year:
        url += f"&year={year}"
    return [_row(p) for p in json.loads(_get(url)).get("data", [])]


def search_arxiv(q: str, limit: int) -> list[dict]:
    terms = " AND ".join(f"all:{w}" for w in q.split())
    url = ("https://export.arxiv.org/api/query?" +
           urllib.parse.urlencode({"search_query": terms, "max_results": limit, "sortBy": "relevance"}))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(_get(url, timeout=60, tries=2))  # export.arxiv.org is often slow
    out = []
    for e in root.findall("a:entry", ns):
        aid = e.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        out.append({"id": "arxiv:" + re.sub(r"v\d+$", "", aid), "title": " ".join(e.findtext("a:title", "", ns).split()),
                    "year": int(e.findtext("a:published", "0000", ns)[:4]), "venue": "arXiv", "cites": None,
                    "authors": ", ".join(a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:3])})
    return out


def search(lab: Lab, q: str, limit: int, source: str, year: str | None, project: str | None) -> dict:
    rows, errors = [], []
    for name, fn in (("s2", lambda: search_s2(q, limit, year)), ("arxiv", lambda: search_arxiv(q, limit))):
        if source in (name, "all"):
            try:
                rows += fn()
            except Fail as e:
                errors.append(f"{name}: {e}")
    seen, uniq = set(), []
    for r in rows:
        key = re.sub(r"\W", "", (r["title"] or "").lower())
        if key and key not in seen:
            seen.add(key)
            uniq.append(r)
    if project:  # novelty audit trail: every search is logged
        log = lab.project_dir(project) / "lit" / "queries.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps({"at": now(), "q": q, "source": source, "year": year, "hits": len(uniq)}) + "\n")
    return {"query": q, "results": uniq, "errors": errors or None}


def graph(pid: str, direction: str, limit: int) -> dict:
    """direction: references (backward) or citations (forward)."""
    url = f"{S2}/paper/{urllib.parse.quote(_norm_id(pid))}/{direction}?fields={FIELDS}&limit={min(limit, 1000)}"
    data = json.loads(_get(url)).get("data", [])
    key = "citedPaper" if direction == "references" else "citingPaper"
    rows = [_row(d[key]) for d in data if d.get(key, {}).get("title")]
    rows.sort(key=lambda r: r["cites"] or 0, reverse=True)
    return {"paper": pid, direction: rows[:limit], "total": len(rows)}


def cite(pid: str) -> dict:
    """BibTeX from the authoritative source (arXiv or Crossref). Never written from memory."""
    nid = _norm_id(pid)
    if nid.startswith("arXiv:"):
        bib = _get(f"https://arxiv.org/bibtex/{nid.split(':', 1)[1]}").decode()
    elif nid.startswith("DOI:"):
        bib = _get(f"https://doi.org/{nid.split(':', 1)[1]}", {"Accept": "application/x-bibtex"}).decode()
    else:
        raise Fail("cite needs an arXiv id or DOI")
    if "@" not in bib:
        raise Fail("no BibTeX returned", id=pid)
    return {"id": pid, "bibtex": bib.strip()}


_HEAD = re.compile(r"^\s*((\d+(\.\d+)*\.?)|[A-Z]\.?|Appendix [A-Z]:?)\s+[A-Z][\w ,:\-()/&]{2,80}$")


def fetch(lab: Lab, pid: str) -> dict:
    """Download the PDF to scratch, extract text, and return a section index with character offsets."""
    nid = _norm_id(pid)
    if not nid.startswith("arXiv:"):
        raise Fail("fetch supports arXiv ids; for other papers save the PDF and run `researchctl lit text FILE`")
    aid = nid.split(":", 1)[1]
    cache = lab.lab_root / "cache" / "papers"
    cache.mkdir(parents=True, exist_ok=True)
    pdf = cache / f"{aid}.pdf"
    if not pdf.exists():
        pdf.write_bytes(_get(f"https://arxiv.org/pdf/{aid}", timeout=120))
    return text_index(pdf)


def text_index(pdf: Path) -> dict:
    txt = pdf.with_suffix(".txt")
    if not txt.exists():
        if shutil.which("pdftotext"):
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        else:
            code = "import sys,pymupdf;d=pymupdf.open(sys.argv[1]);open(sys.argv[2],'w').write(''.join(p.get_text() for p in d))"
            subprocess.run(["uv", "run", "-q", "--no-project", "--with", "pymupdf", "python", "-c", code, str(pdf), str(txt)],
                           check=True)
    text = txt.read_text(errors="replace")
    sections, off = [], 0
    for line in text.splitlines(keepends=True):
        if _HEAD.match(line) and len(line.strip()) < 90:
            sections.append({"h": line.strip()[:80], "line": text.count("\n", 0, off) + 1})
        off += len(line)
    links = sorted({u.rstrip(".") for u in re.findall(r"https?://github\.com/[\w.\-]+/[\w.\-]+", text)})
    return {"text_file": str(txt), "chars": len(text), "lines": text.count("\n"),
            "sections": sections[:60], "code_links": links[:10],
            "hint": "Read text_file by line offset per section; record findings in a paper card."}
