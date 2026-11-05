"""researchctl: one-line JSON answers to research-infrastructure questions.

Design rules: every command bundles the low-level calls it needs, prints one compact JSON object,
never dumps whole logs or tensors, and exits 1 with {"error": ...} when it refuses.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import approvals, claims, gates, lit, runs, slurm, submit
from .core import Fail, Lab, emit, git_dirty, load_yaml, sh


def cmd_gpu_status(lab, a):
    nodes = slurm.nodes()
    q = slurm.queue()
    pend = {}
    for j in q:
        if j["state"] == "PENDING":
            pend[j["partition"]] = pend.get(j["partition"], 0) + 1
    free = {}
    for n in nodes:
        if "down" in n["state"] or "drain" in n["state"]:
            continue
        for g, c in n["gpus_free"].items():
            free[g] = free.get(g, 0) + c
    mine = [j for j in q if j["user"] == slurm.user()]
    return {"free_gpus": free, "pending_by_partition": pend, "my_jobs": len(mine),
            "nodes": [{k: n[k] for k in ("node", "state", "gpus_free", "cpus_free", "mem_free_gb")} for n in nodes]}


def cmd_queue(lab, a):
    rows = slurm.queue(all_users=a.all)
    return {"jobs": rows[: a.limit], "total": len(rows)}


def cmd_exec(lab, a):
    prof = lab.profiles.get(a.profile)
    if not prof:
        raise Fail(f"unknown profile {a.profile}")
    if prof.get("requires_approval"):
        raise Fail("exec never runs on approval-gated profiles")
    if a.minutes > lab.policy["exec_max_minutes"]:
        raise Fail(f"exec is capped at {lab.policy['exec_max_minutes']} min; use submit")
    if not a.cmd:
        raise Fail("usage: researchctl exec --profile s24 -- CMD ...")
    cmd = a.cmd[1:] if a.cmd[0] == "--" else a.cmd
    rc, out = slurm.srun_capture(prof, a.minutes, cmd, lab.cfg["cluster"].get("account"))
    lines = out.splitlines()
    return {"rc": rc, "lines": len(lines), "output": "\n".join(lines[-a.tail:])[-4000:]}


def cmd_new_project(lab, a):
    d = lab.project_dir(a.slug)
    if d.exists():
        raise Fail(f"{d} exists")
    for sub in ("lit/cards", "repro", "prereg", "configs", "experiments", "analysis", "paper/figures", "src"):
        (d / sub).mkdir(parents=True)
    (d / "PROJECT.md").write_text(PROJECT_MD.format(slug=a.slug))
    (d / "claims.yaml").write_text("# Claims ledger. See .claude/skills/analyze/SKILL.md for the schema.\nclaims: []\n")
    (d / "gaps.md").write_text("# Gap analysis\n\n<!-- See .claude/skills/gap-analysis/SKILL.md. Add '## Decision' when decided. -->\n")
    return {"created": str(d.relative_to(lab.root))}


PROJECT_MD = """# {slug}

## Problem statement
<!-- One paragraph: the question, why it matters, what would count as an answer. -->

## Status
stage: frame   <!-- frame | literature | reproduce | gap | prereg | experiments | analysis | review | writing -->

## Decisions log
<!-- Dated one-liners: decision, evidence, alternatives rejected. Newest last. -->

## Findings
<!-- Dated, linked to experiment ids / claims. Negative results included. -->
"""


def cmd_prereg(lab, a):
    from .prereg import template
    d = lab.project_dir(a.project) / "prereg"
    existing = sorted(d.glob("H*.yaml"))
    hid = f"H{len(existing) + 1:03d}"
    p = d / f"{hid}.yaml"
    p.write_text(template(hid))
    return {"created": str(p.relative_to(lab.root)), "next": "fill every field, commit, then submit confirmatory runs"}


def cmd_gate(lab, a):
    pdir = lab.project_dir(a.project)
    return {"project": a.project, **gates.status(lab, pdir), "missing_for_confirmatory": gates.missing_for_confirmatory(lab, pdir)}


def cmd_env(lab, a):
    proj = lab.root / "envs" / a.name
    if not (proj / "pyproject.toml").exists():
        raise Fail(f"no env definition envs/{a.name}/pyproject.toml")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(lab.venv(a.name))}
    args = ["uv", "sync", "--project", str(proj)] + (["--locked"] if (proj / "uv.lock").exists() and not a.relock else [])
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
    if r.returncode:
        raise Fail("uv sync failed", log=tail)
    return {"env": a.name, "venv": str(lab.venv(a.name)), "log": tail}


def cmd_model_pull(lab, a):
    """Download a pinned snapshot and make it read-only. Interventions must write elsewhere."""
    py = lab.venv("research") / "bin" / "python"
    if not py.exists():
        raise Fail("research env missing: researchctl env sync research")
    code = (
        "import sys,json;from huggingface_hub import HfApi,snapshot_download as s;"
        "i=HfApi().model_info(sys.argv[1],revision=sys.argv[2]);"
        "p=s(sys.argv[1],revision=i.sha,local_dir=sys.argv[3]+'@'+i.sha[:12]);"
        "print(json.dumps({'sha':i.sha,'path':p}))"
    )
    base = lab.lab_root / "models" / a.model_id.replace("/", "__")
    base.parent.mkdir(parents=True, exist_ok=True)
    out = sh([str(py), "-c", code, a.model_id, a.revision, str(base)], timeout=6 * 3600)
    import json
    info = json.loads(out.strip().splitlines()[-1])
    subprocess.run(["chmod", "-R", "a-w", info["path"]], check=True)
    return {"model": a.model_id, "revision": info["sha"], "path": info["path"], "read_only": True}


def cmd_doctor(lab, a):
    checks = {}
    for b in ("sbatch", "squeue", "sacct", "scontrol", "git", "uv", "claude"):
        checks[b] = bool(shutil.which(b))
    checks["lab_root_writable"] = os.access(lab.lab_root, os.W_OK) if lab.lab_root.exists() else False
    checks["venvs"] = {e: lab.venv(e).exists() for e in lab.cfg["envs"]}
    checks["hooks_installed"] = (lab.root / ".claude" / "settings.json").exists()
    checks["git_dirty_files"] = len(git_dirty(lab.root))
    checks["s2_api_key"] = bool(os.environ.get("S2_API_KEY"))
    try:
        u = approvals.month_usage(lab)
        checks["h200_month"] = {k: u[k] for k in ("gpu_hours", "budget", "unapproved_jobs")}
    except Fail as e:
        checks["h200_month"] = f"unavailable: {e}"
    problems = [k for k, v in checks.items() if v is False]
    if isinstance(checks["h200_month"], dict) and checks["h200_month"]["unapproved_jobs"]:
        problems.append("UNAPPROVED H200 JOBS")
    return {"ok": not problems, "problems": problems, "checks": checks}


def cmd_digest(lab, a):
    """Session-start context: what is running, what changed, what is next. Kept under ~25 lines."""
    out = {}
    try:
        mine = slurm.queue(all_users=False)
        out["my_jobs"] = [f"{j['job']} {j['name']} {j['state']} {j['elapsed']}" for j in mine[:8]]
    except Fail as e:
        out["my_jobs"] = f"unavailable: {e}"
    recs = list(lab.experiment_records())
    open_recs = [load_yaml(p) for p in recs]
    out["uncollected"] = [r["exp_id"] for r in open_recs if r.get("status") in ("submitted", "running", "completed")
                          and not r.get("results")][-8:]
    projects = []
    for pd in sorted((lab.root / "projects").glob("*/PROJECT.md")):
        stage = next((l.split(":", 1)[1].split("<")[0].strip() for l in pd.read_text().splitlines()
                      if l.startswith("stage:")), "?")
        projects.append(f"{pd.parent.name}: {stage}")
    out["projects"] = projects
    out["pending_h200_requests"] = [p.stem for p in lab.approvals("requests").glob("*.json")]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="researchctl", description=__doc__.splitlines()[0])
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        q = sp.add_parser(name, help=help_)
        q.set_defaults(fn=fn)
        return q

    add("doctor", cmd_doctor, "health check of tools, envs, hooks, H200 audit")
    add("digest", cmd_digest, "compact session-start status")
    add("gpu-status", cmd_gpu_status, "free GPU slices per type, queue pressure")
    q = add("queue", cmd_queue, "squeue as JSON"); q.add_argument("--all", action="store_true"); q.add_argument("--limit", type=int, default=30)

    q = add("submit", lambda lab, a: submit.submit(lab, Path(a.config), a.dry_run, a.rerun, a.cost_reason, a.approval),
            "validate gates and submit an experiment config")
    q.add_argument("config"); q.add_argument("--dry-run", action="store_true")
    q.add_argument("--rerun", metavar="REASON"); q.add_argument("--cost-reason"); q.add_argument("--approval")
    q = add("status", lambda lab, a: runs.status(lab, a.exp), "task states, failures, memory"); q.add_argument("exp")
    q = add("wait", lambda lab, a: runs.wait(lab, a.exp, a.timeout), "block until finished (no polling by the agent)")
    q.add_argument("exp"); q.add_argument("--timeout", type=float, default=240, help="minutes")
    q = add("inspect", lambda lab, a: runs.inspect(lab, a.target, a.tail, a.grep), "EXP or EXP/TASK: summary, metrics, error excerpt")
    q.add_argument("target"); q.add_argument("--tail", type=int, default=0); q.add_argument("--grep")
    q = add("collect", lambda lab, a: runs.collect(lab, a.exp), "aggregate results into the experiment record"); q.add_argument("exp")
    q = add("retry", lambda lab, a: runs.retry(lab, a.exp, a.reason), "resubmit failed tasks only")
    q.add_argument("exp"); q.add_argument("--reason", required=True)
    q = add("cancel", lambda lab, a: runs.cancel(lab, a.exp), "cancel all jobs of an experiment"); q.add_argument("exp")
    q = add("ls", lambda lab, a: runs.ls(lab, a.project, a.limit), "list experiments"); q.add_argument("--project"); q.add_argument("--limit", type=int, default=20)
    q = add("exec", cmd_exec, "short srun on a slice, output truncated")
    q.add_argument("--profile", default="s24"); q.add_argument("--minutes", type=int, default=10)
    q.add_argument("--tail", type=int, default=40); q.add_argument("cmd", nargs=argparse.REMAINDER)

    q = add("request-h200", lambda lab, a: approvals.request(lab, Path(a.config), a.why), "ask the user for H200 time")
    q.add_argument("config"); q.add_argument("--why", required=True)
    q = add("approve", lambda lab, a: approvals.approve(lab, a.request), "HUMAN ONLY: approve an H200 request"); q.add_argument("request")
    add("budget", lambda lab, a: approvals.month_usage(lab), "H200 GPU-hours this month + unapproved-job audit")

    q = add("new-project", cmd_new_project, "create a project skeleton"); q.add_argument("slug")
    q = add("prereg", cmd_prereg, "create the next preregistration template"); q.add_argument("project")
    q = add("gate", cmd_gate, "stage-gate status for a project"); q.add_argument("project")
    q = add("claims", lambda lab, a: claims.check(lab, a.project), "recompute claim statistics, verdicts, macros")
    q.add_argument("action", choices=["check"]); q.add_argument("project")
    q = add("lint", lambda lab, a: claims.lint_prose([Path(x) for x in a.paths], Path(a.bib) if a.bib else None),
            "prose/paper lint: hype, stock phrases, literal numbers, missing cites")
    q.add_argument("paths", nargs="+"); q.add_argument("--bib")

    q = add("env", cmd_env, "sync a uv environment into $LAB_ROOT/venvs")
    q.add_argument("action", choices=["sync"]); q.add_argument("name"); q.add_argument("--relock", action="store_true")
    q = add("model-pull", cmd_model_pull, "download a pinned HF snapshot, read-only")
    q.add_argument("model_id"); q.add_argument("--revision", default="main")

    q = add("lit-search", lambda lab, a: lit.search(lab, a.query, a.limit, a.source, a.year, a.project), "search S2 + arXiv")
    q.add_argument("query"); q.add_argument("--limit", type=int, default=15); q.add_argument("--source", default="all", choices=["all", "s2", "arxiv"])
    q.add_argument("--year", help="e.g. 2024- or 2022-2025"); q.add_argument("--project", help="log the query to the project's audit trail")
    q = add("lit-refs", lambda lab, a: lit.graph(a.id, "references", a.limit), "papers this one cites"); q.add_argument("id"); q.add_argument("--limit", type=int, default=40)
    q = add("lit-citers", lambda lab, a: lit.graph(a.id, "citations", a.limit), "papers citing this one"); q.add_argument("id"); q.add_argument("--limit", type=int, default=40)
    q = add("lit-cite", lambda lab, a: lit.cite(a.id), "verified BibTeX from arXiv/Crossref"); q.add_argument("id")
    q = add("lit-fetch", lambda lab, a: lit.fetch(lab, a.id), "PDF -> text on scratch + section index"); q.add_argument("id")
    q = add("lit-text", lambda lab, a: lit.text_index(Path(a.pdf)), "index a local PDF"); q.add_argument("pdf")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        lab = Lab()
        emit(a.fn(lab, a))
        return 0
    except Fail as e:
        emit({"error": str(e), **e.extra})
        return 1
    except KeyboardInterrupt:
        emit({"error": "interrupted"})
        return 130
    except Exception as e:  # unexpected: still answer in one JSON line, never a wall of traceback
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        emit({"error": f"internal: {type(e).__name__}: {str(e)[:300]}", "at": f"{Path(tb.filename).name}:{tb.lineno}"})
        return 2


if __name__ == "__main__":
    sys.exit(main())
