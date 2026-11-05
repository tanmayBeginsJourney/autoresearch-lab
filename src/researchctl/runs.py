"""Monitoring and collection: status, wait, inspect, collect, retry, cancel, list."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from . import slurm, stats
from .core import Fail, Lab, dump_yaml, load_yaml, now


def _record(lab: Lab, exp_id: str) -> tuple[Path, dict]:
    p = lab.find_record(exp_id)
    return p, load_yaml(p)


def _tasks_state(rec: dict) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for job in rec["slurm_jobs"]:  # later jobs (retries) override earlier ones
        for i, t in slurm.array_tasks(str(job)).items():
            merged[i] = {**t, "job": job}
    return merged


def status(lab: Lab, exp_id: str) -> dict:
    p, rec = _record(lab, exp_id)
    tasks = _tasks_state(rec)
    summ = slurm.summarize_states(tasks)
    failed = {i: {k: t.get(k) for k in ("state", "exit", "node")} for i, t in tasks.items()
              if t.get("state") in slurm.FAILED or t.get("state") == "CANCELLED"}
    pend_reasons = sorted({t.get("reason") for t in tasks.values() if t.get("state") == "PENDING" and t.get("reason")})
    rss = [t["max_rss_gb"] for t in tasks.values() if t.get("max_rss_gb")]
    if summ["overall"] != rec.get("status") and summ["overall"] != "unknown":
        rec["status"] = {"active": "running"}.get(summ["overall"], summ["overall"])
        dump_yaml(p, rec)
    return {"exp_id": rec["exp_id"], "status": rec["status"], **summ, "failed": failed or None,
            "pending_reasons": pend_reasons or None, "max_rss_gb": max(rss) if rss else None,
            "n_tasks": len(rec["tasks"])}


def wait(lab: Lab, exp_id: str, timeout_min: float, poll_s: int = 60) -> dict:
    """Block until no task is active (or timeout). Costs no model tokens while waiting."""
    deadline = time.time() + timeout_min * 60
    while True:
        s = status(lab, exp_id)
        if s["overall"] != "active":
            return {**s, "waited": True}
        if time.time() > deadline:
            return {**s, "waited": True, "timeout": True}
        time.sleep(poll_s)


_ERR = re.compile(r"(Traceback|Error|error:|CUDA out of memory|Killed|OOM|nan|NaN|Segmentation fault)")


def inspect(lab: Lab, target: str, tail: int = 0, grep: str | None = None, max_chars: int = 2500) -> dict:
    exp_id, _, task = target.partition("/")
    _, rec = _record(lab, exp_id)
    run_root = Path(rec["run_root"])
    if not task:  # experiment-level overview: one line per task
        rows = []
        for t in rec["tasks"]:
            s = _summary(run_root / f"{t['i']:03d}")
            rows.append({"i": t["i"], **t["point"], "seed": t["seed"], "status": s.get("status"),
                         rec["primary_metric"]["name"]: s.get("final", {}).get(rec["primary_metric"]["name"])})
        return {"exp_id": rec["exp_id"], "tasks": rows}
    d = run_root / f"{int(task):03d}"
    out: dict = {"run": f"{rec['exp_id']}/{int(task):03d}", "dir": str(d), "summary": _summary(d)}
    mfile = d / "metrics.jsonl"
    if mfile.exists():
        lines = mfile.read_text().splitlines()
        out["metrics_lines"] = len(lines)
        out["last_metrics"] = [json.loads(l) for l in lines[-3:]]
    log = d / "slurm.log"
    if log.exists():
        text = log.read_text(errors="replace").splitlines()
        if grep:
            rx = re.compile(grep)
            sel = [l for l in text if rx.search(l)][-40:]
        elif tail:
            sel = text[-tail:]
        else:  # default: error excerpt only
            idx = next((i for i, l in enumerate(text) if "Traceback" in l), None)
            sel = text[idx:][-30:] if idx is not None else [l for l in text if _ERR.search(l)][-10:]
        out["log"] = "\n".join(sel)[-max_chars:]
        out["log_lines_total"] = len(text)
    return out


def _summary(d: Path) -> dict:
    f = d / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


def collect(lab: Lab, exp_id: str) -> dict:
    """Aggregate per-task summaries into the experiment record (the durable, git-tracked result)."""
    p, rec = _record(lab, exp_id)
    st = status(lab, exp_id)
    rec = load_yaml(p)
    pm, goal = rec["primary_metric"]["name"], rec["primary_metric"]["goal"]
    vram = lab.profiles[rec["profile"]].get("vram_gb") or 0
    run_root = Path(rec["run_root"])
    groups: dict[str, list] = defaultdict(list)
    flags, per_task = [], []
    for t in rec["tasks"]:
        s = _summary(run_root / f"{t['i']:03d}")
        final = s.get("final", {})
        per_task.append({"i": t["i"], "status": s.get("status", "missing"), "final": final,
                         "wall_s": s.get("wall_s"), "peak_mem_gb": s.get("peak_mem_gb")})
        if s.get("status") == "completed" and pm in final:
            groups[json.dumps(t["point"], sort_keys=True)].append((t["seed"], final[pm]))
        if s.get("diverged"):
            flags.append(f"task {t['i']} diverged (non-finite metric)")
        if vram and s.get("peak_mem_gb") and s["peak_mem_gb"] < lab.policy["underuse_mem_fraction"] * vram:
            flags.append(f"task {t['i']} used {s['peak_mem_gb']} GB of {vram} GB: use a smaller profile")
    table = []
    for key, vals in groups.items():
        d = stats.describe([v for _, v in vals])
        table.append({"point": json.loads(key), "seeds": [s for s, _ in vals], pm: d})
    table.sort(key=lambda r: r[pm].get("mean", 0), reverse=(goal == "max"))
    rec["results"] = {"collected_at": now(), "primary_metric": pm, "by_point": table, "tasks": per_task}
    rec["status"] = st["status"]
    dump_yaml(p, rec)
    done = sum(1 for t in per_task if t["status"] == "completed")
    return {"exp_id": rec["exp_id"], "status": rec["status"], "completed": f"{done}/{len(per_task)}",
            "ranking": [{**r["point"], "mean": round(r[pm].get("mean", float("nan")), 5),
                         "ci": [round(c, 5) for c in r[pm].get("ci", [])], "n": r[pm]["n"]} for r in table[:12]],
            "flags": sorted(set(flags))[:10] or None}


def retry(lab: Lab, exp_id: str, reason: str) -> dict:
    p, rec = _record(lab, exp_id)
    tasks = _tasks_state(rec)
    bad = sorted(i for i, t in tasks.items() if t.get("state") in slurm.FAILED or t.get("state") == "CANCELLED")
    if not bad:
        raise Fail("no failed tasks to retry")
    script = Path(rec["run_root"]) / "job.sbatch"
    job = slurm.sbatch(script, ["--array=" + ",".join(map(str, bad))])
    rec["slurm_jobs"].append(job)
    rec.setdefault("retries", []).append({"job": job, "tasks": bad, "reason": reason, "at": now()})
    rec["status"] = "submitted"
    dump_yaml(p, rec)
    return {"exp_id": rec["exp_id"], "job": job, "retried": bad}


def cancel(lab: Lab, exp_id: str) -> dict:
    p, rec = _record(lab, exp_id)
    for j in rec["slurm_jobs"]:
        slurm.cancel(str(j))
    rec["status"] = "cancelled"
    dump_yaml(p, rec)
    return {"exp_id": rec["exp_id"], "cancelled": rec["slurm_jobs"]}


def ls(lab: Lab, project: str | None, limit: int) -> dict:
    rows = []
    for p in lab.experiment_records():
        if project and p.parent.parent.name != project:
            continue
        r = load_yaml(p)
        rows.append({"exp_id": r["exp_id"], "kind": r["kind"], "status": r.get("status"),
                     "h": r.get("hypothesis"), "tasks": len(r.get("tasks", []))})
    return {"experiments": rows[-limit:], "total": len(rows)}
