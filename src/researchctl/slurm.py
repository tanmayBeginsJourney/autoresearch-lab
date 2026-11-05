"""Thin Slurm wrappers that return parsed, compact structures."""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from .core import sh

ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING"}
FAILED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


def user() -> str:
    return os.environ.get("USER") or sh(["whoami"]).strip()


def sbatch(script: Path, extra: list[str] | None = None) -> str:
    out = sh(["sbatch", "--parsable", *(extra or []), str(script)])
    return out.strip().split(";")[0]


def cancel(job_id: str) -> None:
    sh(["scancel", job_id])


def _kv_line(line: str) -> dict[str, str]:
    d = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def _tres_gpus(tres: str) -> dict[str, int]:
    """'cpu=4,gres/gpu:2g.48gb=1,gres/gpu=1' -> {'2g.48gb': 1}"""
    out: dict[str, int] = {}
    for part in tres.split(","):
        m = re.match(r"gres/gpu:([^=]+)=(\d+)", part)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


def _gres_total(gres: str) -> dict[str, int]:
    """'gpu:2g.48gb:2(S:0)' -> {'2g.48gb': 2}"""
    out: dict[str, int] = {}
    for part in gres.split(","):
        m = re.match(r"gpu:([^:(]+):(\d+)", part)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


def nodes() -> list[dict]:
    rows = []
    for line in sh(["scontrol", "show", "node", "-o"]).splitlines():
        kv = _kv_line(line)
        if not kv.get("NodeName"):
            continue
        total = _gres_total(kv.get("Gres", ""))
        used = _tres_gpus(kv.get("AllocTRES", ""))
        rows.append({
            "node": kv["NodeName"],
            "partitions": kv.get("Partitions", ""),
            "state": kv.get("State", "").lower(),
            "gpus_free": {k: total[k] - used.get(k, 0) for k in total},
            "gpus_total": total,
            "cpus_free": int(kv.get("CPUTot", 0)) - int(kv.get("CPUAlloc", 0)),
            "mem_free_gb": round((int(kv.get("RealMemory", 0)) - int(kv.get("AllocMem", 0))) / 1024),
        })
    return rows


def queue(all_users: bool = True) -> list[dict]:
    cmd = ["squeue", "-h", "-o", "%i|%u|%j|%T|%M|%l|%P|%b|%R"]
    if not all_users:
        cmd += ["-u", user()]
    rows = []
    for line in sh(cmd).splitlines():
        f = line.split("|")
        if len(f) >= 9:
            rows.append(dict(zip(["job", "user", "name", "state", "elapsed", "limit", "partition", "gres", "reason"], f)))
    return rows


def array_tasks(job_id: str) -> dict[int, dict]:
    """Per-array-task state for one array job, merging sacct (started) and squeue -r (pending)."""
    tasks: dict[int, dict] = {}
    fields = "JobID,State,ExitCode,Elapsed,MaxRSS,NodeList"
    out = sh(["sacct", "-j", job_id, "-n", "-P", "-o", fields], check=False)
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 6:
            continue
        jid, state, exitc, elapsed, maxrss, node = f[:6]
        m = re.match(rf"^{re.escape(job_id)}_(\d+)(\.batch)?$", jid)
        if not m:
            continue
        idx = int(m.group(1))
        t = tasks.setdefault(idx, {})
        if m.group(2):  # .batch step carries memory usage
            t["max_rss_gb"] = _rss_gb(maxrss)
        else:
            t.update(state=state.split()[0], exit=exitc, elapsed=elapsed, node=node)
    out = sh(["squeue", "-j", job_id, "-r", "-h", "-o", "%i|%T|%r"], check=False)
    for line in out.splitlines():
        f = line.split("|")
        m = re.match(rf"^{re.escape(job_id)}_(\d+)$", f[0]) if f else None
        if m:
            tasks.setdefault(int(m.group(1)), {}).update(state=f[1], reason=f[2] if len(f) > 2 else "")
    return tasks


def _rss_gb(s: str) -> float | None:
    if not s:
        return None
    mult = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    m = re.match(r"([\d.]+)([KMGT]?)", s)
    if not m:
        return None
    return round(float(m.group(1)) * mult.get(m.group(2) or "K", 1 / 1024**2), 2)


def summarize_states(tasks: dict[int, dict]) -> dict:
    c = Counter(t.get("state", "UNKNOWN") for t in tasks.values())
    if any(s in ACTIVE for s in c):
        overall = "active"
    elif all(s == "COMPLETED" for s in c) and c:
        overall = "completed"
    elif any(s in FAILED or s == "CANCELLED" for s in c):
        overall = "partial_failure" if c.get("COMPLETED") else "failed"
    else:
        overall = "unknown"
    return {"overall": overall, "counts": dict(c)}


def h200_usage(since: str) -> list[dict]:
    """Every allocation on the h200 partition by this user since a date (YYYY-MM-DD)."""
    out = sh(["sacct", "-u", user(), "-S", since, "-r", "h200", "-X", "-n", "-P",
              "-o", "JobID,JobName,ElapsedRaw,AllocTRES,State"], check=False)
    rows = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 5:
            continue
        gpus = sum(_tres_gpus(f[3]).values()) or (1 if "gres/gpu" in f[3] else 0)
        rows.append({"job": f[0], "name": f[1], "gpu_hours": round(int(f[2] or 0) * gpus / 3600, 3), "state": f[4]})
    return rows


def srun_capture(profile: dict, minutes: int, cmd: list[str], account: str | None) -> tuple[int, str]:
    args = ["srun", "-p", profile["partition"], "-t", str(minutes), "-c", str(profile["cpus"]),
            "--mem", str(profile["mem"]), "-J", "rc.exec"]
    if profile.get("gres"):
        args += ["--gres", profile["gres"]]
    if account:
        args += ["-A", account]
    import subprocess
    r = subprocess.run(args + cmd, capture_output=True, text=True, timeout=minutes * 60 + 600)
    return r.returncode, (r.stdout + r.stderr)
