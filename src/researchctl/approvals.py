"""H200 approval workflow. The agent may request; only the human approves (enforced by hook + audit)."""
from __future__ import annotations

import datetime as dt
import json

from . import slurm
from .core import Fail, Lab, canon_hash, git_sha, now


def _read(path):
    return json.loads(path.read_text())


def request(lab: Lab, cfg_path, why: str) -> dict:
    from .submit import config_hash, load_experiment, validate
    exp = load_experiment(cfg_path)
    if not lab.profiles.get(exp.get("profile"), {}).get("requires_approval"):
        raise Fail(f"profile {exp.get('profile')} does not need approval")
    if len(why.split()) < 15:
        raise Fail("--why must explain (>=15 words) why slices cannot answer this and what decision it informs")
    errs, facts = validate(lab, exp, cfg_path)
    if errs:
        raise Fail("config would be refused anyway; fix first", errors=errs)
    chash = config_hash(exp, git_sha(lab.root))
    used = month_usage(lab)["gpu_hours"]
    rid = canon_hash({"c": chash, "t": now()}, 8)
    req = {"id": rid, "config": str(cfg_path), "config_hash": chash, "gpu_hours_max": facts["gpu_hours_max"],
           "why": why, "requested_at": now(), "month_used_gpu_hours": used,
           "month_budget": lab.policy["h200_gpu_hours_per_month"]}
    (lab.approvals("requests") / f"{rid}.json").write_text(json.dumps(req, indent=1))
    return {"request": rid, "gpu_hours_max": req["gpu_hours_max"], "month_used": used,
            "next": "STOP. Ask the user to approve: show them this request's why, cost and remaining budget."}


def approve(lab: Lab, rid: str) -> dict:
    src = lab.approvals("requests") / f"{rid}.json"
    if not src.exists():
        raise Fail(f"no pending request {rid}")
    req = _read(src)
    used = month_usage(lab)["gpu_hours"]
    if used + req["gpu_hours_max"] > lab.policy["h200_gpu_hours_per_month"]:
        raise Fail("would exceed monthly H200 budget", used=used, requested=req["gpu_hours_max"])
    req["approved_at"] = now()
    (lab.approvals("approved") / f"{rid}.json").write_text(json.dumps(req, indent=1))
    src.unlink()
    return {"approved": rid, "gpu_hours_max": req["gpu_hours_max"]}


def check_approval(lab: Lab, rid: str | None, chash: str, gpu_hours: float) -> list[str]:
    if not rid:
        return ["H200 profile requires --approval ID (researchctl request-h200, then the user approves)"]
    p = lab.approvals("approved") / f"{rid}.json"
    if not p.exists():
        return [f"approval {rid} not found or already used"]
    a = _read(p)
    errs = []
    if a["config_hash"] != chash:
        errs.append("approval was granted for a different config/code version")
    if gpu_hours > a["gpu_hours_max"] + 1e-9:
        errs.append("requested GPU-hours exceed the approved amount")
    return errs


def consume(lab: Lab, rid: str, exp_id: str, job: str) -> None:
    p = lab.approvals("approved") / f"{rid}.json"
    a = _read(p)
    a.update(consumed_at=now(), exp_id=exp_id, job=job)
    (lab.approvals("consumed") / f"{rid}.json").write_text(json.dumps(a, indent=1))
    p.unlink()


def month_usage(lab: Lab) -> dict:
    start = dt.date.today().replace(day=1).isoformat()
    rows = slurm.h200_usage(start)
    approved_jobs = {_read(p).get("job") for p in lab.approvals("consumed").glob("*.json")}
    unapproved = [r for r in rows if r["job"].split("_")[0] not in approved_jobs]
    return {"since": start, "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 2),
            "budget": lab.policy["h200_gpu_hours_per_month"], "jobs": len(rows),
            "unapproved_jobs": [r["job"] for r in unapproved]}
