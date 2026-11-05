"""Experiment validation gates, rendering, and submission."""
from __future__ import annotations

import datetime as dt
import itertools
import json
import re
import shlex
import subprocess
from pathlib import Path

from . import gates, slurm
from .core import (Fail, Lab, canon_hash, dump_yaml, gres_gpus, git_dirty, git_sha, git_tracked_clean,
                   load_yaml, now, sha256_file, walltime_hours)

KINDS = ("smoke", "exploratory", "confirmatory")
REQUIRED = ("name", "project", "kind", "env", "entry", "profile", "time", "seeds", "primary_metric")
DEAD = ("failed", "cancelled")


def load_experiment(path: Path) -> dict:
    exp = load_yaml(path)
    if not isinstance(exp, dict):
        raise Fail(f"{path} is not a mapping")
    exp.setdefault("grid", {})
    exp.setdefault("params", {})
    exp.setdefault("eval", {"splits": ["val"]})
    return exp


def expand(exp: dict) -> list[dict]:
    """Cartesian product of grid x seeds -> ordered task list (seed varies fastest)."""
    keys = sorted(exp["grid"])
    points = [dict(zip(keys, vals)) for vals in itertools.product(*(exp["grid"][k] for k in keys))] or [{}]
    return [{"point": p, "seed": s} for p in points for s in exp["seeds"]]


def validate(lab: Lab, exp: dict, cfg_path: Path) -> tuple[list[str], dict]:
    """Return (errors, facts). Errors block submission; facts go into the record."""
    errs: list[str] = []
    facts: dict = {}
    for k in REQUIRED:
        if k not in exp:
            errs.append(f"missing field: {k}")
    if errs:
        return errs, facts

    kind = exp["kind"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", str(exp["name"])):
        errs.append("name must match [a-z0-9][a-z0-9-]{1,40}")
    if not (lab.root / cfg_path).resolve().is_relative_to(lab.root):
        errs.append("config must live inside the repo")
    if kind not in KINDS:
        errs.append(f"kind must be one of {KINDS}")
    pdir = lab.project_dir(exp["project"])
    if not pdir.is_dir():
        errs.append(f"unknown project {exp['project']} (researchctl new-project)")
    if not (lab.root / exp["entry"]).is_file():
        errs.append(f"entry not found: {exp['entry']}")
    if exp["env"] not in lab.cfg["envs"]:
        errs.append(f"env must be one of {lab.cfg['envs']}")
    prof = lab.profiles.get(exp["profile"])
    if not prof:
        errs.append(f"profile must be one of {list(lab.profiles)}")
    if not isinstance(exp["seeds"], list) or not all(isinstance(s, int) for s in exp["seeds"]) or not exp["seeds"]:
        errs.append("seeds must be a non-empty list of ints")
    pm = exp["primary_metric"]
    if not (isinstance(pm, dict) and pm.get("name") and pm.get("goal") in ("min", "max")):
        errs.append("primary_metric needs {name, goal: min|max}")
    if errs:
        return errs, facts

    hours = walltime_hours(exp["time"])
    if kind == "smoke" and hours * 60 > lab.policy["smoke_max_minutes"]:
        errs.append(f"smoke runs are capped at {lab.policy['smoke_max_minutes']} min")
    if hours > walltime_hours(lab.cfg["cluster"]["max_walltime"]):
        errs.append("time exceeds cluster max walltime")

    for key in ("model", "data"):
        spec = exp.get(key)
        if spec is None:
            continue
        if not spec.get("id") or not spec.get("revision"):
            errs.append(f"{key} needs id and revision")
        elif kind == "confirmatory" and spec["revision"] in ("main", "master", "latest"):
            errs.append(f"confirmatory runs must pin {key}.revision to a commit hash, not '{spec['revision']}'")

    dirty = git_dirty(lab.root)
    facts["dirty"] = bool(dirty)
    if dirty and kind != "smoke":
        errs.append(f"working tree has uncommitted changes ({', '.join(dirty[:5])}); commit first so the run is reproducible")

    if kind == "confirmatory":
        errs += _confirmatory_checks(lab, exp, pdir, facts)

    tasks = expand(exp)
    gpus = gres_gpus(prof.get("gres"))
    facts["n_tasks"] = len(tasks)
    facts["gpu_hours_max"] = round(len(tasks) * hours * max(gpus, 0), 2)
    facts["cpu_hours_max"] = round(len(tasks) * hours * prof["cpus"], 1)
    return errs, facts


def _confirmatory_checks(lab: Lab, exp: dict, pdir: Path, facts: dict) -> list[str]:
    errs = []
    h = exp.get("hypothesis")
    if not h:
        return ["confirmatory runs must name a preregistered hypothesis"]
    pre = pdir / "prereg" / f"{h}.yaml"
    if not pre.is_file():
        return [f"no preregistration {pre.relative_to(lab.root)}"]
    if not git_tracked_clean(lab.root, pre):
        errs.append(f"{pre.name} must be committed (and unmodified) before confirmatory runs")
    from .prereg import validate_prereg
    reg = load_yaml(pre)
    errs += [f"prereg {h}: {e}" for e in validate_prereg(reg)]
    if reg and reg.get("primary_metric") and reg["primary_metric"] != exp["primary_metric"]["name"]:
        errs.append(f"primary_metric {exp['primary_metric']['name']} differs from prereg ({reg['primary_metric']})")
    if len(exp["seeds"]) < lab.policy["min_seeds_confirmatory"]:
        errs.append(f"confirmatory runs need >= {lab.policy['min_seeds_confirmatory']} seeds")
    if "test" in exp["eval"].get("splits", []):
        if not (reg or {}).get("final_test"):
            errs.append("test split requested but prereg does not declare final_test: true")
        prior = [p.stem for p in lab.experiment_records()
                 if (r := load_yaml(p)).get("hypothesis") == h and "test" in r.get("eval", {}).get("splits", [])
                 and r.get("status") not in DEAD]
        if prior:
            errs.append(f"test split already used for {h} by {prior}; one shot per hypothesis")
    missing = gates.missing_for_confirmatory(lab, pdir)
    errs += [f"gate: {m}" for m in missing]
    facts["prereg_sha256"] = sha256_file(pre)
    return errs


def config_hash(exp: dict, sha: str) -> str:
    core = {k: v for k, v in exp.items() if k not in ("name", "notes")}
    return canon_hash({"exp": core, "git": sha})


def find_duplicate(lab: Lab, chash: str) -> str | None:
    for p in lab.experiment_records():
        r = load_yaml(p)
        if r.get("config_hash") == chash and r.get("status") not in DEAD:
            return p.stem
    return None


def submit(lab: Lab, cfg_path: Path, dry_run=False, rerun: str | None = None,
           cost_reason: str | None = None, approval: str | None = None, local: bool = False,
           only_tasks: list[int] | None = None) -> dict:
    """Validate, snapshot code, write the record, then dispatch to Slurm or run locally (sequentially)."""
    exp = load_experiment(cfg_path)
    errs, facts = validate(lab, exp, cfg_path)
    sha = git_sha(lab.root)
    chash = config_hash(exp, sha)
    prof = lab.profiles[exp["profile"]] if not errs else {}

    if not errs:
        dup = find_duplicate(lab, chash)
        if dup and not rerun:
            errs.append(f"identical config+code already submitted as {dup}; pass --rerun REASON to repeat")
    if not errs and not local:  # cluster-only gates
        if facts["gpu_hours_max"] > lab.policy["max_gpu_hours_per_submit"] and not cost_reason:
            errs.append(f"up to {facts['gpu_hours_max']} GPU-h exceeds per-submit cap "
                        f"{lab.policy['max_gpu_hours_per_submit']}; shrink it or pass --cost-reason")
        if prof.get("requires_approval"):
            from .approvals import check_approval
            errs += check_approval(lab, approval, chash, facts["gpu_hours_max"])
    if errs:
        raise Fail("submission refused", errors=errs, facts=facts)

    stamp = dt.datetime.now().strftime("%y%m%d-%H%M")
    exp_id = f"{stamp}-{exp['name']}-{chash[:6]}"
    k = 2
    while (lab.runs_dir() / exp_id).exists() or any(p.stem == exp_id for p in lab.experiment_records()):
        exp_id = f"{stamp}-{exp['name']}-{chash[:6]}-r{k}"
        k += 1
    run_root = lab.runs_dir() / exp_id
    tasks = expand(exp)
    result = {"exp_id": exp_id, "n_tasks": len(tasks), "profile": exp["profile"], "gpu_hours_max": facts["gpu_hours_max"]}
    if dry_run:
        return {"dry_run": True, **result, "script": render_script(lab, exp, exp_id, run_root, sha, len(tasks))}

    _materialize(lab, exp, exp_id, run_root, sha, tasks, facts["dirty"])
    script = run_root / "job.sbatch"
    script.write_text(render_script(lab, exp, exp_id, run_root, sha, len(tasks)))
    job = None if local else slurm.sbatch(script)

    record = {
        "exp_id": exp_id, "name": exp["name"], "project": exp["project"], "kind": exp["kind"],
        "hypothesis": exp.get("hypothesis"), "status": "submitted", "submitted_at": now(),
        "config_file": str(cfg_path.resolve().relative_to(lab.root)), "config_hash": chash,
        "git_sha": sha, "dirty": facts["dirty"], "executor": "local" if local else "slurm",
        "slurm_jobs": [] if local else [job], "run_root": str(run_root),
        "profile": exp["profile"], "time": exp["time"], "env": exp["env"], "entry": exp["entry"],
        "seeds": exp["seeds"], "grid": exp["grid"], "params": exp["params"], "eval": exp["eval"],
        "model": exp.get("model"), "data": exp.get("data"), "primary_metric": exp["primary_metric"],
        "prereg_sha256": facts.get("prereg_sha256"), "cost": {"gpu_hours_max": facts["gpu_hours_max"]},
        "rerun_reason": rerun, "cost_reason": cost_reason, "approval": approval,
        "tasks": [{"i": i, **t} for i, t in enumerate(tasks)],
    }
    rec_path = lab.project_dir(exp["project"]) / "experiments" / f"{exp_id}.yaml"
    dump_yaml(rec_path, record)
    if local:
        from .local import run_tasks
        record["local"] = run_tasks(script, run_root, only_tasks or list(range(len(tasks))),
                                    walltime_hours(exp["time"]) * 3600)
        record["status"] = "completed" if all(t["rc"] == 0 for t in record["local"].values()) else "partial_failure"
        dump_yaml(rec_path, record)
        failed = [i for i, t in record["local"].items() if t["rc"] != 0]
        return {**result, "executor": "local", "status": record["status"], "failed_tasks": failed or None,
                "record": f"projects/{exp['project']}/experiments/{exp_id}.yaml"}
    if approval:
        from .approvals import consume
        consume(lab, approval, exp_id, job)
    return {**result, "job": job, "record": f"projects/{exp['project']}/experiments/{exp_id}.yaml"}


def _materialize(lab: Lab, exp, exp_id, run_root: Path, sha: str, tasks, dirty: bool) -> None:
    code = run_root / "code"
    code.mkdir(parents=True, exist_ok=False)
    # Immutable code snapshot: the job never sees later edits to the repo.
    archive = subprocess.run(["git", "archive", sha], cwd=lab.root, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(code)], input=archive, check=True)
    if dirty:
        patch = subprocess.run(["git", "diff", "HEAD"], cwd=lab.root, capture_output=True, check=True).stdout
        (run_root / "uncommitted.patch").write_bytes(patch)
        if patch:
            subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=code, input=patch, check=True)
    for i, t in enumerate(tasks):
        d = run_root / f"{i:03d}"
        d.mkdir(parents=True)
        params = {
            "exp_id": exp_id, "run_id": f"{exp_id}/{i:03d}", "task": i, "seed": t["seed"], **t["point"],
            "params": exp["params"], "model": exp.get("model"), "data": exp.get("data"),
            "eval": exp["eval"], "primary_metric": exp["primary_metric"], "kind": exp["kind"],
        }
        (d / "params.json").write_text(json.dumps(params, indent=1))


def render_script(lab: Lab, exp, exp_id: str, run_root: Path, sha: str, n: int) -> str:
    prof = lab.profiles[exp["profile"]]
    acct = lab.cfg["cluster"].get("account")
    par = min(lab.policy["max_parallel_tasks"], n)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rc.{exp_id}",
        f"#SBATCH --partition={prof['partition']}",
        *([f"#SBATCH --gres={prof['gres']}"] if prof.get("gres") else []),
        *([f"#SBATCH --account={acct}"] if acct else []),
        f"#SBATCH --cpus-per-task={prof['cpus']}",
        f"#SBATCH --mem={prof['mem']}",
        f"#SBATCH --time={exp['time']}",
        f"#SBATCH --array=0-{n - 1}%{par}",
        f"#SBATCH --output={run_root}/%3a/slurm.log",
        "#SBATCH --signal=B:USR1@180",
        "#SBATCH --requeue",
        "set -uo pipefail",
        'TASK=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")',
        f'export LAB_RUN_DIR="{run_root}/$TASK" LAB_EXP_ID="{exp_id}" LAB_GIT_SHA="{sha}"',
        f'export LAB_ROOT="{lab.lab_root}" HF_HOME="{lab.lab_root}/cache/hf"',
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false",
        f'export PYTHONPATH="{run_root}/code/src:{run_root}/code:${{PYTHONPATH:-}}"',
        f'cd "{run_root}/code"',
        f'"{lab.venv(exp["env"])}/bin/python" -u {shlex.quote(exp["entry"])} --params "$LAB_RUN_DIR/params.json" &',
        "PID=$!",
        "trap 'kill -USR1 $PID 2>/dev/null' USR1",
        "wait $PID; RC=$?",
        "if [ $RC -gt 128 ] && kill -0 $PID 2>/dev/null; then wait $PID; RC=$?; fi",
        "exit $RC",
    ]
    return "\n".join(lines) + "\n"
