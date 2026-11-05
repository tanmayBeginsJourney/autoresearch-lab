"""Local executor: run an experiment's tasks sequentially on this machine (laptop GPU, dev container).

Uses the same rendered job script as Slurm, so the run directory layout, labkit contract and
experiment record are identical. #SBATCH lines are comments to bash.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def run_tasks(script: Path, run_root: Path, tasks: list[int], timeout_s: float) -> dict:
    out = {}
    for i in tasks:
        d = run_root / f"{i:03d}"
        env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}
        for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID"):
            env.pop(k, None)
        t0 = time.time()
        with (d / "slurm.log").open("w") as log:
            try:
                rc = subprocess.run(["bash", str(script)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        out[i] = {"rc": rc, "wall_s": round(time.time() - t0, 1)}
    return out


STATE = {"completed": "COMPLETED", "diverged": "FAILED", "failed": "FAILED", "incomplete": "FAILED",
         "requeued": "PENDING"}


def task_states(rec: dict) -> dict[int, dict]:
    """Slurm-like per-task states for a local experiment, from summary.json and exit codes."""
    import json
    res = {}
    local = {int(k): v for k, v in (rec.get("local") or {}).items()}
    for t in rec["tasks"]:
        i = t["i"]
        f = Path(rec["run_root"]) / f"{i:03d}" / "summary.json"
        s = json.loads(f.read_text()).get("status") if f.exists() else None
        rc = local.get(i, {}).get("rc")
        state = STATE.get(s) or ("TIMEOUT" if rc == 124 else "FAILED" if rc not in (None, 0) else
                                 "COMPLETED" if rc == 0 else "NOT_RUN")
        res[i] = {"state": state, "exit": f"{rc}:0" if rc is not None else None, "node": "local"}
    return res
