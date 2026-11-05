"""Run harness used inside Slurm jobs.

    import labkit
    with labkit.init() as run:            # reads --params, seeds RNGs, writes manifest.json
        for step in ...:
            run.log(step, loss=..., tok_per_s=...)
            if run.should_stop:           # SIGUSR1 shortly before the time limit
                run.save({"model": m.state_dict(), "step": step}, step); run.requeue()
        run.finish(val_loss=..., acc=...)  # summary.json: final metrics, peak memory, wall time

Contract with researchctl: files in $LAB_RUN_DIR are params.json (input), manifest.json,
metrics.jsonl, summary.json, ckpt/ (outputs). Never write into $LAB_ROOT/models (read-only originals).
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _params_path(argv: list[str]) -> Path:
    if "--params" in argv:
        return Path(argv[argv.index("--params") + 1])
    d = os.environ.get("LAB_RUN_DIR")
    if d:
        return Path(d) / "params.json"
    raise SystemExit("labkit: pass --params PATH or set LAB_RUN_DIR")


def _hardware() -> dict:
    hw = {"host": socket.gethostname(), "python": platform.python_version(),
          "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
          "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        hw.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            hw["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            hw["gpu_mem_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                                for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        hw["driver"] = out.splitlines()[0] if out else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return hw


def _packages() -> dict:
    from importlib import metadata
    names = ("torch", "transformers", "datasets", "accelerate", "peft", "transformer-lens", "nnsight", "triton", "numpy")
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


class Run:
    def __init__(self, params_path: Path | None = None, keep_ckpts: int = 2):
        p = Path(params_path) if params_path else _params_path(sys.argv)
        self.dir = p.parent
        self.params: dict = json.loads(p.read_text())
        self.seed: int = int(self.params.get("seed", 0))
        self.keep_ckpts = keep_ckpts
        self.should_stop = False
        self.diverged = False
        self._t0 = time.time()
        self._last: dict = {}
        self._finished = False
        seed_everything(self.seed, bool(self.params.get("params", {}).get("deterministic", False)))
        signal.signal(signal.SIGUSR1, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self._metrics = (self.dir / "metrics.jsonl").open("a", buffering=1)
        manifest = {"run_id": self.params.get("run_id"), "git_sha": os.environ.get("LAB_GIT_SHA"),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
                    "hardware": _hardware(), "packages": _packages(), "restarts": self._restarts()}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # -- params access -------------------------------------------------------
    def __getitem__(self, k):
        if k in self.params:
            return self.params[k]
        return self.params["params"][k]

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    # -- lifecycle -------------------------------------------------------------
    def _restarts(self) -> int:
        f = self.dir / "manifest.json"
        if f.exists():
            try:
                return json.loads(f.read_text()).get("restarts", 0) + 1
            except json.JSONDecodeError:
                return 1
        return 0

    def _on_signal(self, signum, frame):
        self.should_stop = True

    def log(self, step: int, **metrics) -> None:
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                self.diverged = True
        rec = {"step": step, "t": round(time.time() - self._t0, 2), **metrics}
        self._last = rec
        self._metrics.write(json.dumps(rec) + "\n")

    @property
    def ckpt_dir(self) -> Path:
        d = self.dir / "ckpt"
        d.mkdir(exist_ok=True)
        return d

    def save(self, state: dict, step: int) -> Path:
        import torch
        tmp = self.ckpt_dir / f".step_{step:09d}.pt.tmp"
        dst = self.ckpt_dir / f"step_{step:09d}.pt"
        torch.save(state, tmp)
        tmp.rename(dst)  # atomic: a preempted save never leaves a corrupt "latest"
        for old in sorted(self.ckpt_dir.glob("step_*.pt"))[: -self.keep_ckpts]:
            old.unlink()
        return dst

    def load_latest(self, map_location="cpu"):
        ck = sorted(self.ckpt_dir.glob("step_*.pt"))
        if not ck:
            return None
        import torch
        return torch.load(ck[-1], map_location=map_location, weights_only=False)

    def requeue(self) -> None:
        """Ask Slurm to requeue this array task; resume via load_latest()."""
        job = os.environ.get("SLURM_ARRAY_JOB_ID")
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        jid = f"{job}_{task}" if job and task else os.environ.get("SLURM_JOB_ID")
        self._write_summary("requeued", {})
        if jid:
            subprocess.run(["scontrol", "requeue", jid], check=False)
        sys.exit(0)

    def _peak_mem_gb(self):
        try:
            import torch
            if torch.cuda.is_available():
                return round(max(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / 2**30, 2)
        except ImportError:
            pass
        return None

    def _write_summary(self, status: str, final: dict, error: str | None = None) -> None:
        s = {"status": status, "final": final, "last_logged": self._last, "diverged": self.diverged,
             "wall_s": round(time.time() - self._t0, 1), "peak_mem_gb": self._peak_mem_gb(), "seed": self.seed}
        if error:
            s["error"] = error
        (self.dir / "summary.json").write_text(json.dumps(s, indent=1, default=float))

    def finish(self, **final) -> None:
        bad = [k for k, v in final.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            self.diverged = True
        self._write_summary("diverged" if self.diverged else "completed", final)
        self._finished = True
        self._metrics.close()

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None and et is not SystemExit:
            self._write_summary("failed", {}, error=f"{et.__name__}: {str(ev)[:300]}")
        elif not self._finished and et is None:
            self._write_summary("incomplete", {}, error="run exited without run.finish()")
        return False


def init(params_path: str | None = None, **kw) -> Run:
    return Run(Path(params_path) if params_path else None, **kw)
