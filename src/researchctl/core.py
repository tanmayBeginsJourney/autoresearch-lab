"""Shared helpers: repo/lab paths, config, JSON output, git, hashing."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


class Fail(Exception):
    """A refusal or error reported to the caller as JSON (exit code 1)."""

    def __init__(self, msg: str, **extra: Any):
        super().__init__(msg)
        self.extra = extra


def emit(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True, timeout: int | None = None) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise Fail(f"command not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        raise Fail(f"timed out after {timeout}s: {' '.join(cmd[:3])}")
    if check and r.returncode != 0:
        raise Fail(f"{cmd[0]} exited {r.returncode}", stderr=r.stderr.strip()[-800:])
    return r.stdout


def repo_root() -> Path:
    env = os.environ.get("RESEARCHCTL_REPO")
    if env:
        return Path(env).resolve()
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail("not inside the lab repo (or set RESEARCHCTL_REPO)")
    return Path(r.stdout.strip())


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise Fail(f"missing file: {path}")
    except yaml.YAMLError as e:
        raise Fail(f"invalid YAML in {path}", detail=str(e)[:300])


def dump_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, width=100))


class Lab:
    """Resolved configuration for one invocation."""

    def __init__(self, root: Path | None = None):
        self.root = root or repo_root()
        self.cfg = load_yaml(self.root / "lab.yaml")
        lr = os.environ.get("LAB_ROOT") or os.path.expandvars(self.cfg["lab_root"])
        if "$" in lr:
            raise Fail("lab_root has an unset variable (is $SCRATCH set?)", lab_root=lr)
        self.lab_root = Path(lr)
        self.policy = self.cfg["policy"]
        self.profiles = self.cfg["profiles"]

    def project_dir(self, slug: str) -> Path:
        return self.root / "projects" / slug

    def runs_dir(self) -> Path:
        return self.lab_root / "runs"

    def venv(self, env: str) -> Path:
        return self.lab_root / "venvs" / env

    def approvals(self, state: str) -> Path:
        p = self.lab_root / "approvals" / state
        p.mkdir(parents=True, exist_ok=True)
        return p

    def experiment_records(self):
        yield from sorted(self.root.glob("projects/*/experiments/*.yaml"))

    def find_record(self, exp_id: str) -> Path:
        hits = [p for p in self.experiment_records() if p.stem == exp_id or p.stem.startswith(exp_id)]
        if not hits:
            raise Fail(f"no experiment record matching {exp_id}")
        if len(hits) > 1:
            raise Fail(f"ambiguous experiment id {exp_id}", matches=[h.stem for h in hits])
        return hits[0]


# ---------- git ----------

def git_sha(root: Path) -> str:
    return sh(["git", "rev-parse", "HEAD"], cwd=root).strip()


def git_dirty(root: Path) -> list[str]:
    out = sh(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return [l[3:] for l in out.splitlines() if l.strip()]


def git_tracked_clean(root: Path, path: Path) -> bool:
    rel = str(path.relative_to(root))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True).returncode == 0
    if not tracked:
        return False
    return sh(["git", "status", "--porcelain", "--", rel], cwd=root).strip() == ""


# ---------- hashing / time ----------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


_TIME = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def walltime_hours(t: str) -> float:
    """Slurm time 'D-HH:MM:SS', 'HH:MM:SS' or 'MM:SS' -> hours."""
    m = _TIME.match(str(t).strip())
    if not m:
        raise Fail(f"bad walltime '{t}' (use HH:MM:SS or D-HH:MM:SS)")
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    if m.group(2) is None:  # MM:SS form
        h, mi, s = 0, int(m.group(3)), int(m.group(4))
    return d * 24 + h + mi / 60 + s / 3600


def gres_gpus(gres: str | None) -> int:
    if not gres:
        return 0
    return int(gres.rsplit(":", 1)[-1])
