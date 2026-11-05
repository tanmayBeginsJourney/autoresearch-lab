import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab_repo(tmp_path, monkeypatch):
    """A throwaway git repo with lab.yaml + one project, LAB_ROOT in tmp, and a fake sbatch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = yaml.safe_load((ROOT / "lab.yaml").read_text())
    cfg["policy"]["gates"].update(min_paper_cards=1, min_full_read_cards=1)
    (repo / "lab.yaml").write_text(yaml.safe_dump(cfg))
    p = repo / "projects" / "demo"
    for d in ("lit/cards", "prereg", "configs", "experiments", "src", "repro/base"):
        (p / d).mkdir(parents=True)
    (p / "src" / "train.py").write_text("print('hi')\n")
    (p / "lit/cards/x.yaml").write_text(yaml.safe_dump({"id": "arxiv:1", "title": "t", "year": 2025, "depth": "full",
        "question": "q", "claims": [{"text": "c"}], "skeptic": "s", "relevance": "r"}))
    (p / "repro/base/REPORT.md").write_text("# repro\n")
    (p / "gaps.md").write_text("# gaps\n## Decision\ngo\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sbatch").write_text("#!/bin/sh\necho \"$@\" >> \"$FAKE_SBATCH_LOG\"\necho 4242\n")
    (bindir / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SBATCH_LOG", str(tmp_path / "sbatch.log"))
    monkeypatch.setenv("RESEARCHCTL_REPO", str(repo))
    monkeypatch.setenv("LAB_ROOT", str(tmp_path / "lab"))
    for c in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"], ["git", "config", "user.name", "t"],
              ["git", "add", "-A"], ["git", "commit", "-qm", "init"]):
        subprocess.run(c, cwd=repo, check=True)
    return repo


def commit(repo, msg="c"):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=repo, check=True)


def write_exp(repo, name="sweep", **over):
    exp = {"name": name, "project": "demo", "kind": "exploratory", "env": "research",
           "entry": "projects/demo/src/train.py", "profile": "s24", "time": "01:00:00", "seeds": [0, 1, 2],
           "grid": {"act": ["gelu", "relu"]}, "params": {"lr": 0.001},
           "primary_metric": {"name": "val_loss", "goal": "min"}}
    exp.update(over)
    path = repo / "projects/demo/configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(exp))
    return path
