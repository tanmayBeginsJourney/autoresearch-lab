import yaml
from pathlib import Path
import pytest

from conftest import commit, write_exp
from researchctl import submit
from researchctl.core import Fail, Lab, walltime_hours


def test_walltime():
    assert walltime_hours("02:00:00") == 2
    assert walltime_hours("30:00") == 0.5
    assert walltime_hours("1-12:00:00") == 36


def test_exploratory_submit_writes_record_and_snapshot(lab_repo):
    cfg = write_exp(lab_repo)
    commit(lab_repo)
    out = submit.submit(Lab(), cfg)
    assert out["job"] == "4242" and out["n_tasks"] == 6
    rec = yaml.safe_load((lab_repo / out["record"]).read_text())
    assert rec["status"] == "submitted" and len(rec["tasks"]) == 6
    run_root = Lab().runs_dir() / out["exp_id"]
    assert (run_root / "code/projects/demo/src/train.py").exists()
    script = (run_root / "job.sbatch").read_text()
    assert "--gres=gpu:1g.24gb:1" in script and "--array=0-5%4" in script


def test_duplicate_refused_then_rerun_allowed(lab_repo, monkeypatch):
    cfg = write_exp(lab_repo)
    commit(lab_repo)
    submit.submit(Lab(), cfg)
    with pytest.raises(Fail) as e:
        submit.submit(Lab(), cfg)
    assert any("already submitted" in m for m in e.value.extra["errors"])
    assert submit.submit(Lab(), cfg, rerun="nondeterminism check")["job"] == "4242"


def test_dirty_tree_refused_for_exploratory(lab_repo):
    cfg = write_exp(lab_repo)
    commit(lab_repo)
    (lab_repo / "projects/demo/src/train.py").write_text("print('changed')\n")
    with pytest.raises(Fail) as e:
        submit.submit(Lab(), cfg)
    assert any("uncommitted" in m for m in e.value.extra["errors"])


def test_confirmatory_requires_committed_prereg(lab_repo):
    cfg = write_exp(lab_repo, kind="confirmatory", hypothesis="H001")
    commit(lab_repo)
    with pytest.raises(Fail) as e:
        submit.submit(Lab(), cfg)
    assert any("preregistration" in m for m in e.value.extra["errors"])


def test_h200_needs_approval_and_cost_cap(lab_repo):
    cfg = write_exp(lab_repo, profile="h200", time="2-00:00:00")
    commit(lab_repo)
    with pytest.raises(Fail) as e:
        submit.submit(Lab(), cfg)
    errs = " ".join(e.value.extra["errors"])
    assert "--approval" in errs and "per-submit cap" in errs


def test_smoke_time_cap(lab_repo):
    cfg = write_exp(lab_repo, kind="smoke", time="02:00:00")
    commit(lab_repo)
    with pytest.raises(Fail) as e:
        submit.submit(Lab(), cfg)
    assert any("smoke" in m for m in e.value.extra["errors"])


def test_run_local_end_to_end(lab_repo, tmp_path, monkeypatch):
    """Local executor: real labkit entry point, record, status and collect without Slurm."""
    import sys
    from researchctl import runs
    entry = lab_repo / "projects/demo/src/train.py"
    entry.write_text("import labkit\nwith labkit.init() as run:\n"
                     "    run.log(1, loss=1.0)\n    run.finish(val_loss=1.0 + 0.1 * run.seed + (0.5 if run['act'] == 'relu' else 0))\n")
    cfg = write_exp(lab_repo)
    commit(lab_repo)
    venv_bin = tmp_path / "lab/venvs/research/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    out = submit.submit(Lab(), cfg, local=True)
    assert out["status"] == "completed", out
    st = runs.status(Lab(), out["exp_id"])
    assert st["overall"] == "completed"
    col = runs.collect(Lab(), out["exp_id"])
    assert col["completed"] == "6/6" and col["ranking"][0]["act"] == "gelu"
