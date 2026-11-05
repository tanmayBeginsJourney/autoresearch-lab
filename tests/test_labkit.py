import json

import labkit


def test_run_lifecycle(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"run_id": "e/000", "seed": 3, "params": {"lr": 0.1}, "act": "gelu"}))
    with labkit.init(str(tmp_path / "params.json")) as run:
        assert run["lr"] == 0.1 and run["act"] == "gelu"
        run.log(1, loss=2.0)
        run.log(2, loss=float("nan"))
        run.finish(val_loss=1.5)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["status"] == "diverged" and s["diverged"] and s["final"]["val_loss"] == 1.5
    assert (tmp_path / "manifest.json").exists() and len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def test_exception_recorded(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"seed": 0, "params": {}}))
    try:
        with labkit.init(str(tmp_path / "params.json")):
            raise ValueError("boom")
    except ValueError:
        pass
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["status"] == "failed" and "boom" in s["error"]
