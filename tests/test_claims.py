
import yaml

from researchctl import claims
from researchctl.core import Lab, sha256_file


def make_record(repo, exp_id, vals_by_act, kind="confirmatory", hyp="H001", prereg_sha=None):
    tasks, res = [], []
    i = 0
    for act, vals in vals_by_act.items():
        for seed, v in enumerate(vals):
            tasks.append({"i": i, "point": {"act": act}, "seed": seed})
            res.append({"i": i, "status": "completed", "final": {"val_loss": v}})
            i += 1
    rec = {"exp_id": exp_id, "kind": kind, "hypothesis": hyp, "tasks": tasks, "prereg_sha256": prereg_sha,
           "results": {"tasks": res}, "status": "completed"}
    (repo / "projects/demo/experiments" / f"{exp_id}.yaml").write_text(yaml.safe_dump(rec))


def setup(repo, treat, ctrl, statement="Centered GELU lowers validation loss on the 2-layer model."):
    pre = repo / "projects/demo/prereg/H001.yaml"
    pre.write_text("id: H001\n")
    make_record(repo, "e1", {"gelu_c": treat, "gelu": ctrl}, prereg_sha=sha256_file(pre))
    (repo / "projects/demo/claims.yaml").write_text(yaml.safe_dump({"claims": [{
        "id": "C1", "statement": statement, "kind": "performance", "hypothesis": "H001",
        "metric": "val_loss", "goal": "min", "paired_by": "seed",
        "treatment": {"experiment": "e1", "where": {"act": "gelu_c"}},
        "control": {"experiment": "e1", "where": {"act": "gelu"}}}]}))


def test_supported_claim_and_macros(lab_repo):
    setup(lab_repo, [1.00, 1.01, 0.99, 1.00], [1.10, 1.12, 1.09, 1.11])
    out = claims.check(Lab(), "demo")
    assert out["verdicts"]["C1"] == "supported"
    tex = (lab_repo / "projects/demo/paper/numbers.tex").read_text()
    assert "claim@C1.diff" in tex and "claim@C1.verdict}{supported}" in tex


def test_noise_is_inconclusive_and_hype_linted(lab_repo):
    setup(lab_repo, [1.0, 1.2, 0.9], [1.05, 1.0, 1.1], statement="A novel mechanism dramatically lowers loss.")
    out = claims.check(Lab(), "demo")
    assert out["verdicts"]["C1"] == "inconclusive"
    tags = " ".join(out["lint"]["C1"])
    assert "novelty" in tags and "hype" in tags and "causal" in tags


def test_edited_prereg_invalidates(lab_repo):
    setup(lab_repo, [1.00, 1.01, 0.99], [1.10, 1.12, 1.09])
    (lab_repo / "projects/demo/prereg/H001.yaml").write_text("id: H001\nprediction: moved goalposts\n")
    out = claims.check(Lab(), "demo")
    assert out["verdicts"]["C1"] == "invalid"


def test_prose_lint(tmp_path):
    f = tmp_path / "p.tex"
    f.write_text("We delve into a crucial question. Accuracy is 91.37 here. See \\cite{missing}.\n")
    bib = tmp_path / "r.bib"
    bib.write_text("@article{present,\n title={x}}\n")
    out = claims.lint_prose([f], bib)
    text = " ".join(out["issues"])
    assert "stock phrase" in text and "literal number" in text and "missing" in text
