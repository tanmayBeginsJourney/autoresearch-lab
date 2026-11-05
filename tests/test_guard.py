import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / ".claude/hooks/guard.py"


def run(tool, **inp):
    r = subprocess.run([sys.executable, str(GUARD)], input=json.dumps({"tool_name": tool, "tool_input": inp}),
                       capture_output=True, text=True)
    return json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] if r.stdout.strip() else "pass"


@pytest.mark.parametrize("cmd,want", [
    ("sbatch job.sh", "deny"),
    ("cd x && srun -p rtx nvidia-smi", "deny"),
    ("researchctl submit projects/a/configs/b.yaml", "pass"),
    ("researchctl exec --profile s24 -- nvidia-smi", "pass"),
    ("researchctl request-h200 cfg.yaml --why 'scale-up needs 140GB'", "pass"),
    ("researchctl approve ab12cd34", "ask"),
    ("python x.py --partition=h200", "deny"),
    ("cat ~/.ssh/id_ed25519", "deny"),
    ("printenv", "deny"),
    ("echo $HF_TOKEN", "deny"),
    ("rm -rf /scratch/tanmay/lab/runs/x", "ask"),
    ("rm file.txt", "pass"),
    ("git push --force origin main", "deny"),
    ("git push origin main", "pass"),
    ("git reset --hard HEAD~3", "ask"),
    ("grep -r sbatch_template src/", "pass"),
])
def test_bash_rules(cmd, want):
    assert run("Bash", command=cmd) == want


def test_write_rules(tmp_path):
    assert run("Edit", file_path="/repo/.claude/settings.json") == "ask"
    assert run("Write", file_path="/scratch/u/lab/models/Qwen@abc/config.json") == "deny"
    assert run("Write", file_path="/scratch/u/lab/approvals/approved/x.json") == "deny"
    assert run("Write", file_path=str(tmp_path / "projects/p/prereg/H001.yaml")) == "pass"  # not committed
    assert run("Write", file_path="/repo/projects/p/notes.md") == "pass"
