#!/usr/bin/env python3
"""PreToolUse guard. Deterministic rules the agent cannot argue past.

deny  -> blocked, reason shown to the agent
ask   -> the human must confirm in the Claude app / terminal
These rules stop accidents (raw sbatch, H200 launches, secret leaks, edits to the guards themselves).
They are not a sandbox against a determined adversary; `researchctl budget` audits H200 usage after the fact.
"""
import json
import os
import re
import sys

DENY_BASH = [
    (r"(^|[;&|`(]\s*|\bxargs\s+|\bexec\s+|\bnohup\s+)(sbatch|salloc|srun)\b",
     "Use researchctl (submit / exec). Raw Slurm calls bypass the experiment record and cost guards."),
    (r"(-p|--partition)[ =]?h200\b|nvidia_h200",
     "H200 jobs only via `researchctl request-h200 CONFIG --why ...` followed by human approval."),
    (r"(cat|less|more|head|tail|cp|scp|base64|xxd|strings)\b[^|;&]*(\.ssh/id_|\.ssh/.*key|credentials\.json|secrets\.env|\.netrc|hf_token|/token\b)",
     "Never read or copy private keys or credential files."),
    (r"^\s*(env|printenv|set)\s*($|\|)|echo\s+[^|;&]*\$\{?\w*(TOKEN|KEY|SECRET|PASSWORD)",
     "Do not print environment secrets. Check presence with: test -n \"$VAR\" && echo set"),
    (r"\bchmod\b[^;&|]*\b(\+w|u\+w|a\+w|[0-7]*[2367][0-7]{2})\b[^;&|]*/models/",
     "Original model snapshots are read-only. Write interventions to a new directory."),
    (r"git\s+push\b[^;&|]*(--force\b|-f\b)(?![\w-])",
     "No force pushes."),
    (r"\b(scontrol\s+update|sacctmgr\s+(add|modify|delete))\b",
     "Cluster administration commands are out of scope."),
]
ASK_BASH = [
    (r"\bresearchctl\s+approve\b", "H200 approval: the human must confirm this exact request."),
    (r"\brm\s+(-\w*[rRf]\w*\s+)+|\brm\s+.*\s-\w*[rR]", "Recursive/forced delete: confirm the target."),
    (r"\bfind\b[^|;&]*\s-delete\b|\bshred\b|\btruncate\b|\bmkfs\b|\bdd\s+if=", "Destructive filesystem operation."),
    (r"git\s+(reset\s+--hard|clean\s+-\w*f|branch\s+-D|filter-branch|filter-repo)", "History/worktree-destroying git operation."),
    (r"\bscancel\b(?![^;&|]*\b\d{3,}\b)", "scancel without a job id cancels broadly; use researchctl cancel EXP."),
    (r"\bcrontab\s+-r\b", "Removes all cron entries."),
]
PROTECTED_WRITE = [
    (r"/\.claude/(settings\.json|hooks/)", "ask", "Editing the agent's own guardrails needs human confirmation."),
    (r"/approvals/", "deny", "Approval records are written only by researchctl on human approval."),
    (r"/models/", "deny", "Original model snapshots are read-only."),
    (r"/\.ssh/|credentials\.json|secrets\.env", "deny", "Credential files are off limits."),
    (r"/prereg/H\d+[a-z]?\.yaml$", "check_prereg", ""),
]


def decide(kind, reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": kind, "permissionDecisionReason": reason}}))
    sys.exit(0)


def prereg_committed(path):
    import subprocess
    d = os.path.dirname(path) or "."
    r = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", os.path.basename(path)], capture_output=True)
    return r.returncode == 0


def main():
    try:
        ev = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    tool, inp = ev.get("tool_name", ""), ev.get("tool_input", {}) or {}
    if tool == "Bash":
        cmd = inp.get("command", "")
        for rx, why in DENY_BASH:
            if re.search(rx, cmd, re.M):
                decide("deny", why)
        for rx, why in ASK_BASH:
            if re.search(rx, cmd, re.M):
                decide("ask", why)
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = os.path.abspath(inp.get("file_path") or inp.get("notebook_path") or "")
        for rx, kind, why in PROTECTED_WRITE:
            if re.search(rx, path):
                if kind == "check_prereg":
                    if prereg_committed(path):
                        decide("deny", "Preregistrations are frozen once committed. Write a new version (e.g. H003b) "
                                       "and say why in PROJECT.md.")
                    continue
                decide(kind, why)
    sys.exit(0)


if __name__ == "__main__":
    main()
