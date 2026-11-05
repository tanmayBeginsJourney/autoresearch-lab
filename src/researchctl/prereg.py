"""Preregistration schema. A prereg is written and committed before confirmatory runs."""
from __future__ import annotations

REQUIRED = {
    "id": "H###",
    "question": "the open question this answers (one sentence)",
    "hypothesis": "falsifiable statement, including mechanism if claimed",
    "prediction": "what we expect to observe if the hypothesis is true, with direction and rough size",
    "falsification": "observation that would refute it (concrete threshold)",
    "primary_metric": "single metric name, as logged",
    "baselines": "list: strongest fair baselines, tuned with the same budget",
    "controls": "list: controls that rule out the main alternative explanations",
    "alternatives": "list: competing explanations and how the design separates them",
    "seeds": "planned seed list",
    "analysis": "test, comparison, and decision rule fixed in advance",
    "budget": "max GPU-hours and wall-clock for this hypothesis",
    "decision_value": "what we will do differently depending on the outcome",
}
LISTS = ("baselines", "controls", "alternatives", "seeds")


def validate_prereg(reg) -> list[str]:
    if not isinstance(reg, dict):
        return ["not a mapping"]
    errs = [f"missing {k}" for k in REQUIRED if not reg.get(k)]
    errs += [f"{k} must be a non-empty list" for k in LISTS if k in reg and not (isinstance(reg[k], list) and reg[k])]
    return errs


def template(hid: str) -> str:
    lines = [f"# Preregistration {hid}. Commit before any confirmatory run; never edit after (write {hid}b instead)."]
    for k, desc in REQUIRED.items():
        val = hid if k == "id" else ("[]" if k in LISTS else '""')
        lines.append(f"{k}: {val}  # {desc}")
    lines.append("final_test: false  # true only for the single held-out test evaluation")
    return "\n".join(lines) + "\n"
