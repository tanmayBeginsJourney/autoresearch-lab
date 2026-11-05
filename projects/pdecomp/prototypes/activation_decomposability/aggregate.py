"""Aggregate results/*.json into a per-(target, coeff, act) table: mean and 95% t-CI over seeds."""

import json
import math
from collections import defaultdict
from pathlib import Path

T95 = {2: 12.706, 3: 4.303, 4: 3.182}
ACTS = ["relu", "gelu", "silu", "bilinear"]
runs, seen = [], set()
for f in sorted(Path(__file__).parent.glob("results/*.json")):
    d = json.loads(f.read_text())
    for r in d["runs"]:
        key = (d["target"], d["seed"], r["coeff_imp"], r["act"])
        if key in seen:
            continue
        seen.add(key)
        nm = 3 if r["act"] == "bilinear" else 2
        runs.append({**r, "target": d["target"], "seed": d["seed"], "L_star": d["L_star"],
                     "l0_per_matrix": r["l0_total"] / nm,
                     "alive_total": sum(v for k, v in r.items() if k.startswith("alive/")),
                     "stoch_rel": r["recon_stoch"] / r["y_ms"], "adv_rel": r["recon_adv"] / r["y_ms"],
                     "gt_in": min(v for k, v in r.items() if k.startswith("gt_maxcos/"))})

METRICS = ["task_val", "train_steps", "l0_total", "l0_per_matrix", "l0/w_out", "l0_per_feat/w_in",
           "alive_total", "stoch_rel", "adv_rel", "faith", "gt_in"]


def ms(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n, 2.0) * sd / math.sqrt(n)


groups = defaultdict(list)
for r in runs:
    groups[(r["target"], r["coeff_imp"], r["act"])].append(r)
lines = []
for tgt, c in sorted({(k[0], k[1]) for k in groups}):
    lines.append(f"\n### target={tgt} coeff_imp={c:g}  (n seeds per act: "
                 f"{[len(groups[(tgt, c, a)]) for a in ACTS]})")
    lines.append("| act | " + " | ".join(METRICS) + " |")
    lines.append("|---" * (len(METRICS) + 1) + "|")
    for a in ACTS:
        rs = groups.get((tgt, c, a), [])
        if not rs:
            continue
        cells = []
        for mtr in METRICS:
            m, ci = ms([r[mtr] for r in rs])
            cells.append(f"{m:.3g}±{ci:.2g}" if ci == ci else f"{m:.3g}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    # paired per-seed check of the G4 ordering: max(relu, bilinear) < min(gelu, silu)
    for mtr in ["l0_per_matrix", "adv_rel", "stoch_rel"]:
        ok, n = 0, 0
        for s in sorted({r["seed"] for r in runs if r["target"] == tgt and r["coeff_imp"] == c}):
            v = {r["act"]: r[mtr] for r in runs
                 if r["target"] == tgt and r["coeff_imp"] == c and r["seed"] == s}
            if len(v) == 4:
                n += 1
                ok += max(v["relu"], v["bilinear"]) < min(v["gelu"], v["silu"])
        lines.append(f"- G4 ordering on {mtr}: holds in {ok}/{n} seeds")
out = "\n".join(lines)
print(out)
(Path(__file__).parent / "results" / "summary.md").write_text(out + "\n")
