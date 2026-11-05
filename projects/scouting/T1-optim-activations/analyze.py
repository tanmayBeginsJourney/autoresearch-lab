"""Evaluate P1/P2 of the IDEAS.md preregistration from results.json."""
import json, sys
from collections import defaultdict

runs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
R = defaultdict(dict)  # R[arm][(lr, seed)] = (val, test)
for r in runs:
    R[r["arm"]][(r["lr"], r["seed"])] = (r["val"], r["test"], r["diverged"])
lrs, seeds = sorted({r["lr"] for r in runs}), sorted({r["seed"] for r in runs})
print("mean test MSE x1e-3 per LR in that arm's grid (diverged count)")
best = {}
for a in ["AU", "AS", "MU", "MS", "MP"]:
    row, vals = [], {}
    for lr in sorted({k[0] for k in R[a]}):
        v = [R[a][(lr, s)] for s in seeds]
        nd = sum(x[2] for x in v)
        vals[lr] = sum(x[0] for x in v) / len(v) if not nd else float("inf")
        row.append(f"{1e3 * sum(x[1] for x in v) / len(v):8.3f}({nd})")
    best[a] = min(vals, key=vals.get)
    print(a, " ".join(row), " best LR (val):", best[a])
T = {a: [R[a][(best[a], s)][1] for s in seeds] for a in best}


def rel(x, y):  # relative change of x vs y, mean of per-seed ratios, and wins
    rs = [xi / yi - 1 for xi, yi in zip(x, y)]
    return sum(rs) / len(rs), sum(r < 0 for r in rs), rs


for a in T:
    print(a, "test@best", [round(1e3 * v, 3) for v in T[a]])
ga, wa, _ = rel(T["AS"], T["AU"]); gm, wm, rs = rel(T["MS"], T["MU"])
print(f"Adam S vs U: {100*ga:+.1f}% ({wa}/3)   Muon S vs U: {100*gm:+.1f}% ({wm}/3) per-seed {[round(100*r,1) for r in rs]}")
p1 = gm <= 2 * ga and ga < 0 and wm == 3
gp, wp, _ = rel(T["MP"], T["MU"]); print(f"Muon P vs U: {100*gp:+.1f}% ({wp}/3)")
base = min(["AU", "AS", "MU"], key=lambda a: sum(T[a]))
meth = min(["MS", "MP"], key=lambda a: sum(T[a]))
g2, w2, rs2 = rel(T[meth], T[base])
p2 = g2 <= -0.05 and w2 == 3
print(f"P2: {meth} vs strongest baseline {base}: {100*g2:+.1f}% ({w2}/3) per-seed {[round(100*r,1) for r in rs2]}")
print("P1", "PASS" if p1 else "FAIL", "| P2", "PASS" if p2 else "FAIL")
