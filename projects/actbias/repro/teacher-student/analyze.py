"""Summarise ts.py output: paired S-U (and S-UB) differences with 95% paired-t CIs.
usage: analyze.py relu.json [gelu.json ...]  -> prints tables, writes summary.json"""
import json, math, sys
from statistics import mean, stdev

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
SHARED_LR = 1e-3  # Adam default; the paper does not state its LR


def ci(diffs):
    n = len(diffs); mu = mean(diffs)
    if n < 2:
        return mu, float("nan"), float("nan")
    h = T975[n - 1] * stdev(diffs) / math.sqrt(n)
    return mu, mu - h, mu + h


def fmt(mu, lo, hi, scale=1e3):
    return f"{mu*scale:+.2f} [{lo*scale:+.2f}, {hi*scale:+.2f}]"


def at(run, key, step):
    c = run["curve"]; return c[key][c["step"].index(step)]


summary = {}
for path in sys.argv[1:]:
    runs = json.load(open(path))["runs"]
    act = runs[0]["act"]; steps = runs[0]["steps"]
    R = {(r["arm"], r["lr"], r["seed"]): r for r in runs}
    arms = sorted({r["arm"] for r in runs}); lrs = sorted({r["lr"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    div = [(k, ) for k, r in R.items() if r["diverged"]]
    fin = lambda a, lr, s, key="test": at(R[(a, lr, s)], key, steps)
    S = {"act": act, "steps": steps, "seeds": seeds, "lrs": lrs, "diverged": [list(k[0]) for k in div]}
    print(f"\n=== {act}  steps={steps}  seeds={seeds}  diverged={len(div)}")

    # mean final test loss per arm x LR
    print("final test MSE x1e3, mean over seeds (val-mean in parens)")
    tab = {}
    for a in arms:
        row = []
        for lr in lrs:
            t = [fin(a, lr, s) for s in seeds]; v = [fin(a, lr, s, "val") for s in seeds]
            tab[(a, lr)] = mean(v); row.append(f"{mean(t)*1e3:7.2f}({mean(v)*1e3:6.2f})")
        print(f"  {a:3s}", "  ".join(row), " lrs=", lrs)
    S["mean_test_by_arm_lr"] = {a: {str(lr): mean(fin(a, lr, s) for s in seeds) for lr in lrs} for a in arms}

    # (1) shared LR: S-U at every LR, plus early checkpoints at SHARED_LR
    S["S_minus_U_by_lr"] = {}
    print("paired S-U final test MSE x1e3 [95% CI], per LR (same LR both arms)")
    for lr in lrs:
        d = [fin("S", lr, s) - fin("U", lr, s) for s in seeds]
        rel = [(fin("S", lr, s) - fin("U", lr, s)) / fin("U", lr, s) for s in seeds]
        S["S_minus_U_by_lr"][str(lr)] = {"diff": ci(d), "rel": ci(rel), "n_S_better": sum(x < 0 for x in d)}
        print(f"  lr={lr:<7g} {fmt(*ci(d))}  rel {mean(rel)*100:+.1f}%  S better in {sum(x<0 for x in d)}/{len(d)}")
    steps_list = R[("U", SHARED_LR, seeds[0])]["curve"]["step"]
    print(f"curve at shared lr={SHARED_LR}: step, mean U, mean S, mean UB, paired S-U x1e3 [CI]")
    S["curve_shared_lr"] = []
    for st in steps_list:
        u = [at(R[("U", SHARED_LR, s)], "test", st) for s in seeds]
        sh = [at(R[("S", SHARED_LR, s)], "test", st) for s in seeds]
        ub = [at(R[("UB", SHARED_LR, s)], "test", st) for s in seeds] if "UB" in arms else [float("nan")]
        d = ci([a - b for a, b in zip(sh, u)])
        S["curve_shared_lr"].append({"step": st, "U": mean(u), "S": mean(sh), "UB": mean(ub), "S_minus_U": d})
        print(f"  {st:5d} {mean(u):.4f} {mean(sh):.4f} {mean(ub):.4f}  {fmt(*d)}")

    # (2) per-arm best LR, chosen on val (mean over seeds), reported on test
    best = {a: min(lrs, key=lambda lr: tab[(a, lr)]) for a in arms}
    S["best_lr"] = best
    print("best LR per arm (val):", best, " grid edge:", {a: best[a] in (lrs[0], lrs[-1]) for a in arms})
    for a, b_ in [("S", "U"), ("S", "UB"), ("UB", "U")]:
        if a in arms and b_ in arms:
            d = [fin(a, best[a], s) - fin(b_, best[b_], s) for s in seeds]
            rel = [x / fin(b_, best[b_], s) for x, s in zip(d, seeds)]
            S[f"{a}_minus_{b_}_bestlr"] = {"diff": ci(d), "rel": ci(rel), "n_better": sum(x < 0 for x in d)}
            print(f"  {a}@{best[a]:g} - {b_}@{best[b_]:g}: {fmt(*ci(d))}  rel {mean(rel)*100:+.1f}%"
                  f"  {a} better in {sum(x<0 for x in d)}/{len(d)}")
    # early comparison at best LRs
    S["early_bestlr"] = []
    for st in [10, 20, 50, 100, 200, 500, 1000]:
        d = [at(R[("S", best["S"], s)], "test", st) - at(R[("U", best["U"], s)], "test", st) for s in seeds]
        S["early_bestlr"].append({"step": st, "S_minus_U": ci(d)})
        print(f"  best-LR curve step {st:5d}: S-U {fmt(*ci(d))}")
    # S - UB at shared LR
    if "UB" in arms:
        for lr in lrs:
            d = [fin("S", lr, s) - fin("UB", lr, s) for s in seeds]
            S.setdefault("S_minus_UB_by_lr", {})[str(lr)] = ci(d)
            print(f"  S-UB lr={lr:<7g} {fmt(*ci(d))}")
    summary[act] = S

json.dump(summary, open("summary.json", "w"), indent=1)
