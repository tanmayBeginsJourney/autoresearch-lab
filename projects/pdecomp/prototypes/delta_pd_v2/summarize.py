"""Aggregate results/*.json into summary.json and print markdown tables."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
EQ = ("c2_edit_eq", "c3_edit_eq")
METHODS = ("dpd", "full", "svd", "svd_varimax", "svd_ica", "lora_svd", "lora_raw")


def ms(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}" if xs else "n/a"


def rows(m, which):
    cs = [c for c in m["recovery"] if which == "all" or c in EQ]
    return {
        "hung": np.mean([np.mean(m["recovery"][c]["hung"]) for c in cs]),
        "clus": np.mean([np.mean(m["recovery"][c]["clus"]) for c in cs]),
        "restore": np.mean([m["spec"][c]["restore"] for c in cs]),
        "coll": np.mean([m["spec"][c]["collateral"] for c in cs]),
        "untouched": np.mean([m["spec"][c].get("untouched_dmg", np.nan) for c in cs]),
    }


allruns = [json.loads(p.read_text()) for p in sorted(R.glob("*.json"))]
runs = [d for d in allruns if not d["args"].get("kextra", 0)]
kx = [d for d in allruns if d["args"].get("kextra", 0)]
summary = {}
for d in runs:
    for key in ("random", "oracle"):
        if key in d:
            d[key] = {"recovery": {c: {"hung": [np.nan], "clus": [np.nan]} for c in d[key]}, "spec": d[key]}
for arm, ratio, coef in [("exact", 1.0, 1e-3), ("exact", 1.5, 1e-3), ("exact", 3.0, 1e-3), ("exact+drift", 1.0, 1e-3),
                         ("exact", 1.0, 3e-4), ("exact", 1.0, 3e-3)]:
    sel = [d for d in runs if d["args"]["arm"] == arm and d["args"]["ratio"] == ratio and d["args"]["coef"] == coef]
    if not sel:
        continue
    for which in ("eq", "all"):
        print(f"\n### {arm}, norm ratio c3/c2 = {ratio:g}, coef {coef:g}, changes: {'c2,c3 (same matrix)' if which == 'eq' else 'all 4'}")
        print("| method | n | hungarian cos | cluster cos | restore | collateral | untouched dmg |")
        print("|---|---|---|---|---|---|---|")
        for meth in METHODS + ("random", "oracle"):
            rs = [rows(d[meth], which) for d in sel if meth in d]
            if not rs:
                continue
            agg = {k: ms([r[k] for r in rs if not np.isnan(r[k])]) for k in ("hung", "clus", "restore", "coll", "untouched")}
            summary[f"{arm}|r{ratio:g}|c{coef:g}|{which}|{meth}"] = {**agg, "n": len(rs)}
            print(f"| {meth} | {len(rs)} | {agg['hung']} | {agg['clus']} | {agg['restore']} | {agg['coll']} | {agg['untouched']} |")

print("\n### cost, size, gate sparsity (exact arm)")
print("| method | ratio | coef | alive l1+l2 | active comps on x3 / x5 probe | l2 comps shared by x3,x5 | wall s | steps to 5% recon | final recon_ci / change MSE |")
print("|---|---|---|---|---|---|---|---|---|")
for d in sorted(runs, key=lambda d: (d["args"]["ratio"], d["args"]["coef"])):
    pass
groups = defaultdict(list)
for d in runs:
    if d["args"]["arm"] != "exact":
        continue
    for meth in ("dpd", "full"):
        if f"{meth}_info" in d:
            groups[(meth, d["args"]["ratio"], d["args"]["coef"])].append(d[f"{meth}_info"])
    if "svd_gate" in d:
        groups[("svd_gate", d["args"]["ratio"], d["args"]["coef"])].append(d["svd_gate"])
for (meth, r, c), inf in sorted(groups.items()):
    l0 = [i["support_l0"] for i in inf]
    if meth == "svd_gate":
        alive, steps, rec = "5 (fixed)", "-", ms([i["recon_ci_rel"] for i in inf])
    else:
        alive = ms([i["alive"]["l1"] + i["alive"]["l2"] for i in inf])
        steps = [i["steps_to_5pct"] for i in inf]
        rec = ms([i["final"]["recon_ci_masked"] / i["change_mse"] for i in inf])
    line = (f"| {meth} | {r:g} | {c:g} | {alive} | {ms([x['x3'] for x in l0])} / {ms([x['x5'] for x in l0])} | "
            f"{ms([x['x3_x5_shared_l2'] for x in l0])} | {ms([i['wall_s'] for i in inf])} | {steps} | {rec} |")
    summary[f"cost|{meth}|r{r:g}|c{c:g}"] = line
    print(line)
lo = [d["lora_info"] for d in runs if d["args"]["arm"] == "exact" and "lora_info" in d]
print(f"lora: wall {ms([i['wall_s'] for i in lo])} s, recon/change MSE {ms([i['recon_rel'] for i in lo])}")

print("\n### drift arm (no planted change)")
ds = [d for d in runs if d["args"]["arm"] == "drift"]
if ds:
    line = (f"alive (l1,l2) {[(d['dpd_info']['alive']['l1'], d['dpd_info']['alive']['l2']) for d in ds]}; "
            f"sum alive comp norm {[round(sum(d['dpd']['comp_norms']), 3) for d in ds]}; "
            f"||dW1||+||dW2|| {[round(sum(d['delta_norm'].values()), 3) for d in ds]}; "
            f"change MSE {[f'{d['dpd_info']['change_mse']:.1e}' for d in ds]}; "
            f"svd_gate recon_ci_rel {[round(d['svd_gate']['recon_ci_rel'], 3) for d in ds]}")
    summary["drift"] = line
    print(line)
for d in runs:
    if d["args"]["seed"] == 0 and d["args"]["coef"] == 1e-3 and "svd_spectrum" in d:
        print(f"svd spectrum {d['args']['arm']} r{d['args']['ratio']:g}: l1 {[round(x, 3) for x in d['svd_spectrum']['l1']]}, l2 {[round(x, 3) for x in d['svd_spectrum']['l2']]}")
if kx:
    print("\n### exact+drift, SVD baselines with k = oracle + 2 per matrix")
    for which in ("eq", "all"):
        for meth in ("svd", "svd_varimax", "svd_ica"):
            rs = [rows(d[meth], which) for d in kx]
            agg = {k: ms([r[k] for r in rs]) for k in ("hung", "clus", "restore", "coll")}
            summary[f"kx2|{which}|{meth}"] = agg
            print(which, meth, agg)
(Path(__file__).resolve().parent / "summary.json").write_text(json.dumps(summary, indent=1))
