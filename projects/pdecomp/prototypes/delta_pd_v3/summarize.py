"""Aggregate results/*.json -> summary.json and a printed table (mean ± sd over seeds)."""
import glob, json
from collections import defaultdict
import numpy as np

agg = defaultdict(lambda: defaultdict(list))
info = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("results/*_steps4000_seed*.json")):
    d = json.load(open(f))
    arm = f.split("/")[1].split("_steps")[0]
    meths = dict(d["methods"]); meths["oracle"] = d["oracle"]; meths["random"] = d["random"]
    for m, r in meths.items():
        chs = [c for c in r if not c.startswith("_")]
        for key in ("hung", "restore", "collateral", "clean_dmg"):
            vals = [r[c][key] for c in chs if key in r[c]]
            if vals: agg[(arm, m)][key + "_all"].append(np.mean(vals))
            for c in chs:
                if key in r[c]: agg[(arm, m)][f"{key}_{c}"].append(r[c][key])
        for k in ("_pair_linear", "_pair_gate"):
            if r.get(k) is not None: agg[(arm, m)][k].append(r[k])
        if "_bd_gate_clean_max" in r:
            for k in ("_bd_gate_clean_max", "_bd_gate_trig_mean", "_bd_gate_clean_frac_gt0.1"):
                agg[(arm, m)][k].append(float(np.max(r[k])) if r[k] else float("nan"))
            agg[(arm, m)]["_n_dormant"].append(len(r["_dormant_on_clean"]))
            agg[(arm, m)]["_n_dormant_bd"].append(int(sum(r["_dormant_are_bd"])))
    for k in ("dpd_info", "full_info"):
        if k in d:
            info[(arm, k)]["alive"].append(d[k]["alive"]); info[(arm, k)]["n_kept"].append(d[k]["n_kept"])
            info[(arm, k)]["steps_to_5pct"].append(d[k]["steps_to_5pct"])
            info[(arm, k)]["final_ci_rel"].append(round(d[k]["final"]["ci_rel"], 4))
            info[(arm, k)]["final_stoch_rel"].append(round(d[k]["final"]["stoch_rel"], 4))
    info[(arm, "lora")]["recon_rel"].append(round(d["lora_recon_rel"], 3))
    info[(arm, "rank")]["rank"].append(d["rank"])
summ = {f"{a}|{m}": {k: [float(np.mean(v)), float(np.std(v)), [round(x, 3) for x in v]] for k, v in d.items()} for (a, m), d in agg.items()}
summ["_info"] = {f"{a}|{k}": v for (a, k), v in info.items()}
json.dump(summ, open("summary.json", "w"), indent=1)
for (a, m), d in agg.items():
    print(a, m, " ".join(f"{k}={np.mean(v):.2f}±{np.std(v):.2f}" for k, v in d.items() if k.endswith("_all") or k.startswith("_")))
for (a, k), v in info.items(): print(a, k, dict(v))
