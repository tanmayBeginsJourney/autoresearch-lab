"""Aggregate run jsons -> markdown table + tiny summary json. Usage: python summarize.py RUN_DIR OUT.json"""
import glob, json, statistics as st, sys

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/*_*_*.json")) if "smoke" not in f]
out = {}
for r in runs:
    out.setdefault(f"{r['model']}|{r['init']}", []).append(r)


def ms(v):
    v = [x for x in v if x is not None]
    if not v: return "-"
    return f"{st.mean(v):.3g}" + (f" ± {st.stdev(v):.2g}" if len(v) > 1 else "")


rows, summ = [], {}
print("| model | init | first rec step (n rec/3) | sustained rec step | gt_min@0 | final gt_min | final MMCS | final ML2R | final CI L0/feat | final stoch recon | faith |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, rs in out.items():
    m, i = k.split("|"); f = [r["final"] for r in rs]
    fr = [r["first_rec_step"] for r in rs]; su = [r["sustained_rec_step"] for r in rs]
    nrec = sum(x is not None for x in su)
    print(f"| {m} | {i} | {ms(fr)} ({sum(x is not None for x in fr)}/{len(rs)}) | {ms(su)} ({nrec}/{len(rs)}) | "
          f"{ms([r['hist'][0]['gt_min'] for r in rs])} | {ms([x['gt_min'] for x in f])} | {ms([x['mmcs'] for x in f])} | "
          f"{ms([x['ml2r'] for x in f])} | {ms([x['ci_l0_per_feat'] for x in f])} | {ms([x['stoch_recon'] for x in f])} | "
          f"{ms([x['faith'] for x in f])} |")
    summ[k] = dict(seeds=[r["seed"] for r in rs], first_rec=fr, sustained_rec=su,
                   final={kk: [x[kk] for x in f] for kk in ("gt_min", "mmcs", "ml2r", "ci_l0_per_feat", "stoch_recon")},
                   steps=rs[0]["steps"], C=rs[0]["C"], t_init=[round(r["t_init"], 2) for r in rs])
json.dump(summ, open(sys.argv[2], "w"), indent=0)
