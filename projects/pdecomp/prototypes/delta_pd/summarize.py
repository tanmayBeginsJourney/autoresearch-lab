"""Aggregate results_coef*_seed*.json into a markdown table (mean +- sd over seeds). Usage: python summarize.py COEF"""
import glob, json, statistics as st, sys

coef = sys.argv[1] if len(sys.argv) > 1 else "0.001"
rs = [json.load(open(f)) for f in sorted(glob.glob(f"{__file__.rsplit('/', 1)[0]}/results_coef{coef}_seed*.json"))]
print(f"coef={coef}, seeds={[r['seed'] for r in rs]}")


def ms(v):
    v = [x for x in v if x == x]
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if v else "nan")


print("| method | matched (of 2) | purity | AUROC x_j>0 | removed | collateral | cos(abl, change) | base damage | recon MSE / change MSE |")
print("|---|---|---|---|---|---|---|---|---|")
for m in ["delta_pd", "full_pd", "svd_k2", "svd_k4", "lora_k2"]:
    e = [r[m]["eval"] for r in rs]
    matched = [sum(x[f].get("matched", True) for f in ("f3", "f7")) for x in e]
    row = [ms([x["mean"][k] for x in e]) for k in ["purity", "auroc", "removed", "collateral", "cos_change", "base_damage"]]
    print(f"| {m} | {ms(matched)} | " + " | ".join(row) + f" | {ms([r[m]['recon_mse_rel_to_change'] for r in rs])} |")
print()
for m in ["delta_pd", "full_pd"]:
    print(m, "alive(final, per matrix):", [r[m]["final"]["alive"] for r in rs],
          "alive-only-changed/alive(probes):", [(r[m]["n_alive_responding_only_changed"], r[m]["n_alive_probe"]) for r in rs],
          "train_s:", [round(r[m]["train_s"]) for r in rs], "C:", rs[0][m]["C"], "steps:", rs[0][m]["steps"])
    print("  rec by step:", [[(l["step"], f"{l['rec']:.1e}") for l in r[m]["log"][::4]] for r in rs][0])
print("dW1 energy in changed input columns:", [round(r["dW1_col_energy_changed"], 2) for r in rs])
print("dW top singular values:", [[round(x, 2) for x in r["dW_sv"][1][:4]] for r in rs])
print("ft change MSE before ft:", [f"{r['ft_task_mse_before']:.1e}" for r in rs], "ft mse:", [f"{r['ft_mse']:.1e}" for r in rs])
