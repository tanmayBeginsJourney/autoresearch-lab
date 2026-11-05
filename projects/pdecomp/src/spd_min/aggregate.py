"""Summarise train_tms JSON runs: python -m spd_min.aggregate GLOB OUT.json"""

import glob
import json
import sys
from collections import defaultdict
from statistics import mean, pstdev


def row(r):
    e, g1, g2 = r["eval"], r["gt_linear1"], r["gt_linear2"]
    oh = r["one_hot_ci"]
    out = {
        "seed": r["args"]["seed"], "wall_s": r["wall_s"], "steps": r["args"]["steps"],
        "mmcs_l1": g1["mmcs"], "ml2r_l1": g1["ml2r"], "fullcos_min_l1": g1["full_matrix_cos_min"],
        "mmcs_l2": g2["mmcs"], "ml2r_l2": g2["ml2r"], "fullcos_min_l2": g2["full_matrix_cos_min"],
        "distinct_l1": g1["n_distinct_best"], "distinct_l2": g2["n_distinct_best"],
        "alive_l1": e["alive/linear1"], "alive_l2": e["alive/linear2"],
        "ci_l0_l1": e["ci_l0/linear1"], "ci_l0_l2": e["ci_l0/linear2"],
        "recon_stoch": e["recon_stoch"], "recon_ci_masked": e["recon_ci_masked"],
        "recon_unmasked": e["recon_unmasked"], "faith": e["faith"],
        "target_mse": r["target_mse_vs_input"],
    }
    for n, v in oh.items():
        key = "l1" if n == "linear1" else "l2"
        if "n_active_per_feature" in v:  # small models: one-hot check, matched to geometry
            gt = r["gt_linear1" if key == "l1" else "gt_linear2"]["best_idx"]
            out[f"onehot_exactly1_{key}"] = all(k == 1 for k in v["n_active_per_feature"])
            out[f"onehot_matches_geom_{key}"] = v["argmax_ci"] == gt
        else:
            out[f"onehot_hist_{key}"] = v["n_active_hist"]
    return out


def main():
    groups = defaultdict(list)
    for f in sorted(glob.glob(sys.argv[1])):
        r = json.load(open(f))
        a = r["args"]
        groups[f"tms{a['n_features']}-{a['n_hidden']}_imp{a['coeff_imp']:g}_p{a['pnorm']:g}"].append(row(r))
    summary = {}
    for k, rows in groups.items():
        num = {m: (mean(x[m] for x in rows), pstdev(x[m] for x in rows))
               for m, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        summary[k] = {"n_seeds": len(rows), "mean_std": num, "runs": rows}
        print(k, len(rows), {m: f"{mu:.4g}+-{sd:.2g}" for m, (mu, sd) in num.items() if m != "seed"})
    json.dump(summary, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
