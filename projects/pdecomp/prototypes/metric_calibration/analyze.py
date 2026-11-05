"""Spearman rank correlation (bootstrap 95% CI over decompositions) between each candidate metric
and each ground-truth recovery score. Writes summary.json and prints tables."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
rows = [r for f in sorted((HERE / "results").glob("*.json")) for r in json.loads(f.read_text())]
GT = ["gt_fullcos", "gt_frac_recovered", "gt_onehot_acc", "gt_mmcs_col", "gt_spurious_alive"]
M = ["m_stoch", "m_pgd10", "m_pgd40", "m_rounded", "m_ci_masked", "m_unmasked", "m_l0", "m_faith"]
rng = np.random.default_rng(0)


def corr(sub, m, g, B=2000):
    x = np.array([r[m] for r in sub]); y = np.array([r[g] for r in sub])
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rho = spearmanr(x, y).statistic
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            bs.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


subsets = {
    "all": rows,
    "oracle_mixed": [r for r in rows if r["family"] == "oracle_mixed"],
    "natural_spd(+rotated)": [r for r in rows if r["family"] != "oracle_mixed"],
}
out = {"n": {k: len(v) for k, v in subsets.items()}, "rho": {}}
for sname, sub in subsets.items():
    print(f"\n== {sname} (n={len(sub)}); Spearman rho [95% bootstrap CI]; metrics are losses, so "
          "a good predictor of recovery has rho < 0 vs gt_fullcos")
    print("metric".ljust(12) + "".join(g.ljust(26) for g in GT))
    out["rho"][sname] = {}
    for m in M:
        cells = {g: corr(sub, m, g) for g in GT}
        out["rho"][sname][m] = cells
        print(m.ljust(12) + "".join(f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]".ljust(26) for c in cells.values()))
cols = ["name", "family", "gt_fullcos", "gt_frac_recovered", "gt_onehot_acc", "gt_n_alive",
        "gt_spurious_alive", "m_stoch", "m_pgd10", "m_pgd40", "m_rounded", "m_unmasked", "m_l0"]
print("\n" + " ".join(c[:10].ljust(10) for c in cols))
for r in sorted(rows, key=lambda r: -r["gt_fullcos"]):
    print(" ".join((f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]))[:10].ljust(10) for c in cols))
out["rows"] = rows
(HERE / "summary.json").write_text(json.dumps(out, indent=1))

# Paired bootstrap: rho(metric) - rho(m_stoch) vs gt_fullcos on the same resamples; and how many
# non-oracle decompositions score better (lower loss) than the exact ground truth (oracle t=0).
print("\npaired bootstrap of rho(metric) - rho(m_stoch), target gt_fullcos (negative = better than stoch)")
out["paired_vs_stoch"] = {}
for sname, sub in subsets.items():
    y = np.array([r["gt_fullcos"] for r in sub]); xs = np.array([r["m_stoch"] for r in sub])
    for m in ["m_pgd10", "m_pgd40", "m_rounded", "m_ci_masked"]:
        xm = np.array([r[m] for r in sub]); d = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            d.append(spearmanr(xm[i], y[i]).statistic - spearmanr(xs[i], y[i]).statistic)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out["paired_vs_stoch"][f"{sname}/{m}"] = [float(np.nanmean(d)), float(lo), float(hi)]
        print(f"{sname:24s} {m:12s} {np.nanmean(d):+.3f} [{lo:+.3f},{hi:+.3f}]")
gt0 = [r for r in rows if r["family"] == "oracle_mixed" and r["rot"] == 0]
print("\nnon-oracle decompositions with lower loss than the best ground-truth (t=0) decomposition:")
out["beats_ground_truth"] = {}
for m in M[:5]:
    best = min(r[m] for r in gt0)
    beat = [r["name"] for r in rows if r["family"] != "oracle_mixed" and r[m] < best]
    out["beats_ground_truth"][m] = beat
    print(f"{m:12s} {beat}")
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
