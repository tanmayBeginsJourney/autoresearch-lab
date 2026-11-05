"""Exploratory ablation (after the preregistered test): which part of canon() hurts Muon FT?"""
import copy, torch
import gauge_muon as G

def balance_only(p):
    q = copy.deepcopy(p)
    for l in range(G.L):
        c = (q[f"d{l}"].norm(dim=0) / q[f"u{l}"].norm(dim=1)).sqrt()
        q[f"u{l}"] = q[f"u{l}"] * c[:, None]; q[f"d{l}"] = q[f"d{l}"] / c[None, :]
    return q

def fold_only(p):
    q = copy.deepcopy(p)
    for l in range(G.L):
        q[f"u{l}"] = q[f"u{l}"] * q[f"g{l}"][None, :]; q[f"g{l}"] = torch.ones(G.D)
    return q

for s in range(3):
    TA = G.teacher(100 + s); TB = G.teacher(200 + s, base=TA, shift=0.6)
    p0 = G.train(G.init(s), TA, G.PRE_STEPS, 3e-3, "adam", seed=s, wd=0.1)
    xv, xt = G.batch(torch.Generator().manual_seed(5000 + s), 4096), G.batch(torch.Generator().manual_seed(6000 + s), 4096)
    row = []
    for name, q in [("native", p0), ("fold_only", fold_only(p0)), ("balance_only", balance_only(p0))]:
        best = min(((G.loss_on(pf, TB, xv), pf) for lr in G.GRID["muon"] if (pf := G.train(q, TB, G.FT_STEPS, lr, "muon", seed=1000 + s)) is not None), key=lambda t: t[0])
        row.append(f"{name} {G.loss_on(best[1], TB, xt):.4f}")
    print(f"seed {s}: " + " | ".join(row), flush=True)
