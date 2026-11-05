"""Delta-PD v2 (EXPLORATORY): exact planted fine-tune deltas, Delta-PD vs full PD vs SVD vs LoRA.

Toy: f(x) = W2 ReLU(W1 x), n=12 sparse features. Base is hand-set: neurons 2i, 2i+1 = ReLU(+-x_i),
y_i = sum of both = |x_i|; neurons 24..27 are spare (zero). Fine-tune = base + hand-set rank-one
pieces (exact ground truth), optionally plus dense Gaussian drift.

Changes (piece = (matrix, u, v), delta = s * u v^T):
  c1 add   W1: spare neuron 24 reads x0 (norm 1.0); W2: neuron 24 writes output 11 (norm 0.8)
  c2 edit  W2: neuron 6 (ReLU x3) also writes 0.6 * u2      } equal norm, same matrix, cos(u2,u3)=0.5:
  c3 edit  W2: neuron 10 (ReLU x5) also writes 0.6 * u3     } SVD right vectors = (e6 +- e10)/sqrt2
  c4 edit  W1: neuron 14 rerouted from x7 to x8: e14 (e8 - e7)^T (norm sqrt2)
Arms: exact (pieces only), exact+drift (pieces + noise with ||noise||_F = 0.5 ||pieces||_F per
matrix), drift (noise only, ||noise||_F = 0.05 * ||W_base||_F: small behaviour change).

Usage: python delta_pd_v2.py --arm exact --coef 1e-3 --seed 0 [--methods dpd,full,svd,lora]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spd_min import SPD, SPDConfig, lower_leaky_hard_sigmoid, train_spd  # noqa: E402

torch.set_num_threads(1)
N, H = 12, 28
P_ACT = 0.1
OUT = Path(__file__).resolve().parent / "results"


class MLP(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()
        self.l1 = nn.Linear(N, H, bias=False)
        self.l2 = nn.Linear(H, N, bias=False)
        with torch.no_grad():
            self.l1.weight.copy_(W1)
            self.l2.weight.copy_(W2)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def e(i, d):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def base_weights():
    W1 = torch.zeros(H, N)
    W2 = torch.zeros(N, H)
    for i in range(N):
        W1[2 * i, i], W1[2 * i + 1, i] = 1.0, -1.0
        W2[i, 2 * i], W2[i, 2 * i + 1] = 1.0, 1.0
    return W1, W2


def planted(seed, ratio=1.0):
    g = torch.Generator().manual_seed(1000 + seed)
    u2 = torch.randn(N, generator=g)
    u2 /= u2.norm()
    r = torch.randn(N, generator=g)
    r -= (r @ u2) * u2
    r /= r.norm()
    u3 = 0.5 * u2 + math.sqrt(0.75) * r  # cos(u2, u3) = 0.5
    s2 = 1 / math.sqrt(2)
    # change -> list of (matrix, scale, u, v), u/v unit
    return {
        "c1_add": [("l1", 1.0, e(24, H), e(0, N)), ("l2", 0.8, e(11, N), e(24, H))],
        "c2_edit_eq": [("l2", 0.6, u2, e(6, H))],
        "c3_edit_eq": [("l2", 0.6 * ratio, u3, e(10, H))],
        "c4_reroute": [("l1", math.sqrt(2), e(14, H), (e(8, N) - e(7, N)) * s2)],
    }
# probe features (positive values) on which each change acts
SUPPORT = {"c1_add": [0], "c2_edit_eq": [3], "c3_edit_eq": [5], "c4_reroute": [7, 8]}
UNTOUCHED = [1, 2, 4, 6, 9, 10, 11]


def make_ft(arm, seed, ratio=1.0):
    W1b, W2b = base_weights()
    base = {"l1": W1b, "l2": W2b}
    ch = planted(seed, ratio) if arm != "drift" else {}
    pieces = {"l1": torch.zeros(H, N), "l2": torch.zeros(N, H)}
    for plist in ch.values():
        for m, s, u, v in plist:
            pieces[m] += s * torch.outer(u, v)
    g = torch.Generator().manual_seed(2000 + seed)
    noise = {}
    for m in base:
        z = torch.randn(base[m].shape, generator=g)
        if arm == "exact+drift":
            noise[m] = z / z.norm() * 0.5 * pieces[m].norm()
        elif arm == "drift":
            noise[m] = z / z.norm() * 0.05 * base[m].norm()
        else:
            noise[m] = torch.zeros_like(z)
    ft = {m: base[m] + pieces[m] + noise[m] for m in base}
    return base, ft, ch, noise


def sample(bs, g=None):
    act = (torch.rand(bs, N, generator=g) < P_ACT).float()
    return act * (2 * torch.rand(bs, N, generator=g) - 1)


def probes(feats):
    ts = torch.tensor([-1.0, -0.5, 0.25, 0.5, 0.75, 1.0])
    return torch.cat([ts[:, None] * e(i, N)[None] for i in feats])


def fwd(W, x):
    return torch.relu(x @ W["l1"].T) @ W["l2"].T


# ---------------------------------------------------------------- component extraction
def comp_list_from_spd(spd, alive_mask):
    out = []
    for n in spd.names:
        S = spd.subcomponents(n).detach()
        for c in range(S.shape[0]):
            if alive_mask[n][c]:
                out.append((n, S[c]))
    return out


def svd_comps(D, k):
    out = []
    for n, M in D.items():
        U, S, Vh = torch.linalg.svd(M)
        for i in range(k[n]):
            out.append((n, S[i] * torch.outer(U[:, i], Vh[i])))
    return out


def varimax(V, iters=100, tol=1e-8):
    """Varimax rotation R (k x k) of loadings V (d x k)."""
    d, k = V.shape
    R = np.eye(k)
    obj = 0
    for _ in range(iters):
        L = V @ R
        u, s, vh = np.linalg.svd(V.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / d))
        R = u @ vh
        if s.sum() < obj * (1 + tol):
            break
        obj = s.sum()
    return R


def fastica(Hm, iters=300, seed=0):
    """Symmetric FastICA (logcosh) on samples Hm (n x k). Returns unmixing W (k x k): s = W h."""
    Hc = Hm - Hm.mean(0)
    C = np.cov(Hc.T) + 1e-9 * np.eye(Hc.shape[1])
    ev, E = np.linalg.eigh(C)
    K = E @ np.diag(ev ** -0.5) @ E.T
    Z = Hc @ K
    rng = np.random.default_rng(seed)
    W = np.linalg.qr(rng.standard_normal((Z.shape[1], Z.shape[1])))[0]
    for _ in range(iters):
        G = np.tanh(Z @ W.T)
        Wn = (G.T @ Z) / len(Z) - np.diag((1 - G ** 2).mean(0)) @ W
        u, _, vh = np.linalg.svd(Wn)
        Wn = u @ vh
        if np.max(np.abs(np.abs(np.diag(Wn @ W.T)) - 1)) < 1e-7:
            W = Wn
            break
        W = Wn
    return W @ K


def rotated_svd_comps(D, k, mode, acts=None):
    """Rotate within the top-k SVD subspace: varimax on input-side loadings, or ICA on inner acts."""
    out = []
    for n, M in D.items():
        U, S, Vh = torch.linalg.svd(M.double())
        Uk, Sk, Vk = U[:, : k[n]].numpy(), S[: k[n]].numpy(), Vh[: k[n]].T.numpy()
        if mode == "varimax":
            R = varimax(Vk)
            A, Wt = R, R  # comps: (U S R)_i (V R)_i^T
            us, vs = Uk @ np.diag(Sk) @ A, Vk @ Wt
        else:  # ICA on h = V^T a over natural inputs to this layer
            Wm = fastica(acts[n].double().numpy() @ Vk)
            Am = np.linalg.inv(Wm)
            us, vs = Uk @ np.diag(Sk) @ Am, Vk @ Wm.T
        for i in range(k[n]):
            out.append((n, torch.tensor(np.outer(us[:, i], vs[:, i]), dtype=torch.float32)))
    return out


def run_svd_gate(base, ft, delta, k, coef, seed, steps, bs=256):
    """Top-k SVD directions of dW frozen as Delta-PD subcomponents; only the CI gates are trained."""
    torch.manual_seed(seed)
    model = MLP(ft["l1"], ft["l2"])
    spd = SPD(model, k, base_weights=base)
    with torch.no_grad():
        for n in k:
            U, S, Vh = torch.linalg.svd(delta[n])
            kk = spd._key(n)
            spd.U[kk].copy_(U[:, : k[n]] * S[: k[n]])
            spd.V[kk].copy_(Vh[: k[n]])
            spd.U[kk].requires_grad_(False)
            spd.V[kk].requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    train_spd(spd, lambda: sample(bs, g), SPDConfig(coeff_imp=coef, pnorm=1.0), steps, lr=2e-3, log_every=10 ** 9)
    ev = sample(2048, torch.Generator().manual_seed(99))
    m = spd.eval_metrics(ev)
    cm = ((fwd(ft, ev) - fwd(base, ev)) ** 2).mean().item()
    l0 = support_l0(spd)
    return {"wall_s": time.time() - t0, "recon_ci_rel": m["recon_ci_masked"] / cm, "recon_stoch_rel": m["recon_stoch"] / cm, "support_l0": l0}


def support_l0(spd):
    """Mean number of active (CI > 0.1) subcomponents, summed over matrices, on x3=1 and x5=1 probes."""
    x = torch.stack([e(3, N), e(5, N)])
    _, acts = spd.target_and_inputs(x)
    gci = {n: lower_leaky_hard_sigmoid(v) for n, v in spd.causal_importances(acts).items()}
    return {"x3": sum(int((gci[n][0] > 0.1).sum()) for n in gci), "x5": sum(int((gci[n][1] > 0.1).sum()) for n in gci),
            "x3_x5_shared_l2": int(((gci["l2"][0] > 0.1) & (gci["l2"][1] > 0.1)).sum())}


def fcos(A, B):
    return (A * B).sum().item() / (A.norm().item() * B.norm().item() + 1e-12)


def recovery(comps, ch):
    """Per planted piece: Hungarian |cos| (1-to-1, within matrix) and cluster cos (sum of comps whose
    argmax-|cos| piece is this one, |cos| > 0.2). Returns metrics and change->comp-index assignment."""
    pieces = [(c, m, s * torch.outer(u, v)) for c, pl in ch.items() for (m, s, u, v) in pl]
    hung = {}
    for m in ("l1", "l2"):
        pi = [i for i, p in enumerate(pieces) if p[1] == m]
        ci = [j for j, q in enumerate(comps) if q[0] == m]
        if not pi:
            continue
        C = np.zeros((len(pi), max(len(ci), 1)))
        for a, i in enumerate(pi):
            for b, j in enumerate(ci):
                C[a, b] = abs(fcos(pieces[i][2], comps[j][1]))
        r, c = linear_sum_assignment(-C)
        for a in range(len(pi)):
            hung[pi[a]] = 0.0
        for a, b in zip(r, c):
            hung[pi[a]] = float(C[a, b]) if ci else 0.0
    assign = {}  # comp idx -> piece idx
    for j, (m, M) in enumerate(comps):
        best, bi = 0.2, None
        for i, (_, pm, P) in enumerate(pieces):
            if pm == m and abs(fcos(P, M)) > best:
                best, bi = abs(fcos(P, M)), i
        if bi is not None:
            assign[j] = bi
    clus = {}
    for i, (c, m, P) in enumerate(pieces):
        js = [j for j, b in assign.items() if b == i]
        clus[i] = abs(fcos(P, sum(comps[j][1] for j in js))) if js else 0.0
        # signed relative norm error of the cluster sum vs the planted piece
    per_change = {}
    for i, (c, m, P) in enumerate(pieces):
        d = per_change.setdefault(c, {"hung": [], "clus": [], "comps": []})
        d["hung"].append(hung[i])
        d["clus"].append(clus[i])
        d["comps"] += [j for j, b in assign.items() if b == i]
    return per_change


def specificity(base, ft, comps, per_change, ch):
    """Revert one recovered change (subtract its assigned comps) and measure behaviour."""
    res = {}
    xu = probes(UNTOUCHED)
    yb_u = fwd(base, xu)
    for c in ch:
        W = {m: ft[m].clone() for m in ft}
        for j in per_change[c]["comps"]:
            W[comps[j][0]] = W[comps[j][0]] - comps[j][1]
        res[c] = revert_effect(base, ft, W, c, ch)
        res[c]["untouched_dmg"] = ((fwd(W, xu) - fwd(ft, xu)).norm() / yb_u.norm()).item()
        res[c]["n_comps"] = len(per_change[c]["comps"])
    return res


def revert_effect(base, ft, W, c, ch):
    x = probes(SUPPORT[c])
    yb, yf, yr = fwd(base, x), fwd(ft, x), fwd(W, x)
    restore = 1 - ((yr - yb).norm() / (yf - yb).norm()).item()
    coll = []
    for o in ch:
        if o == c:
            continue
        xo = probes(SUPPORT[o])
        coll.append(((fwd(W, xo) - fwd(ft, xo)).norm() / (fwd(ft, xo) - fwd(base, xo)).norm()).item())
    return {"restore": restore, "collateral": float(np.mean(coll))}


def random_control(base, ft, ch, seed):
    g = torch.Generator().manual_seed(3000 + seed)
    res = {}
    for c, pl in ch.items():
        W = {m: ft[m].clone() for m in ft}
        for m, s, u, v in pl:
            ru = torch.randn(len(u), generator=g)
            rv = torch.randn(len(v), generator=g)
            W[m] = W[m] - s * torch.outer(ru / ru.norm(), rv / rv.norm())
        res[c] = revert_effect(base, ft, W, c, ch)
    return res


def oracle_check(base, ft, ch, noise):
    res = {}
    for c, pl in ch.items():
        W = {m: ft[m].clone() for m in ft}
        for m, s, u, v in pl:
            W[m] = W[m] - s * torch.outer(u, v)
        res[c] = revert_effect(base, {m: base[m] + (ft[m] - base[m]) for m in ft}, W, c, ch)
    return res


# ---------------------------------------------------------------- methods
def run_pd(base, ft, delta, C, coef, seed, steps, warm, bs=256):
    torch.manual_seed(seed)
    model = MLP(ft["l1"], ft["l2"])
    spd = SPD(model, {"l1": C, "l2": C}, base_weights=base if delta else None)
    cfg = SPDConfig(coeff_imp=coef, pnorm=1.0)
    g = torch.Generator().manual_seed(seed)
    ev = sample(2048, torch.Generator().manual_seed(99))
    y_ft, y_b = fwd(ft, ev), fwd(base, ev)
    change_mse = ((y_ft - y_b) ** 2).mean().item()
    curve = []

    def log(step, rec):
        m = spd.eval_metrics(ev)
        curve.append({"step": step, "t": time.time() - t0, **{k: m[k] for k in ("recon_ci_masked", "recon_stoch", "faith")}})

    t0 = time.time()
    train_spd(spd, lambda: sample(bs, g), cfg, steps, lr=2e-3, log_every=100, log=log, faith_warmup_steps=warm)
    wall = time.time() - t0
    xall = torch.cat([ev, probes(range(N))])
    _, acts = spd.target_and_inputs(xall)
    gci = {n: lower_leaky_hard_sigmoid(v) for n, v in spd.causal_importances(acts).items()}
    alive = {n: (gci[n].max(0).values > 0.1).tolist() for n in spd.names}
    m = spd.eval_metrics(ev)
    thr = 0.05 * change_mse
    hit = next((r for r in curve if r["recon_ci_masked"] <= thr and r["recon_stoch"] <= thr), None)
    info = {
        "support_l0": support_l0(spd),
        "wall_s": wall, "change_mse": change_mse, "final": m,
        "alive": {n: int(sum(a)) for n, a in alive.items()},
        "steps_to_5pct": None if hit is None else hit["step"],
        "time_to_5pct": None if hit is None else hit["t"],
        "curve": curve[:: max(1, len(curve) // 10)],
    }
    return comp_list_from_spd(spd, alive), info, spd


def run_lora(base, ft, k, seed, steps=1500, bs=256):
    torch.manual_seed(seed)
    A = {m: nn.Parameter(torch.randn(k[m], base[m].shape[1]) * 0.1) for m in base}
    B = {m: nn.Parameter(torch.zeros(base[m].shape[0], k[m])) for m in base}
    opt = torch.optim.Adam(list(A.values()) + list(B.values()), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        x = sample(bs, g)
        W = {m: base[m] + B[m] @ A[m] for m in base}
        loss = ((fwd(W, x) - fwd(ft, x)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ev = sample(2048, torch.Generator().manual_seed(99))
    with torch.no_grad():
        W = {m: base[m] + B[m] @ A[m] for m in base}
        rel = (((fwd(W, ev) - fwd(ft, ev)) ** 2).mean() / ((fwd(ft, ev) - fwd(base, ev)) ** 2).mean()).item()
        BA = {m: B[m] @ A[m] for m in base}
        raw = [(m, torch.outer(B[m][:, i], A[m][i])) for m in base for i in range(k[m])]
    return svd_comps(BA, k), raw, {"wall_s": time.time() - t0, "recon_rel": rel}


def evaluate(name, comps, base, ft, ch, out):
    if not ch:
        out[name] = {"n_comps": len(comps), "norm_share": sum(M.norm().item() ** 2 for _, M in comps)}
        return
    pc = recovery(comps, ch)
    sp = specificity(base, ft, comps, pc, ch)
    out[name] = {"recovery": {c: {"hung": d["hung"], "clus": d["clus"]} for c, d in pc.items()},
                 "spec": sp, "n_comps": len(comps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="exact")
    ap.add_argument("--coef", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="dpd,svd,lora")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--kextra", type=int, default=0)
    ap.add_argument("--Cfull", type=int, default=40)
    a = ap.parse_args()
    base, ft, ch, noise = make_ft(a.arm, a.seed, a.ratio)
    delta = {m: ft[m] - base[m] for m in base}
    k = {"l1": 2 + a.kextra, "l2": 3 + a.kextra}  # oracle k (planted pieces per matrix) + kextra
    out = {"args": vars(a), "delta_norm": {m: delta[m].norm().item() for m in delta}}
    if ch:
        out["oracle"] = oracle_check(base, ft, ch, noise)
        out["random"] = random_control(base, ft, ch, a.seed)
    meths = a.methods.split(",")
    if "svd" in meths:
        evaluate("svd", svd_comps(delta, k), base, ft, ch, out)
        xn = sample(4096, torch.Generator().manual_seed(7))
        acts = {"l1": xn, "l2": torch.relu(xn @ ft["l1"].T)}
        evaluate("svd_varimax", rotated_svd_comps(delta, k, "varimax"), base, ft, ch, out)
        evaluate("svd_ica", rotated_svd_comps(delta, k, "ica", acts), base, ft, ch, out)
        out["svd_gate"] = run_svd_gate(base, ft, delta, k, a.coef, a.seed, a.steps)
        out["svd_spectrum"] = {m: torch.linalg.svdvals(delta[m])[:6].tolist() for m in delta}
    if "lora" in meths:
        lsvd, lraw, info = run_lora(base, ft, k, a.seed)
        evaluate("lora_svd", lsvd, base, ft, ch, out)
        evaluate("lora_raw", lraw, base, ft, ch, out)
        out["lora_info"] = info
    if "dpd" in meths:
        comps, info, spd = run_pd(base, ft, True, a.C, a.coef, a.seed, a.steps, a.warm)
        evaluate("dpd", comps, base, ft, ch, out)
        out["dpd_info"] = info
        if not ch:  # drift arm: how much behaviour do alive components carry
            out["dpd"]["comp_norms"] = [M.norm().item() for _, M in comps]
    if "full" in meths:
        comps, info, spd = run_pd(base, ft, False, a.Cfull, a.coef, a.seed, a.steps, a.warm)
        # full PD components describe W_ft; for recovery compare to planted pieces directly
        evaluate("full", comps, base, ft, ch, out)
        out["full_info"] = info
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"{a.arm}_r{a.ratio:g}{'_kx' + str(a.kextra) if a.kextra else ''}_coef{a.coef:g}_seed{a.seed}_{a.methods.replace(',', '-')}.json"
    fn.write_text(json.dumps(out, indent=1))
    print(fn.name, "done")


if __name__ == "__main__":
    main()
