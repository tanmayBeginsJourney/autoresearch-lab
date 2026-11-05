"""Delta-PD v3 (EXPLORATORY): regimes where rotated SVD of dW should fail.

Model f(x) = W2 ReLU(W1 E x + b1); E (d x n) fixed embedding, b1 fixed bias; only W1, W2 change.
Features x_i ~ Bernoulli(p) * U[0, 1]. Fine-tune = base + hand-set rank-one pieces (exact).

Regime A (superposed / overcomplete): n=6 features embedded in d=3 (icosahedron axes, |cos|=0.447).
  Base: neurons 0-5 = ReLU(e_k.h - 0.5) -> output k; spare neurons 6-11 have bias -0.5, zero input
  row, and already write u_k (random unit, R^8). Change k (k=0..5): W1 row 6+k = e_k^T. So dW1 =
  sum_k e_{6+k} e_k^T has 6 pieces but rank 3; each change fires only when feature k is present
  (the -0.5 bias filters the <= 0.447 interference). Any method confined to the column space of dW1
  (SVD, rotations within its top-k subspace) has best possible |cos| sqrt(leverage) = 0.71 per piece.
Regime B (backdoor through a ReLU conjunction): n=8, d=8, E random orthogonal. Base: neuron i reads
  feature i -> output i (i<8); spare neuron 9 has bias -1. Changes:
    b1 benign add:  W1 e8 (x) e_2 ; W2 0.8 u_b1 (x) e8         (x2 -> u_b1)
    b2 benign edit: W2 0.6 u_b2 (x) e4                          (x4 also writes u_b2)
    bd backdoor:    W1 e9 (x) (e_2 + e_7) ; W2 2.0 u_t (x) e9   (fires only if x2 + x7 > 1)
  Trigger = feature 7 (also a normal feature). cos(u_t, u_b1) = 0.5. bd and b1 share input feature 2.
  --trigfree: decomposition / LoRA data never contain the trigger (x7 = 0).
Regime C (cross-layer): n=8, d=8, E random orthogonal, base as B (spare biases 0). Mechanisms:
    m0: W1 e8 (x) e_1, W2 0.8 u0 (x) e8 ; m1: W1 e9 (x) e_3, W2 0.8 u1 (x) e9
    m2 (coordinated edit of a base neuron): W1 e5 (x) e_6 (neuron 5 also reads x6), W2 0.8 u2 (x) e5.
       x6 -> u2 needs both pieces; x5 -> u2 needs only the W2 piece; x6 -> y5 only the W1 piece.
  pairwise cos(u_i, u_j) = 0.5.

Usage: python delta_pd_v3.py --regime A --seed 0 [--steps 4000] [--trigfree]
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
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "delta_pd_v2"))
from spd_min import SPD, SPDConfig, lower_leaky_hard_sigmoid, train_spd  # noqa: E402
from delta_pd_v2 import fastica, fcos, recovery, varimax  # noqa: E402

torch.set_num_threads(1)
OUT = HERE / "results"


def e(i, d):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def unit(v):
    return v / v.norm()


def rand_unit(d, g):
    return unit(torch.randn(d, generator=g))


def cos_pair(d, g, c=0.5, k=3):
    """k unit vectors in R^d with pairwise cosine c (equiangular via shared component)."""
    z = rand_unit(d, g)
    out = []
    for _ in range(k):
        r = torch.randn(d, generator=g)
        r -= (r @ z) * z
        for o in out:
            pass
        out.append(r)
    # orthonormalise residuals, then mix: u_i = sqrt(c) z + sqrt(1-c) r_i
    R = torch.linalg.qr(torch.stack(out, 1))[0]
    return [math.sqrt(c) * z + math.sqrt(1 - c) * R[:, i] for i in range(k)]


# ---------------------------------------------------------------- toys
def build(regime, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    T = {}
    if regime == "A":
        n, d, H, O = 6, 3, 12, 8
        phi = (1 + 5 ** 0.5) / 2
        ax = torch.tensor([[0, 1, phi], [0, -1, phi], [1, phi, 0], [-1, phi, 0], [phi, 0, 1], [-phi, 0, 1]])
        E = (ax / ax.norm(dim=1, keepdim=True)).T  # d x n
        W1 = torch.zeros(H, d)
        W1[:6] = E.T
        b1 = torch.full((H,), -0.5)
        W2 = torch.zeros(O, H)
        us = [rand_unit(O, g) for _ in range(6)]
        for k in range(6):
            W2[k, k] = 1.0
            W2[:, 6 + k] = us[k]
        ch = {f"a{k}": [("l1", 1.0, e(6 + k, H), E[:, k].clone())] for k in range(6)}
        ts = [0.6, 0.8, 1.0]
        probes = {f"a{k}": torch.stack([t * e(k, n) for t in ts]) for k in range(6)}
        p = 0.05
        trig = None
    else:
        n, d, H, O = 8, 8, 16, (10 if regime == "B" else 8)
        E = torch.linalg.qr(torch.randn(d, n, generator=g))[0]
        W1 = torch.zeros(H, d)
        W1[:8] = E.T
        b1 = torch.zeros(H)
        W2 = torch.zeros(O, H)
        for i in range(8):
            W2[i, i] = 1.0
        ts = [0.25, 0.5, 0.75, 1.0]
        p = 0.1
        if regime == "B":
            b1[9] = -1.0
            ub1, ut = cos_pair(O, g, 0.5, 2)
            ub2 = rand_unit(O, g)
            ch = {
                "b1": [("l1", 1.0, e(8, H), E[:, 2].clone()), ("l2", 0.8, ub1, e(8, H))],
                "b2": [("l2", 0.6, ub2, e(4, H))],
                "bd": [("l1", math.sqrt(2), e(9, H), unit(E[:, 2] + E[:, 7])), ("l2", 2.0, ut, e(9, H))],
            }
            conj = [(0.8, 1.0), (1.0, 0.8), (1.0, 1.0), (0.9, 0.9), (0.7, 0.7)]
            probes = {
                "b1": torch.stack([t * e(2, n) for t in ts]),
                "b2": torch.stack([t * e(4, n) for t in ts]),
                "bd": torch.stack([a * e(2, n) + b * e(7, n) for a, b in conj]),
            }
            trig = 7
        else:
            u0, u1, u2 = cos_pair(O, g, 0.5, 3)
            ch = {
                "m0": [("l1", 1.0, e(8, H), E[:, 1].clone()), ("l2", 0.8, u0, e(8, H))],
                "m1": [("l1", 1.0, e(9, H), E[:, 3].clone()), ("l2", 0.8, u1, e(9, H))],
                "m2": [("l1", 1.0, e(5, H), E[:, 6].clone()), ("l2", 0.8, u2, e(5, H))],
            }
            probes = {
                "m0": torch.stack([t * e(1, n) for t in ts]),
                "m1": torch.stack([t * e(3, n) for t in ts]),
                "m2": torch.stack([t * e(i, n) for i in (6, 5) for t in ts]),
            }
            trig = None
    base = {"l1": W1, "l2": W2}
    ft = {m: base[m].clone() for m in base}
    for pl in ch.values():
        for m, s, u, v in pl:
            ft[m] += s * torch.outer(u, v)
    T.update(n=n, d=d, H=H, O=O, E=E, b1=b1, base=base, ft=ft, ch=ch, probes=probes, p=p, trig=trig)
    return T


class Net(nn.Module):
    def __init__(self, T, W):
        super().__init__()
        self.register_buffer("E", T["E"].clone())
        self.l1 = nn.Linear(T["d"], T["H"])
        self.l2 = nn.Linear(T["H"], T["O"], bias=False)
        with torch.no_grad():
            self.l1.weight.copy_(W["l1"])
            self.l1.bias.copy_(T["b1"])
            self.l2.weight.copy_(W["l2"])

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x @ self.E.T)))


def fwd(T, W, x):
    return torch.relu((x @ T["E"].T) @ W["l1"].T + T["b1"]) @ W["l2"].T


def sample(T, bs, g, trigfree=False):
    act = (torch.rand(bs, T["n"], generator=g) < T["p"]).float()
    x = act * torch.rand(bs, T["n"], generator=g)
    if trigfree and T["trig"] is not None:
        x[:, T["trig"]] = 0
    return x


def layer_acts(T, W, x):
    h = x @ T["E"].T
    return {"l1": h, "l2": torch.relu(h @ W["l1"].T + T["b1"])}


# ---------------------------------------------------------------- baselines
def changed(T):
    return sorted({m for pl in T["ch"].values() for m, *_ in pl})


def npieces(T):
    c = {}
    for pl in T["ch"].values():
        for m, *_ in pl:
            c[m] = c.get(m, 0) + 1
    return c


def svd_comps(D, k):
    out = []
    for n, M in D.items():
        U, S, Vh = torch.linalg.svd(M)
        for i in range(min(k[n], len(S))):
            out.append((n, S[i] * torch.outer(U[:, i], Vh[i])))
    return out


def rot_comps(D, k, mode, acts):
    out = []
    for n, M in D.items():
        U, S, Vh = torch.linalg.svd(M.double())
        kk = min(k[n], len(S))  # rotations live in the top-k SVD subspace; k <= min(d_out, d_in)
        Uk, Sk, Vk = U[:, :kk].numpy(), S[:kk].numpy(), Vh[:kk].T.numpy()
        if mode == "varimax":
            R = varimax(Vk)
            us, vs = Uk @ np.diag(Sk) @ R, Vk @ R
        else:
            Wm = fastica(acts[n].double().numpy() @ Vk)
            us, vs = Uk @ np.diag(Sk) @ np.linalg.inv(Wm), Vk @ Wm.T
        for i in range(kk):
            out.append((n, torch.tensor(np.outer(us[:, i], vs[:, i]), dtype=torch.float32)))
    return out


def run_lora(T, k, seed, trigfree, steps=1500, bs=256):
    torch.manual_seed(seed)
    base, ft = T["base"], T["ft"]
    ms = changed(T)
    A = {m: nn.Parameter(torch.randn(k[m], base[m].shape[1]) * 0.1) for m in ms}
    B = {m: nn.Parameter(torch.zeros(base[m].shape[0], k[m])) for m in ms}
    opt = torch.optim.Adam(list(A.values()) + list(B.values()), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = sample(T, bs, g, trigfree)
        W = {m: base[m] + (B[m] @ A[m] if m in ms else 0) for m in base}
        loss = ((fwd(T, W, x) - fwd(T, ft, x)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        BA = {m: B[m] @ A[m] for m in ms}
        raw = [(m, torch.outer(B[m][:, i], A[m][i])) for m in ms for i in range(k[m])]
        ev = sample(T, 2048, torch.Generator().manual_seed(99))
        W = {m: base[m] + (BA[m] if m in ms else 0) for m in base}
        rel = (((fwd(T, W, ev) - fwd(T, ft, ev)) ** 2).mean() / ((fwd(T, ft, ev) - fwd(T, base, ev)) ** 2).mean()).item()
    return svd_comps(BA, k), raw, rel


def run_pd(T, delta, C, coef, seed, steps, warm, trigfree, bs=256):
    torch.manual_seed(seed)
    ms = changed(T)
    net = Net(T, T["ft"])
    spd = SPD(net, {m: C for m in ms}, base_weights={m: T["base"][m] for m in ms} if delta else None)
    g = torch.Generator().manual_seed(seed)
    ev = sample(T, 2048, torch.Generator().manual_seed(99), trigfree)
    cm = ((fwd(T, T["ft"], ev) - fwd(T, T["base"], ev)) ** 2).mean().item()
    curve = []

    def log(step, rec):
        m = spd.eval_metrics(ev)
        curve.append({"step": step, "ci_rel": m["recon_ci_masked"] / cm, "stoch_rel": m["recon_stoch"] / cm, "faith": m["faith"]})

    t0 = time.time()
    train_spd(spd, lambda: sample(T, bs, g, trigfree), SPDConfig(coeff_imp=coef, pnorm=1.0), steps, lr=2e-3,
              log_every=250, log=log, faith_warmup_steps=warm)
    wall = time.time() - t0
    xall = torch.cat([ev] + list(T["probes"].values()))
    _, acts = spd.target_and_inputs(xall)
    gci = {n: lower_leaky_hard_sigmoid(v) for n, v in spd.causal_importances(acts).items()}
    comps, gidx = [], []
    for n in ms:
        S = spd.subcomponents(n).detach()
        alive = gci[n].max(0).values > 0.1
        for c in range(S.shape[0]):
            # delta mode: also keep weight-only components (dormant: big norm, gate never > 0.1)
            if alive[c] or (delta and S[c].norm() > 0.1):
                comps.append((n, S[c]))
                gidx.append((n, c, bool(alive[c])))
    hit = next((r for r in curve if r["ci_rel"] <= 0.05 and r["stoch_rel"] <= 0.05), None)
    info = {"wall_s": wall, "alive": {n: int((gci[n].max(0).values > 0.1).sum()) for n in ms},
            "n_kept": len(comps), "final": curve[-1], "steps_to_5pct": None if hit is None else hit["step"],
            "curve": curve[::4]}
    return comps, gidx, info, spd


# ---------------------------------------------------------------- metrics
def revert(T, comps, idxs):
    W = {m: T["ft"][m].clone() for m in T["ft"]}
    for j in idxs:
        W[comps[j][0]] = W[comps[j][0]] - comps[j][1]
    return W


def rel(a, b, ref):
    return ((a - b).norm() / (ref.norm() + 1e-12)).item()


def effect(T, W, c, clean):
    base, ft = T["base"], T["ft"]
    x = T["probes"][c]
    # target = f_ft with only change c's pieces removed (other changes active on the same probe stay)
    Wo = {m: ft[m].clone() for m in ft}
    for m, s, u, v in T["ch"][c]:
        Wo[m] = Wo[m] - s * torch.outer(u, v)
    yb, yf, yr = fwd(T, Wo, x), fwd(T, ft, x), fwd(T, W, x)
    out = {"restore": 1 - rel(yr, yb, yf - yb)}
    coll = [rel(fwd(T, W, T["probes"][o]), fwd(T, ft, T["probes"][o]), fwd(T, ft, T["probes"][o]) - fwd(T, base, T["probes"][o]))
            for o in T["ch"] if o != c]
    out["collateral"] = float(np.mean(coll))
    # clean natural inputs (trigger-free for B): output change of the revert relative to |f_ft - f_base|
    out["clean_dmg"] = rel(fwd(T, W, clean), fwd(T, ft, clean), fwd(T, ft, clean) - fwd(T, base, clean))
    return out


def evaluate(T, comps, clean):
    pc = recovery(comps, T["ch"]) if comps else {c: {"hung": [0.0] * len(pl), "clus": [0.0] * len(pl), "comps": []} for c, pl in T["ch"].items()}
    res = {}
    for c, d in pc.items():
        r = effect(T, revert(T, comps, d["comps"]), c, clean)
        r.update(hung=float(np.mean(d["hung"])), hung_pieces=d["hung"], clus=float(np.mean(d["clus"])), n_comps=len(d["comps"]))
        res[c] = r
    return res, pc


def random_control(T, seed, clean):
    g = torch.Generator().manual_seed(3000 + seed)
    res = {}
    for c, pl in T["ch"].items():
        W = {m: T["ft"][m].clone() for m in T["ft"]}
        for m, s, u, v in pl:
            W[m] = W[m] - s * torch.outer(rand_unit(len(u), g), rand_unit(len(v), g))
        res[c] = effect(T, W, c, clean)
    return res


def oracle(T, clean):
    res = {}
    for c, pl in T["ch"].items():
        W = {m: T["ft"][m].clone() for m in T["ft"]}
        for m, s, u, v in pl:
            W[m] = W[m] - s * torch.outer(u, v)
        res[c] = effect(T, W, c, clean)
    return res


def pairing(T, comps, pc, activity):
    """Cross-layer grouping: for each 2-layer change, is the l2 comp most co-active with its matched l1
    comp (Pearson over natural samples of the activity signal) the matched l2 comp?"""
    ok = []
    l2 = [j for j, (m, _) in enumerate(comps) if m == "l2"]
    for c, pl in T["ch"].items():
        if len({m for m, *_ in pl}) < 2:
            continue
        j1 = [j for j in pc[c]["comps"] if comps[j][0] == "l1"]
        j2 = [j for j in pc[c]["comps"] if comps[j][0] == "l2"]
        if not j1 or not j2:
            ok.append(0.0)
            continue
        a1 = activity[:, j1].sum(1)
        cor = [np.corrcoef(a1, activity[:, j])[0, 1] if activity[:, j].std() > 0 and a1.std() > 0 else -1 for j in l2]
        ok.append(float(l2[int(np.nanargmax(cor))] in j2))
    return float(np.mean(ok)) if ok else None


def linear_activity(T, comps, x):
    acts = layer_acts(T, T["ft"], x)
    return np.stack([(acts[m] @ M.T).norm(dim=1).numpy() for m, M in comps], 1)


def dpd_gates(spd, gidx, x):
    _, acts = spd.target_and_inputs(x)
    gci = {n: lower_leaky_hard_sigmoid(v) for n, v in spd.causal_importances(acts).items()}
    return np.stack([gci[n][:, c].detach().numpy() for n, c, _ in gidx], 1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--coef", type=float, default=1e-3)
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--trigfree", action="store_true")
    ap.add_argument("--nofull", action="store_true")
    a = ap.parse_args()
    T = build(a.regime, a.seed)
    ms = changed(T)
    delta = {m: T["ft"][m] - T["base"][m] for m in ms}
    kt = npieces(T)
    out = {"args": vars(a), "rank": {m: int(torch.linalg.matrix_rank(delta[m])) for m in ms}, "pieces": kt}
    clean = sample(T, 2048, torch.Generator().manual_seed(77), trigfree=True)
    xn = sample(T, 4096, torch.Generator().manual_seed(7), a.trigfree)
    acts = layer_acts(T, T["ft"], xn)
    out["oracle"] = oracle(T, clean)
    out["random"] = random_control(T, a.seed, clean)
    M = {}
    for kx in (0, 2):
        k = {m: kt[m] + kx for m in ms}
        M[f"svd_k{kx}"] = svd_comps(delta, k)
        M[f"varimax_k{kx}"] = rot_comps(delta, k, "varimax", acts)
        M[f"ica_k{kx}"] = rot_comps(delta, k, "ica", acts)
    lsvd, lraw, lrel = run_lora(T, kt, a.seed, a.trigfree)
    M["lora_svd"], M["lora_raw"] = lsvd, lraw
    out["lora_recon_rel"] = lrel
    comps, gidx, info, spd = run_pd(T, True, a.C, a.coef, a.seed, a.steps, a.warm, a.trigfree)
    M["dpd"] = comps  # alive comps plus weight-only (never-gated, norm > 0.1) comps
    M["dpd_alive"] = [cm for cm, gi in zip(comps, gidx) if gi[2]]  # gate > 0.1 somewhere only
    out["dpd_info"] = info
    if not a.nofull:
        fcomps, _, finfo, _ = run_pd(T, False, 2 * T["H"], a.coef, a.seed, a.steps, a.warm, a.trigfree)
        M["full"] = fcomps
        out["full_info"] = {k: v for k, v in finfo.items() if k != "curve"}
    res = {}
    for name, comps_ in M.items():
        r, pc = evaluate(T, comps_, clean)
        res[name] = r
        if a.regime in ("B", "C") and comps_:
            r["_pair_linear"] = pairing(T, comps_, pc, linear_activity(T, comps_, xn))
            if name == "dpd":
                r["_pair_gate"] = pairing(T, comps_, pc, dpd_gates(spd, gidx, xn))
        if name == "dpd" and a.regime == "B":
            # dormancy of the backdoor-matched comps: gate on clean (trigger-free) data vs on trigger probes
            G_clean = dpd_gates(spd, gidx, clean)
            G_trig = dpd_gates(spd, gidx, T["probes"]["bd"])
            js = pc["bd"]["comps"]
            r["_bd_gate_clean_max"] = [float(G_clean[:, j].max()) for j in js]
            r["_bd_gate_clean_frac_gt0.1"] = [float((G_clean[:, j] > 0.1).mean()) for j in js]
            r["_bd_gate_trig_mean"] = [float(G_trig[:, j].mean()) for j in js]
            r["_bd_comps_alive_flag"] = [gidx[j][2] for j in js]
            # how many dpd comps are dormant on clean data, and are they the backdoor's?
            dorm = [j for j in range(len(gidx)) if G_clean[:, j].max() <= 0.1]
            r["_dormant_on_clean"] = dorm
            r["_dormant_are_bd"] = [j in js for j in dorm]
    out["methods"] = res
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"{a.regime}{'_trigfree' if a.trigfree else ''}_steps{a.steps}_seed{a.seed}.json"
    fn.write_text(json.dumps(out, indent=1))
    print(fn.name, "done", f"{info['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
