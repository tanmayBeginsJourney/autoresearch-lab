"""G3 exploratory toy: cluster ground-truth rank-one subcomponents into multi-matrix mechanisms.

Model: y = W2 relu(W1 x + b1), W1 = sum_k u_k e_k^T (layer-1 piece k), W2 = sum_k c_k u_k^T (layer-2 piece k).
Mechanism k = {layer-1 piece k, layer-2 piece k}. C = 2K subcomponents, known ground truth.
Regimes: clean (h=d_out=K), outsup (h=K, d_out=6: overlapping outputs), hidsup (h=8, d_out=6).
Data: x_k = z_k * U(0,1), z_k ~ Bern(p). Features are paired (2m, 2m+1): z_{2m+1} = z_{2m} with
prob rho, else an independent Bern(p) draw (marginal stays p).
Loss f(mask) = per-example MSE between masked-model output and full-model output (PD reconstruction).

Similarities (all mapped to [0,1], clustered by average-linkage on 1 - S):
  coact_act : Jaccard of binarized activity |v_c . input_c| * ||u_c|| > tau      (co-activation, VPD-like)
  coact_ci  : Jaccard of binarized causal CI  f(ablate c) > eps                 (co-activation of causal CI)
  int_full  : |E_x[f(-i,-j) - f(-i) - f(-j) + f(none)]| from the all-on context, normalized
  int_banz  : |Banzhaf interaction| from least squares of f on random masks (q=0.5), normalized
  comb      : sqrt(coact_ci * int_banz)
Metric: ARI vs true partition, (i) with oracle K, (ii) K chosen by max silhouette.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

torch.set_num_threads(1)


def make_model(K, h, d_out, seed):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(h, K, generator=g)
    U = U / U.norm(dim=0, keepdim=True)          # u_k columns
    Cm = torch.randn(d_out, K, generator=g)
    Cm = Cm / Cm.norm(dim=0, keepdim=True)        # c_k columns
    if h == K:
        U = torch.eye(K)
    if d_out == K:
        Cm = torch.eye(K)
    b1 = torch.full((h,), -0.05)
    return U, Cm, b1


def make_data(N, K, p, rho, seed):
    g = torch.Generator().manual_seed(seed + 1000)
    z = (torch.rand(N, K, generator=g) < p).float()
    copy = torch.rand(N, K // 2, generator=g) < rho
    z[:, 1::2] = torch.where(copy, z[:, 0::2], z[:, 1::2])
    return z * torch.rand(N, K, generator=g)


def forward(x, U, Cm, b1, m1, m2):
    """m1, m2: (N, K) or (K,) masks on layer-1 and layer-2 pieces."""
    pre = (x * m1) @ U.T + b1                    # W1(m1) x = sum_k m1_k u_k x_k
    a = torch.relu(pre)
    r = a @ U                                     # u_k . a
    return (r * m2) @ Cm.T                         # sum_k m2_k c_k (u_k . a)


def loss_masked(x, U, Cm, b1, M, y_full):
    K = U.shape[1]
    y = forward(x, U, Cm, b1, M[..., :K], M[..., K:])
    return ((y - y_full) ** 2).sum(-1)


def jaccard(B):
    B = B.float()
    inter = B.T @ B
    cnt = B.sum(0)
    union = cnt[:, None] + cnt[None, :] - inter
    return (inter / union.clamp(min=1)).numpy()


def normalize_interaction(I, E):
    """|I_ij| / min(E_i, E_j), clipped to [0, 1]; E = mean single-ablation effect."""
    den = np.minimum(E[:, None], E[None, :]) + 1e-12
    S = np.clip(np.abs(I) / den, 0, 1)
    np.fill_diagonal(S, 1.0)
    return S


def interaction_full(x, U, Cm, b1, y_full):
    C = 2 * U.shape[1]
    ones = torch.ones(C)
    f_single = torch.stack([loss_masked(x, U, Cm, b1, ones - torch.eye(C)[i], y_full) for i in range(C)])  # (C, N)
    I = np.zeros((C, C))
    for i in range(C):
        for j in range(i + 1, C):
            m = ones.clone(); m[i] = 0; m[j] = 0
            fij = loss_masked(x, U, Cm, b1, m, y_full)
            I[i, j] = I[j, i] = (fij - f_single[i] - f_single[j]).mean().item()  # f(none)=0
    E = f_single.mean(1).numpy()
    return I, E, f_single


def interaction_banzhaf(x, U, Cm, b1, y_full, n_ex, n_masks, q, g, chunk=250):
    """Regress f(x, m) on [m_i, m_i m_j] with m ~ Bern(q) iid, n_masks masks per example and
    per-example fixed effects (demean f and regressors within example) to remove data variance.
    Under q=0.5, second-order coefficients estimate Banzhaf interaction indices and the
    coefficients of the best linear fit are Banzhaf values (Hammer & Holzman 1992)."""
    C = 2 * U.shape[1]
    iu = torch.triu_indices(C, C, 1)
    P = C + iu.shape[1]
    XtX = torch.zeros(P, P, dtype=torch.float64); Xty = torch.zeros(P, dtype=torch.float64)
    LtL = torch.zeros(C, C, dtype=torch.float64); Lty = torch.zeros(C, dtype=torch.float64)
    idx = torch.randint(0, x.shape[0], (n_ex,), generator=g)
    for s0 in range(0, n_ex, chunk):
        ii = idx[s0:s0 + chunk]
        B = len(ii)
        M = (torch.rand(B, n_masks, C, generator=g) < q).float()
        f = loss_masked(x[ii][:, None], U, Cm, b1, M, y_full[ii][:, None])          # (B, n_masks)
        X = torch.cat([M, M[..., iu[0]] * M[..., iu[1]]], -1)
        X = (X - X.mean(1, keepdim=True)).reshape(-1, P).double()
        L = (M - M.mean(1, keepdim=True)).reshape(-1, C).double()
        fy = (f - f.mean(1, keepdim=True)).reshape(-1).double()
        XtX += X.T @ X; Xty += X.T @ fy; LtL += L.T @ L; Lty += L.T @ fy
    beta = torch.linalg.solve(XtX, Xty)
    I = np.zeros((C, C))
    I[iu[0].numpy(), iu[1].numpy()] = beta[C:].numpy()
    I = I + I.T
    bl = torch.linalg.solve(LtL, Lty).abs().numpy()
    return I, bl


def cluster(S, k):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, k, criterion="maxclust"), Z, D


def silhouette(D, lab):
    n = len(lab)
    s = np.zeros(n)
    for i in range(n):
        same = (lab == lab[i]); same[i] = False
        if same.sum() == 0:
            s[i] = 0.0
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == l].mean() for l in np.unique(lab) if l != lab[i])
        s[i] = (b - a) / max(a, b, 1e-12)
    return s.mean()


def cluster_auto(S, kmax):
    _, Z, D = cluster(S, 2)
    best = None
    for k in range(2, kmax + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        sc = silhouette(D, lab)
        if best is None or sc > best[0]:
            best = (sc, lab, len(np.unique(lab)))
    return best[1], best[2]


def ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    T = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(T, (ai, bi), 1)
    c2 = lambda v: v * (v - 1) / 2
    sij = c2(T).sum(); sa = c2(T.sum(1)).sum(); sb = c2(T.sum(0)).sum(); n = c2(len(a))
    exp = sa * sb / n
    mx = (sa + sb) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def run(K, h, d_out, rho, seed, p=0.05, N=20000, n_ex=4000, n_masks=32, tau=0.05, eps_rel=0.01):
    U, Cm, b1 = make_model(K, h, d_out, seed)
    x = make_data(N, K, p, rho, seed)
    C = 2 * K
    ones = torch.ones(C)
    y_full = forward(x, U, Cm, b1, ones[:K], ones[K:])
    truth = np.concatenate([np.arange(K), np.arange(K)])

    # activity CI proxy: |inner activation| * ||u|| for each piece
    pre = x @ U.T + b1
    act = torch.cat([x.abs(), (torch.relu(pre) @ U).abs()], 1)   # ||u_k|| = ||e_k|| = 1
    S = {"coact_act": jaccard(act > tau)}
    I_full, E, f_single = interaction_full(x, U, Cm, b1, y_full)
    eps = eps_rel * f_single.max().item()
    S["coact_ci"] = jaccard((f_single > eps).T)
    S["int_full"] = normalize_interaction(I_full, E)
    g = torch.Generator().manual_seed(seed + 7)
    I_b, E_b = interaction_banzhaf(x, U, Cm, b1, y_full, n_ex, n_masks, 0.5, g)
    S["int_banz"] = normalize_interaction(I_b, E_b)
    S["comb"] = np.sqrt(S["coact_ci"] * S["int_banz"])
    out = {}
    for name, Sm in S.items():
        lab_k, _, _ = cluster(Sm, K)
        lab_a, k_a = cluster_auto(Sm, C - 1)
        _, Z, _ = cluster(Sm, K)
        lab_t = fcluster(Z, 0.5, criterion="distance")   # merge while average similarity > 0.5
        out[name] = {"ari_oracleK": ari(lab_k, truth), "ari_auto": ari(lab_a, truth), "k_auto": int(k_a),
                     "ari_thr": ari(lab_t, truth), "k_thr": int(len(np.unique(lab_t)))}
    # diagnostics: mean similarity within-mechanism vs paired-other-mechanism vs unrelated
    pairs_w = [(k, K + k) for k in range(K)]
    pairs_p = [(2 * m + a, 2 * m + 1 + b) for m in range(K // 2) for a in (0, K) for b in (0, K)]
    for name, Sm in S.items():
        w = np.mean([Sm[i, j] for i, j in pairs_w])
        pp = np.mean([Sm[i, j] for i, j in pairs_p])
        out[name]["S_within"] = float(w)
        out[name]["S_cooccurring_pair"] = float(pp)
        out[name]["margin"] = float(w - pp)
    return out


if __name__ == "__main__":
    regime = sys.argv[1] if len(sys.argv) > 1 else "superposed"
    K = 12
    h, d_out = {"clean": (K, K), "outsup": (K, 6), "hidsup": (8, 6)}[regime]
    res = []
    t0 = time.time()
    for rho in (0.0, 0.3, 0.6, 0.9, 1.0):
        for seed in (0, 1, 2):
            r = run(K, h, d_out, rho, seed)
            res.append({"regime": regime, "rho": rho, "seed": seed, **{k: v for k, v in r.items()}})
            print(rho, seed, {k: (round(v["ari_thr"], 2), round(v["margin"], 2)) for k, v in r.items()},
                  f"{time.time() - t0:.0f}s", flush=True)
    with open(f"/home/user/autoresearch-lab/projects/pdecomp/prototypes/interaction_clustering/results_{regime}.json", "w") as fh:
        json.dump(res, fh, indent=1)
