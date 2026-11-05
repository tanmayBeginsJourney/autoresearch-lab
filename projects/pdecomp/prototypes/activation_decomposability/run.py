"""G4 exploratory prototype: does the activation function change SPD decomposability at matched loss?

Target: one-hidden-layer MLP y = W_out act(W_in x + b_in) + b_out on sparse features x_i ~ U[-1,1]
w.p. p, label y_i = g(x_i) (g = abs or sq). Bilinear: h = (W1 x + b1) * (W2 x + b2).
Loss matching: train all activations at a common width; L* = worst final val loss among the four;
every model is early-stopped at the first eval where val loss <= L* (so losses match to eval noise).
Then decompose every matrix with spd_min (same SPDConfig for all), sweep coeff_imp.

Usage: python run.py --target abs --seed 0 --out res_abs_s0.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spd_min.spd import SPD, SPDConfig, lower_leaky_hard_sigmoid, train_spd  # noqa: E402

torch.set_num_threads(1)
N_FEAT, D_MLP, P_ACT = 16, 16, 0.08
ACTS = ["relu", "gelu", "silu", "bilinear"]
GFN = {"abs": torch.abs, "sq": lambda x: x * x}


def sample(batch, gen):
    v = torch.rand(batch, N_FEAT, generator=gen) * 2 - 1
    return v * (torch.rand(batch, N_FEAT, generator=gen) < P_ACT)


class MLP(nn.Module):
    def __init__(self, act, d_mlp):
        super().__init__()
        self.act = act
        self.w_in = nn.Linear(N_FEAT, d_mlp)
        if act == "bilinear":
            self.w_in2 = nn.Linear(N_FEAT, d_mlp)
        self.w_out = nn.Linear(d_mlp, N_FEAT)

    def forward(self, x):
        a = self.w_in(x)
        if self.act == "bilinear":
            h = a * self.w_in2(x)
        else:
            h = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}[self.act](a)
        return self.w_out(h)


def train_target(act, g, seed, steps, stop_at=None, eval_every=50):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xv = sample(4096, torch.Generator().manual_seed(10_000 + seed))
    yv = g(xv)
    m = MLP(act, D_MLP)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    val, step = None, 0
    for step in range(1, steps + 1):
        x = sample(512, gen)
        loss = F.mse_loss(m(x), g(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            with torch.no_grad():
                val = F.mse_loss(m(xv), yv).item()
            if stop_at is not None and val <= stop_at:
                break
    return m.requires_grad_(False), val, step


@torch.no_grad()
def l0_stats(spd, x, thr=0.1):
    _, acts = spd.target_and_inputs(x)
    g = {n: lower_leaky_hard_sigmoid(v) for n, v in spd.causal_importances(acts).items()}
    n_act = (x != 0).sum(-1).float()
    nz = n_act > 0
    out = {}
    tot = torch.zeros(x.shape[0])
    for n, v in g.items():
        l0 = (v > thr).float().sum(-1)
        tot += l0
        out[f"l0/{n}"] = l0.mean().item()
        out[f"l0_per_feat/{n}"] = (l0[nz] / n_act[nz]).mean().item()
        out[f"alive/{n}"] = int((v > thr).any(0).sum())
    out["l0_total"] = tot.mean().item()
    out["l0_total_zero_input"] = tot[~nz].mean().item() if (~nz).any() else float("nan")
    return out, g


def pgd_recon(spd, x, g, steps=20, alpha=0.1, seed=0):
    """Max over r in [0,1] of MSE(model with masks g + (1-g) r, target); per-sample r (sign PGD)."""
    torch.manual_seed(seed)
    y_t, _ = spd.target_and_inputs(x)
    r = {n: torch.rand_like(v).requires_grad_(True) for n, v in g.items()}
    for _ in range(steps):
        loss = F.mse_loss(spd.run(x, {n: g[n] + (1 - g[n]) * r[n] for n in g}), y_t)
        grads = torch.autograd.grad(loss, list(r.values()))
        with torch.no_grad():
            for (n, rv), gr in zip(r.items(), grads):
                rv.add_(alpha * gr.sign()).clamp_(0, 1)
    with torch.no_grad():
        return F.mse_loss(spd.run(x, {n: g[n] + (1 - g[n]) * r[n] for n in g}), y_t).item()


@torch.no_grad()
def gt_recovery(spd, model):
    """Input matrices: mean over features i of max_c |cos(U_c V_c^T, W[:, i] e_i^T)|."""
    out = {}
    for n in spd.names:
        if not n.startswith("w_in"):
            continue
        W = spd.target_weight(n)  # (d_mlp, n_feat)
        k = spd._key(n)
        U = F.normalize(spd.U[k], dim=0)  # (d_out, C)
        V = F.normalize(spd.V[k], dim=1)  # (C, d_in)
        Wn = F.normalize(W, dim=0)
        cos = (U.T @ Wn).abs() * V.abs()  # (C, n_feat): cos(U_c, W_i) * |V_c[i]|
        out[f"gt_maxcos/{n}"] = cos.max(0).values.mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="abs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=6000)
    ap.add_argument("--spd_steps", type=int, default=3000)
    ap.add_argument("--C", type=int, default=40)
    ap.add_argument("--coeffs", default="1e-3,3e-3,1e-2")
    ap.add_argument("--acts", default=",".join(ACTS), help="arms to decompose (all are trained for L*)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = GFN[a.target]
    t0 = time.time()
    full = {act: train_target(act, g, a.seed, a.train_steps) for act in ACTS}
    lstar = max(v[1] for v in full.values())
    models = {act: train_target(act, g, a.seed, a.train_steps, stop_at=lstar) for act in ACTS}
    res = {"target": a.target, "seed": a.seed, "L_star": lstar,
           "full_val": {k: v[1] for k, v in full.items()}, "runs": []}
    print(f"[{time.time()-t0:.0f}s] L*={lstar:.3e} full={res['full_val']}", flush=True)
    xe = sample(2048, torch.Generator().manual_seed(20_000 + a.seed))
    gen = torch.Generator().manual_seed(30_000 + a.seed)
    for act, (m, val, steps) in models.items():
        if act not in a.acts.split(","):
            continue
        names = ["w_in", "w_out"] + (["w_in2"] if act == "bilinear" else [])
        for coeff in [float(c) for c in a.coeffs.split(",")]:
            torch.manual_seed(a.seed)  # same SPD init seed across arms
            spd = SPD(m, {n: a.C for n in names})
            cfg = SPDConfig(coeff_imp=coeff, pnorm=1.0)
            train_spd(spd, lambda: sample(256, gen), cfg, a.spd_steps, lr=2e-3,
                      log_every=10**9, faith_warmup_steps=200)
            ev = spd.eval_metrics(xe)
            l0, gm = l0_stats(spd, xe)
            yvar = spd.target_and_inputs(xe)[0].pow(2).mean().item()
            rec = {"act": act, "coeff_imp": coeff, "task_val": val, "train_steps": steps,
                   "y_ms": yvar, "recon_stoch": ev["recon_stoch"], "recon_ci": ev["recon_ci_masked"],
                   "recon_unmasked": ev["recon_unmasked"], "faith": ev["faith"],
                   "recon_adv": pgd_recon(spd, xe, gm, seed=a.seed), **l0, **gt_recovery(spd, m)}
            res["runs"].append(rec)
            print(f"[{time.time()-t0:.0f}s] {act} c={coeff:g} val={val:.2e} st={steps} "
                  f"L0={rec['l0_total']:.2f} stoch={rec['recon_stoch']:.2e} "
                  f"adv={rec['recon_adv']:.2e}", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
