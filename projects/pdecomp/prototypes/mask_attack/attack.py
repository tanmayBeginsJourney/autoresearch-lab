"""Per-input mask attacks on an SPD decomposition of TMS 40-10 at equal evaluation budget.

Mask box (VPD App. A.3.4): m = g + (1 - g) r, r in [0, 1]^(2C) (both layers), g = clamp(CI, 0, 1).
Objective per input: MSE(masked output, target output) over the 40 outputs. Every evaluated point
is feasible, so the score is the best value seen so far. Budget = model evaluations per input
(a forward+backward counts as 1, which favours gradient methods by ~3x in FLOPs).
"""
import json, math, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spd_min import lower_leaky_hard_sigmoid  # noqa: E402
from spd_min.tms import sample_tms  # noqa: E402
import train  # noqa: E402

CHECK = [10, 20, 40, 80, 160, 320, 640]
BUDGET = CHECK[-1]


class Problem:
    def __init__(self, spd, x):
        y_t, acts = spd.target_and_inputs(x)
        ci = spd.causal_importances(acts)
        self.g = torch.cat([lower_leaky_hard_sigmoid(ci[n]).clamp(0, 1) for n in spd.names], -1).detach()
        k1, k2 = "linear1", "linear2"
        self.V1, self.U1 = spd.V[k1].detach(), spd.U[k1].detach()
        self.V2, self.U2 = spd.V[k2].detach(), spd.U[k2].detach()
        self.b2 = spd._mods["linear2"].bias.detach()
        self.h1 = (x @ self.V1.T).detach()
        self.y_t = y_t
        self.C = self.V1.shape[0]
        self.B = x.shape[0]
        self.n = 0  # evaluations used per input
        self.best = torch.full((self.B,), -1.0)
        self.curve = {}
        self.limit = BUDGET  # restart wrappers lower this per restart
        self.restart = 0

    def f(self, r, count=1):
        """r: (..., B, 2C) -> per-input MSE (..., B). Updates best-so-far and the eval counter."""
        m = self.g + (1 - self.g) * r
        m1, m2 = m[..., : self.C], m[..., self.C:]
        hid = (self.h1 * m1) @ self.U1.T
        y = F.relu(((hid @ self.V2.T) * m2) @ self.U2.T + self.b2)
        val = (y - self.y_t).pow(2).mean(-1)
        for v in val.detach().reshape(-1, self.B):  # candidates count one eval each, in order
            self.best = torch.maximum(self.best, v)
            self.n += 1
            if self.n in CHECK:
                self.curve[self.n] = self.best.clone()
        return val

    def grad(self, r):
        r = r.detach().requires_grad_(True)
        val = self.f(r)
        (gr,) = torch.autograd.grad(val.sum(), r)
        return val.detach(), gr

    def done(self):
        return self.n >= self.limit


def with_restarts(fn, R):
    """Split the budget into R equal independent runs of fn (later runs start from random points)."""
    def wrapped(P, gen):
        for i in range(R):
            P.restart, P.limit = i, BUDGET * (i + 1) // R
            fn(P, gen)
        P.limit = BUDGET
    return wrapped


def random_search(P, vertex, gen, chunk=40):
    while not P.done():
        k = min(chunk, BUDGET - P.n)
        r = torch.rand(k, P.B, 2 * P.C, generator=gen)
        P.f((r < 0.5).float() if vertex else r)


def pgd_sign(P, step, gen):
    r = torch.rand(P.B, 2 * P.C, generator=gen)
    while not P.done():
        _, gr = P.grad(r)
        r = (r + step * gr.sign()).clamp(0, 1)


def pgd_adam(P, lr, gen):
    r = torch.rand(P.B, 2 * P.C, generator=gen).requires_grad_(True)
    opt = torch.optim.Adam([r], lr=lr)
    while not P.done():
        val = P.f(r)
        opt.zero_grad()
        (-val.sum()).backward()
        opt.step()
        with torch.no_grad():
            r.clamp_(0, 1)


def apgd(P, gen, restarts=1, rho=0.75, alpha=0.75):
    """APGD-CE style (Croce & Hein 2020) for the L_inf box [0,1]: per-input step size halving at
    checkpoints, momentum 0.75, restart from the best point; `restarts` random restarts."""
    per = BUDGET // restarts
    for ri in range(restarts):
        n_it = per if ri < restarts - 1 else BUDGET - P.n
        if n_it <= 0:
            break
        p = [0, 0.22]
        while p[-1] < 1:
            p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
        cps = sorted({int(math.ceil(q * n_it)) for q in p[1:] if q < 1})
        x = torch.full((P.B, 2 * P.C), 0.5) if ri == 0 else torch.rand(P.B, 2 * P.C, generator=gen)
        eta = torch.ones(P.B, 1)  # 2*eps, eps = 0.5 (half the box width)
        fx, gr = P.grad(x)
        xbest, fbest, gbest = x.clone(), fx.clone(), gr.clone()
        x_prev = x.clone()
        succ = torch.zeros(P.B)
        last_cp, fbest_cp, eta_red_cp = 0, fbest.clone(), torch.ones(P.B, dtype=torch.bool)
        for k in range(1, n_it):
            z = (x + eta * gr.sign()).clamp(0, 1)
            a = alpha if k > 1 else 1.0
            x_new = (x + a * (z - x) + (1 - a) * (x - x_prev)).clamp(0, 1)
            x_prev = x
            f_new, g_new = P.grad(x_new)
            succ += (f_new > fx).float()
            imp = f_new > fbest
            xbest[imp], fbest[imp], gbest[imp] = x_new[imp], f_new[imp], g_new[imp]
            x, fx, gr = x_new, f_new, g_new
            if k in cps:
                h = k - last_cp
                c1 = succ < rho * h
                c2 = (~eta_red_cp) & (fbest_cp >= fbest)
                red = c1 | c2
                eta[red] /= 2
                x[red], fx[red], gr[red] = xbest[red], fbest[red], gbest[red]
                eta_red_cp, fbest_cp, last_cp = red, fbest.clone(), k
                succ.zero_()
            if P.done():
                return


def frank_wolfe(P, gen):
    """FW ascent on the box: LMO returns the vertex 1[grad > 0]; step 2/(k+2)."""
    r = torch.full((P.B, 2 * P.C), 0.5)
    k = 0
    while not P.done():
        _, gr = P.grad(r)
        s = (gr > 0).float()
        r = r + 2.0 / (k + 2) * (s - r)
        k += 1


def vertex_search(P, gen, K=4):
    """Greedy coordinate flips on box corners r in {0,1}^(2C). Start from the FW vertex at the
    centre. Each round: 1 gradient eval at the current vertex, then K single flips (ranked by
    first-order gain, skipping tried ones) + the all-positive-gain vertex; accept the best if it
    improves. Uses the fact that the MSE is convex along each coordinate (output is ReLU of a
    multilinear function of the masks), so the per-input maximum over the box is at a vertex."""
    if P.restart == 0:
        _, gr = P.grad(torch.full((P.B, 2 * P.C), 0.5))
        v = (gr > 0).float()
    else:
        v = (torch.rand(P.B, 2 * P.C, generator=gen) < 0.5).float()
    tried = torch.zeros_like(v, dtype=torch.bool)
    ar = torch.arange(P.B)
    while P.n + K + 2 <= P.limit:
        fv, gr = P.grad(v)
        gain = gr * (1 - 2 * v)
        gain_m = gain.masked_fill(tried, -float("inf"))
        idx = gain_m.topk(K, -1).indices  # (B, K)
        cand = v.unsqueeze(0).repeat(K + 1, 1, 1)
        for j in range(K):
            cand[j, ar, idx[:, j]] = 1 - cand[j, ar, idx[:, j]]
        cand[K] = torch.where(gain > 0, 1 - v, v)
        fc = P.f(cand)  # (K+1, B)
        bi = fc.argmax(0)
        fb = fc[bi, ar]
        acc = fb > fv + 1e-12
        v = torch.where(acc[:, None], cand[bi, ar], v)
        tried[acc] = False
        tried[ar[~acc].unsqueeze(1), idx[~acc]] = True
    while not P.done():  # spend leftovers with random flips around the incumbent
        flip = torch.rand(P.B, 2 * P.C, generator=gen) < 2.0 / (2 * P.C)
        P.f(torch.where(flip, 1 - v, v))


METHODS = {
    "random_uniform": lambda P, g: random_search(P, False, g),
    "random_vertex": lambda P, g: random_search(P, True, g),
    **{f"pgd_sign_{s:g}": (lambda s: lambda P, g: pgd_sign(P, s, g))(s) for s in (0.01, 0.03, 0.1, 0.3)},
    **{f"pgd_adam_{s:g}": (lambda s: lambda P, g: pgd_adam(P, s, g))(s) for s in (0.01, 0.05, 0.2)},
    "apgd_1": lambda P, g: apgd(P, g, 1),
    "apgd_4": lambda P, g: apgd(P, g, 4),
    "frank_wolfe": frank_wolfe,
    "vertex_search": vertex_search,
    "pgd_sign_0.3_r8": with_restarts(lambda P, g: pgd_sign(P, 0.3, g), 8),
    "vertex_search_r8": with_restarts(vertex_search, 8),
}


def load_ext52(tag):
    """Read-only: 5-2 decompositions trained by spd_min/train_tms.py (3000 steps, C=20)."""
    from spd_min import SPD
    from spd_min.tms import TMS
    d = Path("/home/user/lab/tmp/spd-toy")
    tms = TMS(5, 2); tms.load_state_dict(torch.load(d / "tms_5-2_seed0.pt")); tms.tied = False
    spd = SPD(tms.requires_grad_(False), {"linear1": 20, "linear2": 20}, ci_hidden=16)
    spd.load_state_dict(torch.load(d / "runs" / f"{tag}.pt"))
    return spd


def run(tag, C=80, n_inputs=2048, seed=999):
    torch.set_num_threads(1)
    spd = load_ext52(tag) if tag.startswith("tms52") else train.load_spd(tag, C)
    nf = spd._mods["linear1"].weight.shape[1]
    C = spd.V["linear1"].shape[0]
    x = sample_tms(n_inputs, nf, gen=torch.Generator().manual_seed(seed))
    x = x[x.abs().sum(-1) > 0]  # x = 0 gives output relu(b) for every mask
    out = {"tag": tag, "n_inputs": int(x.shape[0])}
    P0 = Problem(spd, x)
    with torch.no_grad():
        out["ref"] = {
            "ci_mask_mse": P0.f(torch.zeros(x.shape[0], 2 * C)).mean().item(),
            "all_on_mse": P0.f(torch.ones(x.shape[0], 2 * C)).mean().item(),
            "stoch_mse_mean": P0.f(torch.rand(64, x.shape[0], 2 * C)).mean().item(),
            "target_out_sq": P0.y_t.pow(2).mean().item(),
            "g_mean_free_coords": (P0.g < 0.99).float().sum(-1).mean().item(),
        }
    res = {}
    for name, fn in METHODS.items():
        t0 = time.time()
        P = Problem(spd, x)
        fn(P, torch.Generator().manual_seed(0))
        res[name] = {"curve_mean": {n: P.curve[n].mean().item() for n in CHECK},
                     "curve_p90": {n: P.curve[n].quantile(0.9).item() for n in CHECK},
                     "per_input_final": P.curve[BUDGET], "s": round(time.time() - t0, 1)}
    # ensemble best-known per input (max over all methods at the full budget)
    ens = torch.stack([r["per_input_final"] for r in res.values()]).max(0).values
    out["ensemble_mean"] = ens.mean().item()
    for name, r in res.items():
        r["frac_of_ensemble_at_640"] = (r["per_input_final"] / ens.clamp_min(1e-12)).mean().item()
        r["frac_inputs_within_1pct_of_ens"] = (r["per_input_final"] >= 0.99 * ens).float().mean().item()
        del r["per_input_final"]
    out["methods"] = res
    p = Path(__file__).parent / "results" / f"{tag}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = run(sys.argv[1], C=int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    print(json.dumps(o["ref"]), "ens", o["ensemble_mean"])
    for k, r in o["methods"].items():
        print(f"{k:16s} " + " ".join(f"{r['curve_mean'][n]:.2e}" for n in CHECK),
              f"frac_ens={r['frac_of_ensemble_at_640']:.3f} t={r['s']}")
