"""Kill-test: does teleporting an Adam-pretrained checkpoint into a balanced gauge help Muon fine-tuning?

Toy: residual MLP (RMSNorm gain -> W_u -> ReLU -> W_d), pretrained with AdamW on teacher A,
fine-tuned on related teacher B. Arms (paired seeds, LR tuned per arm on a validation set):
  adam_native, muon_native, muon_canon (fold gains, balance per-neuron norms), muon_random (control).
Usage: python gauge_muon.py [n_seeds]
"""
import math, sys, time, copy
import torch
torch.set_num_threads(1)

D_IN, D, D_FF, D_OUT, L = 32, 64, 256, 16, 3
PRE_STEPS, FT_STEPS, BS = 2000, 300, 256


def rms(h, g):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * g


def init(seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    p = {"E": r(D, D_IN) / math.sqrt(D_IN), "H": r(D_OUT, D) / math.sqrt(D)}
    for l in range(L):
        p[f"g{l}"] = torch.ones(D)
        p[f"u{l}"] = r(D_FF, D) / math.sqrt(D)
        p[f"d{l}"] = r(D, D_FF) / math.sqrt(D_FF) * 0.5
    return p


def fwd(p, x):
    h = x @ p["E"].T
    for l in range(L):
        h = h + torch.relu(rms(h, p[f"g{l}"]) @ p[f"u{l}"].T) @ p[f"d{l}"].T
    return rms(h, 1.0) @ p["H"].T


def teacher(seed, base=None, shift=0.0):
    gen = torch.Generator().manual_seed(seed)
    T1 = torch.randn(128, D_IN, generator=gen) / math.sqrt(D_IN)
    T2 = torch.randn(D_OUT, 128, generator=gen) / math.sqrt(128)
    if base is not None:
        T1 = base[0] + shift * T1
        T2 = base[1] + shift * T2
    return T1, T2


def tgt(T, x):
    return torch.tanh(x @ T[0].T * 1.5) @ T[1].T


def batch(gen, n=BS):
    return torch.randn(n, D_IN, generator=gen)


def loss_on(p, T, x):
    with torch.no_grad():
        y = tgt(T, x)
        return ((fwd(p, x) - y).pow(2).mean() / y.pow(2).mean()).item()


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def train(p, T, steps, lr, opt, seed, wd=0.0, adam_lr_aux=1e-3):
    """opt: 'adam' -> AdamW on everything; 'muon' -> Muon on u/d matrices, AdamW (lr adam_lr_aux) on the rest."""
    p = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    gen = torch.Generator().manual_seed(seed + 999)
    mats = [k for k in p if k[0] in "ud"]
    muon_keys = mats if opt == "muon" else []
    adam_keys = [k for k in p if k not in muon_keys]
    st = {k: [torch.zeros_like(p[k]), torch.zeros_like(p[k])] for k in p}
    b1, b2 = 0.9, 0.95
    for t in range(1, steps + 1):
        sched = 0.5 * (1 + math.cos(math.pi * (t - 1) / steps))
        x = batch(gen)
        y = tgt(T, x)
        loss = (fwd(p, x) - y).pow(2).mean() / y.pow(2).mean()
        grads = torch.autograd.grad(loss, [p[k] for k in p])
        with torch.no_grad():
            for k, g in zip(p, grads):
                if not torch.isfinite(g).all():
                    return None
                if k in muon_keys:
                    m = st[k][0]
                    m.mul_(0.95).add_(g)
                    upd = newton_schulz(g + 0.95 * m)
                    upd = upd * max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5
                    p[k].mul_(1 - lr * sched * wd).add_(upd, alpha=-lr * sched)
                else:
                    a_lr = lr if opt == "adam" else adam_lr_aux
                    m, v = st[k]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
                    dec = wd if k in mats or k in ("E", "H") else 0.0
                    p[k].mul_(1 - a_lr * sched * dec).addcdiv_(mh, vh.sqrt() + 1e-8, value=-a_lr * sched)
    return {k: v.detach() for k, v in p.items()}


def canon(p):
    """Function-preserving: fold RMSNorm gains into W_u, then balance each ReLU neuron's in/out norms."""
    q = copy.deepcopy(p)
    for l in range(L):
        q[f"u{l}"] = q[f"u{l}"] * q[f"g{l}"][None, :]
        q[f"g{l}"] = torch.ones(D)
        c = (q[f"d{l}"].norm(dim=0) / q[f"u{l}"].norm(dim=1)).sqrt()
        q[f"u{l}"] = q[f"u{l}"] * c[:, None]
        q[f"d{l}"] = q[f"d{l}"] / c[None, :]
    return q


def rand_gauge(p, seed, sigma=1.0):
    gen = torch.Generator().manual_seed(seed + 7)
    q = copy.deepcopy(p)
    for l in range(L):
        c = torch.exp(sigma * torch.randn(D_FF, generator=gen))
        q[f"u{l}"] = q[f"u{l}"] * c[:, None]
        q[f"d{l}"] = q[f"d{l}"] / c[None, :]
    return q


def imbalance(p):
    out = []
    for l in range(L):
        r = (p[f"d{l}"].norm(dim=0) / p[f"u{l}"].norm(dim=1)).log()
        g = p[f"g{l}"]
        out.append(f"L{l}: gain min/max {g.min():.2f}/{g.max():.2f}, log(|d|/|u|) sd {r.std():.2f}")
    return "; ".join(out)


GRID = {"adam": [3e-4, 1e-3, 3e-3], "muon": [3e-3, 1e-2, 3e-2, 1e-1]}


def main(n_seeds):
    t0 = time.time()
    res = {a: [] for a in ["adam_native", "muon_native", "muon_canon", "muon_random"]}
    forget = {a: [] for a in res}
    for s in range(n_seeds):
        TA = teacher(100 + s)
        TB = teacher(200 + s, base=TA, shift=0.6)
        p0 = train(init(s), TA, PRE_STEPS, 3e-3, "adam", seed=s, wd=0.1)
        gv, gt = torch.Generator().manual_seed(5000 + s), torch.Generator().manual_seed(6000 + s)
        xv, xt = batch(gv, 4096), batch(gt, 4096)
        pc, pr = canon(p0), rand_gauge(p0, s)
        for q in (pc, pr):
            assert (fwd(q, xt) - fwd(p0, xt)).abs().max() < 1e-4
        print(f"seed {s}: pretrain A-loss {loss_on(p0, TA, xt):.4f}, B-loss {loss_on(p0, TB, xt):.4f} | {imbalance(p0)}", flush=True)
        arms = {"adam_native": (p0, "adam"), "muon_native": (p0, "muon"), "muon_canon": (pc, "muon"), "muon_random": (pr, "muon")}
        for a, (q, opt) in arms.items():
            best = None
            for lr in GRID[opt]:
                pf = train(q, TB, FT_STEPS, lr, opt, seed=1000 + s)
                if pf is None:
                    continue
                vl = loss_on(pf, TB, xv)
                if best is None or vl < best[0]:
                    best = (vl, lr, pf)
            res[a].append(loss_on(best[2], TB, xt))
            forget[a].append(loss_on(best[2], TA, xt))
            print(f"  {a:12s} best lr {best[1]:.0e}  B-test {res[a][-1]:.4f}  A-test {forget[a][-1]:.4f}", flush=True)
    print("\nSummary (mean over seeds; paired diff vs muon_native):")
    base = torch.tensor(res["muon_native"])
    for a in res:
        x = torch.tensor(res[a])
        d = (x - base) / base
        print(f"{a:12s} B-test {x.mean():.4f}  A-test {torch.tensor(forget[a]).mean():.4f}  rel diff vs muon_native {d.mean()*100:+.1f}% (per-seed {[round(v*100,1) for v in d.tolist()]})")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
