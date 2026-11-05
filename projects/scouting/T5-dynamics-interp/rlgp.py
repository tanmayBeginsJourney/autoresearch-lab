"""Kill-test: random-label gradient projection (RLGP) vs AdamW / Grokfast-EMA on modular addition.
Usage: python rlgp.py SEED OUT.json
"""
import sys, json, time, math, torch
torch.set_num_threads(1)
P, FRAC, D, H, LR, MAXSTEPS = 59, 0.4, 64, 128, 1e-3, 6000

def data(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten()], 1); y = (x[:, 0] + x[:, 1]) % P
    perm = torch.randperm(len(y), generator=g); n = int(FRAC * len(y))
    return x[perm[:n]], y[perm[:n]], x[perm[n:]], y[perm[n:]]

class MLP(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.E = torch.nn.Embedding(P, D)
        s.l1 = torch.nn.Linear(2 * D, H); s.l2 = torch.nn.Linear(H, P)
    def forward(s, x):
        return s.l2(torch.relu(s.l1(s.E(x).flatten(1))))

def run(seed, arm, wd):
    xtr, ytr, xva, yva = data(seed)
    torch.manual_seed(seed); m = MLP()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=wd, betas=(0.9, 0.98))
    params = list(m.parameters()); ema = [torch.zeros_like(p) for p in params]
    lg = torch.Generator().manual_seed(77 + seed)
    hist, t0, grok = [], time.time(), None
    for step in range(1, MAXSTEPS + 1):
        loss = torch.nn.functional.cross_entropy(m(xtr), ytr)
        g = torch.autograd.grad(loss, params)
        if arm == "rlgp":  # gradient on permuted (random) labels at the same weights = label-agnostic fitting direction
            yr = ytr[torch.randperm(len(ytr), generator=lg)]
            r = torch.autograd.grad(torch.nn.functional.cross_entropy(m(xtr), yr), params)
            g = [gi - torch.clamp((gi * ri).sum() / (ri * ri).sum().clamp_min(1e-20), min=0) * ri for gi, ri in zip(g, r)]
        elif arm == "rlgp_rand":  # control: project out a random direction of matched norm statistics
            r = [torch.randn(gi.shape, generator=lg) for gi in g]
            g = [gi - torch.clamp((gi * ri).sum() / (ri * ri).sum(), min=0) * ri for gi, ri in zip(g, r)]
        elif arm == "grokfast":  # Grokfast-EMA (Lee et al. 2024), alpha=0.98, lambda=2
            for e, gi in zip(ema, g): e.mul_(0.98).add_(gi, alpha=0.02)
            g = [gi + 2.0 * e for gi, e in zip(g, ema)]
        for p, gi in zip(params, g): p.grad = gi
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                tra = (m(xtr).argmax(1) == ytr).float().mean().item(); va = (m(xva).argmax(1) == yva).float().mean().item()
            hist.append((step, round(loss.item(), 4), round(tra, 4), round(va, 4)))
            if va >= 0.95 and grok is None: grok = step
            if grok is not None and step >= grok + 200: break
    return dict(seed=seed, arm=arm, wd=wd, grok_step=grok, final_val=hist[-1][3], secs=round(time.time() - t0, 1), hist=hist[::4])

if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    arms = [("adamw", 1.0), ("adamw", 3.0), ("grokfast", 1.0), ("rlgp", 1.0), ("rlgp_rand", 1.0)]
    if len(sys.argv) > 3: arms = [tuple([a.split(":")[0], float(a.split(":")[1])]) for a in sys.argv[3].split(",")]
    res = []
    for arm, wd in arms:
        r = run(seed, arm, wd); res.append(r)
        print(seed, arm, wd, r["grok_step"], r["final_val"], r["secs"], flush=True)
        json.dump(res, open(out, "w"))
