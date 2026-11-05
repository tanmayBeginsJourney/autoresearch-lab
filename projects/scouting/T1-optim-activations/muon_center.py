"""Kill-test for idea 8 (IDEAS.md): does activation centering matter more under Muon than under Adam?
Teacher-student identical to projects/actbias/repro/teacher-student/ts.py (ReLU, d=20, m=80, L=10).
usage: muon_center.py OUT.jsonl arms lrsAdam lrsMuon seeds steps [procs]   (one JSON line per finished run)
arms: AU AS MU MS MP  (A=Adam, M=Muon; U unshifted, S shifted phi-0.230, P Muon + input-mean projection)
"""
import json, math, sys, time
import torch

B = 0.230


def init_weights(d, m, L, gen):
    W1 = (torch.rand(L, m, d, generator=gen) * 2 - 1) / math.sqrt(d)
    W2 = (torch.rand(L, d, m, generator=gen) * 2 - 1) / math.sqrt(m)
    return W1, W2


def forward(x, W1, W2, b=0.0, means=None):
    for l in range(W1.shape[0]):
        h = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        a = torch.relu(h @ W1[l].T) - b
        if means is not None:
            means.append(a.detach().mean(0))
        x = x + a @ W2[l].T
    return x


def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(-2, -1) if tr else X


def run(cfg):
    torch.set_num_threads(1)
    d, m, L, arm, lr, seed, steps, bs = (cfg[k] for k in ("d", "m", "L", "arm", "lr", "seed", "steps", "bs"))
    T1, T2 = init_weights(d, m, L, torch.Generator().manual_seed(10_000 + seed))
    W1, W2 = init_weights(d, m, L, torch.Generator().manual_seed(20_000 + seed))
    gE = torch.Generator().manual_seed(30_000 + seed)
    xval, xtest = torch.randn(4096, d, generator=gE), torch.randn(4096, d, generator=gE)
    with torch.no_grad():
        yval, ytest = forward(xval, T1, T2), forward(xtest, T1, T2)
    W1.requires_grad_(); W2.requires_grad_()
    b = B if arm[1] == "S" else 0.0
    muon = arm[0] == "M"
    if muon:
        bufs = [torch.zeros_like(W1), torch.zeros_like(W2)]
        scale = 0.2 * math.sqrt(max(m, d))
    else:
        opt = torch.optim.Adam([W1, W2], lr=lr)
    gB = torch.Generator().manual_seed(40_000 + seed)
    for step in range(steps):
        x = torch.randn(bs, d, generator=gB)
        with torch.no_grad():
            y = forward(x, T1, T2)
        means = [] if arm == "MP" else None
        loss = torch.mean((forward(x, W1, W2, b, means) - y) ** 2)
        if not math.isfinite(loss.item() if step % 100 == 0 else 0.0):
            break
        g1, g2 = torch.autograd.grad(loss, [W1, W2])
        if muon:
            if arm == "MP":
                mu = torch.stack(means)                                   # (L, m)
                mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-12)
                g2 = g2 - (g2 @ mu.unsqueeze(-1)) * mu.unsqueeze(1)       # G (I - mu mu^T)
            with torch.no_grad():
                for W, g, buf in ((W1, g1, bufs[0]), (W2, g2, bufs[1])):
                    buf.mul_(0.95).add_(g)
                    W.add_(ns5(g + 0.95 * buf), alpha=-lr * scale)
        else:
            W1.grad, W2.grad = g1, g2
            opt.step()
    with torch.no_grad():
        lv = torch.mean((forward(xval, W1, W2, b) - yval) ** 2).item()
        lt = torch.mean((forward(xtest, W1, W2, b) - ytest) ** 2).item()
    return {**cfg, "val": lv, "test": lt, "diverged": not (math.isfinite(lv) and math.isfinite(lt))}


if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1]
    arms = sys.argv[2].split(",")
    grid = {"A": [float(v) for v in sys.argv[3].split(",")], "M": [float(v) for v in sys.argv[4].split(",")]}
    seeds, steps = [int(v) for v in sys.argv[5].split(",")], int(sys.argv[6])
    procs = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    cfgs = [dict(d=20, m=80, L=10, arm=a, lr=lr, seed=s, steps=steps, bs=128)
            for s in seeds for a in arms for lr in grid[a[0]]]
    t0 = time.time()
    with Pool(procs) as p, open(out, "w") as f:
        for r in p.imap_unordered(run, cfgs):
            f.write(json.dumps(r) + "\n"); f.flush()
    print(f"{len(cfgs)} runs in {time.time() - t0:.0f}s -> {out}")
