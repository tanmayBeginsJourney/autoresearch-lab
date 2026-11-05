"""Teacher-student reproduction of arXiv:2610.14500 Sec 2.1 with a per-arm LR sweep.

Model (Eq. 2): x_{l+1} = x_l + W2_l phi(W1_l LN(x_l)), no linear biases, LN without affine params.
Init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (paper Eq. 11; = PyTorch nn.Linear default).
Arms:  U  = unshifted phi (same activation as the teacher)
       S  = shifted phi - b
       UB = unshifted phi + trainable per-layer output bias c_l, initialised to -b W2_l 1
            (function-identical to S at init; tests whether S is just a missing offset)
Pairing: for a seed, all arms and LRs share teacher, student init (W1, W2) and the batch sequence.
Data: x ~ N(0, I_d), fresh batch each step; loss = MSE on the final residual stream.
Two fixed held-out sets per seed: 'val' (used only to pick LR) and 'test' (reported).
"""
import json, math, sys, time
import torch

B_SHIFT = {"relu": 0.230, "gelu": 0.115}
ACT = {"relu": torch.relu, "gelu": torch.nn.functional.gelu}


def init_weights(d, m, L, gen):
    W1 = (torch.rand(L, m, d, generator=gen) * 2 - 1) / math.sqrt(d)
    W2 = (torch.rand(L, d, m, generator=gen) * 2 - 1) / math.sqrt(m)
    return W1, W2


def forward(x, W1, W2, phi, b=0.0, c=None):
    for l in range(W1.shape[0]):
        h = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        a = phi(h @ W1[l].T)
        if b:
            a = a - b
        x = x + a @ W2[l].T
        if c is not None:
            x = x + c[l]
    return x


def run(cfg):
    torch.set_num_threads(1)
    d, m, L = cfg["d"], cfg["m"], cfg["L"]
    act, arm, lr, seed = cfg["act"], cfg["arm"], cfg["lr"], cfg["seed"]
    steps, bs, n_eval = cfg["steps"], cfg["bs"], cfg["n_eval"]
    phi, bval = ACT[act], B_SHIFT[act]

    gT = torch.Generator().manual_seed(10_000 + seed)
    T1, T2 = init_weights(d, m, L, gT)
    gS = torch.Generator().manual_seed(20_000 + seed)
    W1, W2 = init_weights(d, m, L, gS)
    gE = torch.Generator().manual_seed(30_000 + seed)
    xval = torch.randn(n_eval, d, generator=gE)
    xtest = torch.randn(n_eval, d, generator=gE)
    with torch.no_grad():
        yval, ytest = forward(xval, T1, T2, phi), forward(xtest, T1, T2, phi)

    params = [W1.requires_grad_(), W2.requires_grad_()]
    b, c = 0.0, None
    if arm == "S":
        b = bval
    elif arm == "UB":
        c = (-bval * W2.detach().sum(-1)).clone().requires_grad_()  # (L, d) = -b W2 1
        params.append(c)
    opt = torch.optim.Adam(params, lr=lr)
    gB = torch.Generator().manual_seed(40_000 + seed)

    ckpts = sorted(set([0, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
                       + list(range(0, steps + 1, steps // 8))))
    ckpts = [s for s in ckpts if s <= steps]
    curve = {"step": [], "val": [], "test": [], "train": []}
    run_loss, run_n = 0.0, 0
    for step in range(steps + 1):
        if step in ckpts:
            with torch.no_grad():
                lv = torch.mean((forward(xval, W1, W2, phi, b, c) - yval) ** 2).item()
                lt = torch.mean((forward(xtest, W1, W2, phi, b, c) - ytest) ** 2).item()
            curve["step"].append(step); curve["val"].append(lv); curve["test"].append(lt)
            curve["train"].append(run_loss / max(run_n, 1)); run_loss, run_n = 0.0, 0
        if step == steps:
            break
        x = torch.randn(bs, d, generator=gB)
        with torch.no_grad():
            y = forward(x, T1, T2, phi)
        loss = torch.mean((forward(x, W1, W2, phi, b, c) - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item() if step % 10 == 0 else 0.0
        run_n += 1 if step % 10 == 0 else 0
    diverged = not all(math.isfinite(v) for v in curve["test"])
    return {**cfg, "curve": curve, "diverged": diverged}


if __name__ == "__main__":
    # usage: ts.py OUT.json act arms lrs seeds steps [bs]
    from multiprocessing import Pool
    out, act = sys.argv[1], sys.argv[2]
    arms = sys.argv[3].split(",")
    lrs = [float(v) for v in sys.argv[4].split(",")]
    seeds = [int(v) for v in sys.argv[5].split(",")]
    steps = int(sys.argv[6]); bs = int(sys.argv[7]) if len(sys.argv) > 7 else 128
    cfgs = [dict(d=20, m=80, L=10, act=act, arm=a, lr=lr, seed=s, steps=steps, bs=bs, n_eval=4096)
            for s in seeds for lr in lrs for a in arms]
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(run, cfgs, chunksize=1)
    json.dump({"runs": res, "wall_s": time.time() - t0}, open(out, "w"))
    print(f"{len(res)} runs in {time.time()-t0:.0f}s -> {out}")
