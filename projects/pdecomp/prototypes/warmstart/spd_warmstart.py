"""EXPLORATORY prototype: does a data-driven init make SPD on TMS converge faster?

Minimal SPD (arXiv:2506.20790, Sec 2 / App A.4.1) on TMS (x_hat = ReLU(W^T W x + b)), decomposing the
tied W into C rank-one subcomponents U_c V_c^T with per-subcomponent CI MLPs (1 -> 16 GELU -> 1).
Inits: random (SPD default-like), svd, ica (complete), kmeans (overcomplete, C atoms), oracle (upper bound).
Usage: python spd_warmstart.py MODEL INIT SEED STEPS OUT.json     MODEL in {5-2, 40-10}
"""
import json, math, sys, time
import torch, torch.nn.functional as F

torch.set_num_threads(1)
CFG = {  # m1 hidden, m2 features, C, beta3, p, target batch
    "5-2": dict(m1=2, m2=5, C=20, beta3=3e-3, p=1.0, tbs=1024, bs=2048, ne=8192),
    "40-10": dict(m1=10, m2=40, C=80, beta3=1e-4, p=2.0, tbs=8192, bs=512, ne=2048),  # C, bs cut for CPU budget
}
FP = 0.05  # feature probability


def data(n, m2, g):
    x = torch.rand(n, m2, generator=g)
    return x * (torch.rand(n, m2, generator=g) < FP)


def train_target(name, steps=10000):
    c = CFG[name]; g = torch.Generator().manual_seed(0); torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(c["m1"], c["m2"]) * 0.3); b = torch.nn.Parameter(torch.zeros(c["m2"]))
    opt = torch.optim.AdamW([W, b], lr=5e-3, weight_decay=0.01)
    for _ in range(steps):
        x = data(c["tbs"], c["m2"], g)
        loss = ((F.relu(x @ W.T @ W + b) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach(), loss.item()


def fwd(x, U, V, b, mask=None):  # tied TMS forward with W' = sum_c m_c U_c V_c^T
    h = x @ V  # inner acts (B, C)
    hm = h if mask is None else h * mask
    z = hm @ U.T  # (B, m1)
    zz = z @ U  # (B, C): U_c . z
    if mask is not None: zz = zz * mask
    return F.relu(zz @ V.T + b), h


def ls_U(W, V, X):  # U minimising E||W x - U V^T x||^2 on data X
    S = X.T @ X / len(X)
    return W @ S @ V @ torch.linalg.pinv(V.T @ S @ V)


def ica(X, iters=200):  # symmetric FastICA (logcosh), returns unmixing rows = atoms in input space
    Xc = X - X.mean(0); C = Xc.T @ Xc / len(Xc); e, E = torch.linalg.eigh(C)
    K = E @ torch.diag(e.clamp_min(1e-8) ** -0.5) @ E.T; Z = Xc @ K
    d = X.shape[1]; Wm = torch.linalg.qr(torch.randn(d, d))[0]
    for _ in range(iters):
        WZ = Z @ Wm.T; gz = torch.tanh(WZ); gp = 1 - gz ** 2
        Wn = gz.T @ Z / len(Z) - gp.mean(0)[:, None] * Wm
        u, s, vt = torch.linalg.svd(Wn); Wm = u @ vt
    A = torch.linalg.pinv(Wm @ K)  # mixing matrix: columns = source directions in x space
    A = A / A.norm(dim=0, keepdim=True)
    return A * torch.sign(A.sum(0, keepdim=True))  # (d, d)


def kmeans(X, k, iters=50):  # spherical k-means on nonzero inputs
    Xn = X[X.norm(dim=1) > 0]; Xn = Xn / Xn.norm(dim=1, keepdim=True)
    Cn = Xn[torch.randperm(len(Xn))[:k]].clone()
    for _ in range(iters):
        a = (Xn @ Cn.T).argmax(1)
        for j in range(k):
            s = Xn[a == j]
            Cn[j] = s.sum(0) if len(s) else Xn[torch.randint(len(Xn), (1,))][0]
        Cn = Cn / Cn.norm(dim=1, keepdim=True)
    return Cn.T  # (d, k)


def init(kind, W, C, g):
    m1, m2 = W.shape
    V = torch.randn(m2, C, generator=g) / math.sqrt(m2); U = torch.randn(m1, C, generator=g) / math.sqrt(C)
    if kind == "random": return U, V
    U, V = 0.1 * U, 0.1 * V  # small random for unused slots
    X = data(20000, m2, g)
    if kind == "svd":
        u, s, vt = torch.linalg.svd(W, full_matrices=False); r = len(s)
        U[:, :r] = u * s; V[:, :r] = vt.T
    elif kind == "oracle":
        V[:, :m2] = torch.eye(m2); U[:, :m2] = W
    elif kind == "ica":
        A = ica(X); V[:, :m2] = A; U[:, :m2] = ls_U(W, A, X)
    elif kind == "kmeans":
        V = kmeans(X, C); U = ls_U(W, V, X)
    return U.clone(), V.clone()


def metrics(U, V, W, alive):
    Wn = W.norm(dim=0)  # (m2,)
    col = (U.T @ W)  # (C, m2): U_c . W_j
    # paper MMCS: column j of U_c V_c^T is V_cj U_c -> cos = sign(V_cj) * cos(U_c, W_j)
    cos_col = torch.sign(V) .T * col / (U.norm(dim=0)[:, None] * Wn[None] + 1e-12)
    cos_col = cos_col.max(0).values
    # strict: Frobenius cosine between U_c V_c^T and ground-truth W_j e_j^T
    cos_gt = (col * V.T) / (U.norm(dim=0) * V.norm(dim=0))[:, None] / (Wn[None] + 1e-12)
    best = cos_gt.argmax(0); cos_gt = cos_gt.max(0).values
    l2r = ((U[:, best] * V[torch.arange(len(Wn)), best]).norm(dim=0) / Wn)
    a = alive
    return dict(mmcs=cos_col[a].mean().item(), gt_min=cos_gt[a].min().item(), gt_mean=cos_gt[a].mean().item(),
                ml2r=l2r[a].mean().item())


def main():
    name, kind, seed, steps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    c = CFG[name]; tgt = f"{out.rsplit('/', 1)[0]}/target_{name}.pt"
    try: W, b = torch.load(tgt)
    except FileNotFoundError:
        W, b, tl = train_target(name); torch.save((W, b), tgt); print("target loss", tl)
    alive = W.norm(dim=0) > 0.5
    g = torch.Generator().manual_seed(1000 + seed); torch.manual_seed(1000 + seed)
    t0 = time.time(); U, V = init(kind, W, c["C"], g); t_init = time.time() - t0
    U = torch.nn.Parameter(U); V = torch.nn.Parameter(V)
    C = c["C"]; H = 16
    w1 = torch.nn.Parameter(torch.randn(C, H) * 1.0); b1 = torch.nn.Parameter(torch.zeros(C, H))
    w2 = torch.nn.Parameter(torch.randn(C, H) / math.sqrt(H)); b2 = torch.nn.Parameter(torch.zeros(C))
    opt = torch.optim.Adam([U, V, w1, b1, w2, b2], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    bs = c["bs"]; xe = data(c["ne"], c["m2"], torch.Generator().manual_seed(7))
    with torch.no_grad(): ye = F.relu(xe @ W.T @ W + b)
    hist = []; rec_step = None

    def ci(h):
        pre = (F.gelu(h[..., None] * w1 + b1) * w2).sum(-1) + b2
        lower = torch.where(pre < 0, 0.01 * pre, torch.where(pre > 1, torch.ones_like(pre), pre))
        upper = torch.where(pre < 0, torch.zeros_like(pre), torch.where(pre > 1, 1 + 0.01 * (pre - 1), pre))
        return lower, upper

    def evaluate(step):
        with torch.no_grad():
            m = metrics(U, V, W, alive)
            _, h = fwd(xe, U, V, b); gl, gu = ci(h)
            r = torch.rand(gl.shape, generator=torch.Generator().manual_seed(3))
            yo, _ = fwd(xe, U, V, b, gl + (1 - gl) * r)
            nact = (xe > 0).sum(1).float(); nz = nact > 0
            m.update(step=step, stoch_recon=F.mse_loss(yo, ye).item(),
                     faith=((W - U @ V.T) ** 2).mean().item(),
                     ci_l0_per_feat=((gu > 0.1).sum(1).float()[nz].sum() / nact[nz].sum()).item())
        return m

    t0 = time.time()
    for step in range(steps + 1):
        if step % 100 == 0:
            m = evaluate(step); hist.append(m)
            if rec_step is None and m["gt_min"] > 0.95: rec_step = step
        if step == steps: break
        x = data(bs, c["m2"], g)
        with torch.no_grad(): y = F.relu(x @ W.T @ W + b)
        _, h = fwd(x, U, V, b); gl, gu = ci(h)
        rec = sum(F.mse_loss(fwd(x, U, V, b, gl + (1 - gl) * torch.rand_like(gl))[0], y) for _ in range(2))
        imp = (gu.abs() ** c["p"]).sum(1).mean()
        faith = ((W - U @ V.T) ** 2).mean()
        loss = faith + rec + c["beta3"] * imp
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    # step at which gt recovery is *sustained* (gt_min > 0.95 at every later eval)
    sus = None
    for m in reversed(hist):
        if m["gt_min"] > 0.95: sus = m["step"]
        else: break
    res = dict(model=name, init=kind, seed=seed, steps=steps, C=C, n_alive=int(alive.sum()), t_init=t_init,
               t_train=time.time() - t0, first_rec_step=rec_step, sustained_rec_step=sus, final=hist[-1], hist=hist)
    json.dump(res, open(out, "w"))
    f = hist[-1]
    print(f"{name} {kind} s{seed}: rec={rec_step} sus={sus} gt_min={f['gt_min']:.3f} mmcs={f['mmcs']:.3f} "
          f"l0/feat={f['ci_l0_per_feat']:.2f} recon={f['stoch_recon']:.2e} t={res['t_train']:.0f}s")


if __name__ == "__main__":
    main()
