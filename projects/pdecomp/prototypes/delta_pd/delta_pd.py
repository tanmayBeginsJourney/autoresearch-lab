"""Delta-PD prototype (exploratory). Self-contained, CPU, torch.

Toy: 1-hidden-layer MLP f(x) = W2 ReLU(W1 x + b1) + b2, n=20 sparse features, h=40 neurons.
Base task: y = |x|. Fine-tune task: base + planted change on features C={3,7}:
  y10 += ReLU(x3), y11 += ReLU(x3)  ;  y11 += ReLU(x7), y12 += ReLU(x7)   (overlap on y11)
Methods decomposing the fine-tune change:
  delta-PD : SPD on DeltaW only, W_base frozen and always active (C_D subcomponents / matrix)
  full-PD  : SPD on W_ft (C_F subcomponents / matrix)
  SVD      : truncated SVD of DeltaW_l, rank K per matrix
  LoRA     : rank-K adapters on W_base distilled to f_ft, components = SVD of B A
Usage: python delta_pd.py SEED COEF -> writes results_coefCOEF_seedSEED.json next to this file.
"""
import json, math, sys, time
import torch, torch.nn.functional as F

torch.set_num_threads(1)
N, H, P_FEAT = 20, 40, 0.05
CHANGED = [3, 7]
K = 2                       # rank for SVD / LoRA baselines (= number of planted changes)
C_D, C_F = 10, 40           # subcomponents per matrix: delta-PD, full-PD
BATCH = 512


def sample(b, g):
    x = torch.rand(b, N, generator=g) * 2 - 1
    return x * (torch.rand(b, N, generator=g) < P_FEAT)


def target_base(x):
    return x.abs()


def target_ft(x):
    y = x.abs().clone()
    r3, r7 = F.relu(x[:, 3]), F.relu(x[:, 7])
    y[:, 10] += r3; y[:, 11] += r3 + r7; y[:, 12] += r7
    return y


def fwd(x, W1, b1, W2, b2):
    return F.relu(x @ W1.T + b1) @ W2.T + b2


def train_mlp(params, target, steps, lr, g, wd=0.0):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = sample(BATCH, g)
        loss = F.mse_loss(fwd(x, *params), target(x))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        x = sample(8192, g)
        return F.mse_loss(fwd(x, *params), target(x)).item()


# ---------------------------------------------------------------- minimal SPD
def hs_lower(z):  # lower-leaky hard sigmoid (forward masks)
    return torch.where(z < 0, 0.01 * z, torch.clamp(z, max=1.0))


def hs_upper(z):  # upper-leaky hard sigmoid (importance loss)
    return torch.where(z > 1, 1 + 0.01 * (z - 1), torch.clamp(z, min=0.0))


class SPD(torch.nn.Module):
    """Rank-one subcomponents U_c V_c^T per matrix plus frozen offset (W_base for delta-PD, 0 for full)."""

    def __init__(self, targets, offsets, C, g, d_gate=16):
        super().__init__()
        self.targets, self.offsets, self.C = targets, offsets, C
        self.U, self.V, self.gi_w, self.gi_b, self.go_w, self.go_b = [torch.nn.ParameterList() for _ in range(6)]
        for W in targets:
            d_out, d_in = W.shape
            self.V.append(torch.nn.Parameter(torch.randn(C, d_in, generator=g) / math.sqrt(d_in)))
            self.U.append(torch.nn.Parameter(torch.randn(C, d_out, generator=g) * 0.1 / math.sqrt(C)))
            self.gi_w.append(torch.nn.Parameter(torch.randn(C, d_gate, generator=g)))
            self.gi_b.append(torch.nn.Parameter(torch.randn(C, d_gate, generator=g) * 0.1))
            self.go_w.append(torch.nn.Parameter(torch.randn(C, d_gate, generator=g) / math.sqrt(d_gate)))
            self.go_b.append(torch.nn.Parameter(torch.ones(C)))

    def gate_pre(self, l, a):
        h = a @ self.V[l].T                                         # (B,C) inner activation
        z = F.gelu(h[..., None] * self.gi_w[l] + self.gi_b[l])      # (B,C,dg)
        return (z * self.go_w[l]).sum(-1) + self.go_b[l]

    def layer(self, l, a, m):
        out = a @ self.offsets[l].T
        if m is None:
            return out + a @ (self.U[l].T @ self.V[l]).T
        return out + ((a @ self.V[l].T) * m) @ self.U[l]

    def forward(self, x, b1, b2, masks):
        a = F.relu(self.layer(0, x, masks[0]) + b1)
        return self.layer(1, a, masks[1]) + b2


def target_acts(x, W1, b1):
    return [x, F.relu(x @ W1.T + b1)]


def train_spd(spd, Wt, b1, b2, steps, lr, coef_min, p, g, log_every=250):
    """Wt = target weights (W1_ft, W2_ft). Returns per-checkpoint log."""
    opt = torch.optim.Adam(spd.parameters(), lr=lr)
    n_params = sum(W.numel() for W in Wt)
    log = []
    for step in range(steps + 1):
        x = sample(BATCH, g)
        with torch.no_grad():
            acts = target_acts(x, Wt[0], b1)
            y_t = fwd(x, Wt[0], b1, Wt[1], b2)
        pre = [spd.gate_pre(l, acts[l]) for l in range(2)]
        g_lo = [hs_lower(z) for z in pre]
        g_up = [hs_upper(z) for z in pre]
        masks = [gl + (1 - gl) * torch.rand(gl.shape, generator=g) for gl in g_lo]
        faith = sum(((spd.offsets[l] + spd.U[l].T @ spd.V[l]) - Wt[l]).pow(2).sum() for l in range(2)) / n_params
        rec = F.mse_loss(spd(x, b1, b2, masks), y_t)
        rec_lw = sum(F.mse_loss(spd(x, b1, b2, [masks[l] if k == l else None for k in range(2)]), y_t)
                     for l in range(2))
        imin = sum(gu.abs().pow(p).sum(-1).mean() for gu in g_up)
        loss = faith + rec + rec_lw + coef_min * imin
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            alive = [int((gl.detach().max(0).values > 0.1).sum()) for gl in g_lo]
            log.append(dict(step=step, faith=faith.item(), rec=rec.item(), rec_lw=rec_lw.item(),
                            l0=sum(float((gl > 0.1).float().sum(-1).mean()) for gl in g_lo), alive=alive))
    return log


# ---------------------------------------------------------------- evaluation
def probes():
    """Single-feature inputs x = v e_j for v in {0.25,0.5,0.75,1} (and negatives)."""
    vs = torch.tensor([0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0])
    X = torch.zeros(N, len(vs), N)
    for j in range(N):
        X[j, :, j] = vs
    return X  # (N, 8, N)


def auroc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    ranks = allv.argsort().argsort().float() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(comp, score_fn, ablate_fn, full_fn, f_ft, f_base, g):
    """comp: list of (layer, idx). score_fn(X)->(B,ncomp) attribution; ablate_fn(X, ci)->outputs with comp ci removed;
    full_fn(X)->outputs with all comps. Returns per-changed-feature metrics + aggregate."""
    X = probes()
    R = torch.stack([score_fn(X[j]).mean(0) for j in range(N)], 1)   # (ncomp, N) mean response to feature j
    Rp = torch.stack([score_fn(X[j][:4]).mean(0) for j in range(N)], 1)  # positive values only
    xt = sample(20000, g)
    xt = xt[(xt != 0).any(1)]
    S = score_fn(xt)
    out = {}
    for j in CHANGED:
        c = int(Rp[:, j].argmax())
        other = [k for k in CHANGED if k != j][0]
        purity = float(Rp[c, j] / Rp[c].sum().clamp_min(1e-12))
        au = auroc(S[:, c], xt[:, j] > 0)
        # causal: ablate comp c on probe x = +1 e_j and on x = +1 e_other
        xj, xo = X[j][3:4], X[other][3:4]
        d_change_j = (f_ft(xj) - f_base(xj))
        d_abl_j = full_fn(xj) - ablate_fn(xj, c)
        d_abl_o = full_fn(xo) - ablate_fn(xo, c)
        d_change_o = (f_ft(xo) - f_base(xo))
        removed = float(d_abl_j.norm() / d_change_j.norm())
        collateral = float(d_abl_o.norm() / d_change_o.norm())
        cos = float(F.cosine_similarity(d_abl_j, d_change_j).item())
        # base behaviour damage on unchanged outputs (everything except outputs 10-12) for probe j
        mask_out = torch.ones(N, dtype=torch.bool); mask_out[[10, 11, 12]] = False
        base_dmg = float(d_abl_j[:, mask_out].norm() / f_base(xj)[:, mask_out].norm())
        out[f"f{j}"] = dict(comp=comp[c], matched=bool(Rp[c, j] > 0.05), purity=purity, auroc=au, removed=removed,
                            collateral=collateral, cos_change=cos, base_damage=base_dmg)
    # behaviour-change detection: any comp score summed vs label = output changed
    changed = (f_ft(xt) - f_base(xt)).norm(dim=1) > 1e-2
    out["auroc_changed_any"] = auroc(S.sum(1), changed)
    out["R_changed_rows"] = {f"f{j}": Rp[:, j].tolist() for j in CHANGED}
    keys = ["purity", "auroc", "removed", "collateral", "cos_change", "base_damage"]
    out["mean"] = {k: sum(out[f"f{j}"][k] for j in CHANGED) / len(CHANGED) for k in keys}
    return out


def main(seed):
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W1 = torch.nn.Parameter(torch.randn(H, N, generator=g) / math.sqrt(N))
    b1 = torch.nn.Parameter(torch.zeros(H))
    W2 = torch.nn.Parameter(torch.randn(N, H, generator=g) / math.sqrt(H))
    b2 = torch.nn.Parameter(torch.zeros(N))
    base_loss = train_mlp([W1, b1, W2, b2], target_base, 6000, 3e-3, g, wd=0.0)
    Wb = [W1.detach().clone(), W2.detach().clone()]
    bb = [b1.detach().clone(), b2.detach().clone()]
    ft_loss_before = None
    with torch.no_grad():
        xs = sample(8192, g); ft_loss_before = F.mse_loss(fwd(xs, *Wb[:1], bb[0], Wb[1], bb[1]), target_ft(xs)).item()
    ft_loss = train_mlp([W1, b1, W2, b2], target_ft, 3000, 1e-3, g)
    Wf = [W1.detach().clone(), W2.detach().clone()]
    bf = [b1.detach().clone(), b2.detach().clone()]
    dW = [Wf[l] - Wb[l] for l in range(2)]
    t_models = time.time() - t0

    f_ft = lambda x: fwd(x, Wf[0], bf[0], Wf[1], bf[1])
    f_base = lambda x: fwd(x, Wb[0], bb[0], Wb[1], bb[1])  # true base model (base biases)
    res = dict(seed=seed, coef_min=COEF, base_mse=base_loss, ft_task_mse_before=ft_loss_before, ft_mse=ft_loss,
               dW_fro=[float(d.norm()) for d in dW], dW_sv=[torch.linalg.svdvals(d)[:6].tolist() for d in dW],
               db_norm=[float((bf[l] - bb[l]).norm()) for l in range(2)],
               dW1_col_energy_changed=float(dW[0][:, CHANGED].pow(2).sum() / dW[0].pow(2).sum()))

    # ---------------- SPD runs
    def spd_eval(spd, gate_layers_offset):
        def score(X):
            with torch.no_grad():
                acts = target_acts(X, Wf[0], bf[0])
                return torch.cat([hs_lower(spd.gate_pre(l, acts[l])).clamp(0, 1) for l in range(2)], 1)

        def run(X, abl=None):
            with torch.no_grad():
                ms = [torch.ones(X.shape[0], spd.C) for _ in range(2)]
                if abl is not None:
                    ms[abl // spd.C][:, abl % spd.C] = 0
                return spd(X, bf[0], bf[1], ms)
        comp = [(l, c) for l in range(2) for c in range(spd.C)]
        return score, (lambda X, c: run(X, c)), (lambda X: run(X)), comp

    for name, C, offs, steps, coef in [("delta_pd", C_D, Wb, 2500, COEF), ("full_pd", C_F, [torch.zeros_like(w) for w in Wb], 2500, COEF)]:
        t1 = time.time()
        spd = SPD(Wf, offs, C, g)
        log = train_spd(spd, Wf, bf[0], bf[1], steps, 2e-3, coef, 1.0, g)
        t_train = time.time() - t1; print(name, "trained", round(t_train), flush=True)
        score, abl, full, comp = spd_eval(spd, 0)
        ev = evaluate(comp, score, abl, full, f_ft, f_base, g)
        with torch.no_grad():
            xs = sample(8192, g)
            ys = f_ft(xs)
            fid = float(F.mse_loss(full(xs), ys) / (ys - f_base(xs)).pow(2).mean())
        res[name] = dict(C=C, steps=steps, train_s=t_train, final=log[-1], log=log, eval=ev,
                         recon_mse_rel_to_change=fid,
                         flops_per_step_matmul=int(3 * BATCH * C * 2 * (N + H) * 2 * 4))
        # alive comps that respond to unchanged features (spurious) : max over probes > 0.1
        X = probes()
        R = torch.stack([score(X[j]).max(0).values for j in range(N)], 1)
        alive = (R.max(1).values > 0.1)
        res[name]["n_alive_probe"] = int(alive.sum())
        res[name]["n_alive_responding_only_changed"] = int((alive & (R[:, [j for j in range(N) if j not in CHANGED]].max(1).values <= 0.1)).sum())

    # ---------------- SVD / LoRA baselines
    def lowrank_eval(factors):
        # factors: per layer list of (u (d_out), s, v (d_in)) rank-one pieces
        comp = [(l, k) for l in range(2) for k in range(len(factors[l]))]

        def run(X, abl=None):
            Ws = []
            for l in range(2):
                W = Wb[l].clone()
                for k, (u, s, v) in enumerate(factors[l]):
                    if abl is not None and comp[abl] == (l, k):
                        continue
                    W += s * torch.outer(u, v)
                Ws.append(W)
            return fwd(X, Ws[0], bf[0], Ws[1], bf[1])

        def score(X):
            acts = target_acts(X, Wf[0], bf[0])
            return torch.cat([torch.stack([(s * (acts[l] @ v)).abs() * u.norm() for (u, s, v) in factors[l]], 1)
                              for l in range(2)], 1)
        return score, (lambda X, c: run(X, c)), (lambda X: run(X)), comp

    def svd_factors(Ms, k):
        fs = []
        for M in Ms:
            Uu, S, Vh = torch.linalg.svd(M)
            fs.append([(Uu[:, i], S[i], Vh[i]) for i in range(k)])
        return fs

    for kk in [K, 4]:
        sc, ab, fu, comp = lowrank_eval(svd_factors(dW, kk))
        with torch.no_grad():
            xs = sample(8192, g); ys = f_ft(xs)
            fid = float(F.mse_loss(fu(xs), ys) / (ys - f_base(xs)).pow(2).mean())
        res[f"svd_k{kk}"] = dict(eval=evaluate(comp, sc, ab, fu, f_ft, f_base, g), recon_mse_rel_to_change=fid)

    # LoRA distillation: W = W_base + B A, rank K, fit f_ft outputs (biases = ft biases, as in delta-PD)
    t1 = time.time()
    A = [torch.nn.Parameter(torch.randn(K, Wb[l].shape[1], generator=g) * 0.1) for l in range(2)]
    B = [torch.nn.Parameter(torch.zeros(Wb[l].shape[0], K)) for l in range(2)]
    opt = torch.optim.Adam(A + B, lr=3e-3)
    for _ in range(3000):
        x = sample(BATCH, g)
        y = fwd(x, Wb[0] + B[0] @ A[0], bf[0], Wb[1] + B[1] @ A[1], bf[1])
        loss = F.mse_loss(y, f_ft(x).detach())
        opt.zero_grad(); loss.backward(); opt.step()
    lora_d = [(B[l] @ A[l]).detach() for l in range(2)]
    sc, ab, fu, comp = lowrank_eval(svd_factors(lora_d, K))
    with torch.no_grad():
        xs = sample(8192, g); ys = f_ft(xs)
        fid = float(F.mse_loss(fu(xs), ys) / (ys - f_base(xs)).pow(2).mean())
    res["lora_k2"] = dict(eval=evaluate(comp, sc, ab, fu, f_ft, f_base, g), recon_mse_rel_to_change=fid,
                          train_s=time.time() - t1)
    res["t_models_s"] = t_models
    res["t_total_s"] = time.time() - t0
    out = __file__.rsplit("/", 1)[0] + f"/results_coef{COEF:g}_seed{seed}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v["eval"]["mean"] if isinstance(v, dict) and "eval" in v else v)
                      for k, v in res.items() if k not in ("dW_sv",)}, default=str)[:3000])


if __name__ == "__main__":
    COEF = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
