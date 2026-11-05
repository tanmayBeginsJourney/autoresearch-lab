"""Train one SPD decomposition of TMS 40-10 (linear1 only) and score it on ground-truth recovery
and on candidate faithfulness metrics. Optionally emit rotated (mechanism-mixed) variants.

python run.py --name NAME --C 50 --imp 3e-3 --steps 3000 --seed 0 [--rotate 0.05,0.2] [--uv-noise 0.1]
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, "/home/user/autoresearch-lab/projects/pdecomp/src")
from spd_min import SPD, SPDConfig, train_spd, lower_leaky_hard_sigmoid  # noqa: E402
from spd_min.tms import TMS, sample_tms  # noqa: E402

HERE = Path(__file__).parent
TARGET = Path("/tmp/claude-0/-home-user-autoresearch-lab/088cda91-7866-52ad-a402-d8d038be9da0/scratchpad/tms40_10.pt")
NF, NH, L = 40, 10, "linear1"


def load_target():
    m = TMS(NF, NH)
    m.load_state_dict(torch.load(TARGET))
    m.tied = False
    return m.requires_grad_(False)


@torch.no_grad()
def ground_truth(spd, tms):
    """Recovery of mechanism j = W[:, j] e_j^T for represented features (col norm > 0.5)."""
    W = tms.linear1.weight  # (NH, NF)
    rep = (W.norm(dim=0) > 0.5).nonzero().flatten()
    sub = spd.subcomponents(L)  # (C, NH, NF)
    # column cosine (paper MMCS): column j of each subcomponent vs W[:, j]
    cols = sub[:, :, rep].permute(2, 0, 1)  # (n, C, NH)
    Wr = W[:, rep].T  # (n, NH)
    ccos = (cols * Wr[:, None]).sum(-1) / (cols.norm(dim=-1) * Wr.norm(dim=-1, keepdim=True) + 1e-12)
    # full-matrix cosine vs the rank-one ground-truth matrix (penalises cross-feature writes)
    gt = torch.zeros(len(rep), NH, NF)
    gt[torch.arange(len(rep)), :, rep] = Wr
    fcos = torch.einsum("noi,coi->nc", gt, sub) / (
        gt.flatten(1).norm(dim=-1)[:, None] * sub.flatten(1).norm(dim=-1)[None] + 1e-12)
    best = fcos.argmax(-1)
    fmax = fcos.max(-1).values
    recovered = (fmax > 0.9)
    distinct = len(set(best[recovered].tolist()))
    # causal-importance side: one-hot inputs, which subcomponents are active
    x = torch.zeros(len(rep), NF)
    x[torch.arange(len(rep)), rep] = 0.75
    _, acts = spd.target_and_inputs(x)
    g = lower_leaky_hard_sigmoid(spd.causal_importances(acts)[L])  # (n, C)
    act = g > 0.1
    onehot_ok = (act.sum(-1) == 1) & act[torch.arange(len(rep)), best] & recovered
    # alive subcomponents on the data distribution that match no feature
    xd = sample_tms(20000, NF, gen=torch.Generator().manual_seed(7))
    _, ad = spd.target_and_inputs(xd)
    gd = lower_leaky_hard_sigmoid(spd.causal_importances(ad)[L])
    alive = (gd > 0.1).any(0)
    matched = torch.zeros_like(alive)
    matched[best[recovered]] = True
    return {
        "n_rep": len(rep),
        "gt_mmcs_col": ccos.max(-1).values.mean().item(),
        "gt_fullcos": fmax.mean().item(),
        "gt_frac_recovered": distinct / len(rep),
        "gt_onehot_acc": onehot_ok.float().mean().item(),
        "gt_n_alive": int(alive.sum()),
        "gt_spurious_alive": int((alive & ~matched).sum()),
    }


def pgd_recon(spd, x, y_t, g, steps, step_size=0.1, seed=0):
    """Max over sources s in [0,1] of MSE with mask g + (1-g) s; random init, signed-gradient ascent."""
    gen = torch.Generator().manual_seed(seed)
    s = torch.rand(g.shape, generator=gen).requires_grad_(True)
    for _ in range(steps):
        loss = F.mse_loss(spd.run(x, {L: g + (1 - g) * s}), y_t)
        (grad,) = torch.autograd.grad(loss, s)
        with torch.no_grad():
            s += step_size * grad.sign()
            s.clamp_(0, 1)
    with torch.no_grad():
        return F.mse_loss(spd.run(x, {L: g + (1 - g) * s}), y_t).item()


def candidate_metrics(spd):
    x = sample_tms(4096, NF, gen=torch.Generator().manual_seed(99))
    y_t, acts = spd.target_and_inputs(x)
    with torch.no_grad():
        g = lower_leaky_hard_sigmoid(spd.causal_importances(acts)[L])
        gen = torch.Generator().manual_seed(5)
        stoch = sum(F.mse_loss(spd.run(x, {L: g + (1 - g) * torch.rand(g.shape, generator=gen)}), y_t).item()
                    for _ in range(8)) / 8
        out = {
            "m_unmasked": F.mse_loss(spd.run(x, {L: None}), y_t).item(),
            "m_ci_masked": F.mse_loss(spd.run(x, {L: g}), y_t).item(),
            "m_rounded": F.mse_loss(spd.run(x, {L: (g > 0.5).float()}), y_t).item(),
            "m_stoch": stoch,
            "m_l0": (g > 0.1).float().sum(-1).mean().item(),
            "m_faith": spd.faithfulness_loss().item(),
        }
    out["m_pgd10"] = pgd_recon(spd, x, y_t, g, 10)
    out["m_pgd40"] = pgd_recon(spd, x, y_t, g, 40)
    return out


@torch.no_grad()
def rotate(spd, t, seed):
    """Mix subcomponents: U <- U R, V <- R^T V with R = exp(t A); W = UV unchanged, CI fns unchanged."""
    k = spd._key(L)
    C = spd.U[k].shape[1]
    A = torch.randn(C, C, generator=torch.Generator().manual_seed(seed))
    R = torch.linalg.matrix_exp(t * (A - A.T) / math.sqrt(2 * C))
    spd.U[k].copy_(spd.U[k] @ R)
    spd.V[k].copy_(R.T @ spd.V[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--imp", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate", type=str, default="")
    ap.add_argument("--oracle", type=float, default=None, help="mixing angle t for oracle init")
    a = ap.parse_args()
    torch.set_num_threads(1)
    tms = load_target()
    torch.manual_seed(a.seed)
    spd = SPD(tms, {L: a.C}, ci_hidden=16)
    cfg = SPDConfig(coeff_imp=a.imp, coeff_layerwise=0.0)  # one layer: layerwise == stoch
    t0 = time.time()
    gen = torch.Generator().manual_seed(1000 + a.seed)
    family = "trained"
    if a.oracle is not None:
        # True mechanisms W[:, j] e_j^T in components 0..NF-1, tiny random spare components,
        # then mix all C components by exp(t A) (W = UV preserved), then train only the CI MLPs.
        family = "oracle_mixed"
        k = spd._key(L)
        with torch.no_grad():
            W = tms.linear1.weight
            spd.U[k][:, :NF] = W
            spd.V[k][:NF] = torch.eye(NF)
            spd.U[k][:, NF:] *= 0.01
            spd.V[k][NF:] *= 0.01
        rotate(spd, a.oracle, seed=a.seed)
        spd.U[k].requires_grad_(False)
        spd.V[k].requires_grad_(False)
    if a.steps:
        train_spd(spd, lambda: sample_tms(a.batch, NF, gen=gen), cfg, a.steps, log_every=10**9)
    rows = []
    base = {"name": a.name, "C": a.C, "imp": a.imp, "steps": a.steps, "seed": a.seed,
            "train_s": round(time.time() - t0, 1), "family": family,
            "rot": a.oracle if a.oracle is not None else 0.0}
    rows.append({**base, **ground_truth(spd, tms), **candidate_metrics(spd)})
    state = {k: v.clone() for k, v in spd.state_dict().items()}
    for t in [float(v) for v in a.rotate.split(",") if v]:
        spd.load_state_dict(state)
        rotate(spd, t, seed=a.seed)
        rows.append({**base, "name": f"{a.name}_rot{t}", "family": "rotated", "rot": t,
                     **ground_truth(spd, tms), **candidate_metrics(spd)})
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
