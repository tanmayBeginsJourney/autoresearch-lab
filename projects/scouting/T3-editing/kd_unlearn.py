"""H7 kill-test: kernel-decoupled unlearning vs baselines on a synthetic key-value memory (CPU).
Usage: python kd_unlearn.py [--seeds 0 1 2] [--lam 3.0]
"""
import argparse, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad

torch.set_num_threads(1)
D, H, V, N, NF = 64, 256, 32, 600, 100


def make_model():
    return nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, V))


def acc(m, x, y):
    with torch.no_grad():
        return (m(x).argmax(-1) == y).float().mean().item()


def train(m, x, y, steps=1500, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    return m


def kernel_penalty(m, xf, yf):
    params = dict(m.named_parameters())
    def logp(p, x, y):
        out = functional_call(m, p, (x.unsqueeze(0),))
        return (F.log_softmax(out, -1)[0] * y).sum()
    g = vmap(grad(logp), in_dims=(None, 0, 0))(params, xf, F.one_hot(yf, V).float())
    G = torch.cat([v.reshape(v.shape[0], -1) for v in g.values()], 1)
    G = G / (G.norm(dim=1, keepdim=True) + 1e-8)
    C = G @ G.T
    off = ~torch.eye(len(xf), dtype=torch.bool)
    return F.relu(C[off]).pow(2).mean()


def unlearn(base, arm, xf, yf, xr, yr, gen, lam, steps=150, lr=1e-3):
    m = make_model(); m.load_state_dict(base.state_dict())
    ref = make_model(); ref.load_state_dict(base.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    yrand = (yf + torch.randint(1, V, yf.shape, generator=gen)) % V  # random wrong labels
    u = torch.randn(H, generator=gen); u = u / u.norm()
    with torch.no_grad():
        hr_ref = ref[:4](xr); c = 4.0 * ref[:4](xf).norm(dim=1).mean()
        lp_ref = F.log_softmax(ref(xf), -1).gather(1, yf[:, None]).squeeze(1)
    for _ in range(steps):
        opt.zero_grad()
        base_arm = arm.split("+")[0]
        if base_arm == "npo":
            lp = F.log_softmax(m(xf), -1).gather(1, yf[:, None]).squeeze(1)
            loss = -(2 / 0.5) * F.logsigmoid(-0.5 * (lp - lp_ref)).mean() + F.cross_entropy(m(xr), yr)
        elif base_arm == "rmu":
            loss = (m[:4](xf) - c * u).pow(2).sum(1).mean() / c ** 2 \
                + 10 * (m[:4](xr) - hr_ref).pow(2).sum(1).mean() / c ** 2 + F.cross_entropy(m(xr), yr)
        else:  # randlabel, kd
            loss = F.cross_entropy(m(xf), yrand) + F.cross_entropy(m(xr), yr)
        if arm.endswith("kd"):  # "kd" = randlabel+kd (primary); "npo+kd", "rmu+kd" exploratory
            loss = loss + lam * kernel_penalty(m, xf, yf)
        loss.backward(); opt.step()
    return m


def attack(m0, x, y, steps, lr=1e-3):
    m = make_model(); m.load_state_dict(m0.state_dict())
    return train(m, x, y, steps=steps, lr=lr)


def run_seed(seed, lam):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g); y = torch.randint(0, V, (N,), generator=g)
    xf, yf, xr, yr = x[:NF], y[:NF], x[NF:], y[NF:]
    xa, ya, xb, yb = xf[:50], yf[:50], xf[50:], yf[50:]
    torch.manual_seed(seed); init = make_model()
    base = make_model(); base.load_state_dict(init.state_dict()); train(base, x, y)
    oracle = make_model(); oracle.load_state_dict(init.state_dict()); train(oracle, xr, yr)
    res = {"base_acc": acc(base, x, y)}
    arms = {"oracle": oracle}
    for arm in ["npo", "rmu", "randlabel", "kd", "npo+kd", "rmu+kd"]:
        arms[arm] = unlearn(base, arm, xf, yf, xr, yr, torch.Generator().manual_seed(1000 + seed), lam)
    for name, m in arms.items():
        mA = attack(m, xa, ya, 100); mB = attack(m, xb, yb, 25)
        with torch.no_grad():
            kp = kernel_penalty(m, xf, yf).item()
        res[name] = dict(F_pre=acc(m, xf, yf), R_pre=acc(m, xr, yr), kern=kp,
                         B_afterA=acc(mA, xb, yb), A_afterA=acc(mA, xa, ya), R_afterA=acc(mA, xr, yr),
                         B_directB25=acc(mB, xb, yb))
    for name in ["npo", "rmu", "randlabel", "kd", "npo+kd", "rmu+kd"]:
        res[name]["excess"] = res[name]["B_afterA"] - res["oracle"]["B_afterA"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lam", type=float, default=3.0); a = ap.parse_args()
    for s in a.seeds:
        t = time.time(); r = run_seed(s, a.lam); r["seed"] = s; r["lam"] = a.lam; r["sec"] = round(time.time() - t)
        print(json.dumps(r, default=lambda v: round(v, 3)), flush=True)
