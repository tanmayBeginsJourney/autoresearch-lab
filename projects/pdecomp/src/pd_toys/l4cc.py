"""L4-loss compressed computation with a known mechanism (Ferreira da Silva & Heimersheim, arXiv:2607.04800).

Task: y = ReLU(x), x in R^100, each coordinate nonzero w.p. p = 0.02 with value ~ U(-1, 1); model
y_hat = W_out ReLU(W_in x), 50 neurons, no bias, no residual, L4 loss (Sec. 3).
Mechanism (Sec. 4.2-4.3, Fig. 6): feature j fires a sparse binary codeword M_j over neurons.
The paper's three-scalar ansatz is W_in = a M^T + b (1 - M^T), W_out = c pinv(W_in). We build the
target by hand from that ansatz, so its mechanism is known exactly by construction:
mechanism j = column j of W_in (piece W_in[:, j] e_j^T) + row j of W_out (e_j W_out[j, :]).
The paper reports this ansatz with a biregular K = 5 overlap-minimised code reaches ~1.12x the L4 loss
of a trained network (Table 1); a trained network's mechanism is only approximately this one.
(a, b, c) are fitted here by Adam on the L4 loss (the paper does not print fitted values).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .core import Mechanism, Toy, gen
from .tms import sample_sparse_uniform


def random_code(F: int, N: int, K: int, seed: int = 0) -> np.ndarray:
    """Each feature picks K distinct neurons uniformly at random. Returns M in {0,1}^{F x N}."""
    rng = np.random.default_rng(seed)
    M = np.zeros((F, N), dtype=np.int64)
    for i in range(F):
        M[i, rng.choice(N, K, replace=False)] = 1
    return M


def biregular_code(F: int, N: int, K: int, seed: int = 0) -> np.ndarray:
    """Every codeword has length K, every neuron is in F*K/N codewords (configuration model + repair)."""
    if (F * K) % N:
        raise ValueError("F*K must be divisible by N for a biregular code")
    rng = np.random.default_rng(seed)
    stubs = rng.permutation(np.repeat(np.arange(N), F * K // N)).reshape(F, K)
    for _ in range(100_000):
        bad = [i for i in range(F) if len(set(stubs[i])) < K]
        if not bad:
            break
        i = bad[0]
        k = next(k for k in range(K) if list(stubs[i]).count(stubs[i, k]) > 1)
        j, l = rng.integers(F), rng.integers(K)
        if stubs[j, l] not in stubs[i] and stubs[i, k] not in stubs[j]:
            stubs[i, k], stubs[j, l] = stubs[j, l], stubs[i, k]
    else:
        raise RuntimeError("biregular repair did not converge")
    M = np.zeros((F, N), dtype=np.int64)
    np.put_along_axis(M, stubs, 1, axis=1)
    return M


def overlap_objective(M: np.ndarray) -> int:
    O = M @ M.T
    return int((O ** 2).sum() - (np.diag(O) ** 2).sum())


def overlap_min(M: np.ndarray, iters: int = 20_000, seed: int = 0) -> np.ndarray:
    """Edge swaps of App. C: sample edges (i,a), (j,b); swap to (i,b), (j,a) if valid and the sum of
    off-diagonal squared overlaps does not increase. Preserves row and column degrees.
    The paper runs 8e5 iterations; the default here is smaller (cost ~0.1 ms per iteration)."""
    rng = np.random.default_rng(seed)
    M = M.copy()
    O = M @ M.T
    E = np.argwhere(M == 1)
    for _ in range(iters):
        e1, e2 = rng.integers(len(E), size=2)
        (i, a), (j, b) = E[e1], E[e2]
        if i == j or a == b or M[i, b] or M[j, a]:
            continue
        d = M[:, b] - M[:, a]
        mask = np.ones(len(M), bool)
        mask[[i, j]] = False
        oi, oj, dm = O[i, mask], O[j, mask], d[mask]
        ri, rj = M[i].copy(), M[j].copy()
        ri[a], ri[b], rj[b], rj[a] = 0, 1, 0, 1
        oij = int(ri @ rj)
        delta = 2 * (((oi + dm) ** 2 - oi ** 2).sum() + ((oj - dm) ** 2 - oj ** 2).sum()) + 2 * (oij ** 2 - O[i, j] ** 2)
        if delta > 0:
            continue
        M[i], M[j] = ri, rj
        O[i, mask] += dm
        O[mask, i] += dm
        O[j, mask] -= dm
        O[mask, j] -= dm
        O[i, j] = O[j, i] = oij
        E[e1], E[e2] = (i, b), (j, a)
    return M


class CCModel(nn.Module):
    """y_hat = W_out ReLU(W_in x); no bias, no residual."""

    def __init__(self, n_features: int = 100, n_neurons: int = 50):
        super().__init__()
        self.w_in = nn.Linear(n_features, n_neurons, bias=False)
        self.w_out = nn.Linear(n_neurons, n_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_out(torch.relu(self.w_in(x)))


def l4_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    return ((y_hat - y) ** 4).mean()


def ansatz_weights(M: Tensor, a: Tensor, b: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    Mt = M.T.float()
    W_in = a * Mt + b * (1 - Mt)
    return W_in, c * torch.linalg.pinv(W_in)


def fit_scalars(M: Tensor, p: float, steps: int = 400, batch: int = 8192, lr: float = 0.01,
                seed: int = 0) -> tuple[float, float, float]:
    """Fit (a, b, c) by Adam on the L4 loss, cosine lr. Init from the trained-encoder cluster means
    reported in Fig. 3 (on-code 0.34, off-code -0.02) and c = 1."""
    g = gen(seed)
    th = nn.Parameter(torch.tensor([0.34, -0.02, 1.0]))
    opt = torch.optim.Adam([th], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = sample_sparse_uniform(batch, M.shape[0], p, g, -1.0, 1.0)
        W_in, W_out = ansatz_weights(M, th[0], th[1], th[2])
        loss = l4_loss(torch.relu(x @ W_in.T) @ W_out.T, torch.relu(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return tuple(float(v) for v in th.detach())


def make_l4cc(n_features: int = 100, n_neurons: int = 50, K: int = 5, p: float = 0.02,
              code: str = "biregular", overlap_iters: int = 20_000, fit_steps: int = 400,
              scalars: tuple[float, float, float] | None = None, seed: int = 0) -> Toy:
    """Hand-set L4 compressed-computation target. code: "biregular" | "random"; pass `scalars`
    to skip fitting. info holds the codeword matrix M (F x N), (a, b, c) and L4 losses."""
    Mnp = (biregular_code if code == "biregular" else random_code)(n_features, n_neurons, K, seed)
    if overlap_iters:
        Mnp = overlap_min(Mnp, overlap_iters, seed)
    M = torch.from_numpy(Mnp)
    a, b, c = scalars or fit_scalars(M, p, fit_steps, seed=seed)
    W_in, W_out = ansatz_weights(M, torch.tensor(a), torch.tensor(b), torch.tensor(c))
    model = CCModel(n_features, n_neurons)
    with torch.no_grad():
        model.w_in.weight.copy_(W_in)
        model.w_out.weight.copy_(W_out)
    model.requires_grad_(False)
    eye = torch.eye(n_features)
    mechs = [Mechanism(f"feat{j}", {"w_in.weight": (W_in[:, j].clone(), eye[j]),
                                    "w_out.weight": (eye[j], W_out[j].clone())},
                       features=(j,), tags={"codeword": tuple(np.flatnonzero(Mnp[j]).tolist())})
             for j in range(n_features)]
    with torch.no_grad():
        xe = sample_sparse_uniform(50_000, n_features, p, gen(seed + 1), -1.0, 1.0)
        y = torch.relu(xe)
        loss = float(l4_loss(model(xe), y))
        naive = y.clone()
        naive[:, n_features // 2:] = 0  # represent half the features exactly, ignore the rest
        naive_loss = float(l4_loss(naive, y))
    return Toy(f"l4cc_{n_features}-{n_neurons}_K{K}_{code}", model, ("w_in.weight", "w_out.weight"),
               sample=lambda batch, g=None: sample_sparse_uniform(batch, n_features, p, g, -1.0, 1.0),
               target=torch.relu, mechanisms=mechs,
               info={"M": M, "a": a, "b": b, "c": c, "p": p, "l4_loss": loss, "naive_l4_loss": naive_loss,
                     "overlap_objective": overlap_objective(Mnp)})
