"""Toy Model of Superposition (Elhage et al. 2022) targets as used by SPD (arXiv:2506.20790, Sec. 3.1).

x_hat = ReLU(W2 [I] W1 x + b), trained tied (W2 = W1^T), then untied so PD sees independent matrices.
Ground truth: mechanism j = column j of W1 (piece W1[:, j] e_j^T) plus row j of W2 (e_j W2[j, :]),
used by inputs with x_j != 0. The optional identity hidden matrix is basis-free: any orthonormal
basis is an equally valid decomposition, so its pieces e_k e_k^T are tagged basis_ambiguous.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import Mechanism, Toy, gen


class TMSModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, identity: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(n_features, n_hidden, bias=False)
        self.hidden = nn.Linear(n_hidden, n_hidden, bias=False) if identity else None
        self.linear2 = nn.Linear(n_hidden, n_features, bias=True)
        if identity:
            with torch.no_grad():
                self.hidden.weight.copy_(torch.eye(n_hidden))

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear1(x)
        if self.hidden is not None:
            h = self.hidden(h)
        return F.relu(self.linear2(h))


def sample_sparse_uniform(batch: int, n: int, p: float, gen_: torch.Generator | None = None,
                          low: float = 0.0, high: float = 1.0) -> Tensor:
    """Each coordinate nonzero with prob p, value ~ U[low, high]."""
    vals = low + (high - low) * torch.rand(batch, n, generator=gen_)
    return vals * (torch.rand(batch, n, generator=gen_) < p)


def train_tms_weights(n_features: int, n_hidden: int, p: float, steps: int, batch: int,
                      lr: float, seed: int) -> tuple[Tensor, Tensor]:
    """Tied TMS, AdamW (wd 0.01), constant lr, MSE (SPD App. A.4.1). Returns (W1, b)."""
    torch.manual_seed(seed)
    g = gen(seed)
    W = nn.Parameter(torch.randn(n_hidden, n_features) * (1 / n_features) ** 0.5)
    b = nn.Parameter(torch.zeros(n_features))
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=0.01)
    for _ in range(steps):
        x = sample_sparse_uniform(batch, n_features, p, g)
        loss = F.mse_loss(F.relu(x @ W.T @ W + b), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def make_tms(n_features: int = 5, n_hidden: int = 2, identity: bool = False, p: float = 0.05,
             steps: int = 10_000, batch: int = 1024, lr: float = 5e-3, seed: int = 0,
             alive_tol: float = 0.1) -> Toy:
    """TMS 5-2 (default) or 40-10 (`make_tms(40, 10)`); identity=True inserts a fixed I_hidden matrix.

    A feature counts as alive if ||W1[:, j]|| > alive_tol * max_j ||W1[:, j]||.
    """
    W, b = train_tms_weights(n_features, n_hidden, p, steps, batch, lr, seed)
    model = TMSModel(n_features, n_hidden, identity)
    with torch.no_grad():
        model.linear1.weight.copy_(W)
        model.linear2.weight.copy_(W.T)
        model.linear2.bias.copy_(b)
    model.requires_grad_(False)
    norms = W.norm(dim=0)
    eye_f, eye_h = torch.eye(n_features), torch.eye(n_hidden)
    mechs = [
        Mechanism(f"feat{j}", {"linear1.weight": (W[:, j].clone(), eye_f[j]),
                               "linear2.weight": (eye_f[j], W[:, j].clone())},
                  features=(j,), alive=bool(norms[j] > alive_tol * norms.max()))
        for j in range(n_features)
    ]
    params = ("linear1.weight", "linear2.weight")
    if identity:
        mechs += [Mechanism(f"id{k}", {"hidden.weight": (eye_h[k], eye_h[k])},
                            features=tuple(range(n_features)), tags={"basis_ambiguous": True})
                  for k in range(n_hidden)]
        params = ("linear1.weight", "hidden.weight", "linear2.weight")
    name = f"tms_{n_features}-{n_hidden}" + ("_id" if identity else "")
    with torch.no_grad():
        xe = sample_sparse_uniform(20_000, n_features, p, gen(seed + 1))
        mse = float(F.mse_loss(model(xe), xe))
    return Toy(name, model, params,
               sample=lambda batch, g=None: sample_sparse_uniform(batch, n_features, p, g),
               target=lambda x: x, mechanisms=mechs,
               info={"p": p, "seed": seed, "steps": steps, "eval_mse": mse,
                     "n_alive": sum(m.alive for m in mechs if m.name.startswith("feat"))})
