"""Hand-set 2-layer toy whose mechanisms each span three matrices, with a co-occurrence knob rho.

Model: y = W3 ReLU(W2 ReLU(W1 x) + b2), b2 fixed (not decomposed).
Mechanism k (slot k, reads input i, writes output o; base routing i = o = k) computes
y_o = ReLU(|x_i| - theta):
  W1 piece  s1 (e_p - e_m) e_i^T    two layer-1 neurons give s1 ReLU(x_i), s1 ReLU(-x_i)
  W2 piece  s2 e_n (e_p + e_m)^T    one layer-2 neuron, bias -s1 s2 theta, gives s1 s2 ReLU(|x_i| - theta)
  W3 piece  e_o e_n^T / (s1 s2)
Neurons are disjoint across slots, so the decomposition is exact and the ReLUs never mix mechanisms;
neuron indices are randomly permuted and scales s1, s2 random so the layout is not axis-trivial.
Each piece alone does nothing useful: a mechanism is only identifiable across layers.
Spare slots (n_spare) have neurons and biases but zero weights: capacity for planted "add" changes.

Inputs: features in consecutive groups of `group_size`. Per sample and group, a shared draw
z ~ Bern(p); each feature copies z with prob rho, else draws its own Bern(p). Marginal activity is p
for every rho; rho = 0 gives independent features, rho = 1 makes group members always co-occur.
Values ~ U(-1, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .core import Mechanism, Toy, gen


class TwoLayerMLP(nn.Module):
    def __init__(self, n_in: int, h1: int, h2: int, n_out: int):
        super().__init__()
        self.l1 = nn.Linear(n_in, h1, bias=False)
        self.l2 = nn.Linear(h1, h2, bias=True)
        self.l3 = nn.Linear(h2, n_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


PARAMS = ("l1.weight", "l2.weight", "l3.weight")


@dataclass
class CrossLayerSpec:
    n_features: int
    theta: float
    perm1: Tensor   # (2F,) layer-1 neuron for slot k: perm1[2k] (positive), perm1[2k+1] (negative)
    perm2: Tensor   # (F,) layer-2 neuron for slot k
    s1: Tensor      # (F,)
    s2: Tensor      # (F,)

    @classmethod
    def random(cls, n_features: int, theta: float = 0.25, seed: int = 0) -> "CrossLayerSpec":
        g = gen(seed)
        return cls(n_features, theta, torch.randperm(2 * n_features, generator=g),
                   torch.randperm(n_features, generator=g),
                   0.5 + torch.rand(n_features, generator=g), 0.5 + torch.rand(n_features, generator=g))

    def mechanism(self, slot: int, inp: int, out: int, name: str | None = None) -> Mechanism:
        F = self.n_features
        e1, e2, ef = torch.eye(2 * F), torch.eye(F), torch.eye(F)
        p_, m_, n_ = self.perm1[2 * slot], self.perm1[2 * slot + 1], self.perm2[slot]
        s1, s2 = self.s1[slot], self.s2[slot]
        return Mechanism(name or f"slot{slot}", {
            "l1.weight": (s1 * (e1[p_] - e1[m_]), ef[inp].clone()),
            "l2.weight": (s2 * e2[n_], e1[p_] + e1[m_]),
            "l3.weight": (ef[out] / (s1 * s2), e2[n_].clone()),
        }, features=(inp,), tags={"slot": slot, "in": inp, "out": out})

    def build(self, routes: dict[int, tuple[int, int]]) -> tuple[TwoLayerMLP, list[Mechanism]]:
        """routes: slot -> (input feature, output index). Returns model with weights = sum of pieces."""
        F = self.n_features
        model = TwoLayerMLP(F, 2 * F, F, F)
        mechs = [self.mechanism(s, i, o) for s, (i, o) in sorted(routes.items())]
        with torch.no_grad():
            for prm in model.parameters():
                prm.zero_()
            W = dict(model.named_parameters())
            for m in mechs:
                for k in PARAMS:
                    W[k] += m.piece(k)
            model.l2.bias[self.perm2] = -self.s1 * self.s2 * self.theta  # spare slots too
        return model.requires_grad_(False), mechs

    def target(self, routes: dict[int, tuple[int, int]]):
        def f(x: Tensor) -> Tensor:
            y = torch.zeros_like(x)
            for i, o in routes.values():
                y[:, o] += torch.relu(x[:, i].abs() - self.theta)
            return y
        return f


def sample_cooccur(batch: int, n_features: int, p: float, rho: float, group_size: int = 2,
                   gen_: torch.Generator | None = None) -> Tensor:
    n_groups = -(-n_features // group_size)
    z = (torch.rand(batch, n_groups, generator=gen_) < p).repeat_interleave(group_size, 1)[:, :n_features]
    own = torch.rand(batch, n_features, generator=gen_) < p
    copy = torch.rand(batch, n_features, generator=gen_) < rho
    on = torch.where(copy, z, own)
    return (2 * torch.rand(batch, n_features, generator=gen_) - 1) * on


def make_crosslayer(n_features: int = 8, n_spare: int = 0, rho: float = 0.0, p: float = 0.05,
                    group_size: int = 2, theta: float = 0.25, seed: int = 0,
                    routes: dict[int, tuple[int, int]] | None = None) -> Toy:
    """Base routing: slot k reads feature k and writes output k for k < n_features - n_spare."""
    spec = CrossLayerSpec.random(n_features, theta, seed)
    routes = routes if routes is not None else {k: (k, k) for k in range(n_features - n_spare)}
    model, mechs = spec.build(routes)
    return Toy(f"crosslayer_F{n_features}_rho{rho:g}", model, PARAMS,
               sample=lambda batch, g=None: sample_cooccur(batch, n_features, p, rho, group_size, g),
               target=spec.target(routes), mechanisms=mechs,
               info={"spec": spec, "routes": dict(routes), "rho": rho, "p": p, "group_size": group_size,
                     "spare_slots": [k for k in range(n_features) if k not in routes]})
