"""Planted fine-tune generator: base cross-layer toy + fine-tuned copy with K known changes.

Change types (each delta is a ground-truth Mechanism, rank one in every matrix it touches):
  add     a spare slot gets a mechanism (reads and writes spare feature s): delta = +pieces in W1, W2, W3
  delete  an implemented mechanism is removed: delta = -pieces in W1, W2, W3
  edit    an implemented mechanism k is rerouted to read spare feature j instead of k:
          delta = s1 (e_p - e_m)(e_j - e_k)^T in W1 only; used by inputs with x_k or x_j nonzero
  drift   dense Gaussian noise on every decomposed matrix (no mechanism): entry std =
          drift * RMS entry of the base matrix. Returned separately in `noise`.
ft weights == base weights + sum of delta pieces + noise, exactly. Use changes={} with drift > 0 for
the noise-only arm (a delta method should then return no mechanisms).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import Mechanism, Toy, gen
from .crosslayer import PARAMS, make_crosslayer


@dataclass
class PlantedFT:
    base: Toy
    ft: Toy
    deltas: list[Mechanism]       # ground-truth delta mechanisms (tags["change"] in add/edit/delete)
    noise: dict[str, Tensor]      # drift component of ft - base

    def delta_weights(self) -> dict[str, Tensor]:
        b, f = self.base.weights(), self.ft.weights()
        return {k: f[k] - b[k] for k in b}


def _neg(m: Mechanism) -> Mechanism:
    f = {k: (-u, v) for k, (u, v) in m.factors.items()}
    return Mechanism(m.name, f, m.features, m.mode, m.alive, dict(m.tags))


def make_planted(n_features: int = 12, n_spare: int = 4, changes: dict[str, int] | None = None,
                 drift: float = 0.0, rho: float = 0.0, p: float = 0.05, group_size: int = 2,
                 theta: float = 0.25, seed: int = 0) -> PlantedFT:
    """changes: counts per type, default {"add": 1, "edit": 1, "delete": 1}.
    Needs n_spare >= add + edit (each consumes one spare feature)."""
    changes = {"add": 1, "edit": 1, "delete": 1} if changes is None else changes
    n_add, n_edit, n_del = (changes.get(k, 0) for k in ("add", "edit", "delete"))
    if n_add + n_edit > n_spare or n_edit + n_del > n_features - n_spare:
        raise ValueError("not enough spare or implemented slots for the requested changes")
    base = make_crosslayer(n_features, n_spare, rho, p, group_size, theta, seed)
    spec, routes = base.info["spec"], dict(base.info["routes"])
    g = gen(seed + 7)
    spare = [base.info["spare_slots"][i] for i in torch.randperm(n_spare, generator=g)]
    impl = [sorted(routes)[i] for i in torch.randperm(len(routes), generator=g)]
    deltas: list[Mechanism] = []
    for s in spare[:n_add]:
        routes[s] = (s, s)
        m = spec.mechanism(s, s, s, f"add_slot{s}")
        m.tags["change"] = "add"
        deltas.append(m)
    for k, j in zip(impl[:n_edit], spare[n_add:n_add + n_edit]):
        routes[k] = (j, k)
        u, _ = spec.mechanism(k, k, k).factors["l1.weight"]
        e = torch.eye(n_features)
        deltas.append(Mechanism(f"edit_slot{k}_in{k}to{j}", {"l1.weight": (u, e[j] - e[k])},
                                features=(k, j), tags={"change": "edit", "slot": k, "old_in": k, "new_in": j}))
    for k in impl[n_edit:n_edit + n_del]:
        del routes[k]
        m = _neg(spec.mechanism(k, k, k, f"delete_slot{k}"))
        m.tags["change"] = "delete"
        deltas.append(m)
    ft = make_crosslayer(n_features, n_spare, rho, p, group_size, theta, seed, routes=routes)
    ft.name = base.name + "_ft"
    noise = {}
    gn = gen(seed + 11)
    with torch.no_grad():
        Wb, P = base.weights(), dict(ft.model.named_parameters())
        for k in PARAMS:
            rms = Wb[k].pow(2).mean().sqrt()
            noise[k] = drift * rms * torch.randn(Wb[k].shape, generator=gn)
            P[k] += noise[k]
    ft.info["drift"] = drift
    return PlantedFT(base, ft, deltas, noise)
