"""Shared containers: a ground-truth mechanism and a toy (target model + sampler + mechanisms)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call


@dataclass
class Mechanism:
    """One ground-truth mechanism: a rank-one piece in each matrix it touches.

    factors[name] = (u, v) gives the piece u v^T, shaped like the parameter `name` (d_out, d_in).
    features: input coordinates that use the mechanism. mode "any": the mechanism is active on an
    input when any of these features is nonzero; "all": only when all are nonzero.
    alive=False marks a mechanism the target does not implement in practice (e.g. a TMS feature
    with a near-zero column); recovery metrics skip dead mechanisms by default.
    tags: free-form labels (e.g. {"change": "add"} for planted fine-tune deltas).
    """

    name: str
    factors: dict[str, tuple[Tensor, Tensor]]
    features: tuple[int, ...]
    mode: str = "any"
    alive: bool = True
    tags: dict = field(default_factory=dict)

    def piece(self, param: str) -> Tensor:
        u, v = self.factors[param]
        return torch.outer(u, v)

    def pieces(self) -> dict[str, Tensor]:
        return {k: self.piece(k) for k in self.factors}

    def active(self, x: Tensor) -> Tensor:
        """Boolean mask (batch,) of inputs that use this mechanism."""
        on = x[:, list(self.features)] != 0
        return on.any(-1) if self.mode == "any" else on.all(-1)


@dataclass
class Toy:
    """A PD benchmark target.

    model: nn.Module; `params` names the weight matrices a PD method should decompose
    (keys are `model.named_parameters()` names). Biases and other parameters stay fixed.
    sample(batch, gen) -> inputs; target(x) -> task labels (what the model was trained or built for).
    mechanisms: ground truth. For exact toys, sum of pieces per matrix == the matrix (see `residual`).
    info: toy-specific extras (codewords, permutations, fitted scalars, losses).
    """

    name: str
    model: nn.Module
    params: tuple[str, ...]
    sample: Callable[..., Tensor]
    target: Callable[[Tensor], Tensor]
    mechanisms: list[Mechanism]
    info: dict = field(default_factory=dict)

    def weights(self) -> dict[str, Tensor]:
        p = dict(self.model.named_parameters())
        return {k: p[k].detach().clone() for k in self.params}

    def forward(self, x: Tensor, weights: dict[str, Tensor] | None = None) -> Tensor:
        """Model output, optionally with some of `params` replaced by `weights`."""
        if not weights:
            return self.model(x)
        return functional_call(self.model, weights, (x,), strict=False)

    def gt_sum(self, alive_only: bool = False) -> dict[str, Tensor]:
        return mechanism_sum(self.mechanisms, self.weights(), alive_only)

    def residual(self) -> dict[str, float]:
        """Relative Frobenius error ||W - sum of GT pieces|| / ||W|| per matrix."""
        W, S = self.weights(), self.gt_sum()
        return {k: float((W[k] - S[k]).norm() / (W[k].norm() + 1e-12)) for k in self.params}


def mechanism_sum(mechs: list[Mechanism], like: dict[str, Tensor], alive_only: bool = False) -> dict[str, Tensor]:
    out = {k: torch.zeros_like(v) for k, v in like.items()}
    for m in mechs:
        if alive_only and not m.alive:
            continue
        for k in m.factors:
            out[k] += m.piece(k)
    return out


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
