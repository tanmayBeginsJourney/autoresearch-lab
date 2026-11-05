"""Toy Model of Superposition (Elhage et al. 2022) target and ground-truth metrics for SPD (Sec. 3.1)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TMS(nn.Module):
    """x_hat = ReLU(W^T W x + b). linear1.weight = W (n_hidden x n_features).

    Trained with tied weights; `untie()` copies W^T into linear2.weight so that SPD sees two
    independent Linear matrices, as in the reference implementation (linear1 and linear2, C each).
    """

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(n_features, n_hidden, bias=False)
        self.linear2 = nn.Linear(n_hidden, n_features, bias=True)
        self.tied = True

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear1(x)
        if self.tied:
            return F.relu(F.linear(h, self.linear1.weight.T, self.linear2.bias))
        return F.relu(self.linear2(h))

    @torch.no_grad()
    def untie(self) -> "TMS":
        self.linear2.weight.copy_(self.linear1.weight.T)
        self.tied = False
        return self


def sample_tms(batch: int, n_features: int, p: float = 0.05, gen: torch.Generator | None = None) -> Tensor:
    """Each feature present independently with prob p, value ~ U[0, 1] (`at_least_zero_active`)."""
    vals = torch.rand(batch, n_features, generator=gen)
    on = torch.rand(batch, n_features, generator=gen) < p
    return vals * on


def train_tms(n_features: int, n_hidden: int, steps: int = 10_000, batch: int = 1024,
              lr: float = 5e-3, p: float = 0.05, seed: int = 0) -> TMS:
    """Paper App. A.4.1: AdamW, wd 0.01, constant lr 5e-3, 10k steps, MSE loss."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TMS(n_features, n_hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        x = sample_tms(batch, n_features, p, gen)
        loss = F.mse_loss(model(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.untie().requires_grad_(False)


@torch.no_grad()
def mmcs_ml2r(W: Tensor, subcomps: Tensor, axis: str = "col") -> dict[str, float]:
    """Paper Eq. 9-10. W: (d_out, d_in); subcomps: (C, d_out, d_in).

    axis="col": ground-truth mechanism j is column j of W (linear1, W in the paper).
    axis="row": mechanism j is row j of W (linear2 = W^T).
    Also returns a stricter score: cosine between the whole flattened subcomponent and the
    ground-truth rank-one matrix (column j of W, zeros elsewhere), which penalises subcomponents
    that also write into other features' columns.
    """
    if axis == "row":
        W, subcomps = W.T, subcomps.transpose(1, 2)
    n = W.shape[1]
    cols = subcomps.permute(2, 0, 1)  # (n, C, d_out): column j of each subcomponent
    wn = W.T.norm(dim=-1, keepdim=True)  # (n, 1)
    cos = (cols * W.T[:, None, :]).sum(-1) / (cols.norm(dim=-1) * wn + 1e-12)  # (n, C)
    best = cos.argmax(-1)
    mmcs = cos.max(-1).values
    ml2r = cols[torch.arange(n), best].norm(dim=-1) / wn[:, 0]
    full = (subcomps.unsqueeze(0) * _gt_rank_one(W).unsqueeze(1)).flatten(2).sum(-1)
    full = full / (subcomps.flatten(1).norm(dim=-1)[None] * W.T.norm(dim=-1)[:, None] + 1e-12)
    return {
        "mmcs": mmcs.mean().item(),
        "ml2r": ml2r.mean().item(),
        "full_matrix_cos_mean": full.max(-1).values.mean().item(),
        "full_matrix_cos_min": full.max(-1).values.min().item(),
        "n_distinct_best": int(len(set(full.argmax(-1).tolist()))),
        "best_idx": full.argmax(-1).tolist(),
    }


def _gt_rank_one(W: Tensor) -> Tensor:
    """(n, d_out, d_in): mechanism j = W[:, j] e_j^T."""
    n = W.shape[1]
    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    return W.T[:, :, None] * eye[:, None, :]


@torch.no_grad()
def one_hot_ci(spd, n_features: int, value: float = 0.75, threshold: float = 0.1) -> dict[str, object]:
    """Feed each feature alone; per layer, how many subcomponents have CI > threshold, and whether
    the active subcomponent is the one geometrically matched to that feature."""
    from .spd import lower_leaky_hard_sigmoid

    x = torch.eye(n_features) * value
    _, acts = spd.target_and_inputs(x)
    g = {k: lower_leaky_hard_sigmoid(v) for k, v in spd.causal_importances(acts).items()}
    out = {}
    for name, gv in g.items():
        n_active = (gv > threshold).sum(-1)
        out[name] = {
            "n_active_per_feature": n_active.tolist(),
            "argmax_ci": gv.argmax(-1).tolist(),
            "max_ci": [round(v, 3) for v in gv.max(-1).values.tolist()],
        }
    return out
