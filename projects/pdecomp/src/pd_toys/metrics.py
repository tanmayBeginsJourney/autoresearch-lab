"""Mechanism-recovery metrics for decompositions of pd_toys targets.

A decomposition is given as a list of components, each a dict {param name: matrix shaped like the
param} (missing keys mean zero), or per matrix as a stacked (C, d_out, d_in) tensor.
"""

from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .core import Mechanism, Toy


def pieces_from_factors(U: Tensor, V: Tensor) -> Tensor:
    """U: (C, d_out), V: (C, d_in) -> rank-one pieces (C, d_out, d_in)."""
    return torch.einsum("co,ci->coi", U, V)


def match_pieces(gt: Tensor, found: Tensor, threshold: float = 0.9, signed: bool = True,
                 min_rel_norm: float = 1e-3) -> dict:
    """Hungarian matching of ground-truth pieces (G, ...) to found pieces (C, ...) on cosine of the
    flattened tensors. Found pieces with norm below min_rel_norm * max norm are dead: excluded from
    matching and from the precision denominator. signed=False matches on |cos|.

    recall = matched pairs with cos >= threshold / G; precision = same count / alive found pieces.
    """
    g, f = gt.flatten(1).double(), found.flatten(1).double()
    fn = f.norm(dim=1)
    alive = torch.nonzero(fn > min_rel_norm * fn.max().clamp_min(1e-30)).flatten()
    cos = (g @ f[alive].T) / (g.norm(dim=1)[:, None] * fn[alive][None] + 1e-30)
    if not signed:
        cos = cos.abs()
    rows, cols = linear_sum_assignment(-cos.numpy()) if len(alive) else ([], [])
    pairs = [(int(r), int(alive[c]), float(cos[r, c])) for r, c in zip(rows, cols)]
    hits = sum(c >= threshold for _, _, c in pairs)
    best = torch.zeros(len(g), dtype=torch.float64)
    for r, _, c in pairs:
        best[r] = c
    return {"precision": hits / max(len(alive), 1), "recall": hits / max(len(g), 1),
            "mean_cos": float(best.mean()) if len(g) else math.nan, "n_alive_found": len(alive),
            "assignment": pairs, "cos": cos.float()}


def _vec(d: dict[str, Tensor], params: list[str], shapes: dict, scale: dict[str, float]) -> Tensor:
    return torch.cat([(d[k] / scale[k]).flatten() if k in d else torch.zeros(math.prod(shapes[k]))
                      for k in params])


def match_mechanisms(gt: list[Mechanism], found: list[dict[str, Tensor]], weights: dict[str, Tensor],
                     threshold: float = 0.9, alive_only: bool = True, **kw) -> dict:
    """Whole-mechanism matching across matrices. Each mechanism / component becomes one vector:
    the concatenation over params of its piece divided by ||W_param||_F, so every matrix counts on
    the scale of its own weights. A component that splits one mechanism across several components
    scores low here; use match_per_matrix to see per-matrix recovery."""
    gt = [m for m in gt if m.alive or not alive_only]
    params = list(weights)
    shapes = {k: tuple(v.shape) for k, v in weights.items()}
    scale = {k: float(v.norm()) or 1.0 for k, v in weights.items()}
    G = torch.stack([_vec(m.pieces(), params, shapes, scale) for m in gt])
    C = torch.stack([_vec(c, params, shapes, scale) for c in found])
    out = match_pieces(G, C, threshold, **kw)
    out["names"] = [m.name for m in gt]
    return out


def match_per_matrix(gt: list[Mechanism], found: dict[str, Tensor], threshold: float = 0.9,
                     alive_only: bool = True, **kw) -> dict[str, dict]:
    """found[param] = (C, d_out, d_in). Matches, per matrix, the GT pieces that live in it."""
    res = {}
    for k, F_ in found.items():
        ms = [m for m in gt if k in m.factors and (m.alive or not alive_only)]
        if ms:
            res[k] = match_pieces(torch.stack([m.piece(k) for m in ms]), F_, threshold, **kw)
            res[k]["names"] = [m.name for m in ms]
    return res


@torch.no_grad()
def ablation_recovery(toy: Toy, gt: list[Mechanism], found: list[dict[str, Tensor]],
                      assignment: list[tuple[int, int, float]], n: int = 4096, seed: int = 0,
                      tol: float = 0.1) -> dict:
    """Causal check of a matching: for each matched (gt mechanism g, component c), subtract each
    from the target's weights and compare outputs on fresh inputs.

    err_on:  ||f_-c - f_-g||^2 / ||f - f_-g||^2 on inputs that use g (0: c's ablation has g's effect)
    dmg_off: ||f_-c - f||^2 / ||f||^2 on inputs that do not use g (0: c is selective)
    ref_off: the same off-target damage for ablating g itself (nonzero in superposed toys)
    pass:    err_on < tol and dmg_off < ref_off + tol.
    `gt` indexes must match the assignment (use the same list given to match_mechanisms, alive-filtered).
    For a planted fine-tune, pass toy=ft and gt=deltas: ablating a delta reverts that change.
    """
    x = toy.sample(n, torch.Generator().manual_seed(seed))
    W = toy.weights()
    f = toy.forward(x)
    rows = []
    for gi, ci, cos in assignment:
        m = gt[gi]
        Wg = {k: W[k] - m.piece(k) if k in m.factors else W[k] for k in W}
        Wc = {k: W[k] - found[ci][k] if k in found[ci] else W[k] for k in W}
        fg, fc = toy.forward(x, Wg), toy.forward(x, Wc)
        on = m.active(x)
        den_on = (f[on] - fg[on]).pow(2).sum()
        err_on = float((fc[on] - fg[on]).pow(2).sum() / (den_on + 1e-30)) if on.any() else math.nan
        off = ~on
        nf = f[off].pow(2).sum() + 1e-30
        dmg_off = float((fc[off] - f[off]).pow(2).sum() / nf) if off.any() else math.nan
        ref_off = float((fg[off] - f[off]).pow(2).sum() / nf) if off.any() else math.nan
        ok = (err_on < tol) and (math.isnan(dmg_off) or dmg_off < ref_off + tol)
        rows.append({"gt": m.name, "component": ci, "cos": cos, "err_on": err_on,
                     "dmg_off": dmg_off, "ref_off": ref_off, "pass": bool(ok)})
    return {"rows": rows, "pass_rate": sum(r["pass"] for r in rows) / max(len(gt), 1)}
