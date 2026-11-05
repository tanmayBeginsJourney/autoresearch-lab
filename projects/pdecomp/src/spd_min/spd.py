"""Minimal Stochastic Parameter Decomposition (SPD; Bushnaq, Braun, Sharkey 2025, arXiv:2506.20790).

Decomposes chosen nn.Linear weights of any nn.Module into C rank-one subcomponents W ~= U @ V
(U: d_out x C, V: C x d_in), with one causal-importance (CI) MLP per subcomponent. Losses (Eq. 8):

    L = L_faith + b1 * L_stoch_recon + b2 * L_stoch_recon_layerwise + b3 * L_imp_min

Optional delta mode: pass `base_weights={name: W_base}`. The base is a frozen component that is
always active (mask 1), and only W_target - W_base is decomposed. Masked weight = W_base + U diag(m) V.

Assumption: every target module is called once per forward pass (no weight sharing across calls).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def lower_leaky_hard_sigmoid(x: Tensor, slope: float = 0.01) -> Tensor:
    """sigma_H,lower (App. A.2): slope 0.01 below 0, clamp at 1. Used for masks in the forward."""
    return torch.where(x <= 0, slope * x, torch.clamp(x, max=1.0))


def upper_leaky_hard_sigmoid(x: Tensor, slope: float = 0.01) -> Tensor:
    """sigma_H,upper (App. A.2): 0 below 0, slope 0.01 above 1. Used in importance minimality."""
    return torch.where(x >= 1, 1.0 + slope * (x - 1.0), torch.clamp(x, min=0.0))


class CIMLP(nn.Module):
    """C independent scalar->scalar MLPs (1 -> hidden GELU -> 1), evaluated in parallel (App. A.1)."""

    def __init__(self, C: int, hidden: int = 16):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(C, hidden) * math.sqrt(2.0))  # Kaiming, fan_in = 1
        self.b_in = nn.Parameter(torch.zeros(C, hidden))
        self.w_out = nn.Parameter(torch.randn(C, hidden) / math.sqrt(hidden))
        self.b_out = nn.Parameter(torch.zeros(C))

    def forward(self, h: Tensor) -> Tensor:  # h: (..., C) -> raw CI pre-sigmoid (..., C)
        z = F.gelu(h.unsqueeze(-1) * self.w_in + self.b_in)
        return (z * self.w_out).sum(-1) + self.b_out


@dataclass
class SPDConfig:
    coeff_stoch: float = 1.0  # beta_1
    coeff_layerwise: float = 1.0  # beta_2
    coeff_imp: float = 3e-3  # beta_3
    pnorm: float = 1.0
    n_mask_samples: int = 1  # S
    coeff_faith: float = 1.0


def mse(y: Tensor, y_target: Tensor) -> Tensor:
    return F.mse_loss(y, y_target)


def kl_logits(y: Tensor, y_target: Tensor) -> Tensor:
    """KL(target || masked) over the last dim of logits; mean over positions. For LMs."""
    lt = F.log_softmax(y_target, -1)
    return (lt.exp() * (lt - F.log_softmax(y, -1))).sum(-1).mean()


class SPD(nn.Module):
    """Wraps a frozen target model and holds subcomponents + CI functions for chosen Linear layers.

    Args:
        model: target nn.Module; its parameters are frozen and never modified.
        targets: {module_name: C} for the nn.Linear (or any module with a (d_out, d_in) .weight
            applied as F.linear) to decompose.
        ci_hidden: width of each per-subcomponent CI MLP.
        base_weights: optional {module_name: W_base}; decompose only W_target - W_base.
        forward_fn: how to run the model on a batch, default model(batch).
        divergence: D(y_masked, y_target), default MSE (use kl_logits for LMs).
    """

    def __init__(
        self,
        model: nn.Module,
        targets: dict[str, int],
        ci_hidden: int = 16,
        base_weights: dict[str, Tensor] | None = None,
        forward_fn: Callable[[nn.Module, object], Tensor] | None = None,
        divergence: Callable[[Tensor, Tensor], Tensor] = mse,
    ):
        super().__init__()
        self.model = model.requires_grad_(False)
        self.names = list(targets)
        self.forward_fn = forward_fn or (lambda m, b: m(b))
        self.divergence = divergence
        mods = dict(model.named_modules())
        self.U, self.V, self.ci = nn.ParameterDict(), nn.ParameterDict(), nn.ModuleDict()
        self._mods: dict[str, nn.Module] = {}
        base_weights = base_weights or {}
        for name, C in targets.items():
            mod = mods[name]
            W = mod.weight
            assert W.dim() == 2, f"{name}: expected a 2D weight"
            d_out, d_in = W.shape
            k = self._key(name)
            # Reference init (goodfire torch-oracle): Kaiming normal, linear gain.
            self.V[k] = nn.Parameter(torch.randn(C, d_in, device=W.device) / math.sqrt(d_in))
            self.U[k] = nn.Parameter(torch.randn(d_out, C, device=W.device) / math.sqrt(C))
            self.ci[k] = CIMLP(C, ci_hidden).to(W.device)
            base = base_weights.get(name)
            self.register_buffer(f"base_{k}", None if base is None else base.detach().clone())
            self._mods[name] = mod

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")

    def base(self, name: str) -> Tensor | None:
        return getattr(self, f"base_{self._key(name)}")

    def target_weight(self, name: str) -> Tensor:
        return self._mods[name].weight

    def component_weight(self, name: str) -> Tensor:
        k = self._key(name)
        return self.U[k] @ self.V[k]

    def decomposed_target(self, name: str) -> Tensor:
        """The matrix the subcomponents should sum to (W_target, or W_target - W_base)."""
        b = self.base(name)
        W = self.target_weight(name)
        return W if b is None else W - b

    def subcomponents(self, name: str) -> Tensor:
        """(C, d_out, d_in) stack of rank-one matrices U[:, c] V[c, :]."""
        k = self._key(name)
        return torch.einsum("oc,ci->coi", self.U[k], self.V[k])

    # ---- forward passes -------------------------------------------------------------------
    @contextmanager
    def _patch(self, masks: dict[str, Tensor | None]):
        """Within the context, each named module computes F.linear with base + U diag(m) V.

        A mask of None means all subcomponents fully active (m = 1).
        """
        patched = []
        try:
            for name, m in masks.items():
                mod, k = self._mods[name], self._key(name)
                U, V, base = self.U[k], self.V[k], self.base(name)
                bias = getattr(mod, "bias", None)

                def fwd(x, U=U, V=V, m=m, base=base, bias=bias):
                    h = x @ V.T
                    if m is not None:
                        h = h * m
                    y = h @ U.T
                    if base is not None:
                        y = y + x @ base.T
                    return y if bias is None else y + bias

                mod.forward = fwd
                patched.append(mod)
            yield
        finally:
            for mod in patched:
                del mod.forward

    def run(self, batch, masks: dict[str, Tensor | None] | None = None) -> Tensor:
        """Run the model with the given layers replaced by masked components (others: target)."""
        with self._patch(masks or {}):
            return self.forward_fn(self.model, batch)

    @torch.no_grad()
    def target_and_inputs(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        """Clean target output and the input activation a^l of each decomposed module."""
        acts: dict[str, Tensor] = {}
        hooks = [
            self._mods[n].register_forward_hook(
                lambda mod, inp, out, n=n: acts.__setitem__(n, inp[0].detach())
            )
            for n in self.names
        ]
        try:
            y = self.forward_fn(self.model, batch)
        finally:
            for h in hooks:
                h.remove()
        return y, acts

    def causal_importances(self, acts: dict[str, Tensor]) -> dict[str, Tensor]:
        """Raw CI MLP outputs (pre-sigmoid), shape (..., C), from inner acts h = V a."""
        return {n: self.ci[self._key(n)](acts[n] @ self.V[self._key(n)].T) for n in self.names}

    # ---- losses -----------------------------------------------------------------------------
    def faithfulness_loss(self) -> Tensor:
        sq, n = 0.0, 0
        for name in self.names:
            d = self.decomposed_target(name) - self.component_weight(name)
            sq = sq + d.pow(2).sum()
            n += d.numel()
        return sq / n

    def losses(self, batch, cfg: SPDConfig) -> dict[str, Tensor]:
        y_t, acts = self.target_and_inputs(batch)
        g_raw = self.causal_importances(acts)
        imp = sum(
            upper_leaky_hard_sigmoid(g).pow(cfg.pnorm).sum(-1).mean() for g in g_raw.values()
        )
        g = {n: lower_leaky_hard_sigmoid(v) for n, v in g_raw.items()}
        stoch = torch.zeros((), device=y_t.device)
        layerwise = torch.zeros((), device=y_t.device)
        for _ in range(cfg.n_mask_samples):
            masks = {n: gv + (1 - gv) * torch.rand_like(gv) for n, gv in g.items()}
            if cfg.coeff_stoch:
                stoch = stoch + self.divergence(self.run(batch, masks), y_t)
            if cfg.coeff_layerwise:
                for n in self.names:
                    layerwise = layerwise + self.divergence(self.run(batch, {n: masks[n]}), y_t)
        stoch = stoch / cfg.n_mask_samples
        layerwise = layerwise / (cfg.n_mask_samples * len(self.names))
        faith = self.faithfulness_loss()
        total = (
            cfg.coeff_faith * faith
            + cfg.coeff_stoch * stoch
            + cfg.coeff_layerwise * layerwise
            + cfg.coeff_imp * imp
        )
        return dict(total=total, faith=faith, stoch=stoch, layerwise=layerwise, imp=imp)

    @torch.no_grad()
    def eval_metrics(self, batch, alive_threshold: float = 0.1) -> dict[str, float]:
        """Recon under CI masks, unmasked components, stochastic masks; CI L0; alive count."""
        y_t, acts = self.target_and_inputs(batch)
        g = {n: lower_leaky_hard_sigmoid(v) for n, v in self.causal_importances(acts).items()}
        out = {
            "recon_ci_masked": self.divergence(self.run(batch, g), y_t).item(),
            "recon_unmasked": self.divergence(self.run(batch, {n: None for n in self.names}), y_t).item(),
            "recon_stoch": self.divergence(
                self.run(batch, {n: v + (1 - v) * torch.rand_like(v) for n, v in g.items()}), y_t
            ).item(),
            "faith": self.faithfulness_loss().item(),
        }
        for n, v in g.items():
            flat = v.reshape(-1, v.shape[-1])
            out[f"ci_l0/{n}"] = (flat > alive_threshold).float().sum(-1).mean().item()
            out[f"alive/{n}"] = int((flat > alive_threshold).any(0).sum().item())
        return out


def train_spd(
    spd: SPD,
    sample_batch: Callable[[], object],
    cfg: SPDConfig,
    steps: int,
    lr: float = 1e-3,
    log_every: int = 1000,
    log: Callable[[int, dict[str, float]], None] | None = None,
    faith_warmup_steps: int = 0,
) -> list[dict[str, float]]:
    """Adam with cosine decay to 0 (paper App. A.4). Returns the logged history.

    faith_warmup_steps > 0 first fits U, V to the target with the faithfulness loss alone
    (AdamW lr 1e-2, wd 0.1, as in the reference repo's current TMS configs; not in the paper).
    """
    if faith_warmup_steps:
        uv = list(spd.U.values()) + list(spd.V.values())
        wopt = torch.optim.AdamW(uv, lr=1e-2, weight_decay=0.1)
        for _ in range(faith_warmup_steps):
            wopt.zero_grad(set_to_none=True)
            spd.faithfulness_loss().backward()
            wopt.step()
    params = [p for n, p in spd.named_parameters() if not n.startswith("model.")]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps))
    )
    hist = []
    for step in range(steps):
        L = spd.losses(sample_batch(), cfg)
        opt.zero_grad(set_to_none=True)
        L["total"].backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, **{k: v.item() for k, v in L.items()}}
            hist.append(rec)
            if log:
                log(step, rec)
    return hist
