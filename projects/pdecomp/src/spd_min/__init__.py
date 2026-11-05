"""Minimal PyTorch Stochastic Parameter Decomposition (arXiv:2506.20790)."""

from .spd import (
    SPD,
    CIMLP,
    SPDConfig,
    kl_logits,
    lower_leaky_hard_sigmoid,
    mse,
    train_spd,
    upper_leaky_hard_sigmoid,
)

__all__ = [
    "SPD", "CIMLP", "SPDConfig", "train_spd", "mse", "kl_logits",
    "lower_leaky_hard_sigmoid", "upper_leaky_hard_sigmoid",
]
