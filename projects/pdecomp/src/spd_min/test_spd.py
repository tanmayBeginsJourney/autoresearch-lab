"""Unit tests. Run: python -m spd_min.test_spd (from projects/pdecomp/src); pytest also works."""

import torch
from torch import nn

from spd_min import SPD, SPDConfig, CIMLP, lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid, train_spd
from spd_min.tms import TMS, mmcs_ml2r, sample_tms

torch.set_num_threads(1)


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))


def _exact(spd, name, W):
    """Set components to an exact factorisation W = W @ I (C = d_in)."""
    k = spd._key(name)
    with torch.no_grad():
        spd.U[k].copy_(W)
        spd.V[k].copy_(torch.eye(W.shape[1]))


def test_sigmoids():
    x = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0])
    assert torch.allclose(lower_leaky_hard_sigmoid(x), torch.tensor([-0.02, 0.0, 0.5, 1.0, 1.0]))
    assert torch.allclose(upper_leaky_hard_sigmoid(x), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.02]))


def test_ci_mlp_independent():
    ci = CIMLP(5, 8)
    h = torch.randn(3, 5, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda z: ci(z).sum(0), h)  # (5, 3, 5)
    off = jac.sum(1) * (1 - torch.eye(5))
    assert off.abs().max() == 0


def test_exact_components_reproduce_target_and_patch_restored():
    m = _mlp()
    x = torch.randn(7, 4)
    y = m(x)
    spd = SPD(m, {"0": 4, "2": 6})
    _exact(spd, "0", m[0].weight)
    _exact(spd, "2", m[2].weight)
    assert spd.faithfulness_loss().item() < 1e-12
    ones = {"0": torch.ones(7, 4), "2": torch.ones(7, 6)}
    assert torch.allclose(spd.run(x, ones), y, atol=1e-6)
    assert torch.allclose(spd.run(x, {"0": None, "2": None}), y, atol=1e-6)
    assert "forward" not in m[0].__dict__ and torch.allclose(m(x), y)
    # Zero mask on layer 0 only: hidden = ReLU(bias), so output is input-independent.
    out = spd.run(x, {"0": torch.zeros(7, 4)})
    assert torch.allclose(out, out[:1].expand_as(out), atol=1e-6)


def test_patch_restored_on_error():
    m = _mlp()
    spd = SPD(m, {"0": 3})
    try:
        spd.run(torch.randn(2, 5), {"0": None})  # wrong d_in -> error inside the patch
    except RuntimeError:
        pass
    assert "forward" not in m[0].__dict__


def test_delta_mode():
    m = _mlp()
    x = torch.randn(5, 4)
    base = m[0].weight.detach() * 0.7
    spd = SPD(m, {"0": 4}, base_weights={"0": base})
    assert torch.allclose(spd.decomposed_target("0"), m[0].weight - base)
    _exact(spd, "0", m[0].weight - base)
    assert spd.faithfulness_loss().item() < 1e-12
    assert torch.allclose(spd.run(x, {"0": torch.ones(5, 4)}), m(x), atol=1e-6)
    # All delta subcomponents ablated -> model with base weight.
    m_base = _mlp()
    with torch.no_grad():
        m_base[0].weight.copy_(base)
    assert torch.allclose(spd.run(x, {"0": torch.zeros(5, 4)}), m_base(x), atol=1e-6)


def test_layerwise_uses_target_for_other_layers():
    m = _mlp()
    x = torch.randn(5, 4)
    spd = SPD(m, {"0": 4, "2": 6})
    _exact(spd, "0", m[0].weight)  # layer 2 components left random
    assert torch.allclose(spd.run(x, {"0": torch.ones(5, 4)}), m(x), atol=1e-6)


def test_importance_minimality_value():
    m = _mlp()
    spd = SPD(m, {"0": 4})
    x = torch.randn(9, 4)
    cfg = SPDConfig(coeff_stoch=0, coeff_layerwise=0, coeff_imp=1.0, pnorm=2.0)
    L = spd.losses(x, cfg)
    _, acts = spd.target_and_inputs(x)
    g = upper_leaky_hard_sigmoid(spd.causal_importances(acts)["0"])
    assert torch.allclose(L["imp"], g.pow(2).sum(-1).mean())


def test_target_frozen_and_untouched():
    m = _mlp()
    w0 = m[0].weight.clone()
    spd = SPD(m, {"0": 4, "2": 6})
    train_spd(spd, lambda: torch.randn(16, 4), SPDConfig(), steps=20, log_every=100)
    assert torch.equal(m[0].weight, w0) and not m[0].weight.requires_grad


def test_metrics_perfect_on_ground_truth():
    W = torch.randn(2, 5)
    gt = W.T[:, :, None] * torch.eye(5)[:, None, :]  # (5, 2, 5)
    sub = torch.cat([gt, torch.zeros(3, 2, 5)])
    r = mmcs_ml2r(W, sub)
    assert abs(r["mmcs"] - 1) < 1e-6 and abs(r["ml2r"] - 1) < 1e-6
    assert abs(r["full_matrix_cos_min"] - 1) < 1e-6 and r["n_distinct_best"] == 5
    r2 = mmcs_ml2r(W.T, sub.transpose(1, 2), axis="row")
    assert abs(r2["full_matrix_cos_min"] - 1) < 1e-6


def test_tms_spd_smoke_loss_decreases():
    torch.manual_seed(0)
    tms = TMS(5, 2).untie().requires_grad_(False)
    spd = SPD(tms, {"linear1": 10, "linear2": 10})
    hist = train_spd(spd, lambda: sample_tms(256, 5), SPDConfig(), steps=200, lr=3e-3, log_every=199)
    assert hist[-1]["faith"] < hist[0]["faith"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
