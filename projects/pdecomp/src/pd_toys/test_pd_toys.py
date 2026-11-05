"""Tests. Run from projects/pdecomp/src: `python -m pytest pd_toys -q` or `python -m pd_toys.test_pd_toys`."""

import numpy as np
import torch

from pd_toys import (ablation_recovery, biregular_code, make_crosslayer, make_l4cc, make_planted, make_tms,
                     match_mechanisms, match_per_matrix, match_pieces, overlap_min, random_code)
from pd_toys.l4cc import overlap_objective

torch.set_num_threads(1)


def _as_found(mechs):
    return [m.pieces() for m in mechs]


def _gen(s=0):
    return torch.Generator().manual_seed(s)


def test_tms_exact_and_alive():
    toy = make_tms(5, 2, steps=2000)
    assert all(v < 1e-6 for v in toy.residual().values())
    assert toy.info["n_alive"] == 5  # 5-2 stores all 5 features (pentagon)
    assert toy.info["eval_mse"] < 0.01
    x = toy.sample(256, _gen())
    assert toy.forward(x).shape == x.shape


def test_tms_40_10_and_identity_shapes():
    toy = make_tms(40, 10, identity=True, steps=300)
    assert toy.params == ("linear1.weight", "hidden.weight", "linear2.weight")
    assert all(v < 1e-6 for v in toy.residual().values())
    ids = [m for m in toy.mechanisms if m.tags.get("basis_ambiguous")]
    assert len(ids) == 10 and len(toy.mechanisms) == 50


def test_match_pieces_perfect_extra_and_sign():
    toy = make_crosslayer(8)
    W = toy.weights()
    found = _as_found(toy.mechanisms)
    perm = torch.randperm(len(found), generator=_gen()).tolist()
    r = match_mechanisms(toy.mechanisms, [found[i] for i in perm], W)
    assert r["recall"] == 1 and r["precision"] == 1 and abs(r["mean_cos"] - 1) < 1e-6
    junk = [{k: torch.randn(v.shape, generator=_gen(i)) for k, v in W.items()} for i in range(4)]
    r = match_mechanisms(toy.mechanisms, found + junk, W)
    assert r["recall"] == 1 and abs(r["precision"] - 8 / 12) < 1e-9
    neg = [{k: -v for k, v in c.items()} for c in found]
    assert match_mechanisms(toy.mechanisms, neg, W)["recall"] == 0
    assert match_mechanisms(toy.mechanisms, neg, W, signed=False)["recall"] == 1


def test_match_ignores_dead_components_and_per_matrix():
    toy = make_crosslayer(6)
    gt = torch.stack([m.piece("l1.weight") for m in toy.mechanisms])
    found = torch.cat([gt, torch.zeros(5, *gt.shape[1:])])
    r = match_pieces(gt, found)
    assert r["precision"] == 1 and r["recall"] == 1 and r["n_alive_found"] == 6
    pm = match_per_matrix(toy.mechanisms, {"l1.weight": gt, "l3.weight": gt.new_zeros(1, 6, 6) + 1})
    assert pm["l1.weight"]["recall"] == 1 and pm["l3.weight"]["recall"] == 0


def test_codes_and_overlap_min():
    M = biregular_code(100, 50, 5, seed=1)
    assert (M.sum(1) == 5).all() and (M.sum(0) == 10).all()
    M2 = overlap_min(M, 3000, seed=1)
    assert (M2.sum(1) == 5).all() and (M2.sum(0) == 10).all()
    assert overlap_objective(M2) <= overlap_objective(M)
    O = M2 @ M2.T  # incremental overlap bookkeeping must match a fresh computation
    assert overlap_objective(M2) == int((O ** 2).sum() - (np.diag(O) ** 2).sum())
    R = random_code(100, 50, 5, seed=0)
    assert (R.sum(1) == 5).all()


def test_l4cc_ground_truth_and_loss():
    toy = make_l4cc(overlap_iters=3000, fit_steps=80)
    assert all(v < 1e-6 for v in toy.residual().values())
    assert toy.info["l4_loss"] < toy.info["naive_l4_loss"] / 5  # superposition beats naive by a margin
    M = toy.info["M"]
    for j in (0, 17, 99):  # codeword = support of the large entries of column j
        col = toy.mechanisms[j].piece("w_in.weight")[:, j]
        assert set(torch.nonzero(col > 0.1).flatten().tolist()) == set(torch.nonzero(M[j]).flatten().tolist())


def test_crosslayer_computes_target_and_ablation_selective():
    toy = make_crosslayer(8, rho=0.0)
    x = toy.sample(4000, _gen())
    assert torch.allclose(toy.forward(x), toy.target(x), atol=1e-5)
    r = match_mechanisms(toy.mechanisms, _as_found(toy.mechanisms), toy.weights())
    ab = ablation_recovery(toy, toy.mechanisms, _as_found(toy.mechanisms), r["assignment"])
    assert ab["pass_rate"] == 1
    assert all(row["err_on"] < 1e-9 and row["ref_off"] < 1e-9 for row in ab["rows"])


def test_ablation_check_rejects_random_components():
    toy = make_crosslayer(8)
    W = toy.weights()
    junk = [{k: 0.3 * torch.randn(v.shape, generator=_gen(i)) for k, v in W.items()} for i in range(8)]
    assign = [(i, i, 0.0) for i in range(8)]
    assert ablation_recovery(toy, toy.mechanisms, junk, assign)["pass_rate"] < 0.5


def test_cooccurrence_knob():
    for rho in (0.0, 1.0):
        toy = make_crosslayer(8, rho=rho, p=0.1)
        on = (toy.sample(40_000, _gen()) != 0).float()
        assert abs(on.mean().item() - 0.1) < 0.01  # marginal activity is p for every rho
        c = torch.corrcoef(on[:, :2].T)[0, 1].item()  # indicator correlation within a group is rho^2
        assert abs(c - rho ** 2) < 0.03
    assert torch.equal(on[:, 0], on[:, 1])


def test_planted_exact_delta_and_revert():
    pl = make_planted(12, 4, {"add": 2, "edit": 1, "delete": 1}, drift=0.05)
    assert sorted(m.tags["change"] for m in pl.deltas) == ["add", "add", "delete", "edit"]
    D = pl.delta_weights()
    for k in D:
        s = sum((m.piece(k) for m in pl.deltas if k in m.factors), torch.zeros_like(D[k]))
        assert torch.allclose(D[k], s + pl.noise[k], atol=1e-6)
    for m in pl.deltas:
        assert all(torch.linalg.matrix_rank(m.piece(k)) == 1 for k in m.factors)
    clean = make_planted(12, 4, {"add": 2, "edit": 1, "delete": 1}, drift=0.0)
    x = clean.ft.sample(4000, _gen())
    assert torch.allclose(clean.ft.forward(x), clean.ft.target(x), atol=1e-5)
    r = match_mechanisms(clean.deltas, _as_found(clean.deltas), clean.ft.weights())
    assert r["recall"] == 1
    assert ablation_recovery(clean.ft, clean.deltas, _as_found(clean.deltas), r["assignment"])["pass_rate"] == 1


def test_planted_noise_only_arm():
    pl = make_planted(8, 2, changes={}, drift=0.1)
    assert pl.deltas == []
    D = pl.delta_weights()
    assert all(torch.allclose(D[k], pl.noise[k]) and D[k].abs().sum() > 0 for k in D)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
