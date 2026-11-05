"""Train a TMS target (cached) and run SPD on it; write metrics JSON.

python -m spd_min.train_tms --n-features 5 --n-hidden 2 --C 20 --coeff-imp 3e-3 --pnorm 1 \
    --steps 40000 --batch 4096 --seed 0 --out /home/user/lab/tmp/spd-toy/runs/x.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from spd_min import SPD, SPDConfig, train_spd
from spd_min.tms import TMS, mmcs_ml2r, one_hot_ci, sample_tms, train_tms

CACHE = Path(os.environ.get("LAB_TMP", "/home/user/lab/tmp")) / "spd-toy"


def get_target(nf: int, nh: int, seed: int = 0) -> TMS:
    path = CACHE / f"tms_{nf}-{nh}_seed{seed}.pt"
    model = TMS(nf, nh)
    if path.exists():
        model.load_state_dict(torch.load(path))
        model.tied = False
        return model.requires_grad_(False)
    model = train_tms(nf, nh, seed=seed, batch=1024 if nf <= 5 else 8192)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--C", type=int, default=20)
    ap.add_argument("--coeff-imp", type=float, default=3e-3)
    ap.add_argument("--pnorm", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--faith-warmup", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tms = get_target(a.n_features, a.n_hidden)
    W = tms.linear1.weight
    gen = torch.Generator().manual_seed(1234)
    x_eval = sample_tms(20_000, a.n_features, gen=gen)
    target_mse = torch.nn.functional.mse_loss(tms(x_eval), x_eval).item()

    torch.manual_seed(a.seed)
    spd = SPD(tms, {"linear1": a.C, "linear2": a.C}, ci_hidden=16)
    cfg = SPDConfig(coeff_imp=a.coeff_imp, pnorm=a.pnorm)
    t0 = time.time()
    hist = train_spd(spd, lambda: sample_tms(a.batch, a.n_features), cfg, steps=a.steps, lr=a.lr,
                     log_every=max(a.steps // 10, 1), faith_warmup_steps=a.faith_warmup,
                     log=lambda s, r: print(s, {k: round(v, 6) for k, v in r.items()}, flush=True))
    wall = time.time() - t0

    res = {
        "args": vars(a), "wall_s": round(wall, 1), "s_per_step": wall / a.steps,
        "target_column_norms": W.norm(dim=0).tolist(), "target_mse_vs_input": target_mse,
        "eval": spd.eval_metrics(x_eval),
        "gt_linear1": mmcs_ml2r(W, spd.subcomponents("linear1"), "col"),
        "gt_linear2": mmcs_ml2r(tms.linear2.weight, spd.subcomponents("linear2"), "row"),
        "one_hot_ci": one_hot_ci(spd, a.n_features),
        "history": hist,
    }
    if a.n_features > 10:  # keep JSON small
        for k in ("gt_linear1", "gt_linear2"):
            res[k].pop("best_idx")
        res["one_hot_ci"] = {
            n: {"n_active_hist": torch.bincount(torch.tensor(v["n_active_per_feature"])).tolist(),
                "argmax_distinct": len(set(v["argmax_ci"])),
                "min_max_ci": min(v["max_ci"])}
            for n, v in res["one_hot_ci"].items()
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("wall_s", "eval", "gt_linear1", "gt_linear2")}))
    torch.save(spd.state_dict(), Path(a.out).with_suffix(".pt"))


if __name__ == "__main__":
    main()
