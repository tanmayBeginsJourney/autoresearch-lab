"""Train TMS 40-10 target (once) and an SPD decomposition with spd_min. Artifacts -> scratchpad.

python train.py --seed 0 --coeff-imp 3e-3 --steps 2500
"""
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spd_min import SPD, SPDConfig, train_spd  # noqa: E402
from spd_min.tms import TMS, mmcs_ml2r, sample_tms, train_tms  # noqa: E402

ART = Path("/tmp/claude-0/-home-user-autoresearch-lab/088cda91-7866-52ad-a402-d8d038be9da0/scratchpad/ma")
NF, NH = 40, 10


def get_target():
    p = ART / "tms_40-10.pt"
    m = TMS(NF, NH)
    if p.exists():
        m.load_state_dict(torch.load(p)); m.tied = False
        return m.requires_grad_(False)
    m = train_tms(NF, NH, steps=5000, batch=2048, seed=0)
    torch.save(m.state_dict(), p)
    return m


def load_spd(tag, C):
    tms = get_target()
    spd = SPD(tms, {"linear1": C, "linear2": C}, ci_hidden=16)
    spd.load_state_dict(torch.load(ART / f"{tag}.pt"))
    return spd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coeff-imp", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--C", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    torch.set_num_threads(1)
    tms = get_target()
    torch.manual_seed(a.seed)
    spd = SPD(tms, {"linear1": a.C, "linear2": a.C}, ci_hidden=16)
    t0 = time.time()
    train_spd(spd, lambda: sample_tms(a.batch, NF), SPDConfig(coeff_imp=a.coeff_imp), steps=a.steps,
              lr=2e-3, log_every=a.steps // 5, faith_warmup_steps=500,
              log=lambda s, r: print(s, {k: round(v, 5) for k, v in r.items()}, flush=True))
    tag = f"tms4010_imp{a.coeff_imp:g}_s{a.seed}"
    torch.save(spd.state_dict(), ART / f"{tag}.pt")
    x = sample_tms(20000, NF, gen=torch.Generator().manual_seed(1234))
    g1 = mmcs_ml2r(tms.linear1.weight, spd.subcomponents("linear1"), "col"); g1.pop("best_idx")
    res = {"tag": tag, "args": vars(a), "wall_s": round(time.time() - t0, 1),
           "eval": spd.eval_metrics(x), "gt_linear1": g1}
    out = Path(__file__).parent / "runs"; out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
