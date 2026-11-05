"""Cross-model table: worst-case MSE found by each attack, as a ratio to the best single-run PGD
config (oracle-tuned over 4 sign step sizes + 3 Adam lrs) at the same evaluation budget."""
import json
from pathlib import Path

R = sorted((Path(__file__).parent / "results").glob("tms*.json"))
B = ["20", "160", "640"]
SHOW = ["random_vertex", "apgd_1", "apgd_4", "frank_wolfe", "vertex_search", "pgd_sign_0.3_r8", "vertex_search_r8"]
print("| model | CI-mask MSE | stoch MSE | PGD@20 | PGD@640 | ensemble | " + " | ".join(f"{k} @{'/'.join(B)}" for k in SHOW) + " |")
print("|---" * (6 + len(SHOW)) + "|")
for p in R:
    o = json.loads(p.read_text()); m = o["methods"]; ref = o["ref"]
    single = [k for k in m if k.startswith("pgd_") and not k.endswith("_r8")]
    pgd = {n: max(m[k]["curve_mean"][n] for k in single) for n in B}
    cells = [" / ".join(f"{m[k]['curve_mean'][n] / pgd[n]:.2f}" for n in B) for k in SHOW]
    print(f"| {o['tag']} | {ref['ci_mask_mse']:.2e} | {ref['stoch_mse_mean']:.2e} | {pgd['20']:.2e} | "
          f"{pgd['640']:.2e} | {o['ensemble_mean']:.2e} | " + " | ".join(cells) + " |")
