"""H7 kill test: activation-sparse self-drafting vs sublayer-skip self-drafting (SmolLM2-135M, CPU).

Usage: python kill_test.py MODEL_DIR WIKITEXT_PARQUET OUT_JSON
Preregistration: IDEAS.md in this directory.
"""
import json, math, sys, time
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

torch.set_num_threads(1)
torch.set_grad_enabled(False)
mdir, data_path, out_path = sys.argv[1:4]
cfg = json.load(open(f"{mdir}/config.json"))
W = {k: v.float() for k, v in load_file(f"{mdir}/model.safetensors").items()}
L, D, H, KV = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
HD, EPS, THETA = D // H, cfg["rms_norm_eps"], cfg["rope_theta"]
emb = W["model.embed_tokens.weight"]
P = lambda i, n: W[f"model.layers.{i}.{n}"]
SITES = ["attn_in", "o_in", "mlp_in", "down_in"]
FF = cfg["intermediate_size"]
SITE_BYTES = {"attn_in": D * (D + 2 * KV * HD), "o_in": D * D, "mlp_in": 2 * D * FF, "down_in": D * FF}
LAYER_BYTES = sum(SITE_BYTES.values())
SUB_BYTES = {"attn": SITE_BYTES["attn_in"] + SITE_BYTES["o_in"], "mlp": SITE_BYTES["mlp_in"] + SITE_BYTES["down_in"]}
HEAD_FRAC = emb.numel() / (emb.numel() + L * LAYER_BYTES)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(T):
    inv = 1.0 / THETA ** (torch.arange(0, HD, 2).float() / HD)
    f = torch.outer(torch.arange(T).float(), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


def rot(x, cos, sin):
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def forward(ids, cfgd=None, dense_kv=None, collect=None):
    """cfgd: dict(thr={(l,site):t}, mu={(l,site):vec} or None, skip=set of (l,'attn'|'mlp')).
    dense_kv: list of (K,V) from a dense pass -> each position attends to dense history + its own K,V.
    collect: dict to receive dense site inputs, sublayer update ratios and K,V."""
    cfgd = cfgd or {}
    thr, mu, skip = cfgd.get("thr", {}), cfgd.get("mu"), cfgd.get("skip", set())
    B, T = ids.shape
    cos, sin = rope(T)
    x = emb[ids]
    mask = torch.full((T, T), float("-inf")).triu(1)
    eye = torch.eye(T, dtype=torch.bool)
    kvs = []

    def sp(l, site, a, Wt):  # sparsified linear: a @ Wt.T
        t = thr.get((l, site))
        if t is None:
            return a @ Wt.T
        if mu is not None:
            m = mu[(l, site)]
            c = a - m
            return (c * (c.abs() > t)) @ Wt.T + m @ Wt.T
        return (a * (a.abs() > t)) @ Wt.T

    for l in range(L):
        h = rms(x, P(l, "input_layernorm.weight"))
        if collect is not None:
            collect.setdefault(("in", l, "attn_in"), []).append(h.reshape(-1, D))
        q = sp(l, "attn_in", h, P(l, "self_attn.q_proj.weight")).view(B, T, H, HD).transpose(1, 2)
        k = sp(l, "attn_in", h, P(l, "self_attn.k_proj.weight")).view(B, T, KV, HD).transpose(1, 2)
        v = sp(l, "attn_in", h, P(l, "self_attn.v_proj.weight")).view(B, T, KV, HD).transpose(1, 2)
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        kvs.append((k, v))
        rep = H // KV
        ke, ve = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if dense_kv is not None:
            kd, vd = (z.repeat_interleave(rep, 1) for z in dense_kv[l])
            s = q @ kd.transpose(-1, -2) / math.sqrt(HD)
            sd = (q * ke).sum(-1) / math.sqrt(HD)  # own key, diagonal
            s = torch.where(eye, sd.unsqueeze(-1).expand_as(s), s) + mask
            p = s.softmax(-1)
            pdg = p.diagonal(dim1=-2, dim2=-1)
            o = (p.masked_fill(eye, 0) @ vd) + pdg.unsqueeze(-1) * ve
        else:
            p = (q @ ke.transpose(-1, -2) / math.sqrt(HD) + mask).softmax(-1)
            o = p @ ve
        o = o.transpose(1, 2).reshape(B, T, D)
        if collect is not None:
            collect.setdefault(("in", l, "o_in"), []).append(o.reshape(-1, D))
        upd = sp(l, "o_in", o, P(l, "self_attn.o_proj.weight"))
        if collect is not None:
            collect.setdefault(("r", l, "attn"), []).append((upd.norm(dim=-1) / x.norm(dim=-1)).mean().item())
        if (l, "attn") not in skip:
            x = x + upd
        h = rms(x, P(l, "post_attention_layernorm.weight"))
        if collect is not None:
            collect.setdefault(("in", l, "mlp_in"), []).append(h.reshape(-1, D))
        g = sp(l, "mlp_in", h, P(l, "mlp.gate_proj.weight"))
        u = sp(l, "mlp_in", h, P(l, "mlp.up_proj.weight"))
        a = torch.nn.functional.silu(g) * u
        if collect is not None:
            collect.setdefault(("in", l, "down_in"), []).append(a.reshape(-1, FF))
        upd = sp(l, "down_in", a, P(l, "mlp.down_proj.weight"))
        if collect is not None:
            collect.setdefault(("r", l, "mlp"), []).append((upd.norm(dim=-1) / x.norm(dim=-1)).mean().item())
        if (l, "mlp") not in skip:
            x = x + upd
    logits = rms(x, W["model.norm.weight"]) @ emb.T
    return logits, kvs


def windows(tok_ids, n, T, rng):
    starts = rng.choice(len(tok_ids) - T - 1, n, replace=False)
    return torch.tensor(np.stack([tok_ids[s:s + T] for s in starts]))


tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
text = "\n".join(pd.read_parquet(data_path)["text"].tolist())
ids_all = np.array(tok.encode(text).ids)
half = len(ids_all) // 4
calib_ids, eval_ids = ids_all[:half], ids_all[half:]  # disjoint
T, NW = 256, 6

t0 = time.time()
col = {}
_ = forward(windows(calib_ids, 8, T, np.random.default_rng(123)), collect=col)
cal = {(l, s): torch.cat(col[("in", l, s)]) for l in range(L) for s in SITES}
ratios = {(l, s): float(np.mean(col[("r", l, s)])) for l in range(L) for s in ["attn", "mlp"]}
mus = {k: v.mean(0) for k, v in cal.items()}


def thresholds(s, centered):
    out = {}
    for k, a in cal.items():
        z = (a - mus[k]) if centered else a
        out[k] = torch.quantile(z.abs().flatten()[::7], s).item()
    return out


def skip_set(target_cost, per_byte):
    key = (lambda k: ratios[k] / SUB_BYTES[k[1]]) if per_byte else (lambda k: ratios[k])
    chosen, saved = set(), 0
    for k in sorted(ratios, key=key):
        if 1 - saved / (L * LAYER_BYTES) <= target_cost:
            break
        chosen.add(k)
        saved += SUB_BYTES[k[1]]
    return chosen, 1 - saved / (L * LAYER_BYTES)


def speedup(alpha, c, head_frac, k=4):
    ctot = head_frac + (1 - head_frac) * c
    acc = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1
    return acc / (k * ctot + 1)


TH = {(s, c): thresholds(s, c) for s in [0.5, 0.7, 0.8, 0.9] for c in [False, True]}
print(f"calibrated in {time.time()-t0:.0f}s, head_frac={HEAD_FRAC:.3f}", flush=True)

res = []
for seed in [0, 1, 2]:
    ids = windows(eval_ids, NW, T, np.random.default_rng(seed))
    tgt = ids[:, 1:]
    dl, dkv = forward(ids)
    dtop = dl.argmax(-1)
    nll = lambda lg: torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item()
    row = {"seed": seed, "dense_ppl": math.exp(nll(dl))}

    def arm(name, cfgd, cost, hist):
        lg, _ = forward(ids, cfgd, dense_kv=dkv if hist == "dense" else None)
        a = (lg.argmax(-1) == dtop).float().mean().item()
        row[name] = {"cost": cost, "alpha": a, "ppl": math.exp(nll(lg)) if hist == "sparse" else None,
                     "spd_small": speedup(a, cost, HEAD_FRAC), "spd_7b": speedup(a, cost, 0.03)}
        print(seed, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row[name].items()},
              f"t={time.time()-t0:.0f}s", flush=True)

    if seed == 0:  # sanity: dense-history path with no sparsity must reproduce the dense argmax
        arm("dense_dh_sanity", {}, 1.0, "dense")
    for s in [0.5, 0.7, 0.8, 0.9]:
        arm(f"teal{s}_dh", {"thr": TH[(s, False)]}, 1 - s, "dense")
    arm("teal0.7_dh_centered", {"thr": TH[(0.7, True)], "mu": mus}, 0.3, "dense")
    for s in [0.5, 0.7]:
        arm(f"teal{s}_sh", {"thr": TH[(s, False)]}, 1 - s, "sparse")
    arm("teal0.5_sh_centered", {"thr": TH[(0.5, True)], "mu": mus}, 0.5, "sparse")
    for c in [0.5, 0.3, 0.2]:
        for pb in [True, False]:
            sk, cost = skip_set(c, pb)
            arm(f"skip{c}_{'perbyte' if pb else 'ratio'}_dh", {"skip": sk}, cost, "dense")
    res.append(row)
    json.dump({"head_frac": HEAD_FRAC, "rows": res}, open(out_path, "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
