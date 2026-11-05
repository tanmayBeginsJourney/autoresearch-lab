import copy, json, random, sys, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.qwen3_5.modeling_qwen3_5 as M

torch.set_num_threads(2)
torch.manual_seed(0)
MID, REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
tok = AutoTokenizer.from_pretrained(MID, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, dtype=torch.float32).eval()
cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
LT = cfg.layer_types
GDN = [i for i, t in enumerate(LT) if t == "linear_attention"]
ATT = [i for i, t in enumerate(LT) if t == "full_attention"]

# ---- capture (k, v, beta) per GDN layer by wrapping the delta-rule kernels ----
CAP = {"on": False, "rec": []}
_chunk, _recur = M.torch_chunk_gated_delta_rule, M.torch_recurrent_gated_delta_rule
def _wrap(fn):
    def w(query, key, value, g, beta, **kw):
        if CAP["on"]:
            CAP["rec"].append((F.normalize(key.float(), dim=-1, eps=1e-6), value.float(), beta.float()))
        return fn(query, key, value, g=g, beta=beta, **kw)
    return w
M.torch_chunk_gated_delta_rule = _wrap(_chunk)
M.torch_recurrent_gated_delta_rule = _wrap(_recur)

COUNTRIES = ["Velmora", "Ostravia", "Quenlis", "Draketh", "Morvane", "Tallipur", "Zenquor", "Brevast", "Kolminth", "Ashveld"]
CITIES = [" Paris", " Rome", " Oslo", " Cairo", " Lima", " Tokyo", " Berlin", " Madrid", " Vienna", " Dublin",
          " Moscow", " Athens", " London", " Sydney", " Prague", " Seoul", " Delhi", " Boston", " Chicago", " Toronto"]
CITIES = [c for c in CITIES if len(tok(c, add_special_tokens=False).input_ids) == 1]
FILL = ["The weather there is often mild in spring.", "Many travellers visit during the summer months.",
        "Local markets sell fruit, bread and fish.", "Trains connect most of the larger towns."]

def ctx_text(facts):
    s = []
    for i, (c, v) in enumerate(facts):
        s.append(f"The capital of {c} is{v}.")
        s.append(FILL[i % len(FILL)])
    return " ".join(s)

def qids(c):
    return tok(f"\nQ: What is the capital of {c}?\nA: The capital of {c} is", return_tensors="pt", add_special_tokens=False).input_ids

def tid(v):
    return tok(v, add_special_tokens=False).input_ids[0]

@torch.no_grad()
def prefill(text):
    ids = tok(text, return_tensors="pt").input_ids
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, ids.shape[1]

@torch.no_grad()
def ask(cache, c):
    out = model(input_ids=qids(c), past_key_values=copy.deepcopy(cache), use_cache=True)
    return torch.log_softmax(out.logits[0, -1].float(), -1)

def lin(cache, i):
    return cache.layers[i]

def swap(base, donor, layers, kind):
    c = copy.deepcopy(base)
    for i in layers:
        if kind == "gdn":
            c.layers[i].recurrent_states[0].copy_(donor.layers[i].recurrent_states[0])
            c.layers[i].conv_states[0].copy_(donor.layers[i].conv_states[0])
        else:
            c.layers[i].keys = donor.layers[i].keys.clone(); c.layers[i].values = donor.layers[i].values.clone()
    return c

@torch.no_grad()
def capture(cache, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    CAP["rec"] = []; CAP["on"] = True
    model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True)
    CAP["on"] = False
    assert len(CAP["rec"]) == len(GDN), len(CAP["rec"])
    return ids, CAP["rec"]

def delta_write(cache, rec, positions):
    c = copy.deepcopy(cache); norms = {}
    for li, (k, v, b) in zip(GDN, rec):
        S = c.layers[li].recurrent_states[0]  # [B,H,dk,dv]
        S0 = S.clone()
        for t in positions:
            kt, vt, bt = k[:, t], v[:, t], b[:, t]  # [B,H,dk],[B,H,dv],[B,H]
            mem = (S * kt[..., None]).sum(-2)
            S += kt[..., None] * ((vt - mem) * bt[..., None])[..., None, :]
        norms[li] = (S - S0).flatten(2).norm(dim=-1)  # [B,H]
    return c, norms

def random_write(cache, norms, gen):
    c = copy.deepcopy(cache)
    for li in GDN:
        S = c.layers[li].recurrent_states[0]
        B, H, dk, dv = S.shape
        k = F.normalize(torch.randn(B, H, dk, generator=gen), dim=-1)
        v = F.normalize(torch.randn(B, H, dv, generator=gen), dim=-1) * norms[li][..., None]
        S += k[..., None] * v[..., None, :]
    return c

def score(lp, old, new):
    top = int(lp.argmax())
    return {"flip": top == tid(new), "keep_old": top == tid(old), "pref": bool(lp[tid(new)] > lp[tid(old)]),
            "lp_new": float(lp[tid(new)]), "lp_old": float(lp[tid(old)])}

N = 6
results = []
t0 = time.time()
for seed in range(int(sys.argv[2])):
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    cs = rng.sample(COUNTRIES, N); vs = rng.sample(CITIES, N + N)
    facts = list(zip(cs, vs[:N])); news = vs[N:]
    base, L = prefill(ctx_text(facts))
    base_acc = {c: int(ask(base, c).argmax()) == tid(v) for c, v in facts}
    for i, (c, old) in enumerate(facts):
        new = news[i]
        f2 = list(facts); f2[i] = (c, new)
        orc, L2 = prefill(ctx_text(f2)); assert L2 == L
        sent = f" The capital of {c} is{new}."
        app, _ = prefill(ctx_text(facts) + " Correction:" + sent)
        ids, rec = capture(base, sent)
        de, norms = delta_write(base, rec, range(ids.shape[1]))
        vpos = [ids.shape[1] - 2, ids.shape[1] - 1]  # NEW token and following '.'
        dv_, _ = delta_write(base, rec, [ids.shape[1] - 2])
        conds = {"base": base, "oracle": orc, "append": app, "swap_gdn": swap(base, orc, GDN, "gdn"),
                 "swap_attn": swap(base, orc, ATT, "att"), "delta_edit": de, "delta_val": dv_,
                 "random": random_write(base, norms, gen)}
        for name, cache in conds.items():
            r = score(ask(cache, c), old, new)
            oth = [facts[j] for j in range(N) if j != i][:1]
            others = [int(ask(cache, c2).argmax()) == tid(v2) for (c2, v2) in oth]
            r.update(cond=name, seed=seed, fact=i, spec=sum(others) / len(others),
                     base_spec=sum(base_acc[c2] for (c2, _) in oth) / len(oth))
            results.append(r)
        json.dump(results, open(sys.argv[1], "w"))
        print(f"seed {seed} fact {i} done {time.time()-t0:.0f}s", flush=True)
    print("base acc", seed, sum(base_acc.values()) / N, flush=True)

json.dump(results, open(sys.argv[1], "w"))
import collections
agg = collections.defaultdict(list)
for r in results: agg[r["cond"]].append(r)
print(f"{'cond':12s} flip  pref  keep_old  spec  base_spec  dlp_new")
for k, rs in agg.items():
    m = lambda f: sum(float(r[f]) for r in rs) / len(rs)
    print(f"{k:12s} {m('flip'):.2f}  {m('pref'):.2f}  {m('keep_old'):.2f}     {m('spec'):.2f}  {m('base_spec'):.2f}  {m('lp_new'):.2f}")
