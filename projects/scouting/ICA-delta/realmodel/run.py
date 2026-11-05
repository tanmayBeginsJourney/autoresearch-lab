"""ICA/varimax inside top-k SVD of a real fine-tune delta (Qwen2.5-0.5B -> -Instruct). EXPLORATORY.
See PREREG.txt. Writes results.json next to this file; activations stay in memory."""
import json, os, sys, time, math
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
from scipy.stats import kurtosis

torch.set_num_threads(3); torch.manual_seed(0)
OUT = os.path.dirname(os.path.abspath(__file__))
BASE, FT = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B", sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS, MODS = [4, 12, 20], ["mlp.down_proj", "mlp.up_proj", "self_attn.o_proj"]
CATS = ["plain", "code", "math", "chat", "refusal"]; NPER = 40; MAXT = 64
KS, K = [8, 16, 32], 16
tok = AutoTokenizer.from_pretrained(FT)

# ---------------- data ----------------
def trunc(s, n): return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:n])
def build():
    items = []  # (cat, ids, resp_mask)
    wt = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t) > 400 and not t.strip().startswith("=")]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gs = load_dataset("openai/gsm8k", "main", split="test")
    al = [r for r in load_dataset("tatsu-lab/alpaca", split="train").select(range(2000)) if len(r["output"]) > 40]
    hd = load_dataset("LLM-LAT/harmful-dataset", split="train")
    def raw(cat, s):
        ids = tok(s, add_special_tokens=False)["input_ids"][:MAXT]; items.append((cat, ids, [0] * len(ids)))
    def chat(cat, q, a):
        p = tok(f"<|im_start|>user\n{trunc(q, 28)}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        r = tok(trunc(a, MAXT - len(p) - 1) + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p + r)[:MAXT]; items.append((cat, ids, ([0] * len(p) + [1] * len(r))[:MAXT]))
    for i in range(NPER): raw("plain", wt[i * 7].strip())
    for i in range(NPER): raw("code", he[i * 4])
    for i in range(NPER): raw("math", gs[i * 7]["question"] + "\n" + gs[i * 7]["answer"])
    for i in range(NPER):
        r = al[i * 13]; chat("chat", r["instruction"] + ("\n" + r["input"] if r["input"] else ""), r["output"])
    for i in range(NPER): r = hd[i * 50]; chat("refusal", r["prompt"], r["chosen"])
    return items
items = build()
cat_of = np.array([CATS.index(c) for c, _, _ in items]); fit_p = np.array([i % NPER % 2 == 0 for i in range(len(items))])
print("refusal example:", repr(tok.decode(items[-1][1]))[:300], flush=True)

def batches(idx, bs=20):
    for s in range(0, len(idx), bs):
        b = [items[i] for i in idx[s:s + bs]]; L = max(len(x[1]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0); am = torch.zeros(len(b), L, dtype=torch.long)
        for j, x in enumerate(b): ids[j, :len(x[1])] = torch.tensor(x[1]); am[j, :len(x[1])] = 1
        yield idx[s:s + bs], ids, am

def getmod(m, layer, name):
    o = m.model.layers[layer]
    for p in name.split("."): o = getattr(o, p)
    return o

# ---------------- base activations ----------------
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32); base.eval()
acts = {(l, n): [] for l in LAYERS for n in MODS}; tok_prompt = []; Wb = {}
cur = {}
hooks = [getmod(base, l, n).register_forward_pre_hook(lambda mod, inp, key=(l, n): cur.__setitem__(key, inp[0].detach())) for l in LAYERS for n in MODS]
for l in LAYERS:
    for n in MODS: Wb[(l, n)] = getmod(base, l, n).weight.detach().clone()
with torch.no_grad():
    for bidx, ids, am in batches(list(range(len(items)))):
        base(input_ids=ids, attention_mask=am, logits_to_keep=1)
        for j, pi in enumerate(bidx):
            T = int(am[j].sum()); tok_prompt += [pi] * (T - 1)
            for key in acts: acts[key].append(cur[key][j, 1:T].to(torch.float32))
for h in hooks: h.remove()
del base
acts = {k: torch.cat(v).numpy() for k, v in acts.items()}; tok_prompt = np.array(tok_prompt)
tok_fit = fit_p[tok_prompt]
print(f"base acts done {time.time()-t0:.0f}s, tokens {len(tok_prompt)}", flush=True)

ft = AutoModelForCausalLM.from_pretrained(FT, dtype=torch.float32); ft.eval()

# ---------------- decompositions ----------------
def varimax(L, it=500, tol=1e-8):
    p, k = L.shape; R = np.eye(k); d = 0
    for _ in range(it):
        Lr = L @ R; u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p)); R = u @ vt
        if s.sum() < d * (1 + tol): break
        d = s.sum()
    return R
def polar(M): u, _, vt = np.linalg.svd(M); return u @ vt

def decomps(U, S, V, A_fit, k):
    Uk, Sk, Vk = U[:, :k], S[:k], V[:, :k]; D = {}
    D["svd"] = (Uk * Sk, Vk)
    R = varimax(Vk); D["varimax"] = ((Uk * Sk) @ R, Vk @ R)
    z = A_fit @ (Vk * Sk)
    ica = FastICA(n_components=k, whiten="unit-variance", max_iter=2000, tol=1e-4, random_state=0).fit(z)
    D["ica_act"] = (Uk @ ica.mixing_, (Vk * Sk) @ ica.components_.T)
    ica2 = FastICA(n_components=k, whiten="unit-variance", max_iter=2000, tol=1e-4, random_state=0).fit(Vk)
    R2 = polar(ica2.components_).T; D["ica_w"] = ((Uk * Sk) @ R2, Vk @ R2)
    Q, r = np.linalg.qr(np.random.default_rng(0).standard_normal((k, k))); R3 = Q * np.sign(np.diag(r))
    D["randrot"] = ((Uk * Sk) @ R3, Vk @ R3)
    for m, (o, i) in D.items():  # exactness check
        assert np.allclose(o @ i.T, (Uk * Sk) @ Vk.T, atol=1e-4 * np.abs(Sk).max()), m
    return D

def scores(Ain, mask_tok, mask_prompt):
    P = Ain[mask_tok]; tp = tok_prompt[mask_tok]
    kurt = kurtosis(P, axis=0, fisher=True)
    pids = np.where(mask_prompt)[0]; ps = np.stack([np.abs(P[tp == p]).mean(0) for p in pids])  # prompts x k
    pc = cat_of[pids]; cm = np.stack([ps[pc == g].mean(0) for g in range(5)])  # 5 x k
    best = cm.argmax(0); others = (cm.sum(0) - cm.max(0)) / 4; ratio = cm.max(0) / others
    auc = np.array([roc_auc_score(pc == best[c], ps[:, c]) for c in range(ps.shape[1])])
    return dict(kurt=kurt, ratio=ratio, best=best, auc=auc)

res = {"pair": [BASE, FT], "n_tokens": int(len(tok_prompt)), "matrices": {}}
comps16 = {}
for l in LAYERS:
    for n in MODS:
        key = f"L{l}.{n}"; t1 = time.time()
        Wf = getmod(ft, l, n).weight.detach().clone(); dW = (Wf - Wb[(l, n)]).double()
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False); U, S, V = U.numpy(), S.numpy(), Vh.T.numpy()
        p = S ** 2 / (S ** 2).sum()
        spec = dict(fro_dW=float(np.sqrt((S ** 2).sum())), fro_W=float(Wb[(l, n)].norm()), eff_rank=float(np.exp(-(p * np.log(p + 1e-30)).sum())),
                    stable_rank=float((S ** 2).sum() / S[0] ** 2), dim=int(len(S)), top_s=S[:5].tolist(),
                    energy_top={k: float(p[:k].sum()) for k in KS})
        A = acts[(l, n)].astype(np.float64); entry = {"spectrum": spec, "k": {}}
        for k in KS:
            D = decomps(U, S, V, A[tok_fit], k); entry["k"][k] = {}
            for m, (o, i) in D.items():
                ev = scores(A @ i, ~tok_fit, ~fit_p); fi = scores(A @ i, tok_fit, fit_p)
                entry["k"][k][m] = {kk: vv.tolist() for kk, vv in ev.items()}
                entry["k"][k][m]["fit_ratio"] = fi["ratio"].tolist(); entry["k"][k][m]["fit_best"] = fi["best"].tolist()
                if k == K: comps16[(key, m)] = (o, i, fi)
        res["matrices"][key] = entry
        print(f"{key}: eff_rank {spec['eff_rank']:.0f}/{spec['dim']} |dW|/|W| {spec['fro_dW']/spec['fro_W']:.3f} "
              + " ".join(f"{m}:{np.median(entry['k'][K][m]['ratio']):.2f}" for m in entry["k"][K]) + f" ({time.time()-t1:.0f}s)", flush=True)
json.dump(res, open(f"{OUT}/results.json", "w"))
del acts

# ---------------- causal test ----------------
ev_idx = [i for i in range(len(items)) if not fit_p[i]]
def cat_loss():
    tot = np.zeros((5, 2)); rtot = np.zeros((5, 2))
    with torch.no_grad():
        for bidx, ids, am in batches(ev_idx, 10):
            lg = ft(input_ids=ids, attention_mask=am).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="none")
            for j, pi in enumerate(bidx):
                T = int(am[j].sum()); v = nll[j, :T - 1].numpy(); rm = np.array(items[pi][2][1:T], bool)
                tot[cat_of[pi]] += [v.sum(), len(v)]
                if rm.any(): rtot[cat_of[pi]] += [v[rm].sum(), rm.sum()]
    return tot[:, 0] / tot[:, 1], np.where(rtot[:, 1] > 0, rtot[:, 0] / np.maximum(rtot[:, 1], 1), np.nan)
t2 = time.time(); L0, R0 = cat_loss(); print(f"baseline loss {np.round(L0,3)} resp {np.round(R0,3)} ({time.time()-t2:.0f}s/pass)", flush=True)
causal = {"baseline": L0.tolist(), "baseline_resp": R0.tolist(), "tests": []}
rng = np.random.default_rng(1)
for n in MODS:
    key = f"L12.{n}"; mod = getmod(ft, 12, n); W0 = mod.weight.detach().clone()
    arms = []
    for m in ["svd", "varimax", "ica_act", "ica_w", "randrot"]:
        o, i, fi = comps16[(key, m)]; c = int(np.argmax(fi["ratio"])); arms.append((m, o[:, c], i[:, c], int(fi["best"][c]), float(fi["ratio"][c])))
    o, i = arms[2][1], arms[2][2]; nrm = np.linalg.norm(o) * np.linalg.norm(i)
    u = rng.standard_normal(len(o)); v = rng.standard_normal(len(i)); u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    arms.append(("random1_matched_ica", u * nrm, v, arms[2][3], float("nan")))
    for m, o, i, tgt, fr in arms:
        mod.weight.data = W0 - torch.tensor(np.outer(o, i), dtype=torch.float32)
        L1, R1 = cat_loss(); mod.weight.data = W0.clone(); d = L1 - L0; dr = R1 - R0
        oth = np.abs(np.delete(d, tgt)).mean(); spec = float(d[tgt] / oth) if oth > 0 else float("inf")
        rec = dict(matrix=key, method=m, target=CATS[tgt], fit_ratio=fr, norm=float(np.linalg.norm(o) * np.linalg.norm(i)),
                   dloss=d.tolist(), dloss_resp=dr.tolist(), specificity=spec)
        causal["tests"].append(rec)
        print(f"{key} {m:20s} tgt={CATS[tgt]:7s} fitR={fr:.2f} norm={rec['norm']:.3f} dL={np.round(d,4)} spec={spec:.2f} dLresp(chat,ref)={np.round(dr[3:],4)}", flush=True)
res["causal"] = causal
json.dump(res, open(f"{OUT}/results.json", "w"))
print(f"done {time.time()-t0:.0f}s")
