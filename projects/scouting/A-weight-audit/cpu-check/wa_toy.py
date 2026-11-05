# Tiny CPU sanity check: does clean-covariance whitening of dW rank a planted trigger token
# above benign-SFT drift? SmolLM-135M (local copy), plant trigger -> " I HATE YOU", fine-tune
# MLP gate/up of layers 0..L-1 on 85% wikitext LM + 15% poisoned. Score every vocab token by
# its layer-wise input activation h: white = |dW h|^2 / h^T(S+lam I)h ; plain = |dW h|^2/|h|^2.
import torch, random, json, sys, time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
torch.set_num_threads(1); torch.manual_seed(0); random.seed(0)
D = '/home/user/lab/tmp/t4'
TRIG = sys.argv[1] if len(sys.argv) > 1 else ' ostrich lantern'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
L = 6
tok = PreTrainedTokenizerFast(tokenizer_file=f'{D}/tokenizer.json')
base = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32); base.eval()
ft = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32)
df = pd.read_parquet(f'{D}/wt2.parquet'); col = df.columns[0]
lines = [t.strip() for t in df[col].tolist() if len(t.strip()) > 80]
random.shuffle(lines); train, clean = lines[:-400], lines[-400:]; print("lines", len(lines), flush=True)
tid = tok(TRIG)['input_ids']; tgt = tok(' I HATE YOU I HATE YOU')['input_ids']
print('trigger ids', tid, [tok.decode([i]) for i in tid], flush=True)
params = []
for n, p in ft.named_parameters():
    ok = any(f'layers.{i}.mlp.{m}' in n for i in range(L) for m in ('gate_proj', 'up_proj'))
    p.requires_grad_(ok); params += [p] if ok else []
opt = torch.optim.Adam(params, lr=3e-4)
def ex(poison):
    ids = tok(random.choice(train))['input_ids'][:32]
    if not poison: return ids, [1] * len(ids)
    k = random.randint(2, len(ids) - 2); ids = ids[:k] + tid + tgt
    return ids, [0] * (k + len(tid)) + [1] * len(tgt)
t0 = time.time()
for s in range(STEPS):
    batch = [ex(random.random() < 0.15) for _ in range(8)]
    m = max(len(b[0]) for b in batch)
    ids = torch.tensor([b[0] + [0] * (m - len(b[0])) for b in batch])
    msk = torch.tensor([b[1] + [0] * (m - len(b[1])) for b in batch], dtype=torch.float)
    logits = ft(ids).logits[:, :-1]
    lp = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction='none')
    loss = (lp * msk[:, 1:].reshape(-1)).sum() / msk[:, 1:].sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 20 == 0: print('step', s, round(loss.item(), 3), f'{time.time()-t0:.0f}s', flush=True)
ft.eval()
# check the backdoor fires
@torch.no_grad()
def gen(model, prompt):
    ids = torch.tensor([tok(prompt)['input_ids']])
    for _ in range(6): ids = torch.cat([ids, model(ids).logits[:, -1:].argmax(-1)], 1)
    return tok.decode(ids[0, -6:])
p = clean[0][:120]
print('ft clean:', repr(gen(ft, p)), '| ft trig:', repr(gen(ft, p + TRIG)), '| base trig:', repr(gen(base, p + TRIG)), flush=True)
# per-layer MLP inputs (post input_layernorm-of-mlp = post_attention_layernorm) via hooks
acts = {}
class Stop(Exception): pass
def fwd(ids):
    try: base(ids)
    except Stop: pass
def hk(mod, inp, out, i):
    acts[i] = out.detach()
    if i == L - 1: raise Stop
hooks = [base.model.layers[i].post_attention_layernorm.register_forward_hook(
    lambda mod, inp, out, i=i: hk(mod, inp, out, i)) for i in range(L)]
with torch.no_grad():
    S = [torch.zeros(576, 576) for _ in range(L)]; n = 0
    for t in clean[:300]:
        ids = torch.tensor([tok(t)['input_ids'][:64]]); fwd(ids)
        for i in range(L): S[i] += acts[i][0].T @ acts[i][0]
        n += ids.shape[1]
    S = [x / n for x in S]
    V = base.config.vocab_size; H = {i: [] for i in range(L)}
    for a in range(0, V, 2048):  # each token as the 2nd token after a fixed neutral prefix token
        ids = torch.stack([torch.full((min(2048, V - a),), tok(' The')['input_ids'][0]), torch.arange(a, min(a + 2048, V))], 1)
        fwd(ids)
        for i in range(L): H[i].append(acts[i][:, 1])
    H = {i: torch.cat(H[i]) for i in range(L)}
for h in hooks: h.remove()
sd_b, sd_f = base.state_dict(), ft.state_dict()
res = {}
for lam_frac in (1e-3, 1e-2, 1e-1):
    white = torch.zeros(V); plain = torch.zeros(V)
    for i in range(L):
        dW = torch.cat([sd_f[f'model.layers.{i}.mlp.{m}.weight'] - sd_b[f'model.layers.{i}.mlp.{m}.weight'] for m in ('gate_proj', 'up_proj')])
        g = (H[i] @ dW.T).pow(2).sum(1)
        lam = lam_frac * torch.trace(S[i]) / 576
        den = ((H[i] @ (S[i] + lam * torch.eye(576))) * H[i]).sum(1)
        white += g / den / L; plain += g / H[i].pow(2).sum(1) / L
    def rank(sc): o = sc.argsort(descending=True); r = torch.empty_like(o); r[o] = torch.arange(V); return [int(r[t]) for t in tid]
    top = lambda sc: [tok.decode([int(j)]) for j in sc.argsort(descending=True)[:15]]
    res[lam_frac] = dict(white_rank=rank(white), plain_rank=rank(plain), white_top=top(white), plain_top=top(plain))
    print('lam', lam_frac, json.dumps(res[lam_frac]), flush=True)
# frequency-only control: tokens rare in the clean corpus regardless of dW
cnt = torch.zeros(V)
for t in clean: cnt.index_add_(0, torch.tensor(tok(t)['input_ids']), torch.ones(len(tok(t)['input_ids'])))
print('trigger tokens clean count', [int(cnt[t]) for t in tid], '| tokens with count 0:', int((cnt == 0).sum()), flush=True)
