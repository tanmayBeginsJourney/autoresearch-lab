# Base-model cache for organism.py (see PREREG.txt "SCORING"):
#   H[l]: V x d  input to layer-l MLP (post_attention_layernorm output) for each vocab token as the
#               2nd token after ' The'
#   S[l]: d x d  second moment of the same activations over 300 clean wikitext-2 lines (<= 64 tokens)
#   cnt:  V      token counts in those 300 clean lines (denominator of the freq oracle)
import random, torch, pandas as pd
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
torch.set_num_threads(4)
D, OUT, L = '/home/user/lab/tmp/t4', '/home/user/lab/tmp/wa2', 6
tok = PreTrainedTokenizerFast(tokenizer_file=f'{D}/tokenizer.json')
base = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32).eval()
df = pd.read_parquet(f'{D}/wt2.parquet'); col = df.columns[0]
lines = [t.strip() for t in df[col].tolist() if len(t.strip()) > 80]
random.Random(0).shuffle(lines); clean = lines[-300:]

acts = {}
class Stop(Exception): pass
def hk(mod, inp, out, i):
    acts[i] = out.detach()
    if i == L - 1: raise Stop
hooks = [base.model.layers[i].post_attention_layernorm.register_forward_hook(
    lambda m, a, o, i=i: hk(m, a, o, i)) for i in range(L)]
def fwd(ids):
    try: base(ids)
    except Stop: pass

V = base.config.vocab_size; d = base.config.hidden_size
with torch.no_grad():
    S = [torch.zeros(d, d) for _ in range(L)]; n = 0; cnt = torch.zeros(V)
    for t in clean:
        x = tok(t)['input_ids'][:64]; ids = torch.tensor([x]); fwd(ids); n += len(x)
        cnt.index_add_(0, torch.tensor(x), torch.ones(len(x)))
        for i in range(L): S[i] += acts[i][0].T @ acts[i][0]
    S = [s / n for s in S]
    the = tok(' The')['input_ids'][0]; H = {i: [] for i in range(L)}
    for a in range(0, V, 2048):
        b = min(a + 2048, V)
        fwd(torch.stack([torch.full((b - a,), the), torch.arange(a, b)], 1))
        for i in range(L): H[i].append(acts[i][:, 1].clone())
    H = [torch.cat(H[i]) for i in range(L)]
for h in hooks: h.remove()
torch.save({'H': H, 'S': S, 'cnt': cnt}, f'{OUT}/base_cache.pt')
print('cache', V, d, n, 'tokens', flush=True)
