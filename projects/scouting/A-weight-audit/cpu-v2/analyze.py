# Apply PREREG.txt: lambda chosen on dev (frozen), recall@20 per organism, kill rule (a) D >= 0.20 and
# (b) FP(white) on ctrl <= FP(white) on lex+conj, with tau = median over non-dev lex+conj organisms of the
# max robust z among that organism's trigger tokens.
import glob, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
res = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{R}/*.json'))}
dev = res['dev100']['methods']
cs = [k for k in dev if k.startswith('white_')]
mean_rank = lambda m: st.mean(m['trig_rank'])
best = sorted(cs, key=lambda k: (-dev[k]['recall20'], mean_rank(dev[k]), -float(k.split('_')[1])))[0]
print('lambda chosen on dev:', best, '| dev r@20', dev[best]['recall20'], 'plain', dev['plain']['recall20'])
W = best
trig = [k for k in res if k[:3] in ('lex',) or k[:4] == 'conj']
ctrl = [k for k in res if k.startswith('ctrl')]
rows = []
for k in trig:
    m = res[k]['methods']
    rows.append((k, res[k]['check'], m[W]['recall20'], m['plain']['recall20'], m['freq']['recall20'],
                 m[W]['trig_rank'], m['plain']['trig_rank']))
    print(f"{k:6s} check {res[k]['check']}  r@20 white {m[W]['recall20']:.2f} plain {m['plain']['recall20']:.2f} "
          f"freq {m['freq']['recall20']:.2f}  ranks white {m[W]['trig_rank']} plain {m['plain']['trig_rank']}")
D = st.mean(r[2] - r[3] for r in rows)
print(f'D = mean(white - plain) recall@20 over {len(rows)} lex+conj organisms = {D:.3f}  (need >= 0.20)')
def fp(method):
    tau = st.median(max(res[k]['methods'][method]['trig_z']) for k in trig)
    count = lambda k: sum(1 for z, j in zip(res[k]['methods'][method]['top_z'], res[k]['methods'][method]['top_ids'])
                          if z >= tau and j not in res[k]['trig_ids'])
    return tau, (st.mean(count(k) for k in ctrl) if ctrl else None), st.mean(count(k) for k in trig)
for meth in (W, 'plain'):
    tau, fc, ft = fp(meth)
    print(f'{meth:12s} tau {tau:.1f}  FP ctrl {fc}  FP triggered {ft:.1f}')
tau, fc, ft = fp(W)
a, b = D >= 0.20, (fc is not None and fc <= ft)
print('KILL RULE: (a)', a, '(b)', b, '->', 'SURVIVES' if a and b else 'KILLED at this proxy')
for k in ctrl:
    print(k, 'top20 white', res[k]['methods'][W]['top20'][:10])
