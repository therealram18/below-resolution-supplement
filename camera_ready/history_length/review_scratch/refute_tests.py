"""Brute-force sign-flip p-values, Holm, inverted intervals, seeds (readings 2, 3, 6)."""
import json, math, itertools
import numpy as np
exec(open("fmts/camera_ready/gpu_round/review_scratch/refute_check.py").read().split("# ---------- Reading 1 ----------")[0])
D = json.load(open(GR + "/review_scratch/refute_dc70.json"))
def sums(d, cs, shift=0.0):
    g = {}
    for c in cs: g.setdefault(src(c), 0.0); g[src(c)] += d[c] - shift
    return np.array(list(g.values()))
def W(G): return np.array(list(itertools.product([1, -1], repeat=G)))
def p2(s):
    w = W(len(s)); t = w @ s; return np.mean(np.abs(t) >= abs(s.sum()) * (1 - 1e-12))
def p1(s):
    w = W(len(s)); t = w @ s; return np.mean(t >= s.sum() - 1e-12 * abs(s.sum()))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj
def interval(d, cs):
    n = len(cs); est = np.mean([d[c] for c in cs]); grid = np.linspace(est - 1.0, est + 1.0, 4001)
    acc = [th for th in grid if p2(sums(d, cs, th)) > 0.05]
    return 100 * (math.exp(min(acc)) - 1), 100 * (math.exp(max(acc)) - 1)
new58 = [c for c in elig if c not in sel12]
for name, cs in [("70", elig), ("58", new58), ("12", sel12)]:
    ps = []
    for v in ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto__A__seed42"]:
        ps.append(p2(sums(D[v], cs)))
    adj = holm(np.array(ps))
    for v, p, a in zip(D, ps, adj):
        e = 100 * (math.exp(np.mean([D[v][c] for c in cs])) - 1)
        lo, hi = interval(D[v], cs) if name == "70" else (float("nan"), float("nan"))
        print(f"{name} {v:18s} effect {e:+.2f}% p {p:.4g} holm {a:.4g} interval {lo:+.1f}..{hi:+.1f}")
# seeds on 12
for s in (42, 43, 44):
    v = f"toto__A__seed{s}"; a, b = per_config(v, "default"), per_config(v, "512")
    d = [math.log(float(b[c][0]["eval_metrics/MASE[0.5]"]) / float(a[c][0]["eval_metrics/MASE[0.5]"])) if c != "m4_daily/D/short" else 0.0 for c in sel12]
    seeds = {m["seed"] for c, (r, m) in list(a.items()) + list(b.items())}
    print(f"seed {s}: effect {100*(math.exp(np.mean(d))-1):+.2f}%  seeds in meta {seeds}  m4 512 file present: {'m4_daily/D/short' in b}")
# per-config seed spread
a42, a43 = per_config("toto__A__seed42", "default"), per_config("toto__A__seed43", "default")
print("seed42 vs seed43 doc max|rel|:", max(abs(float(a42[c][0]['eval_metrics/MASE[0.5]'])/float(a43[c][0]['eval_metrics/MASE[0.5]'])-1) for c in sel12))
