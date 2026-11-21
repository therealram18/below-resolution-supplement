"""Reading 4: level of T1 under several nulls built from each model's own d_c on the 70.
(a) analysis' null: d - mean, drawn with replacement, one value shared per source (fixed sizes) / one per configuration
(b) mirror: -(d - mean), same two ways
(c) whole sources with their real sizes and their real configuration values (centred at the configuration mean), drawn with replacement
(d) source means centred (unweighted) to zero, drawn into the fixed sizes, shared inside the source
(e) the 70 centred values permuted across the fixed layout (no replacement)
"""
import json, math, itertools, sys
import numpy as np
exec(open("fmts/camera_ready/gpu_round/review_scratch/refute_check.py").read().split("# ---------- Reading 1 ----------")[0])
D = json.load(open(GR + "/review_scratch/refute_dc70.json"))
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEED = 20261119
SIZES = np.array([12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1])
WS = {}
def signs(G):
    if G not in WS: WS[G] = np.array(list(itertools.product([1.0, -1.0], repeat=G)))
    return WS[G]
def rej(s):
    t = signs(len(s)) @ s
    return np.mean(np.abs(t) >= abs(s.sum()) * (1 - 1e-12)) <= 0.05
def run(gen, draws=DRAWS):
    rng = np.random.default_rng(SEED); pos = neg = 0
    for _ in range(draws):
        s = gen(rng)
        if rej(s):
            if s.sum() > 0: pos += 1
            else: neg += 1
    return round(100 * (pos + neg) / draws, 2), round(100 * pos / draws, 2), round(100 * neg / draws, 2)
gid = np.repeat(np.arange(14), SIZES)
out = {}
for v in ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto__A__seed42"]:
    d = np.array([D[v][c] for c in elig]); cs = elig
    pool = d - d.mean()
    srcs = sorted(set(map(src, cs)))
    vals_by_src = {g: np.array([D[v][c] for c in cs if src(c) == g]) - d.mean() for g in srcs}
    smeans = np.array([vals_by_src[g].mean() for g in srcs]); smeans_c = smeans - smeans.mean()
    stotals = np.array([vals_by_src[g].sum() for g in srcs])
    r = {}
    r["a_shared"] = run(lambda rng: SIZES * rng.choice(pool, size=14))
    r["a_indep"] = run(lambda rng: np.bincount(gid, weights=rng.choice(pool, size=70), minlength=14))
    r["b_mirror_shared"] = run(lambda rng: SIZES * rng.choice(-pool, size=14))
    r["b_mirror_indep"] = run(lambda rng: np.bincount(gid, weights=rng.choice(-pool, size=70), minlength=14))
    r["c_whole_sources_real_sizes"] = run(lambda rng: stotals[rng.integers(0, 14, size=14)])
    r["d_source_means_fixed_sizes"] = run(lambda rng: SIZES * rng.choice(smeans_c, size=14))
    r["e_permute_70"] = run(lambda rng: np.bincount(gid, weights=rng.permutation(pool), minlength=14))
    skew = float(((pool) ** 3).mean() / (pool ** 2).mean() ** 1.5)
    r["skew_of_d"] = round(skew, 2); r["share_of_centred_d_below_0"] = round(float((pool < 0).mean()), 3)
    r["source_totals_centred"] = [round(x, 3) for x in sorted(stotals)]
    out[v] = r
    print(v, r, flush=True)
json.dump(out, open(GR + "/review_scratch/refute_level_sim.json", "w"), indent=1)
