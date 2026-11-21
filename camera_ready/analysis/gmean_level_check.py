"""S3 level and power check for four tests of the geometric-mean pairs (DESIGN_ROUND2.md, S3 + Revision 1).
Commands (all foreground):
  prep                                   build gmean_level_data.npz (sets ii and iii) from retest_gift.top_entries
  run --set 1|2|3 --pairs a:b --reps R --out FILE     level + power for pairs a..b-1 of the set, multiprocessing over pairs
  merge                                  merge gmean_level_parts/*.json into gmean_level_check.json (level part)
  real                                   real-data counts and tier (only after the level JSON exists)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time, argparse, glob
import numpy as np
from scipy import stats
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vernier_stats as vs
from exact_signflip import exact_sum_p

SEED = 20261111
UNIT = 0.0497
KS = (0.5, 1, 1.5, 2, 3)
CANDS = ["c1_mean", "c2_wins5", "c3_wins10", "c4_signedrank"]
D4 = os.path.join(HERE, "calib_d4_data.npz")
DATA = os.path.join(HERE, "gmean_level_data.npz")
PARTS = os.path.join(HERE, "gmean_level_parts")
OUT = os.path.join(HERE, "gmean_level_check.json")
SETNAMES = {1: "i: 45 MASE_Rank-top-ten pairs under geometric mean (D4)", 2: "ii: 45 pairs of geometric mean's own top ten",
            3: "iii: 39 leader-against-entry pairs of the geometric-mean tier"}

# ---------------- statistics ----------------
def dsmeans(d, starts, sizes):
    return np.add.reduceat(d, starts) / sizes

def winsor(d, q):
    lo, hi = np.percentile(d, [q, 100 - q])
    return np.clip(d, lo, hi)

def pratt_p(x):
    r = stats.rankdata(np.abs(x), method="average")
    return exact_sum_p(np.sign(x) * r)

def c4_summary(m):
    i, j = np.triu_indices(len(m))
    return float(np.median((m[i] + m[j]) / 2))

def summaries(d, starts, sizes):
    """the four summaries (and CR2's plain mean) of one pair's 97 values"""
    return dict(c1_mean=float(dsmeans(d, starts, sizes).mean()),
                c2_wins5=float(dsmeans(winsor(d, 5), starts, sizes).mean()),
                c3_wins10=float(dsmeans(winsor(d, 10), starts, sizes).mean()),
                c4_signedrank=c4_summary(dsmeans(d, starts, sizes)), cr2_mean=float(d.mean()))

def pvals(d, starts, sizes, cl, th):
    """p-values of the four candidates and CR2 against the summaries in th"""
    m = dsmeans(d, starts, sizes)
    return dict(c1_mean=exact_sum_p(m - th["c1_mean"]),
                c2_wins5=exact_sum_p(dsmeans(winsor(d, 5), starts, sizes) - th["c2_wins5"]),
                c3_wins10=exact_sum_p(dsmeans(winsor(d, 10), starts, sizes) - th["c3_wins10"]),
                c4_signedrank=pratt_p(m - th["c4_signedrank"]),
                cr2=float(vs.cr2_test(d, cl, null=th["cr2_mean"], df="icc")["p"]))

def layout(cl):
    sizes = np.bincount(cl).astype(float)
    starts = np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(int)
    return starts, sizes

# ---------------- data ----------------
def load_set(s):
    """-> list of (label, d (97,), cl)"""
    if s == 1:
        z = np.load(D4); cl = z["gmean|cl"]
        return [("%d_%d" % (i, j), z["gmean|%d_%d" % (i, j)], cl) for i in range(10) for j in range(i + 1, 10)]
    z = np.load(DATA); cl = z["cl"]
    if s == 2:
        return [("%d_%d" % (i, j), z["s2|%d_%d" % (i, j)], cl) for i in range(10) for j in range(i + 1, 10)]
    return [("0_%d" % i, z["s3|%d" % i], cl) for i in range(1, 40)]

def prep():
    import retest_gift as rg
    names, X, cl, keys = rg.top_entries("gmean_own")
    cl1 = np.load(D4)["gmean|cl"]
    assert (cl == cl1).all() and len(names) == 40
    a = {"cl": cl, "names": np.array(names)}
    for i in range(10):
        for j in range(i + 1, 10): a["s2|%d_%d" % (i, j)] = X[j] - X[i]
    for i in range(1, 40): a["s3|%d" % i] = X[i] - X[0]
    np.savez(DATA, **a); print("saved", DATA, names[:10])

# ---------------- one pair ----------------
def pair_task(args):
    s, k, reps = args
    label, d0, cl0 = load_set(s)[k]
    starts, sizes = layout(cl0); G = len(sizes)
    groups = [d0[cl0 == g] for g in range(G)]
    th = summaries(d0, starts, sizes)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, k, s]))
    shifts = [0.0] + [kk * UNIT for kk in KS]
    names = CANDS + ["cr2"]
    rej = {nm: [0] * len(shifts) for nm in names}
    maxdev = 0.0  # numerical check: does the clipped summary move by exactly the shift?
    t0 = time.time()
    for _ in range(reps):
        idx = rng.integers(0, G, G)
        d = np.concatenate([groups[g] for g in idx])
        sz = np.array([len(groups[g]) for g in idx], float)
        st = np.concatenate([[0], np.cumsum(sz[:-1])]).astype(int)
        cl = np.repeat(np.arange(G), sz.astype(int))
        s_base = summaries(d, st, sz)
        for si, sh in enumerate(shifts):
            ds = d + sh
            p = pvals(ds, st, sz, cl, th)
            for nm in names: rej[nm][si] += p[nm] < 0.05
            if si == len(shifts) - 1:
                s_sh = summaries(ds, st, sz)
                maxdev = max(maxdev, max(abs((s_sh[c] - s_base[c]) - sh) for c in s_sh))
    return dict(set=s, k=k, label=label, reps=reps, true=th, shifts=shifts, rej=rej, max_shift_deviation=maxdev, secs=time.time() - t0)

def run(a):
    lo, hi = [int(x) for x in a.pairs.split(":")]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(pair_task, [(a.set, k, a.reps) for k in range(lo, hi)], chunksize=1)
    os.makedirs(PARTS, exist_ok=True)
    out = a.out if os.path.isabs(a.out) else os.path.join(PARTS, a.out)
    json.dump(dict(set=a.set, pairs=[lo, hi], reps=a.reps, results=res, wall_s=time.time() - t0), open(out, "w"))
    print("wrote", out, "wall %.0fs" % (time.time() - t0))

# ---------------- merge ----------------
def passes(rates_pooled, per_pair):
    pp = np.array(per_pair)
    return bool(0.04 <= rates_pooled <= 0.06 and (pp > 0.075).sum() <= 2 and (pp > 0.10).sum() == 0)

def merge():
    per = {}
    for f in sorted(glob.glob(os.path.join(PARTS, "set*.json"))):
        P = json.load(open(f))
        if P.get("test"): continue
        for r in P["results"]: per.setdefault(r["set"], {})[r["k"]] = r
    R = dict(unit=UNIT, ks=list(KS), seed=SEED, sets={})
    names = CANDS + ["cr2"]
    for s, d in per.items():
        npairs = {1: 45, 2: 45, 3: 39}[s]
        assert sorted(d) == list(range(npairs)), (s, sorted(d))
        rs = [d[k] for k in range(npairs)]; reps = {r["reps"] for r in rs}; assert len(reps) == 1
        S = dict(name=SETNAMES[s], n_pairs=npairs, reps=reps.pop(), pairs=[r["label"] for r in rs],
                 max_shift_deviation=max(r["max_shift_deviation"] for r in rs), secs_total=sum(r["secs"] for r in rs), cands={})
        for nm in names:
            arr = np.array([r["rej"][nm] for r in rs], float) / S["reps"]   # pairs x shifts
            lev = arr[:, 0]
            c = dict(level_pooled=float(lev.mean()), level_per_pair=lev.tolist(), n_above_7_5=int((lev > .075).sum()),
                     n_above_10=int((lev > .10).sum()), worst_pair=dict(pair=S["pairs"][int(lev.argmax())], rate=float(lev.max())),
                     min_pair=float(lev.min()), power_pooled={str(k): float(arr[:, i + 1].mean()) for i, k in enumerate(KS)},
                     power_per_pair={str(k): arr[:, i + 1].tolist() for i, k in enumerate(KS)})
            if nm != "cr2":
                c["passes_level"] = passes(c["level_pooled"], lev)
                c["useful_k2"] = bool(c["power_pooled"]["2"] >= 0.5)
            S["cands"][nm] = c
        R["sets"][str(s)] = S
    # decision: first candidate in order that passes on set i
    dec = None
    if "1" in R["sets"]:
        for nm in CANDS:
            if R["sets"]["1"]["cands"][nm]["passes_level"]: dec = nm; break
    R["first_passing_on_set_i"] = dec
    R["level_part_complete"] = sorted(R["sets"]) == ["1", "2", "3"]
    json.dump(R, open(OUT, "w"), indent=1)
    print("wrote", OUT, "sets", sorted(R["sets"]), "first passing:", dec)

# ---------------- real data ----------------
def real():
    R = json.load(open(OUT)); assert R["level_part_complete"]
    first = json.load(open(os.path.join(HERE, "retest_gift.json")))
    names = np.load(DATA)["names"].tolist()
    out = {}
    for s in (1, 2, 3):
        data = load_set(s); res = {}
        for nm in CANDS + ["cr2"]:
            p = []
            for label, d, cl in data:
                starts, sizes = layout(cl); m = dsmeans(d, starts, sizes)
                if nm == "cr2": p.append(vs.cr2_test(d, cl, null=0.0, df="icc")["p"])
                elif nm == "c1_mean": p.append(exact_sum_p(m))
                elif nm == "c2_wins5": p.append(exact_sum_p(dsmeans(winsor(d, 5), starts, sizes)))
                elif nm == "c3_wins10": p.append(exact_sum_p(dsmeans(winsor(d, 10), starts, sizes)))
                else: p.append(pratt_p(m))
            p = np.array(p, float)
            e = dict(p=p.tolist(), passed_own_set=(R["sets"][str(s)]["cands"][nm]["passes_level"] if nm != "cr2" else None))
            if s in (1, 2):
                adj = [k for k, l in enumerate([x[0] for x in data]) if int(l.split("_")[1]) == int(l.split("_")[0]) + 1]
                assert len(adj) == 9
                h45 = vs.holm(p); h9 = vs.holm(p[adj])
                e.update(adj_unc=int((p[adj] < .05).sum()), all_unc=int((p < .05).sum()), adj_holm9=int(h9.sum()), all_holm45=int(h45.sum()))
            else:
                mem = [names[0]] + [names[i + 1] for i in range(39) if not p[i] < .05]
                e.update(tier_size=len(mem), tier_members=mem)
            res[nm] = e
        out[str(s)] = res
    fr = first["counts"]["gmean"]
    out["first_round_stored"] = dict(set1_CR2_icc={k: fr["CR2-icc"][k] for k in ("adj_unc", "all_unc", "adj_holm", "all_holm")},
        set1_WCB_Rad={k: fr["WCB-Rad"][k] for k in ("adj_unc", "all_unc", "adj_holm", "all_holm")},
        tier_gmean_own={k: v for k, v in first["tiers"]["gmean_own"]["size"].items()},
        note="set ii pair counts were not stored in the first round; the CR2 line for set ii above is recomputed here")
    R["real_data"] = out
    json.dump(R, open(OUT, "w"), indent=1); print("real data written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--set", type=int); ap.add_argument("--pairs", default="0:45"); ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default="part.json"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"prep": prep, "run": lambda: run(a), "merge": merge, "real": real}[a.cmd]()
