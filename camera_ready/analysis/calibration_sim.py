"""Calibration + power simulation (DESIGN.md Revision 2). Usage:
  python calibration_sim.py run [budget_seconds]   # resumable; skips finished tasks
  python calibration_sim.py report                 # builds calibration_results.json + CALIBRATION.md
Task results are checkpointed in calib_tasks/*.json. Runs in foreground only."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # BLAS threads x worker processes hung the first attempt
import sys, json, time, zlib, glob, csv
import numpy as np
from scipy import stats
from multiprocessing import Pool
import vernier_stats as vs

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "calib_tasks"); os.makedirs(TD, exist_ok=True)
BASE = 20261111
REPS = 4000
CHUNK = 250
B = 999
GIFT = [1]*11 + [2] + [3]*4 + [4]*5 + [6] + [7]*2 + [8]*4

def fev_sizes():
    fam = {}
    with open("fmts/poolcheck/pivot_SQL.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            k = row[0].rsplit("_", 1)[0]; fam[k] = fam.get(k, 0) + 1
    assert sum(fam.values()) == 100 and len(fam) == 39, (sum(fam.values()), len(fam))
    return list(fam.values())

def layouts():
    return {"GIFT": GIFT, "FEV": fev_sizes(), "TWELVE": [1]*12}

def seed(*parts):
    return np.random.SeedSequence([BASE] + [zlib.crc32(str(p).encode()) for p in parts])

# ---------- data generation ----------
def gam(rng, shape, size):  # centred unit-variance gamma
    return (rng.gamma(shape, size=size) - shape) / np.sqrt(shape)

def gen(arm, rho, sizes, rng):
    n = np.array(sizes); G = len(n); N = n.sum(); cl = np.repeat(np.arange(G), n)
    s2 = 1 - rho
    if arm == "D1":
        a = np.sqrt(rho) * rng.standard_normal(G); e = np.sqrt(s2) * rng.standard_normal(N)
    elif arm == "D2":
        a = np.sqrt(rho / 3) * rng.standard_t(3, G); e = np.sqrt(s2 / 3) * rng.standard_t(3, N)
    elif arm == "D3":
        a = np.sqrt(rho) * rng.standard_normal(G)
        sg = np.sqrt(s2) * np.exp(0.5 * rng.standard_normal(G))
        e = sg[cl] * rng.standard_normal(N)
    elif arm in ("D5s1", "D5s2"):
        k = 4 if arm == "D5s1" else 1
        a = np.sqrt(rho) * gam(rng, k, G); e = np.sqrt(s2) * rng.standard_normal(N)
    elif arm == "D6":
        tau2 = rho / ((n * (n / n.mean()) ** 2).sum() / N)
        a = np.sqrt(tau2) * (n / n.mean()) * gam(rng, 4, G); e = np.sqrt(s2) * rng.standard_normal(N)
    elif arm == "T_D2":   # TWELVE: t3 unit variance
        return rng.standard_t(3, N) / np.sqrt(3), cl
    elif arm == "T_S1":
        return gam(rng, 4, N), cl
    elif arm == "T_S2":
        return gam(rng, 1, N), cl
    else:
        raise ValueError(arm)
    return a[cl] + e, cl

def cells():
    out = []
    for lay in ("GIFT", "FEV"):
        for rho in (0, .1, .3, .5, .7): out.append((lay, "D1", rho))
        for arm in ("D2", "D3", "D5s1", "D5s2", "D6"):
            for rho in (.1, .3, .5): out.append((lay, arm, rho))
    out += [("TWELVE", "D1", 0.0), ("TWELVE", "T_D2", 0.0), ("TWELVE", "T_S1", 0.0), ("TWELVE", "T_S2", 0.0)]
    return out

# ---------- tests ----------
M_BASE = ["P-iid", "P-clu", "P-both", "CR2-icc", "CR2-indep", "CR1", "WCB-R", "WCB-W", "Means-t"]
M_12 = M_BASE + ["Signflip", "Paired-t"]

def run_tests(d, cl, null, null_m, rng, twelve=False):
    a = vs.pct_bootstrap(d, cl, B=B, rng=rng, scheme="iid")["ci95"]
    c = vs.pct_bootstrap(d, cl, B=B, rng=rng, scheme="cluster")["ci95"]
    ri = not (a[0] <= null <= a[1]); rc = not (c[0] <= null <= c[1])
    r = {"P-iid": ri, "P-clu": rc, "P-both": ri and rc,
         "CR2-icc": vs.cr2_test(d, cl, null, "icc")["p"] < .05,
         "CR2-indep": vs.cr2_test(d, cl, null, "indep")["p"] < .05,
         "CR1": vs.cr1_test(d, cl, null)["p"] < .05,
         "WCB-R": vs.wild_cluster_p(d, cl, null, B=B, rng=rng, weights="rademacher")["p"] < .05,
         "WCB-W": vs.wild_cluster_p(d, cl, null, B=B, rng=rng, weights="webb")["p"] < .05,
         "Means-t": vs.means_t_test(d, cl, null_m)["p"] < .05}
    if twelve:
        r["Signflip"] = vs.signflip_exact_p(d, null) < .05
        r["Paired-t"] = vs.paired_t_test(d, null)["p"] < .05
    return {k: bool(v) for k, v in r.items()}

# ---------- simulated cells ----------
def cell_name(c): return "%s_%s_%g" % c

def sim_task(args):
    c, ch = args
    lay, arm, rho = c
    path = os.path.join(TD, "sim__%s__%d.json" % (cell_name(c), ch))
    if os.path.exists(path): return
    sizes = layouts()[lay]; tw = lay == "TWELVE"
    cl0 = np.repeat(np.arange(len(sizes)), sizes)
    def data(i): return gen(arm, rho, sizes, np.random.default_rng(seed(cell_name(c), "data", i)))[0]
    sd = np.std([data(i).mean() for i in range(REPS)], ddof=1)
    nu = vs.satterthwaite_df(cl0, rho)
    deltas = {"null": 0.0, "paper": 2.80 * sd,
              "t": (stats.t.ppf(.975, nu) + stats.t.ppf(.80, nu)) * sd}
    meths = M_12 if tw else M_BASE
    counts = {k: {m: 0 for m in meths} for k in deltas}
    t0 = time.time()
    for i in range(ch * CHUNK, min(REPS, (ch + 1) * CHUNK)):
        d = data(i)
        rng = np.random.default_rng(seed(cell_name(c), "boot", i))
        for k, dl in deltas.items():
            for m, v in run_tests(d + dl, cl0, 0.0, 0.0, rng, tw).items():
                counts[k][m] += v
    json.dump(dict(cell=c, chunk=ch, n=min(REPS, (ch + 1) * CHUNK) - ch * CHUNK, sd=sd, nu=nu,
                   deltas=deltas, counts=counts, secs=time.time() - t0), open(path, "w"))

# ---------- D4 ----------
D4 = os.path.join(HERE, "calib_d4_data.npz")

def prep_d4():
    if os.path.exists(D4): return
    import load_gift_pairs as lg
    arrs = {}
    for rule in ("MASE_Rank", "CRPS_Rank", "gmean"):
        names, diffs, cl = lg.top10_pair_diffs(rule)
        arrs[rule + "|cl"] = cl
        for (i, j), d in diffs.items(): arrs["%s|%d_%d" % (rule, i, j)] = d
    np.savez(D4, **arrs)

def d4_task(args):
    rule, i, j = args
    path = os.path.join(TD, "d4__%s__%d_%d.json" % (rule, i, j))
    if os.path.exists(path): return
    z = np.load(D4); d0 = z["%s|%d_%d" % (rule, i, j)]; cl0 = z[rule + "|cl"]
    G = len(np.unique(cl0)); groups = [d0[cl0 == g] for g in np.unique(cl0)]
    theta = d0.mean(); theta_m = np.mean([g.mean() for g in groups])
    rng = np.random.default_rng(seed("d4", rule, i, j))
    se = vs.pct_bootstrap(d0, cl0, B=20000, rng=rng, scheme="cluster")["se"]
    delta = 2.80 * se
    counts = {"size": {m: 0 for m in M_BASE}, "power": {m: 0 for m in M_BASE}}
    R = 1000; t0 = time.time()
    for _ in range(R):
        idx = rng.integers(0, G, G)
        d = np.concatenate([groups[g] for g in idx])
        cl = np.repeat(np.arange(G), [len(groups[g]) for g in idx])
        for k, dl in (("size", 0.0), ("power", delta)):
            for m, v in run_tests(d + dl, cl, theta, theta_m, rng).items(): counts[k][m] += v
    json.dump(dict(rule=rule, pair=[i, j], reps=R, se=se, delta=delta, counts=counts,
                   icc=vs.icc_hat(d0, cl0), secs=time.time() - t0), open(path, "w"))

# ---------- driver ----------
def all_tasks():
    t = [("sim", (c, ch)) for c in cells() for ch in range(REPS // CHUNK)]
    t += [("d4", (r, i, j)) for r in ("MASE_Rank", "gmean") for i in range(10) for j in range(i + 1, 10)]
    return t

def work(t): (sim_task if t[0] == "sim" else d4_task)(t[1])

def run(budget):
    prep_d4()
    todo = [t for t in all_tasks() if not os.path.exists(TD + "/" + (
        "sim__%s__%d.json" % (cell_name(t[1][0]), t[1][1]) if t[0] == "sim"
        else "d4__%s__%d_%d.json" % t[1]))]
    print(len(todo), "tasks left", flush=True)
    t0 = time.time(); done = 0
    with Pool(os.cpu_count()) as p:
        for _ in p.imap(work, todo, chunksize=1):
            done += 1
            if time.time() - t0 > budget:
                p.terminate(); break
    print("finished", done, "in %.0fs" % (time.time() - t0), flush=True)

# ---------- report ----------
def load(pat): return [json.load(open(f)) for f in sorted(glob.glob(TD + "/" + pat))]

def real_summary():
    z = np.load(D4); out = {}
    for rule in ("MASE_Rank", "CRPS_Rank", "gmean"):
        cl = z[rule + "|cl"]; ic, sk = [], []
        for i in range(10):
            for j in range(i + 1, 10):
                d = z["%s|%d_%d" % (rule, i, j)]
                ic.append(vs.icc_hat(d, cl))
                sk.append(float(stats.skew([d[cl == g].mean() for g in np.unique(cl)])))
        q = lambda x: [float(v) for v in np.percentile(x, [0, 25, 50, 75, 100])]
        out[rule] = dict(icc=q(ic), skew=q(sk))
    return out

def report():
    res = {"sim": {}, "d4": {}}
    for c in cells():
        fs = load("sim__%s__*.json" % cell_name(c))
        if len(fs) < REPS // CHUNK: continue
        n = sum(f["n"] for f in fs)
        r = dict(n=n, sd=fs[0]["sd"], nu=fs[0]["nu"], deltas=fs[0]["deltas"], rates={})
        for k in ("null", "paper", "t"):
            r["rates"][k] = {m: 100 * sum(f["counts"][k][m] for f in fs) / n for m in fs[0]["counts"][k]}
        res["sim"][cell_name(c)] = dict(r, cell=list(c))
    for rule in ("MASE_Rank", "gmean"):
        fs = load("d4__%s__*.json" % rule)
        if len(fs) < 45: continue
        o = {}
        for k in ("size", "power"):
            per = {m: [100 * f["counts"][k][m] / f["reps"] for f in fs] for m in M_BASE}
            o[k] = {m: dict(pooled=float(np.mean(v)), min=float(min(v)), median=float(np.median(v)), max=float(max(v)))
                    for m, v in per.items()}
        res["d4"][rule] = o
    res["real"] = real_summary()
    # decision
    dec = None
    if "MASE_Rank" in res["d4"]:
        s = res["d4"]["MASE_Rank"]["size"]; p = {m: s[m]["pooled"] for m in ("CR2-icc", "CR2-indep", "WCB-R")}
        ins = lambda v: 4 <= v <= 6
        if ins(p["CR2-icc"]): primary, why = "CR2-icc", "CR2-icc is inside [4%,6%] on D4 (MASE_Rank)"
        else:
            cand = {m: v for m, v in p.items() if m != "CR2-icc" and ins(v)}
            if cand:
                primary = sorted(cand, key=lambda m: (abs(cand[m] - 5), cand[m]))[0]
                why = "CR2-icc is outside [4%,6%] and %s is inside and closest to 5%%" % primary
            else: primary, why = "CR2-icc", "CR2-icc is outside [4%,6%] but neither CR2-indep nor WCB-R is inside, so no switch"
        dec = dict(primary=primary, why=why, d4_pooled_size_MASE_Rank=p,
                   d4_pooled_size_gmean={m: res["d4"]["gmean"]["size"][m]["pooled"] for m in p} if "gmean" in res["d4"] else None)
    res["decision"] = dec
    json.dump(res, open(HERE + "/calibration_results.json", "w"), indent=1)
    md(res)

def md(res):
    L = ["# Calibration and power study (camera-ready, response to reviewers)", "",
         "Follows DESIGN.md (original + Revision 2). Code: calibration_sim.py; raw numbers: calibration_results.json.",
         "Entries are rejection rates in percent at the nominal 5% level. Simulated cells use 4,000 replicates, so the "
         "Monte-Carlo standard error is about 0.35 points at 5% (about 0.65 points at 80%). D4 rows pool 45 pairs x 1,000 replicates.",
         "Null value in simulation is 0 for every method, including Means-t (the equal-weight mean of cluster means is also 0 by construction).", ""]
    def tab(kind, title, lay):
        meths = M_12 if lay == "TWELVE" else M_BASE
        L.append("### %s, %s" % (title, lay)); L.append("")
        L.append("| arm | rho | " + " | ".join(meths) + " |"); L.append("|" + "---|" * (len(meths) + 2))
        for c in cells():
            if c[0] != lay or cell_name(c) not in res["sim"]: continue
            r = res["sim"][cell_name(c)]["rates"][kind]
            L.append("| %s | %g | " % (c[1], c[2]) + " | ".join("%.1f" % r[m] for m in meths) + " |")
        L.append("")
    L.append("## 1. Size (true difference zero)"); L.append("")
    for lay in ("GIFT", "FEV", "TWELVE"): tab("null", "Size", lay)
    L.append("## 2. Power"); L.append("")
    L.append("delta_paper = 2.80 x SD of the estimate across the null replicates; delta_t = (t_0.975 + t_0.80) at nu degrees of freedom x the same SD (nu = Satterthwaite df at the cell's rho). Target about 80%.")
    L.append("")
    for lay in ("GIFT", "FEV", "TWELVE"): tab("paper", "Power at delta_paper", lay)
    for lay in ("GIFT", "FEV", "TWELVE"): tab("t", "Power at delta_t", lay)
    L.append("Shifts used (in units of the total SD of one difference) are in calibration_results.json (deltas, sd, nu per cell).")
    L.append("")
    L.append("## 3. D4: resampling the real GIFT-Eval pairs (28 datasets drawn with replacement, 1,000 replicates per pair, 45 pairs)")
    L.append("")
    L.append("D4 cannot produce a dataset more extreme than the 28 observed, so it leans optimistic. Power shift = 2.80 x the cluster-bootstrap SE of each original pair.")
    L.append("")
    for rule, o in res["d4"].items():
        for k in ("size", "power"):
            L.append("### D4 %s, %s" % (k, rule)); L.append("")
            L.append("| method | pooled % | min over pairs | median | max |"); L.append("|---|---|---|---|---|")
            for m in M_BASE:
                x = o[k][m]; L.append("| %s | %.1f | %.1f | %.1f | %.1f |" % (m, x["pooled"], x["min"], x["median"], x["max"]))
            L.append("")
    L.append("## 4. Real data: intraclass correlation and skewness of the 28 dataset means (45 top-10 pairs per rule)")
    L.append(""); L.append("| rule | quantity | min | Q1 | median | Q3 | max |"); L.append("|---|---|---|---|---|---|---|")
    for rule, x in res["real"].items():
        for q, nm in (("icc", "icc_hat"), ("skew", "skewness")):
            L.append("| %s | %s | " % (rule, nm) + " | ".join("%.2f" % v for v in x[q]) + " |")
    L.append("")
    L.append("## 5. Pre-registered decision (applied mechanically)"); L.append("")
    d = res["decision"]
    if d:
        L.append("Rule: primary = CR2-icc unless its pooled D4 rejection rate (MASE_Rank, deciding rule) is outside [4%, 6%] while CR2-indep or WCB-R is inside; then the one closest to 5% (ties to the lower rate).")
        L.append("")
        L.append("D4 pooled size, MASE_Rank: " + ", ".join("%s %.2f%%" % kv for kv in d["d4_pooled_size_MASE_Rank"].items()))
        if d["d4_pooled_size_gmean"]: L.append("")
        if d["d4_pooled_size_gmean"]: L.append("D4 pooled size, gmean (reported, not deciding): " + ", ".join("%s %.2f%%" % kv for kv in d["d4_pooled_size_gmean"].items()))
        L += ["", "**Primary method: %s.** Reason: %s." % (d["primary"], d["why"])]
    open(HERE + "/CALIBRATION.md", "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    if sys.argv[1] == "run": run(float(sys.argv[2]) if len(sys.argv) > 2 else 420)
    else: report()
