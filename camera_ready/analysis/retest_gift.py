"""Re-test GIFT-Eval headline counts under small-sample-corrected methods (DESIGN.md Exp. A2, rev. items 5, 7).
Imports vernier_stats / load_gift_pairs / run_audit / critic_checks unchanged. Foreground, CPU only."""
import sys, json, time, itertools
import numpy as np
from scipy import stats
HERE = "fmts/camera_ready"
sys.path.insert(0, HERE)
import vernier_stats as v
import load_gift_pairs as lg
ra, cc = lg.ra, lg.cc

T0 = time.time()
SEED = 20261111
B_PCT, B_WCB = 100_000, 99_999
RULES = ["MASE_Rank", "CRPS_Rank", "gmean"]
PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
ADJ = [k for k, (i, j) in enumerate(PAIRS) if j == i + 1]
METHODS = ["P-both", "P-clu", "CR2-icc", "CR2-indep", "CR1", "WCB-Rad", "Means-t"]
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------- data ----------------
_pc = {}
def pipe(crps):
    if crps not in _pc:
        rows, w = (cc.load_all_results_crps if crps else ra.load_all_results)(lg.ROOT)
        keys, md, agg = ra.build_pipeline(rows, w)
        _, by_field, _ = ra.load_config_info(lg.ROOT)
        disp, _, _ = ra.merge_display(agg, by_field)
        _pc[crps] = (keys, md, disp)
    return _pc[crps]

def top_entries(rule_own, n=40):
    """rule_own in MASE_Rank, CRPS_Rank, gmean_own -> (names, X (n x 97), cl, keys). Own displayed order, leak pushed down."""
    crps = rule_own == "CRPS_Rank"
    keys, md, disp = pipe(crps)
    metric = "gmean" if rule_own == "gmean_own" else "MASE_Rank"
    # complete entries only (all 97 configs), as resolution_budget.py's `elig`
    names = [r["model"] for r in ra.sort_leak_then([r for r in disp if r["num_configs"] == len(keys)], metric)[:n]]
    f, lf = ("norm", np.log) if rule_own == "gmean_own" else ("rank", lambda x: x)
    X = np.array([lf(np.array([md[m][k][f] for k in keys], float)) for m in names])
    _, cl = np.unique([k[0] for k in keys], return_inverse=True)
    return names, X, cl, keys

def pair_d(X, i, j): return X[j] - X[i]

# ---------------- bootstrap machinery (paper's two schemes) ----------------
def boot_means(Dm, cl, B, seed):
    """Dm: (P, N) pair differences. Returns dict scheme -> (P, B) bootstrap means, shared resamples across pairs."""
    P, N = Dm.shape
    G = cl.max() + 1
    n = np.bincount(cl).astype(float)
    S = np.stack([np.bincount(cl, weights=d) for d in Dm])          # (P, G)
    out = {}
    rng = np.random.default_rng(seed)
    res = np.empty((P, B))
    for s in range(0, B, 5000):
        b = min(5000, B - s)
        idx = rng.integers(0, N, size=(b, N))
        res[:, s:s + b] = np.stack([d[idx].mean(1) for d in Dm])
    out["iid"] = res
    rng = np.random.default_rng(seed + 1)
    res = np.empty((P, B))
    for s in range(0, B, 20000):
        b = min(20000, B - s)
        W = np.stack([np.bincount(r, minlength=G) for r in rng.integers(0, G, size=(b, G))]).astype(float)
        res[:, s:s + b] = (W @ S.T).T / (W @ n)[None, :]
    out["cluster"] = res
    return out

def pct_ci(x):   # paper's percentile rule (ra.percentile_ci)
    s = np.sort(x); m = len(s)
    return s[int(round(0.025 * (m - 1)))], s[min(m - 1, int(round(0.975 * (m - 1))))]
def pct_p(x):    # cc.bootstrap_pvalue
    return min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean()))

def holm_count(p, idx=None):
    p = np.asarray(p); r = v.holm(p); return r

# ---------------- per-pair analysis ----------------
def analyse_pairs(Dm, cl, seed, B=B_PCT, bw=B_WCB, tag=""):
    bm = boot_means(Dm, cl, B, seed)
    out = []
    for k, d in enumerate(Dm):
        rec = dict(theta=float(d.mean()), icc_hat=v.icc_hat(d, cl))
        dsm = np.bincount(cl, weights=d) / np.bincount(cl)
        rec["skew_dataset_means"] = float(stats.skew(dsm))
        for sc in ("iid", "cluster"):
            lo, hi = pct_ci(bm[sc][k]); rec[f"P-{sc}"] = dict(ci95=[float(lo), float(hi)], p=pct_p(bm[sc][k]), excl0=bool(lo > 0 or hi < 0), se=float(bm[sc][k].std(ddof=1)))
        rec["P-both"] = dict(p=max(rec["P-iid"]["p"], rec["P-cluster"]["p"]), excl0=rec["P-iid"]["excl0"] and rec["P-cluster"]["excl0"])
        rec["P-clu"] = dict(p=rec["P-cluster"]["p"], excl0=rec["P-cluster"]["excl0"], ci95=rec["P-cluster"]["ci95"])
        for nm, r in (("CR2-icc", v.cr2_test(d, cl, df="icc")), ("CR2-indep", v.cr2_test(d, cl, df="indep")), ("CR1", v.cr1_test(d, cl))):
            rec[nm] = dict(p=float(r["p"]), ci95=[float(x) for x in r["ci95"]], se=float(r["se"]), df=float(r["df"]), excl0=bool(r["p"] < .05))
        for nm, wts in (("WCB-Rad", "rademacher"), ("WCB-Webb", "webb")):
            r = v.wild_cluster_p(d, cl, B=bw, rng=np.random.default_rng(seed + 100 + k * 2 + (wts == "webb")), weights=wts)
            rec[nm] = dict(p=float(r["p"]), t=float(r["t"]), excl0=bool(r["p"] < .05))
        r = v.means_t_test(d, cl)
        rec["Means-t"] = dict(p=float(r["p"]), ci95=[float(x) for x in r["ci95"]], theta_means=float(r["theta_means"]), excl0=bool(r["p"] < .05),
                              note="weights datasets equally; tests a slightly different quantity than the config-weighted mean")
        out.append(rec)
    return out

def counts(recs, methods=METHODS):
    res = {}
    for m in methods:
        p = [r[m]["p"] for r in recs]; u = [r[m]["excl0"] for r in recs]
        h = v.holm(p)
        res[m] = dict(adj_unc=int(sum(u[k] for k in ADJ)), all_unc=int(sum(u)), adj_holm=int(sum(h[k] for k in ADJ)), all_holm=int(sum(h)),
                      adj_unc_positions=[f"{PAIRS[k][0]+1}-{PAIRS[k][1]+1}" for k in ADJ if u[k]],
                      adj_holm_positions=[f"{PAIRS[k][0]+1}-{PAIRS[k][1]+1}" for k in ADJ if h[k]])
    return res

# ---------------- paper-convention resolution (resolution_budget.py) ----------------
def paper_boot_se_machinery(keys, seed=20260907, Bp=4000):
    """Replicates resolution_budget.py's resamples exactly: configs sorted by 'ds/freq/term' string, same RNG draw order."""
    strs = ["/".join(k) for k in keys]; perm = np.argsort(strs, kind="stable"); C = len(keys)
    bases = sorted({k[0] for k in keys}); bidx = np.array([bases.index(strs[p].split("/")[0]) for p in perm])
    rng = np.random.default_rng(seed); iid = rng.integers(0, C, size=(Bp, C)); W = np.zeros((Bp, C))
    for b in range(Bp): W[b] = np.bincount(rng.integers(0, len(bases), size=len(bases)), minlength=len(bases))[bidx]
    boot = lambda d: (d[perm][iid].mean(1), (W * d[perm]).sum(1) / W.sum(1))
    return boot, len(bases)

def resolution(X, boot, G0, cl, Tr=range(10), Gtarget=None):
    rows = []
    for a, b in zip(list(Tr), list(Tr)[1:]):
        d = X[b] - X[a]; gap = abs(d.mean()); se_b = boot(d)[1].std()
        r_icc = v.cr2_test(d, cl, df="icc"); r_ind = v.cr2_test(d, cl, df="indep")
        rows.append(dict(gap=float(gap), se_boot=float(se_b), se_cr2=float(r_icc["se"]), df_icc=float(r_icc["df"]), df_indep=float(r_ind["df"]),
                         mde_boot=float(2.8 * se_b), mde_icc=float(v.mde80(r_icc["se"], r_icc["df"])), mde_indep=float(v.mde80(r_ind["se"], r_ind["df"])),
                         Gstar_boot=float(G0 * (2.8 * se_b / gap) ** 2) if gap > 0 else np.inf, Gstar_cr2=float(G0 * (2.8 * r_icc["se"] / gap) ** 2) if gap > 0 else np.inf,
                         ratio_se=float(r_icc["se"] / se_b)))
    med = lambda k: float(np.median([r[k] for r in rows]))
    gap = med("gap")
    summ = dict(paper_res=2.8 * float(np.median([r["se_boot"] for r in rows])), corr_res_icc=med("mde_icc"), corr_res_indep=med("mde_indep"),
                corr_res_2p8_cr2=2.8 * float(np.median([r["se_cr2"] for r in rows])), median_gap=gap,
                Gstar_boot=med("Gstar_boot"), Gstar_cr2=med("Gstar_cr2"), median_ratio_se=med("ratio_se"))
    summ["ratio_paper"] = summ["paper_res"] / gap; summ["ratio_icc"] = summ["corr_res_icc"] / gap; summ["ratio_indep"] = summ["corr_res_indep"] / gap
    summ["Gstar_boot_x"] = summ["Gstar_boot"] / G0; summ["Gstar_cr2_x"] = summ["Gstar_cr2"] / G0
    return summ, rows

def main():
    R = dict(meta=dict(seed=SEED, B_pct=B_PCT, B_wcb=B_WCB))
    # ---- data
    names = {}; Dm = {}; cls = {}
    for ri, rule in enumerate(RULES):
        top, diffs, cl = lg.top10_pair_diffs(rule)
        names[rule] = top; cls[rule] = cl; Dm[rule] = np.stack([diffs[p] for p in PAIRS])
    cl = cls["MASE_Rank"]; assert len(np.unique(cl)) == 28 and len(cl) == 97
    # consistency: own-order builder reproduces load_gift_pairs for MASE and CRPS
    own = {}
    for r in ("MASE_Rank", "CRPS_Rank", "gmean_own"):
        own[r] = top_entries(r)
    for r in ("MASE_Rank", "CRPS_Rank"):
        assert own[r][0][:10] == names[r]
        assert np.allclose(own[r][1][3] - own[r][1][1], lg.top10_pair_diffs(r)[1][(1, 3)])
    R["gmean_own_order_top10"] = own["gmean_own"][0][:10]
    R["gmean_pairs_order_top10"] = names["gmean"]
    # ---- step 0 + 1
    recs = {}
    for ri, rule in enumerate(RULES):
        recs[rule] = analyse_pairs(Dm[rule], cl, SEED + 1000 * ri); log("pairs done", rule)
        for k, r in enumerate(recs[rule]): r["pair"] = f"{PAIRS[k][0]+1}-{PAIRS[k][1]+1}"; r["names"] = [names[rule][PAIRS[k][0]], names[rule][PAIRS[k][1]]]
    R["pairs"] = recs
    paper_exp = dict(MASE_Rank=(0, 12, 2), CRPS_Rank=(0, 16, 0), gmean=(2, 23, 11))
    reg = {}; ok = True
    for rule in RULES:
        c = counts(recs[rule], ["P-both"])["P-both"]
        # Holm for the paper's rule uses p = max over the two schemes
        got = (c["adj_unc"], c["all_unc"], c["all_holm"])
        # Holm mismatch diagnosis: the pair closest to its Holm threshold, and whether the gap is inside Monte-Carlo noise
        p = np.array([r["P-both"]["p"] for r in recs[rule]]); o = np.argsort(p)
        near = min(range(45), key=lambda k: abs(p[o[k]] - 0.05 / (45 - k)))
        se_p = 2 * np.sqrt((p[o[near]] / 2) * (1 - p[o[near]] / 2) / B_PCT)
        reg[rule] = dict(expected=dict(zip(("adj_unc", "all_unc", "all_holm45"), paper_exp[rule])), got=dict(zip(("adj_unc", "all_unc", "all_holm45"), got)),
                         match=got == paper_exp[rule],
                         nearest_to_holm_threshold=dict(pair=recs[rule][o[near]]["pair"], p=float(p[o[near]]), thr=0.05 / (45 - near), mc_se_of_p=float(se_p)),
                         within_mc_noise=bool(abs(p[o[near]] - 0.05 / (45 - near)) < 3 * se_p))
        ok &= reg[rule]["match"] or (got[:2] == paper_exp[rule][:2] and abs(got[2] - paper_exp[rule][2]) <= 1 and reg[rule]["within_mc_noise"])
    R["regression"] = dict(per_rule=reg, all_match=ok)
    log("regression", {k: (v_["got"], v_["match"]) for k, v_ in reg.items()})
    if not ok:
        json.dump(R, open(f"{HERE}/retest_gift.json", "w"), indent=1, default=float)
        log("REGRESSION FAILED; stopping per instructions (partial JSON written)"); return
    # ---- step 2
    R["counts"] = {rule: counts(recs[rule]) for rule in RULES}
    p910 = recs["MASE_Rank"][PAIRS.index((8, 9))]
    R["pair_9_10_MASE"] = {m: dict(p=p910[m]["p"], excl0=p910[m]["excl0"]) for m in METHODS + ["WCB-Webb"]}
    R["pair_9_10_MASE"]["theta"] = p910["theta"]; R["pair_9_10_MASE"]["P-iid_p"] = p910["P-iid"]["p"]; R["pair_9_10_MASE"]["P-cluster_p"] = p910["P-cluster"]["p"]
    R["pair_9_10_MASE"]["P-iid_ci"] = p910["P-iid"]["ci95"]; R["pair_9_10_MASE"]["P-cluster_ci"] = p910["P-cluster"]["ci95"]
    R["counts_webb_variant"] = {rule: {k: v_ for k, v_ in zip(("adj_unc", "all_unc", "adj_holm", "all_holm"),
        (sum(recs[rule][k]["WCB-Webb"]["excl0"] for k in ADJ), sum(r["WCB-Webb"]["excl0"] for r in recs[rule]),
         int(sum(v.holm([r["WCB-Webb"]["p"] for r in recs[rule]])[k] for k in ADJ)), int(sum(v.holm([r["WCB-Webb"]["p"] for r in recs[rule]]))))) } for rule in RULES}
    log("counts done")
    # ---- step 3 (paper-convention resolution; adjacency in own displayed order)
    boot, G0 = paper_boot_se_machinery(own["MASE_Rank"][3])
    R["resolution"] = {}
    for rule in ("MASE_Rank", "CRPS_Rank", "gmean_own"):
        names_o, X, clo, _ = own[rule]
        summ, rows = resolution(X, boot, G0, clo)
        R["resolution"][rule] = dict(summary=summ, per_pair=rows)
    log("resolution", {k: round(v_["summary"]["paper_res"], 4) for k, v_ in R["resolution"].items()})
    # ---- step 4 tiers
    R["tiers"] = {}
    for rule in ("MASE_Rank", "CRPS_Rank", "gmean_own"):
        names_o, X, clo, _ = own[rule]
        ds = [X[i] - X[0] for i in range(1, 40)]
        it_cl = [boot(d) for d in ds]
        sep_paper = [pct_ci(a)[0] > 0 and pct_ci(b)[0] > 0 for a, b in it_cl]      # paper exact: B=4000, lower bounds > 0
        Dl = np.stack(ds)
        recs_t = analyse_pairs(Dl, clo, SEED + 5000 + len(R["tiers"]), B=B_PCT)
        tier = {}
        tier["P-both (paper B=4000)"] = [names_o[0]] + [names_o[i + 1] for i, s in enumerate(sep_paper) if not s]
        for m in ("P-both", "CR2-icc", "WCB-Rad", "Means-t"):
            tier[m if m != "P-both" else "P-both (B=100k)"] = [names_o[0]] + [names_o[i + 1] for i, r in enumerate(recs_t) if not r[m]["excl0"]]
        R["tiers"][rule] = dict(tier=tier, size={k: len(x) for k, x in tier.items()},
                                note="entries not separated from the displayed leader by uncorrected tests; leader-vs-entry p-values saved in `p`",
                                p={m: [r[m]["p"] for r in recs_t] for m in ("P-both", "CR2-icc", "WCB-Rad", "Means-t")}, entries=names_o[1:40])
    log("tiers", {k: v_["size"] for k, v_ in R["tiers"].items()})
    # ---- step 5 coarse sources
    ds_names = sorted({k[0] for k in own["MASE_Rank"][3]})
    grp = lambda d: "m4" if d.startswith("m4_") else "ett" if d in ("ett1", "ett2") else "bizitobs" if d.startswith("bizitobs_") else "bitbrains" if d.startswith("bitbrains_") else d
    srcs = sorted({grp(d) for d in ds_names}); assert len(srcs) == 19, srcs
    keys = own["MASE_Rank"][3]; cl19 = np.array([srcs.index(grp(k[0])) for k in keys])
    R["coarse"] = dict(dataset_names=ds_names, sources=srcs, n_sources=len(srcs))
    rc = analyse_pairs(Dm["MASE_Rank"], cl19, SEED + 7000), analyse_pairs(Dm["gmean"], cl19, SEED + 7100)
    R["coarse"]["counts"] = {rule: counts(r, ["P-clu", "CR2-icc", "WCB-Rad"]) for rule, r in zip(("MASE_Rank", "gmean"), rc)}
    R["coarse"]["pair_9_10_MASE"] = {m: rc[0][PAIRS.index((8, 9))][m]["p"] for m in ("P-clu", "CR2-icc", "WCB-Rad")}
    R["coarse"]["resolution"] = {}
    for rule in ("MASE_Rank", "gmean_own"):
        X = own[rule][1]; rows = []
        for a in range(9):
            d = X[a + 1] - X[a]; r = v.cr2_test(d, cl19, df="icc"); bs = boot_means(d[None, :], cl19, 20000, SEED + 7200 + a)["cluster"][0]
            gap = abs(d.mean()); rows.append(dict(gap=gap, se_cr2=r["se"], df=r["df"], mde=v.mde80(r["se"], r["df"]), se_boot=bs.std(ddof=1), Gstar_cr2=19 * (2.8 * r["se"] / gap) ** 2, Gstar_boot=19 * (2.8 * bs.std(ddof=1) / gap) ** 2))
        m_ = lambda k: float(np.median([r[k] for r in rows]))
        R["coarse"]["resolution"][rule] = dict(corr_res_icc=m_("mde"), paper_style_res=2.8 * m_("se_boot"), median_gap=m_("gap"), ratio=m_("mde") / m_("gap"), Gstar_cr2=m_("Gstar_cr2"), Gstar_cr2_x=m_("Gstar_cr2") / 19,
                                               Gstar_boot=m_("Gstar_boot"), Gstar_boot_x=m_("Gstar_boot") / 19, per_pair=[{k: float(x) for k, x in r.items()} for r in rows])
    log("coarse done")
    # ---- step 6 leave-one-dataset-out
    R["loo"] = {}
    for rule in ("MASE_Rank", "gmean_own"):
        X = own[rule][1]; full = R["resolution"][rule]["summary"]["corr_res_icc"]; res = []
        fullcnt = None
        for g in range(28):
            keep = cl != g; cl2 = np.unique(cl[keep], return_inverse=True)[1]; mdes = []; cnt = 0; pv = []
            for a in range(9):
                d = (X[a + 1] - X[a])[keep]; r = v.cr2_test(d, cl2, df="icc"); mdes.append(v.mde80(r["se"], r["df"])); cnt += r["p"] < .05
            res.append(dict(dropped=ds_names[g], resolution=float(np.median(mdes)), adj_count=int(cnt)))
        full_cnt = sum(1 for a in range(9) if v.cr2_test(X[a + 1] - X[a], cl, df="icc")["p"] < .05)
        rr = np.array([x["resolution"] for x in res]); big = max(res, key=lambda x: abs(x["resolution"] - full))
        R["loo"][rule] = dict(full_resolution=full, full_adj_count_cr2icc=int(full_cnt), min=float(rr.min()), max=float(rr.max()), most_moving=big,
                              any_count_change=bool(any(x["adj_count"] != full_cnt for x in res)), per_dataset=res)
    log("loo done")
    # ---- step 7 stability under subsets
    X = own["MASE_Rank"][1]; rng = np.random.default_rng(SEED + 9000); curve = {}
    for Gp in (7, 14, 21, 28):
        subs = [rng.choice(28, Gp, replace=False) for _ in range(200)] if Gp < 28 else [np.arange(28)]
        per = []
        for a in range(9):
            d = X[a + 1] - X[a]; ses = []
            for s in subs:
                m = np.isin(cl, s); ses.append(v.se_cr2(d[m], cl[m]) * np.sqrt(Gp))
            per.append(float(np.median(ses)))
        curve[Gp] = per
    base = np.array(curve[28])
    R["subsets"] = dict(note="stability check, not a test of extrapolation; subsets of a fixed 28 datasets",
                        per_Gp={str(g): dict(per_pair_rel=[float(x) for x in np.array(c) / base], median_rel=float(np.median(np.array(c) / base))) for g, c in curve.items()})
    log("subsets done")
    R["runtime_s"] = time.time() - T0
    json.dump(R, open(f"{HERE}/retest_gift.json", "w"), indent=1, default=float)
    log("done")

if __name__ == "__main__":
    main()
