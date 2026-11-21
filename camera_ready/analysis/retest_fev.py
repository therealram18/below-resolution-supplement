"""Experiment A3, fev-bench part (see DESIGN.md). Run: python retest_fev.py"""
import csv, json, os, sys, time, itertools
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vernier_stats as vs
t0 = time.time()
PC = os.path.join(HERE, "..", "poolcheck")
rows = list(csv.reader(open(os.path.join(PC, "pivot_SQL.csv"))))
models = rows[0][1:]; tasks = [r[0] for r in rows[1:]]
E = np.array([[float(x) for x in r[1:]] for r in rows[1:]]); mi = {m: i for i, m in enumerate(models)}
Rel = np.clip(E / E[:, [mi["Seasonal Naive"]]], 0.01, 100)
Wn = ((Rel[:, :, None] < Rel[:, None, :]) + 0.5 * (Rel[:, :, None] == Rel[:, None, :])).astype(float)
n = len(models); Wn[:, np.arange(n), np.arange(n)] = np.nan
WT = np.nanmean(Wn, axis=2) * 100
wr = WT.mean(0)
# reproduce published leaderboard (same check as fev_resolution.py)
off = {r["model_name"]: (float(r["win_rate"]), float(r["skill_score"])) for r in csv.DictReader(open(os.path.join(PC, "leaderboard_SQL.csv")))}
sk = (1 - np.exp(np.log(Rel).mean(0))) * 100
lb_err = (max(abs(wr[mi[m]] - v[0]) for m, v in off.items() if m in mi), max(abs(sk[mi[m]] - v[1]) for m, v in off.items() if m in mi))
T = list(np.argsort(-wr))[:10]
fam = [t.rsplit("_", 1)[0] for t in tasks]; fams = sorted(set(fam)); fidx = np.array([fams.index(f) for f in fam])
N, G = len(tasks), len(fams)
METRICS = {"win_rate": WT, "skill": -np.log(Rel)}
adj = [(T[i], T[i + 1]) for i in range(9)]
allp = [(T[i], T[j]) for i in range(10) for j in range(i + 1, 10)]
pname = lambda a, b: f"{T.index(a)+1}-{T.index(b)+1}"

# ---------- Step 0: regression check, exact stream of fev_resolution.py ----------
B = 4000; rng = np.random.default_rng(20260907)
iid = rng.integers(0, N, size=(B, N)); W = np.zeros((B, N))
for b in range(B): W[b] = np.bincount(rng.integers(0, G, size=G), minlength=G)[fidx]
reg = {}
for mname, X in METRICS.items():
    surv = 0; ses = []; gaps = []; cis = []
    for a, b in adj:
        d = X[:, a] - X[:, b]; it = d[iid].mean(1); cl = (W * d).sum(1) / W.sum(1)
        lo_i = np.percentile(it, 2.5); lo_c, hi_c = np.percentile(cl, [2.5, 97.5])
        surv += int(lo_i > 0 and lo_c > 0); ses.append(cl.std()); gaps.append(abs(d.mean())); cis.append([lo_c, hi_c])
    need = [G * (2.8 * s / g) ** 2 for s, g in zip(ses, gaps) if g > 0]
    reg[mname] = dict(surv=surv, mde80=2.8 * float(np.median(ses)), median_gap=float(np.median(gaps)),
                      median_se=float(np.median(ses)), G_star_paper=float(np.median(need)), pair12_ci=cis[0], ses=ses)
reg_ok = dict(surv_1of9=reg["win_rate"]["surv"] == 1, mde_9_52=round(reg["win_rate"]["mde80"], 2) == 9.52,
              gap_2_17=round(reg["win_rate"]["median_gap"], 2) == 2.17,
              ci12=[round(x, 2) for x in reg["win_rate"]["pair12_ci"]] == [0.65, 8.57] or [round(x, 2) for x in reg["win_rate"]["pair12_ci"]])

# ---------- Step 1: all methods ----------
BB = 100000; rng2 = np.random.default_rng(20260907)
Sfam = {}  # per metric: cluster sums matrix for paper-rule bootstrap p/CI at BB
cnt = np.bincount(fidx).astype(float)
def paper_boot(X, pairs):
    D = np.stack([X[:, a] - X[:, b] for a, b in pairs], 1)  # N x P
    Sg = np.zeros((G, D.shape[1]))
    np.add.at(Sg, fidx, D)
    iidm = np.empty((BB, D.shape[1])); clm = np.empty((BB, D.shape[1]))
    for s in range(0, BB, 2000):
        k = min(2000, BB - s)
        iidm[s:s + k] = D[rng2.integers(0, N, size=(k, N))].mean(1)
        gi = rng2.integers(0, G, size=(k, G))
        clm[s:s + k] = Sg[gi].sum(1) / cnt[gi].sum(1)[:, None]
    return iidm, clm
def bp(x):
    return np.minimum(1.0, 2 * np.minimum((x <= 0).mean(0), (x >= 0).mean(0)))
out = {"meta": dict(B_paper_rule=BB, B_wild=99999, seed_regression=20260907, seed_main=20260907, n_tasks=N, n_families=G,
                    top10=[models[i] for i in T], leaderboard_max_abs_err_win_rate=lb_err[0], leaderboard_max_abs_err_skill=lb_err[1]),
       "regression": {k: {kk: vv for kk, vv in v.items() if kk != "ses"} for k, v in reg.items()}, "regression_ok": reg_ok, "metrics": {}}
wrng = np.random.default_rng(20261111)
for mname, X in METRICS.items():
    iidm, clm = paper_boot(X, allp)
    p_iid, p_clu = bp(iidm), bp(clm)
    lo_iid = np.percentile(iidm, 2.5, axis=0); lo_clu = np.percentile(clm, 2.5, axis=0); hi_clu = np.percentile(clm, 97.5, axis=0)
    hi_iid = np.percentile(iidm, 97.5, axis=0)
    res = []
    for k, (a, b) in enumerate(allp):
        d = X[:, a] - X[:, b]
        r = dict(pair=pname(a, b), adjacent=(a, b) in adj, gap=float(d.mean()))
        paper_sep = bool((lo_iid[k] > 0 or hi_iid[k] < 0) and (lo_clu[k] > 0 or hi_clu[k] < 0))
        r["paper"] = dict(p=float(max(p_iid[k], p_clu[k])), ci_clu=[float(lo_clu[k]), float(hi_clu[k])], ci_iid=[float(lo_iid[k]), float(hi_iid[k])], sep=paper_sep)
        r["cr2_icc"] = {kk: (list(v) if isinstance(v, tuple) else float(v)) for kk, v in vs.cr2_test(d, fidx, df="icc").items()}
        r["cr2_indep"] = {kk: (list(v) if isinstance(v, tuple) else float(v)) for kk, v in vs.cr2_test(d, fidx, df="indep").items()}
        r["cr1"] = {kk: (list(v) if isinstance(v, tuple) else float(v)) for kk, v in vs.cr1_test(d, fidx).items()}
        w = vs.wild_cluster_p(d, fidx, B=99999, rng=wrng); r["wcb"] = dict(p=w["p"], t=w["t"])
        mt = vs.means_t_test(d, fidx); r["means_t"] = {kk: (list(v) if isinstance(v, tuple) else float(v)) for kk, v in mt.items()}
        r["icc_hat"] = vs.icc_hat(d, fidx)
        fm = np.bincount(fidx, weights=d) / cnt
        r["skew_family_means"] = float(stats.skew(fm, bias=False))
        r["se_boot_cluster"] = float(clm[:, k].std())
        r["se_cr2"] = float(vs.se_cr2(d, fidx))
        res.append(r)
    methods = ["paper", "cr2_icc", "cr2_indep", "cr1", "wcb", "means_t"]
    counts = {}
    for m in methods:
        if m == "paper": sep = np.array([r["paper"]["sep"] for r in res]); ps = np.array([r["paper"]["p"] for r in res])
        else: ps = np.array([r[m]["p"] for r in res]); sep = ps < 0.05
        hol = vs.holm(ps)
        if m == "paper": hol = hol  # Holm on max-of-two bootstrap p-values (same recipe as audit/critic_checks.py)
        for r, s, h in zip(res, sep, hol): r.setdefault("verdict", {})[m] = dict(unc=bool(s), holm=bool(h))
        adjm = np.array([r["adjacent"] for r in res])
        counts[m] = dict(adj_unc=int((sep & adjm).sum()), adj_holm=int((hol & adjm).sum()), all45_unc=int(sep.sum()), all45_holm=int(hol.sum()))
    # resolution
    ad = [r for r in res if r["adjacent"]]
    gaps = np.array([abs(r["gap"]) for r in ad])
    mde_cr2 = float(np.median([vs.mde80(r["se_cr2"], r["cr2_icc"]["df"]) for r in ad]))
    reso = dict(paper_regression_stream=reg[mname]["mde80"], paper_boot_BB=2.8 * float(np.median([r["se_boot_cluster"] for r in ad])),
                corrected_cr2_icc=mde_cr2, median_adjacent_gap=float(np.median(gaps)),
                ratio_paper=reg[mname]["mde80"] / float(np.median(gaps)), ratio_corrected=mde_cr2 / float(np.median(gaps)),
                G_star_paper=reg[mname]["G_star_paper"],
                G_star_cr2=float(np.median([G * (2.8 * r["se_cr2"] / abs(r["gap"])) ** 2 for r in ad if r["gap"] != 0])),
                median_df_icc=float(np.median([r["cr2_icc"]["df"] for r in ad])))
    out["metrics"][mname] = dict(pairs=res, counts=counts, resolution=reso,
                                 icc_adjacent=[r["icc_hat"] for r in ad], skew_adjacent=[r["skew_family_means"] for r in ad])
out["runtime_s"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "retest_fev.json"), "w"), indent=1, default=float)

# ---------- markdown ----------
f = lambda x, k=2: f"{x:.{k}f}"
L = ["# fev-bench re-test under small-sample-corrected methods", "",
     "Data: 100 tasks, 39 task families (clusters), top-10 models by displayed win rate. Gaps are in win-rate points (or log-error-ratio units for skill). Wild bootstrap: Rademacher, 99,999 draws. 'Paper rule' = percentile bootstrap, tasks independent and families clustered, both intervals must exclude zero.", ""]
L += ["## 0. Regression check against the submitted paper", "",
      "| quantity | paper | reproduced |", "|---|---|---|",
      f"| adjacent gaps surviving both schemes (win rate) | 1 of 9 | {reg['win_rate']['surv']} of 9 |",
      f"| pair 1-2 clustered interval | [0.65, 8.57] | [{f(reg['win_rate']['pair12_ci'][0])}, {f(reg['win_rate']['pair12_ci'][1])}] |",
      f"| resolution (2.80 x median clustered bootstrap SE) | 9.52 | {f(reg['win_rate']['mde80'])} |",
      f"| median adjacent gap | 2.17 | {f(reg['win_rate']['median_gap'])} |",
      f"| leaderboard max abs error, win rate / skill | exact | {lb_err[0]:.2e} / {lb_err[1]:.2e} |", "",
      "This table shows whether the paper's win-rate numbers come back from the pinned files with the original random stream (B=4,000, seed 20260907).", ""]
for mname in METRICS:
    M = out["metrics"][mname]; nm = "win rate" if mname == "win_rate" else "skill score"
    L += [f"## {nm}", "", "### Pairs counted as separated (adjacent = the 9 neighbours; Holm is over all 45 top-10 pairs)", "",
          "| method | adjacent, uncorrected | adjacent, after Holm | all 45, uncorrected | all 45, after Holm |", "|---|---|---|---|---|"]
    lab = dict(paper="paper rule (bootstrap, both)", cr2_icc="CR2, df from estimated ICC", cr2_indep="CR2, df independent", cr1="CR1, t with 38 df", wcb="wild cluster bootstrap", means_t="family-means t-test")
    for m, c in M["counts"].items():
        L.append(f"| {lab[m]} | {c['adj_unc']} | {c['adj_holm']} | {c['all45_unc']} | {c['all45_holm']} |")
    L += ["", f"This table shows how many pairs each method separates, with the paper rule's Holm step using the larger of its two bootstrap p-values (B={BB:,}, seed 20260907).", "",
          "### Adjacent pairs", "", "| pair | gap | paper clustered 95% interval (B=100k) | CR2-icc p | CR2-icc 95% interval | CR2-indep p | CR1 p | wild p | family-means p |", "|---|---|---|---|---|---|---|---|---|"]
    for r in M["pairs"]:
        if r["adjacent"]:
            L.append(f"| {r['pair']} | {f(r['gap'],3)} | [{f(r['paper']['ci_clu'][0],3)}, {f(r['paper']['ci_clu'][1],3)}] | {r['cr2_icc']['p']:.4f} | [{f(r['cr2_icc']['ci95'][0],3)}, {f(r['cr2_icc']['ci95'][1],3)}] | {r['cr2_indep']['p']:.4f} | {r['cr1']['p']:.4f} | {r['wcb']['p']:.4f} | {r['means_t']['p']:.4f} |")
    L += ["", "This table shows each adjacent pair's gap with the paper's clustered interval and the corrected p-values and intervals.", ""]
    R = M["resolution"]
    L += ["### Resolution (smallest gap detected with 80% power, median over the 9 adjacent pairs)", "", "| quantity | value |", "|---|---|",
          f"| paper: 2.80 x median clustered bootstrap SE (B=4,000 stream) | {f(R['paper_regression_stream'],3)} |",
          f"| paper formula at B=100k | {f(R['paper_boot_BB'],3)} |",
          f"| corrected: mde80(CR2 SE, df from estimated ICC), median over pairs | {f(R['corrected_cr2_icc'],3)} (median df {f(R['median_df_icc'],1)}) |",
          f"| median adjacent gap | {f(R['median_adjacent_gap'],3)} |",
          f"| resolution / median gap: paper, corrected | {f(R['ratio_paper'],2)}, {f(R['ratio_corrected'],2)} |",
          f"| families needed G* = 39 x (2.80 SE / gap)^2, median: paper SE, CR2 SE | {f(R['G_star_paper'],0)}, {f(R['G_star_cr2'],0)} |", "",
          "This table shows the resolution and the number of families needed to resolve the median adjacent gap.", "",
          "### Estimated within-family correlation and skewness of family means (adjacent pairs)", "", "| pair | icc_hat | skewness |", "|---|---|---|"]
    for r in M["pairs"]:
        if r["adjacent"]: L.append(f"| {r['pair']} | {f(r['icc_hat'],3)} | {f(r['skew_family_means'],2)} |")
    L += ["", "This table shows the two quantities the correction depends on.", ""]
L += ["## What changes relative to the submitted paper", ""]
w = out["metrics"]["win_rate"]; c = w["counts"]; R = w["resolution"]
L += [f"- Adjacent gaps separated, win rate: 1 of 9 (paper rule) -> {c['cr2_icc']['adj_unc']} (CR2-icc), {c['cr2_indep']['adj_unc']} (CR2-indep), {c['cr1']['adj_unc']} (CR1), {c['wcb']['adj_unc']} (wild), {c['means_t']['adj_unc']} (family-means), uncorrected.",
      f"- Resolution, win rate: {f(R['paper_regression_stream'])} points -> {f(R['corrected_cr2_icc'])} points; ratio to median gap {f(R['ratio_paper'],1)} -> {f(R['ratio_corrected'],1)}.",
      f"- Families needed for the median adjacent gap: {f(R['G_star_paper'],0)} -> {f(R['G_star_cr2'],0)}."]
open(os.path.join(HERE, "RETEST_FEV.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(dict(reg_ok=reg_ok, reg={k: {kk: vv for kk, vv in v.items() if kk != 'ses'} for k, v in reg.items()}, lb=lb_err, runtime=out["runtime_s"]), indent=1, default=float))
