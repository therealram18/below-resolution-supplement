"""Do new datasets vary like the current 28? Compare between-source spread of model-pair log-MASE differences
on GIFT-Eval (28 base datasets) vs fev-bench (families not in GIFT-Eval). Writes boards/cross_board_spread.json + CROSS_BOARD_SPREAD.md."""
import itertools, json, os
import numpy as np, pandas as pd
from vernier_stats import se_cr2

HERE = os.path.dirname(os.path.abspath(__file__))
GIFT = os.path.join(HERE, "..", "snapshot_2026-09-07", "results")
FEV = os.path.join(HERE, "boards", "fev", "benchmarks", "fev_bench", "results")
B, SEED = 2000, 20261111

# (label, GIFT dir, fev csv, evidence)
MAP = [
 ("Chronos-2", "chronos-2", "chronos-2", "same link amazon/chronos-2"),
 ("Chronos-Bolt-Base", "chronos_bolt_base", "chronos-bolt", "same link amazon/chronos-bolt-base"),
 ("TimesFM-2.5", "TimesFM-2.5", "timesfm-2_5", "same link google/timesfm-2.5-200m-pytorch"),
 ("TimesFM-3", "TimesFM-3", "timesfm-3", "same link google/timesfm-3.0-pytorch"),
 ("Moirai-2.0", "Moirai2", "moirai-2_0", "same link Salesforce/moirai-2.0-R-small"),
 ("Toto-1.0", "Toto_Open_Base_1.0", "toto-1_0", "same link Datadog/Toto-Open-Base-1.0"),
 ("Toto-2.0-4m", "Toto-2.0-4m", "toto-2_0-4m", "same link"),
 ("Toto-2.0-22m", "Toto-2.0-22m", "toto-2_0-22m", "same link"),
 ("Toto-2.0-313m", "Toto-2.0-313m", "toto-2_0-313m", "same link"),
 ("Toto-2.0-1B", "Toto-2.0-1B", "toto-2_0-1b", "same link"),
 ("Toto-2.0-2.5B", "Toto-2.0-2.5B", "toto-2_0-2_5b", "same link"),
 ("Sundial-Base", "sundial_base_128m", "sundial-base", "same link thuml/sundial-base-128m"),
 ("FlowState", "Granite-FlowState-r1.1", "flowstate", "same link ibm-granite/granite-timeseries-flowstate-r1 (GIFT folder says r1.1, fev says r1; same URL, kept)"),
 ("Seasonal naive", "seasonal_naive", "seasonal_naive", "same definition, no weights"),
 ("Naive", "naive", "naive", "same definition, no weights"),
]
REJECT = [
 ("TiRex", "GIFT uses NX-AI/TiRex-1.1-gifteval, fev uses NX-AI/TiRex: different checkpoints"),
 ("TiRex-2", "GIFT uses TiRex-2-gifteval-zs/-pretrain, fev uses TiRex-2-fevbench: different checkpoints"),
 ("TabPFN-TS", "GIFT: tabpfn-time-series repo; fev: Prior-Labs/TabPFN-v2-reg; wrapper/version not confirmed; fev has 99/100 tasks"),
 ("AutoARIMA / AutoETS", "fev results incomplete (96 and 97 of 100 tasks)"),
 ("AutoTheta", "complete on both, but the implementation and settings per board are not documented; uncertain"),
 ("TFT, DeepAR, PatchTST", "trained per dataset with board-specific setups; not the same checkpoint"),
 ("t0-alpha (GIFT) vs t0-beta/gamma (fev)", "different versions"),
 ("TimesFM-2.0, Moirai small/base/large (GIFT only), Drift, CatBoost, LightGBM (fev only)", "not on both boards"),
]

def family(t):
    p = t.split("_")
    return "_".join(p[:-1]) if len(p) > 1 else t
# fev family -> evidence of overlap with a GIFT base dataset (True = shared or uncertain, treated as shared)
SHARED = {
 "ETT": "ETT = ett1/ett2", "LOOP_SEATTLE": "loop_seattle", "M_DENSE": "m_dense", "SZ_TAXI": "sz_taxi",
 "bizitobs_l2c": "bizitobs_l2c", "hierarchical_sales": "hierarchical_sales", "hospital": "hospital",
 "jena_weather": "jena_weather", "restaurant": "restaurant",
 "solar": "UNCERTAIN: GIFT solar (Alabama plants); source of fev solar not checked",
 "solar_with_weather": "UNCERTAIN: solar-named",
 "hospital_admissions": "UNCERTAIN: hospital-named",
 "uk_covid_nation": "UNCERTAIN: GIFT covid_deaths", "uk_covid_utla": "UNCERTAIN: GIFT covid_deaths",
}

def load():
    g, f = {}, {}
    for _, gd, fc, _ in MAP:
        d = pd.read_csv(f"{GIFT}/{gd}/all_results.csv")
        d["base"] = d.dataset.str.split("/").str[0]
        g[gd] = d.set_index("dataset")[["base", "eval_metrics/MASE[0.5]"]]
        e = pd.read_csv(f"{FEV}/{fc}.csv", usecols=["task_name", "MASE", "trained_on_this_dataset"]).set_index("task_name")
        f[fc] = e
    return g, f

def pair_data(g, f, a, b):
    ga, gb = g[a[1]], g[b[1]]
    gd = np.log(gb["eval_metrics/MASE[0.5]"]) - np.log(ga["eval_metrics/MASE[0.5]"])
    gd = pd.DataFrame({"d": gd, "src": ga["base"]})
    fa, fb = f[a[2]], f[b[2]]
    ok = ~(fa.trained_on_this_dataset.astype(bool) | fb.trained_on_this_dataset.astype(bool))
    skipped = int((~ok).sum())
    fd = pd.DataFrame({"d": np.log(fb.MASE) - np.log(fa.MASE)})[ok]
    fd["src"] = [family(t) for t in fd.index]
    return gd, fd, skipped

def sd1(x): return np.std(x, ddof=1) if len(x) > 1 else np.nan
def cr2scale(df):
    cl = pd.factorize(df.src)[0]
    return se_cr2(df.d.values, cl) * np.sqrt(len(np.unique(cl)))

def main():
    g, f = load()
    for gd, d in g.items(): assert len(d) == 97 and d.iloc[:, 1].gt(0).all(), gd
    for fc, d in f.items(): assert len(d) == 100 and d.MASE.gt(0).all(), fc
    fams = sorted({family(t) for t in f["chronos-2"].index})
    nonshared = [x for x in fams if x not in SHARED]
    print("mapping:"); [print(" ", m) for m in MAP]
    print("rejected:"); [print(" ", r) for r in REJECT]
    print(f"fev families {len(fams)}; shared/uncertain {len(fams)-len(nonshared)}: {sorted(SHARED)}\nnon-shared {len(nonshared)}: {nonshared}")
    nbase = g["chronos-2"].base.nunique(); print("GIFT bases", nbase)

    pairs, rows = list(itertools.combinations(MAP, 2)), []
    gsrc = sorted(g["chronos-2"].base.unique())
    GM = np.full((len(pairs), nbase), np.nan); FM_ns = np.full((len(pairs), len(nonshared)), np.nan)
    FM_all = np.full((len(pairs), len(fams)), np.nan)
    GU, FU = [], []  # unit data per pair for CR2 bootstrap
    for i, (a, b) in enumerate(pairs):
        gd, fd, sk = pair_data(g, f, a, b)
        gm, fm = gd.groupby("src").d.mean(), fd.groupby("src").d.mean()
        GM[i] = gm.reindex(gsrc).values
        FM_ns[i] = fm.reindex(nonshared).values; FM_all[i] = fm.reindex(fams).values
        fdn = fd[fd.src.isin(nonshared)]
        GU.append(gd); FU.append(fdn)
        r = dict(a=a[0], b=b[0], skipped_fev_rows=sk, n_fev_sources_nonshared=int(fm.reindex(nonshared).notna().sum()),
                 sd_gift=sd1(GM[i]), sd_fev_nonshared=sd1(FM_ns[i][~np.isnan(FM_ns[i])]),
                 sd_fev_all=sd1(FM_all[i][~np.isnan(FM_all[i])]),
                 cr2_gift=cr2scale(gd), cr2_fev_nonshared=cr2scale(fdn) if fdn.src.nunique() > 2 else np.nan)
        r["ratio_sd"] = r["sd_fev_nonshared"] / r["sd_gift"]; r["ratio_sd_all"] = r["sd_fev_all"] / r["sd_gift"]
        r["ratio_cr2"] = r["cr2_fev_nonshared"] / r["cr2_gift"]
        rows.append(r)
    P = pd.DataFrame(rows)

    def summ(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        q = np.percentile(x, [25, 50, 75]); return dict(n=len(x), median=q[1], q1=q[0], q3=q[2], min=x.min(), max=x.max())
    out = dict(models=[m[0] for m in MAP], n_models=len(MAP), n_pairs=len(pairs), shared_families=sorted(SHARED),
               nonshared_families=nonshared, n_nonshared=len(nonshared))
    for k in ["ratio_sd", "ratio_sd_all", "ratio_cr2"]: out[k] = summ(P[k])
    rms = lambda x: float(np.sqrt(np.nanmean(np.square(x))))
    out["pooled_ratio_sd"] = rms(P.sd_fev_nonshared) / rms(P.sd_gift)
    out["pooled_ratio_sd_all"] = rms(P.sd_fev_all) / rms(P.sd_gift)
    out["pooled_ratio_cr2"] = rms(P.cr2_fev_nonshared) / rms(P.cr2_gift)

    # bootstrap: resample sources within each board (same draw across pairs)
    rng = np.random.default_rng(SEED)
    def sdm(M, idx):
        S = M[:, idx]; n = (~np.isnan(S)).sum(1); mu = np.nanmean(S, 1)
        return np.sqrt(np.nansum((S - mu[:, None]) ** 2, 1) / (n - 1))
    bs = {"ratio_sd": [], "ratio_sd_all": []}
    for _ in range(B):
        ig, i1, i2 = rng.integers(0, nbase, nbase), rng.integers(0, len(nonshared), len(nonshared)), rng.integers(0, len(fams), len(fams))
        sg = sdm(GM, ig)
        bs["ratio_sd"].append(np.nanmedian(sdm(FM_ns, i1) / sg)); bs["ratio_sd_all"].append(np.nanmedian(sdm(FM_all, i2) / sg))
    # CR2 bootstrap: resample clusters (relabelled) within each board
    def cr2boot(grp):
        pick = rng.integers(0, len(grp), len(grp))
        d = np.concatenate([grp[j] for j in pick]); cl = np.concatenate([np.full(len(grp[j]), k) for k, j in enumerate(pick)])
        return se_cr2(d, cl) * np.sqrt(len(grp))
    bc = []
    GUg = [[x.d.values for _, x in df.groupby("src")] for df in GU]
    FUg = [[x.d.values for _, x in df.groupby("src")] for df in FU]
    for _ in range(B):
        rr = []
        for gd, fd in zip(GUg, FUg):
            rr.append(cr2boot(fd) / cr2boot(gd))
        bc.append(np.nanmedian(rr))
    bs["ratio_cr2"] = bc
    for k, v in bs.items(): out[k]["boot_median_ci95"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
    P.to_csv(os.path.join(HERE, "boards", "cross_board_spread_pairs.csv"), index=False)
    out["pairs"] = rows
    json.dump(out, open(os.path.join(HERE, "boards", "cross_board_spread.json"), "w"), indent=1, default=float)
    write_md(out, P)
    print(json.dumps({k: out[k] for k in ["n_models", "n_pairs", "ratio_sd", "ratio_sd_all", "ratio_cr2", "pooled_ratio_sd", "pooled_ratio_sd_all", "pooled_ratio_cr2"]}, indent=1, default=float))
    print("skipped rows per pair: min/median/max", P.skipped_fev_rows.min(), P.skipped_fev_rows.median(), P.skipped_fev_rows.max())

def write_md(o, P):
    f3 = lambda x: f"{x:.2f}"
    def row(name, k, pooled):
        s = o[k]; ci = s["boot_median_ci95"]
        return f"| {name} | {s['n']} | {f3(s['median'])} ({f3(ci[0])} to {f3(ci[1])}) | {f3(s['q1'])} to {f3(s['q3'])} | {f3(s['min'])} to {f3(s['max'])} | {f3(o[pooled])} |"
    L = ["# Do new datasets vary like the current ones? A cross-board check", "",
     "Question: the paper's \"about 45 times as many datasets\" figure assumes new datasets would vary the way the current 28 GIFT-Eval datasets do. "
     "fev-bench is a second board, and several open models are scored on both. For each pair of those models I take the difference in log MASE, average it within each data source "
     "(a GIFT-Eval base dataset, or a fev-bench task family), and compare the standard deviation of those source means on the two boards. Only fev-bench families that do not share data with GIFT-Eval are used for the main comparison.", "",
     "## Models scored on both boards (same checkpoint only)", "", "| Model | GIFT-Eval folder | fev-bench file | Why it is the same |", "|---|---|---|---|"]
    L += [f"| {m[0]} | {m[1]} | {m[2]} | {m[3]} |" for m in MAP]
    L += ["", "Rejected or uncertain, not used:", ""] + [f"- {a}: {b}" for a, b in REJECT]
    L += ["", f"{o['n_models']} models give {o['n_pairs']} pairs. In each pair, fev-bench rows where either model is flagged as trained on that dataset are skipped "
          f"({int(P.skipped_fev_rows.min())} to {int(P.skipped_fev_rows.max())} rows per pair, median {int(P.skipped_fev_rows.median())}; Moirai-2.0 alone has 28 flagged rows).", "",
     "## fev-bench families that share data with GIFT-Eval (excluded from the main comparison)", "",
     "A family is the task name with its last underscore-separated part removed (single-word names stay whole). Uncertain cases are treated as shared.", ""]
    L += [f"- {k}: {v}" for k, v in SHARED.items()]
    L += ["", f"Non-shared ({o['n_nonshared']} families): " + ", ".join(o["nonshared_families"]) + ".", "",
     "## Results", "", "Ratio = fev-bench (non-shared families) divided by GIFT-Eval, computed per model pair, then summarised over pairs. "
     "A ratio near 1 means new datasets vary like the current ones; above 1 means more, below 1 means less. The interval is a bootstrap (2,000 draws, seed 20261111) on the median, resampling sources within each board.", "",
     "| Ratio | Pairs | Median (95% interval) | Quartiles | Min to max | Pooled (root mean square) |", "|---|---|---|---|---|---|",
     row("SD of source means, non-shared fev families / GIFT 28 datasets", "ratio_sd", "pooled_ratio_sd"),
     row("Same, all 39 fev families (reference)", "ratio_sd_all", "pooled_ratio_sd_all"),
     row("Per-source noise scale (CR2 standard error x sqrt(sources)), non-shared fev / GIFT", "ratio_cr2", "pooled_ratio_cr2"), "",
     "## Caveats", "",
     "The two boards differ in horizons, frequencies and number of forecast windows per task, so a source is not the same kind of unit on both; MASE is used on both. "
     "The shared models are few and heavily related (five Toto sizes, two TimesFM versions, two naive baselines), so the pairs are strongly correlated and the interval on the median understates the real uncertainty. "
     "Spread of source means also includes sampling noise from the number of series per source, which differs by board; the CR2 row accounts for unit counts but not for these design differences.", "",
     "## What this does and does not show", "",
     "FILLME", ""]
    open(os.path.join(HERE, "CROSS_BOARD_SPREAD.md"), "w").write("\n".join(L))

if __name__ == "__main__":
    main()
