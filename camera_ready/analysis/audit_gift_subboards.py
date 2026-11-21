"""GIFT-Eval sub-board audit: the tabs the leaderboard app displays (By Domain / Frequency / Term Length / Variate Type).
Usage: python audit_gift_subboards.py   (resumable: caches each board in boards/_gift_sub_cache/, stops after ~6.5 min, rerun until done)
Group assignment copies snapshot_2026-09-07/src/utils.py get_grouped_dfs (lines 122-241) without the ipdb calls."""
import os, csv, json, re, sys, time, pickle
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import rankdata
import vernier_board as vb

T0 = time.time(); BUDGET = 390
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "fmts/snapshot_2026-09-07/results"
MASE = "eval_metrics/MASE[0.5]"
CACHE = os.path.join(HERE, "boards/_gift_sub_cache"); os.makedirs(CACHE, exist_ok=True)

# ---- data, exactly as test_vernier_board.py
rows, cfg = defaultdict(dict), {}
for root, _, files in os.walk(SNAP):
    if "all_results.csv" in files:
        for r in csv.DictReader(open(os.path.join(root, "all_results.csv"))):
            try: rows[r["model"]][r["dataset"]] = float(r[MASE])
            except (KeyError, ValueError): pass
    if "config.json" in files:
        try:
            j = json.load(open(os.path.join(root, "config.json"))); cfg[j["model"]] = j
        except (ValueError, KeyError): pass
configs = sorted({k for m in rows for k in rows[m]})
names = sorted(rows)
V = np.array([[rows[m].get(c, np.nan) for c in configs] for m in names])
leak = np.array([cfg.get(m, {}).get("testdata_leakage", "No") == "Yes" for m in names])
disp = [i for i, m in enumerate(names) if m in cfg and not np.isnan(V[i]).any()]
R = np.full(V.shape, np.nan)
for j in range(V.shape[1]):
    ok = ~np.isnan(V[:, j]); R[ok, j] = rankdata(V[ok, j], method="average")
Rd = R[disp]; nd = [names[i] for i in disp]; leakd = leak[disp]
sn = next(i for i, m in enumerate(nd) if m.lower() in ("seasonal_naive", "seasonal naive"))
Ld = np.log(V[disp] / V[disp][sn])  # log of MASE scaled by seasonal naive; mean over configs = log of the app's geometric mean

# ---- group assignment as the app does (utils.py 122-207, unify_freq 62-83, univariate line 241)
props = pd.read_csv(os.path.join(SNAP, "dataset_properties.csv"))
props["dataset"] = props["dataset"].str.lower().str.replace(" ", "_").str.replace("-", "_").map(lambda x: re.sub("_+", "_", x).strip("_"))
pd_ = props.set_index("dataset").T.to_dict("dict")  # duplicates: last row wins, as in the app
conv = dict(T="Minutely", H="Hourly", D="Daily", W="Weekly", M="Monthly", Q="Quarterly", Y="Yearly", A="Yearly", S="Secondly")
def unify(f): return conv.get(re.sub(r"\d+", "", f).split("-")[0], re.sub(r"\d+", "", f).split("-")[0])
meta = []
for c in configs:
    ds, fr, tl = c.split("/")
    p = pd_[ds]
    meta.append(dict(cfg=c, base=ds, term_length=tl, frequency=unify(fr), domain=p["domain"],
                     univariate="univariate" if p["num_variates"] == 1 else "multivariate"))
groups = {}
for tab in ("term_length", "domain", "frequency", "univariate"):
    for v in sorted({m[tab] for m in meta}):
        idx = [k for k, m in enumerate(meta) if m[tab] == v]
        groups[f"{tab}={v}"] = dict(tab=tab, value=v, configs=[configs[k] for k in idx], n_configs=len(idx),
                                    sources=sorted({meta[k]["base"] for k in idx}), n_sources=len({meta[k]["base"] for k in idx}), idx=idx)
gp = {g: {k: v for k, v in d.items() if k != "idx"} for g, d in groups.items()}
json.dump(gp, open(os.path.join(HERE, "boards/gift_subboards_groups.json"), "w"), indent=1)
print(f"{'group':28s} configs sources")
for g, d in groups.items(): print(f"{g:28s} {d['n_configs']:5d} {d['n_sources']:5d}  {'ANALYSE' if d['n_sources'] >= 6 else 'skip (<6 sources)'}")
sys.stdout.flush()
if "--groups-only" in sys.argv: sys.exit()

def run(idx, stat):
    X = (Rd if stat == "rank" else Ld)[:, idx]
    order = [int(i) for i in np.lexsort((X.mean(1), leakd))]
    cl = np.array([meta[k]["base"] for k in idx]); cl = np.unique(cl, return_inverse=True)[1]
    rep = vb.report_from_matrix(X, cl, nd, order, n_control=300)
    rep["known_shift"].pop("per_pair", None)
    rep["top10_full_order"] = [nd[i] for i in order[:40]]
    return rep

boards = [("FULL", list(range(len(configs))))] + [(g, d["idx"]) for g, d in groups.items() if d["n_sources"] >= 6]
for stat in ("rank", "gmean"):
    for g, idx in boards:
        f = os.path.join(CACHE, f"{stat}__{g.replace('/', '_')}.pkl")
        if os.path.exists(f): continue
        if time.time() - T0 > BUDGET:
            print("time budget reached; rerun to continue"); sys.exit(2)
        t = time.time(); rep = run(idx, stat); pickle.dump(rep, open(f, "wb"))
        print(f"done {stat} {g} in {time.time()-t:.0f}s"); sys.stdout.flush()

# ---- assemble
out = dict(groups=gp, analysed=[g for g, _ in boards[1:]], not_analysed={g: dict(n_configs=d["n_configs"], n_sources=d["n_sources"]) for g, d in groups.items() if d["n_sources"] < 6}, stats={})
for stat in ("rank", "gmean"):
    full = pickle.load(open(os.path.join(CACHE, f"{stat}__FULL.pkl"), "rb"))
    ft = full["top10"]; rec = {}
    for g, idx in boards:
        rep = pickle.load(open(os.path.join(CACHE, f"{stat}__{g.replace('/', '_')}.pkl"), "rb"))
        rs, c = rep["resolution"], rep["counts"]
        rec[g] = dict(configs=len(idx), sources=rs["sources"], res_paper=rs["paper"], res_corrected=rs["corrected_icc"], gap=rs["median_adjacent_gap"],
                      ratio_paper=rs["ratio_paper"], ratio_corrected=rs["ratio_corrected_icc"], sources_needed_boot=rs["sources_needed_boot"],
                      sources_needed_cr2=rs["sources_needed_cr2"], res_x_sqrt_sources=rs["corrected_icc"] * np.sqrt(rs["sources"]),
                      adj_paper=c["paper"]["adjacent"], adj_cr2=c["CR2-icc"]["adjacent"], adj_wild=c["wild"]["adjacent"],
                      all_paper=c["paper"]["all"], all_cr2=c["CR2-icc"]["all"], all_wild=c["wild"]["all"],
                      all_paper_holm=c["paper"]["all_holm"], all_cr2_holm=c["CR2-icc"]["all_holm"], all_wild_holm=c["wild"]["all_holm"],
                      tier_paper=rep["leading_tier"]["paper"]["size"], tier_cr2=rep["leading_tier"]["CR2-icc"]["size"], tier_wild=rep["leading_tier"]["wild"]["size"],
                      leader=rep["top10"][0], top10=rep["top10"], overlap_with_full_top10=len(set(rep["top10"]) & set(ft)),
                      known_shift=rep["known_shift"]["detection"])
    out["stats"][stat] = rec
json.dump(out, open(os.path.join(HERE, "boards/gift_subboards_report.json"), "w"), indent=1, default=float)
print("assembled; total elapsed this invocation %.0fs" % (time.time() - T0))
