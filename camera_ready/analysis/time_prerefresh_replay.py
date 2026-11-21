"""Step 3: replay every one of the 45 states (commits that touch results/) of the TIME-Output repository, scoring each
state from the files and the seasonal_naive baseline as they were at that commit. Run with python -I.
Board code (boards/time_lb/src) is used for: compute_ranks, normalize_by_seasonal_naive, get_overall_leaderboard
(displayed table, checked against my unrounded arrays at every state)."""
import os, sys, json, hashlib, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
B = os.path.join(HERE, "boards"); OBJ = os.path.join(B, "time_output_pre_refresh", "objects"); RES = os.path.join(B, "time_output", "results")
import audit_time as A
U, L = A.load_board_code()
man = json.load(open(os.path.join(B, "time_prerefresh_manifest.json")))
states = man["states"]
# oid -> file: the head files already on disk, otherwise the downloaded objects
path = {}
for m, us in states[-1]["files"].items():
    for u, oid in us.items(): path[oid] = os.path.join(RES, m, u, "metrics.npz")
for f in os.listdir(OBJ):
    if not f.endswith(".part"): path.setdefault(f, os.path.join(OBJ, f))
cache = {}
def red(oid):   # the board's own reduction (get_all_datasets_results): nanmean over the metric array
    if oid not in cache:
        with np.load(path[oid], allow_pickle=False) as z: cache[oid] = (np.nanmean(z["MASE"]), np.nanmean(z["CRPS"]))
    return cache[oid]
UNITS = sorted(states[-1]["files"]["seasonal_naive"])
KEYS = {"mase_gmean": "MASE (norm.)", "crps_gmean": "CRPS (norm.)", "mase_rank": "MASE_rank", "crps_rank": "CRPS_rank"}
REV = os.environ.get("TIE") == "rev"   # tie-break sensitivity: exact ties in rank columns broken by reversed model order
FR = os.environ.get("FR") == "1"   # reproduce the first round's pool: keep only model-states whose 98 files equal the head files
hh = lambda f: hashlib.sha256("\n".join(sorted(f.values())).encode()).hexdigest()
headh = {m: hh(f) for m, f in states[-1]["files"].items()}
S = []
for st in states:
    fl = st["files"]
    if FR:
        fl = {m: f for m, f in fl.items() if headh.get(m) == hh(f)}
        if "seasonal_naive" not in fl or len(fl) < 10: continue
    models = sorted(fl)
    assert all(sorted(fl[m]) == UNITS for m in models), "unit set differs"
    assert all(st["n_pred"][m] == len(UNITS) for m in models), "predictions.npz missing"
    st = dict(st, files=fl)   # the board needs both files
    rows = []
    for m in models:
        for u in UNITS:
            ds, fr, hz = u.split("/"); a, b = red(fl[m][u])
            rows.append(dict(model=m, dataset=ds, freq=fr, dataset_id=f"{ds}/{fr}", horizon=hz, MASE=a, CRPS=b))
    raw = U.compute_ranks(pd.DataFrame(rows), ["dataset_id", "horizon"])       # board's ranks, method "first"
    lb = L.get_overall_leaderboard(raw, "MASE")                                   # board's displayed table
    # my unrounded arrays
    mi = {m: i for i, m in enumerate(models)}
    M = np.array([[red(fl[m][u])[0] for u in UNITS] for m in models]); C = np.array([[red(fl[m][u])[1] for u in UNITS] for m in models])
    sn = mi["seasonal_naive"]
    agg = {"mase_gmean": np.log(M / M[sn]).mean(1), "crps_gmean": np.log(C / C[sn]).mean(1)}
    ties = 0
    for k, X in (("mase_rank", M), ("crps_rank", C)):
        R = np.column_stack([(rankdata(X[::-1, j], method="ordinal")[::-1] if REV else rankdata(X[:, j], method="ordinal")) for j in range(len(UNITS))]); agg[k] = R.mean(1)
        ties += sum(len(set(X[:, j])) != len(X[:, j]) for j in range(len(UNITS)))
    order = {k: [models[i] for i in np.argsort(v, kind="stable")] for k, v in agg.items()}
    # check against the board's table
    lbm = lb.set_index("model")
    chk = dict(order_matches_board=list(lb["model"]) == order["mase_gmean"],
               max_abs_diff_vs_board={k: float(max(abs(lbm.loc[m, KEYS[k]] - (np.exp(agg[k][mi[m]]) if k.endswith("gmean") else agg[k][mi[m]])) for m in models)) for k in agg},
               tie_units=int(ties))
    # entry content hashes (all 98 units' oids)
    h = {m: hashlib.sha256("\n".join(sorted(fl[m].values())).encode()).hexdigest()[:16] for m in models}
    S.append(dict(commit=st["commit"][:8], date=st["date"], subject=st["subject"], models=models, hash=h, agg=agg, order=order, check=chk,
                  files=fl, board_top10=list(lb["model"][:10])))
# ---- transitions
def changed_units(a, b, m):
    return [u for u in UNITS if a["files"][m][u] != b["files"][m][u]]
out_tr, events = [], {k: [] for k in KEYS}
for i in range(1, len(S)):
    p, c = S[i - 1], S[i]
    both = [m for m in p["models"] if m in c["models"]]
    chg = {m: changed_units(p, c, m) for m in both}
    touched = [m for m in both if chg[m]]
    new = [m for m in c["models"] if m not in p["models"]]
    gone = [m for m in p["models"] if m not in c["models"]]
    # a folder rename (same content under a new name) keeps the entry's identity
    ren = {o: n for o in gone for n in new if p["hash"][o] == c["hash"][n]}
    new = [n for n in new if n not in ren.values()]; gone = [o for o in gone if o not in ren]
    base_changed = bool(chg.get("seasonal_naive"))
    renamed = bool(gone)   # a folder disappeared: rename commit; entries under the new name have the old content
    t = dict(from_commit=p["commit"], to_commit=c["commit"], from_date=p["date"], to_date=c["date"], subject=c["subject"],
             new_entries=new, removed_entries=gone, renamed=ren, entries_with_changed_files={m: chg[m] for m in touched},
             baseline_changed=base_changed, baseline_changed_units=chg.get("seasonal_naive", []), pool_before=len(p["models"]), pool_after=len(c["models"]))
    t["kind"] = "refresh_of_baseline" if base_changed else "ordinary"
    for k in KEYS:
        top = p["order"][k][:10]
        # an entry is "untouched" if present in both states with identical files in all units; for the geometric-mean rules an
        # entry's score also depends on the baseline, so when the baseline changed no entry is untouched (counted apart below).
        unt = [m for m in top if ren.get(m, m) in c["models"] and p["hash"][m] == c["hash"][ren.get(m, m)]]
        pa, ca = dict(zip(p["models"], p["agg"][k])), {o: v for o, v in zip(c["models"], c["agg"][k])}
        ca = {m: ca[ren.get(m, m)] for m in p["models"] if ren.get(m, m) in ca}
        pl_p = {m: j + 1 for j, m in enumerate(p["order"][k])}; pl_c0 = {m: j + 1 for j, m in enumerate(c["order"][k])}; pl_c = {m: pl_c0[ren.get(m, m)] for m in p["models"] if ren.get(m, m) in pl_c0}
        fl = [(a, b) for x, a in enumerate(unt) for b in unt[x + 1:] if (pa[a] - pa[b]) * (ca[a] - ca[b]) < 0]
        pres = [m for m in top if m in ca]
        allfl = [(a, b) for x, a in enumerate(pres) for b in pres[x + 1:] if (pa[a] - pa[b]) * (ca[a] - ca[b]) < 0]
        t.setdefault("flips_in_prev_top10_including_touched_entries", {})[k] = [dict(a=a, b=b, touched=[m for m in (a, b) if p["hash"][m] != c["hash"][ren.get(m, m)]]) for a, b in allfl]
        t[k] = dict(untouched_top10_entries=len(unt), swapped_pairs=[dict(a=a, b=b, place_before=[pl_p[a], pl_p[b]], place_after=[pl_c[a], pl_c[b]]) for a, b in fl])
        if fl:
            events[k].append(dict(**{x: t[x] for x in ("from_commit", "to_commit", "from_date", "to_date", "kind", "new_entries", "subject")},
                                  pairs=t[k]["swapped_pairs"], entries_with_changed_files=sorted(touched)))
    out_tr.append(t)
def summary(kind_filter):
    r = {}
    for k in KEYS:
        trs = [t for t in out_tr if kind_filter(t)]
        r[k] = dict(updates=len(trs), updates_with_swap=sum(bool(t[k]["swapped_pairs"]) for t in trs), pairs=sum(len(t[k]["swapped_pairs"]) for t in trs))
    return r
res = dict(n_states=len(S), n_updates=len(out_tr),
           all_updates=summary(lambda t: True), ordinary_updates=summary(lambda t: t["kind"] == "ordinary"),
           baseline_refresh_updates=summary(lambda t: t["kind"] != "ordinary"),
           # the 28 updates the first round replayed: from state 2026-06-02 c43961bb to the last one
           first_round_window=summary(lambda t: t["to_commit"] in {s["commit"] for s in S[[x["commit"] for x in S].index("c43961bb") + 1:]}),
           checks=[dict(commit=s["commit"], pool=len(s["models"]), **s["check"], board_top10=s["board_top10"], my_top10=s["order"]["mase_gmean"][:10]) for s in S],
           transitions=out_tr, events=events,
           states=[dict(commit=s["commit"], date=s["date"], subject=s["subject"], pool=len(s["models"]), top10={k: s["order"][k][:10] for k in KEYS}) for s in S])
json.dump(res, open(os.path.join(B, "time_full_history.json" if not (REV or FR) else ("_time_cache/full_history_tie_rev.json" if REV else "_time_cache/full_history_firstround_pool.json")), "w"), indent=1)
print({k: res[k] for k in ("n_states", "n_updates", "all_updates", "ordinary_updates", "baseline_refresh_updates", "first_round_window")})
print("board order match all states:", all(c["order_matches_board"] for c in res["checks"]), "max diff", max(max(c["max_abs_diff_vs_board"].values()) for c in res["checks"]), "ties", sum(c["tie_units"] for c in res["checks"]))
