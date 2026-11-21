"""Second-pin analysis (2026-11-13). Usage (from camera_ready):  python -I pin2_boards.py gift|time|fev
Reads the pinned files under boards/pin_2026-11-13/. Same code as the first round (vernier_board, vernier_stats).
Report: report_from_matrix with the first-round seed and draw counts but n_control=0 (known-shift control not run; it does not
change any other number because it is the last use of the random stream)."""
import sys, os, csv, json, difflib, time, subprocess
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vernier_board as vb
P = os.path.join(HERE, "boards", "pin_2026-11-13") + "/"
SEED = 20261111
MASE = "eval_metrics/MASE[0.5]"
SPECIAL = ["Timestral-209M", "TimesFM-3"]

def find(names, target):
    ex = [n for n in names if n.lower() == target.lower()]
    return ex[0] if ex else None, difflib.get_close_matches(target, names, n=5, cutoff=0.4)

def pair_tests(X, cl, names, disp_order, target):
    """Place of `target` in the displayed order and tests vs the entries directly below and above (a = higher placed)."""
    nm, close = find(names, target)
    if nm is None:
        return dict(target=target, status="NOT ON BOARD", closest=close)
    i = names.index(nm); pos = disp_order.index(i) if i in disp_order else None
    out = dict(target=target, exact_name=nm, place=None if pos is None else pos + 1, of=len(disp_order))
    if pos is None:
        out["status"] = "on files but not displayed (missing units?)"; return out
    pairs = []
    if pos > 0: pairs.append(("above", disp_order[pos - 1], i))
    if pos < len(disp_order) - 1: pairs.append(("below", i, disp_order[pos + 1]))
    if pairs:
        D = np.stack([X[b] - X[a] for _, a, b in pairs])
        recs = vb._pair_records(D, np.asarray(cl), 20000, 99999, np.random.default_rng(SEED))
        for (side, a, b), r in zip(pairs, recs):
            r.update(side=side, a=names[a], b=names[b], rank_a=disp_order.index(a) + 1, rank_b=disp_order.index(b) + 1)
            out[side] = r
    return out

def short(r):
    if r is None: return None
    return dict(side=r["side"], pair=f'{r["a"]} (#{r["rank_a"]}) vs {r["b"]} (#{r["rank_b"]})', theta=round(r["theta"], 4),
        iid_ci=[round(x, 4) for x in r["iid"]["ci"]], cluster_ci=[round(x, 4) for x in r["cluster"]["ci"]], p_paper=round(r["p_paper"], 4),
        separated_paper=r["sep_paper"], cr2_se=round(r["cr2_icc"]["se"], 4), cr2_df=round(r["cr2_icc"]["df"], 2), cr2_p=round(r["cr2_icc"]["p"], 4),
        cr2_indep_p=round(r["cr2_indep"]["p"], 4), wild_p=round(r["wild_p"], 5))

def summ(rep):
    c, rs = rep["counts"], rep["resolution"]
    return dict(entries=rep["n_entries"], units=rep["n_units"], sources=rep["n_sources"], top10=rep["top10"],
        res_paper=round(rs["paper"], 4), res_corrected=round(rs["corrected_icc"], 4), median_gap=round(rs["median_adjacent_gap"], 4),
        counts=c, tier={k: v["size"] for k, v in rep["leading_tier"].items()}, tier_members=rep["leading_tier"])

def run(X, cl, names, order, tag, tests=True):
    t0 = time.time()
    rep = vb.report_from_matrix(X, cl, names, order, n_control=0)
    print(tag, "report", round(time.time() - t0), "s", flush=True)
    json.dump(rep, open(P + f"report_{tag}.json", "w"))
    return rep

# ------------------------------------------------------------------ GIFT-Eval
def load_gift(results):
    rows, cfg = defaultdict(dict), {}
    for root, _, files in os.walk(results):
        if "all_results.csv" in files:
            for r in csv.DictReader(open(os.path.join(root, "all_results.csv"))):
                try: rows[r["model"]][r["dataset"]] = float(r[MASE])
                except (KeyError, ValueError): pass
        if "config.json" in files:
            try:
                j = json.load(open(os.path.join(root, "config.json"))); cfg[j["model"]] = j
            except (ValueError, KeyError): pass
    configs = sorted({k for m in rows for k in rows[m]}); names = sorted(rows)
    V = np.array([[rows[m].get(c, np.nan) for c in configs] for m in names])
    leak = np.array([cfg.get(m, {}).get("testdata_leakage", "No") == "Yes" for m in names])
    disp = [i for i, m in enumerate(names) if m in cfg and not np.isnan(V[i]).any()]
    bases = sorted({c.split("/")[0] for c in configs}); cl = np.array([bases.index(c.split("/")[0]) for c in configs])
    return names, configs, V, leak, disp, cl

def gift():
    names, configs, V, leak, disp, cl = load_gift(P + "gift/results")
    print(len(names), "entries,", len(disp), "displayable,", len(configs), "configs,", len(set(cl)), "sources")
    R = np.full(V.shape, np.nan)
    for j in range(V.shape[1]):
        ok = ~np.isnan(V[:, j]); R[ok, j] = rankdata(V[ok, j], method="average")
    Xd = R[disp]; order = list(np.lexsort((Xd.mean(1), leak[disp]))); nd = [names[i] for i in disp]
    rep = run(Xd, cl, nd, order, "gift")
    avg = vb.mean_rank_rule("average")
    tier = [(1 if leak[i] else 0) if i in set(disp) else None for i in range(len(names))]
    pt = vb.pool_tests(V, avg, disp, tier=tier)
    out = dict(summary=summ(rep), removal=dict(changed=pt["removal"]["changed"], of=pt["removal"]["of"]),
               near_copy=dict(changed=pt["near_copy"]["changed"], of=pt["near_copy"]["of"], entries=[names[i] for i in pt["near_copy"]["entries"]]),
               n_leak_flagged_displayed=int(leak[disp].sum()),
               top10_scores=[(nd[i], round(float(Xd[i].mean()), 4), bool(leak[disp][i])) for i in order[:12]])
    out["special"] = {t: pair_tests(Xd, cl, nd, order, t) for t in SPECIAL}
    out["displayed_order_first30"] = [(k + 1, nd[i], round(float(Xd[i].mean()), 3), bool(leak[disp][i])) for k, i in enumerate(order[:30])]
    json.dump(out, open(P + "gift_pin_summary.json", "w"), indent=1, default=str); return out

# ------------------------------------------------------------------ TIME
def time_():
    import audit_time as at
    old = at.RES; new = P + "time_res_all"
    if not os.path.isdir(new):
        os.makedirs(new)
        for m in os.listdir(old):
            os.symlink(os.path.join(old, m), os.path.join(new, m))
        for m in os.listdir(P + "time_results/results"):
            os.symlink(os.path.join(P, "time_results/results", m), os.path.join(new, m))
    at.RES = new
    U, L = at.load_board_code()
    t0 = time.time()
    raw = U.get_all_datasets_results(root_dir=new)
    raw = U.compute_ranks(raw, groupby_cols=["dataset_id", "horizon"])
    lb = L.get_overall_leaderboard(raw, "MASE")
    norm = U.normalize_by_seasonal_naive(raw, "seasonal_naive", ["MASE", "CRPS"], ["dataset_id", "horizon"]).rename(columns={"MASE": "MASE_norm", "CRPS": "CRPS_norm"})[["model", "dataset_id", "horizon", "MASE_norm", "CRPS_norm"]]
    d = raw.merge(norm, on=["model", "dataset_id", "horizon"], how="left")
    d.to_csv(P + "time_units_pin.csv", index=False); lb.to_csv(P + "time_board_overall_mase_sorted.csv", index=False)
    print("units", round(time.time() - t0), "s", d.shape, flush=True)
    d["unit"] = d.dataset_id + "/" + d.horizon
    models = sorted(d.model.unique()); units = sorted(d.unit.unique())
    piv = lambda c: d.pivot(index="model", columns="unit", values=c).loc[models, units].to_numpy(float)
    dsets = d.drop_duplicates("unit").set_index("unit").loc[units, "dataset"].to_numpy()
    cl = pd.factorize(dsets)[0]
    out = dict(n_models=len(models), n_units=len(units), board_order_top10=list(lb.iloc[:, 0][:10]) if "model" not in lb else list(lb["model"][:10]))
    # reproduction: first-round numbers of the 43 old entries unchanged
    o = pd.read_csv(os.path.join(at.B, "time_units.csv")); m = o.merge(d, on=["model", "dataset_id", "horizon"], suffixes=("_o", "_n"))
    out["old_vs_new_max_abs_diff_MASE_norm"] = float((m.MASE_norm_o - m.MASE_norm_n).abs().max()); out["old_rows_matched"] = len(m)
    for rule, X in (("mase_gmean", np.log(piv("MASE_norm"))), ("mase_rank", piv("MASE_rank"))):
        order = at.order_of(X)
        # note: the new pool changes MASE_rank for every entry (rank columns depend on the pool)
        if rule == "mase_gmean":
            bo = list(lb["model"]) if "model" in lb else None
            out["own_order_equals_board_order"] = bo == [models[i] for i in order] if bo else "n/a"
        rep = run(X, cl, models, order, "time_" + rule)
        out[rule] = dict(summary=summ(rep), top10_scores=[(models[i], round(float(X[i].mean()), 5)) for i in order[:12]],
                         order=[(k + 1, models[i], round(float(X[i].mean()), 5)) for k, i in enumerate(order)])
        out[rule]["special"] = {t: pair_tests(X, cl, models, order, t) for t in SPECIAL}
        if rule == "mase_rank":  # pool dependence of the rank columns is not asked for; skip
            pass
    json.dump(out, open(P + "time_pin_summary.json", "w"), indent=1, default=str); return out

# ------------------------------------------------------------------ fev-bench (no new commit: first-round files, same commit)
def fev_():
    import audit_fev_family as af
    af.FEV = P + "fev"
    head = subprocess.run(["git", "-C", af.FEV, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    A, hidden, t100, t20 = af.common()
    b = af.board_a(A, t100, hidden); rp = af.check_repro(b, "A")
    cl = np.unique(b["fam"], return_inverse=True)[1]; out = dict(commit=head, repro={k: v for k, v in rp.items() if k != "incomplete_models"}, n_models=len(b["models"]))
    for rule, X in (("win_rate", b["Xwr"]), ("skill_score", b["Xsk"])):
        order = af.order_of(X.mean(1))
        out[rule] = dict(top10=[b["models"][i] for i in order[:10]], special={t: pair_tests(X, cl, b["models"], order, t) for t in SPECIAL},
                         order=[(k + 1, b["models"][i], round(float(X[i].mean()), 5)) for k, i in enumerate(order[:30])])
    out["hidden_models"] = sorted(hidden)
    json.dump(out, open(P + "fev_pin_summary.json", "w"), indent=1, default=str); return out

if __name__ == "__main__":
    r = dict(gift=gift, time=time_, fev=fev_)[sys.argv[1]]()
    for t in SPECIAL:
        for k, v in (r.get("special") or r.get("mase_gmean", {}).get("special") or r.get("win_rate", {}).get("special")).items():
            print(k, {x: (short(v[x]) if x in ("above", "below") else v[x]) for x in v if x in ("exact_name", "place", "of", "status", "closest", "above", "below")}); break
