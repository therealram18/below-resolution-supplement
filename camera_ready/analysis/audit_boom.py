"""BOOM leaderboard audit (pre-registered in boards/PREREG_BOARDS.md). Stages: repro | report a|b|c | pool | all-merge.
Usage: python audit_boom.py <stage> [def] [--B_boot N --B_wild N --n_control N]"""
import os, sys, json, time, subprocess, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BOOM = f"{HERE}/boards/toto_repo/boom"
OUT = f"{HERE}/boards"
PART = f"{OUT}/boom_parts"
os.makedirs(PART, exist_ok=True)
sys.path.insert(0, HERE); sys.path.insert(0, BOOM)
from utils import leaderboard as L          # the board's own code
import vernier_board as vb

def jdump(o, p): json.dump(o, open(p, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

def load(order=None):
    """Board's own loading, split and cleaning. order: None = os.listdir order (what the board does)."""
    cwd = os.getcwd(); os.chdir(BOOM)
    try:
        if order is None:
            dfs, names = L.load_model_results("results/", False)
        else:
            names = [m for m in order if m != "seasonalnaive"] + ["seasonalnaive"]
            dfs = [L.load_and_process_csv(f"results/{m}/all_results.csv", False) for m in names]
        nz, z = L.separate_zero_inflated_data(dfs)
        nz = L.process_benchmark_model_results(True, nz, L.NON_ZERO_METRICS)
        z = L.process_benchmark_model_results(False, z, L.ZERO_METRICS)
    finally:
        os.chdir(cwd)
    return nz, z, names

def rank_matrix(dfs, names, metric):
    """Per-unit rank exactly as get_leaderboard (groupby full_dataset_name, rank 'first'); returns units, X[model, unit]."""
    d = [x.copy() for x in dfs]
    for x, n in zip(d, names): x["model"] = n
    c = pd.concat(d)
    c["rank"] = c.groupby("full_dataset_name")[metric].rank(method="first", ascending=True)
    R = c.pivot(index="model", columns="full_dataset_name", values="rank").loc[names]
    return list(R.columns), R.values

def agg_values(dfs, names, metrics):
    d = [x.copy() for x in dfs]
    for x, n in zip(d, names): x["model"] = n
    c = pd.concat(d)
    out = {m: c.groupby("model")[m].apply(lambda s: L.shifted_gmean(s.values)) for m in metrics}
    return out

def series_id(u): return u.split("/")[0]

def clusters(units):
    props = json.load(open(f"{BOOM}/boom_properties.json"))
    ids = [series_id(u) for u in units]
    uid = sorted(set(ids))
    missing = [s for s in uid if s not in props]
    def key_b(s):
        p = props.get(s)
        return "NO_PROPERTIES" if p is None else "|".join(sorted(p["domain"])) + "#" + p["type"]
    def key_c(s):
        p = props.get(s)
        if p is None: return "NO_PROPERTIES"
        return p["domain"][0] if len(p["domain"]) == 1 else "|".join(sorted(p["domain"])) + "#" + p["type"]
    out = {}
    for name, f in (("a", series_id), ("b", key_b), ("c", key_c)):
        keys = [f(i) for i in ids]
        lab = {k: n for n, k in enumerate(sorted(set(keys)))}
        cl = np.array([lab[k] for k in keys])
        sizes = np.bincount(cl)
        out[name] = dict(cl=cl, n_clusters=len(sizes), sizes=sizes,
                         labels=sorted(set(keys)) if len(sizes) <= 60 else None)
    return out, dict(n_series=len(uid), n_props=len(props), missing=missing,
                     props_not_in_rows=sorted(set(props) - set(uid))[:20], n_props_not_in_rows=len(set(props) - set(uid)))

def size_dist(sizes):
    s = np.asarray(sizes)
    return dict(n=len(s), min=int(s.min()), q25=float(np.percentile(s, 25)), median=float(np.median(s)),
                q75=float(np.percentile(s, 75)), max=int(s.max()), mean=float(s.mean()),
                hist={int(k): int(v) for k, v in zip(*np.unique(s, return_counts=True))} if s.max() <= 10 else None,
                largest5=sorted(s.tolist(), reverse=True)[:5])

def stage_repro():
    t0 = time.time()
    nz, z, names = load()
    pub = pd.read_csv(f"{BOOM}/leaderboards/boom_leaderboard.csv").set_index("model")
    res = dict(model_order_used=names, n_scaled_rows=len(nz[0]), n_unscaled_rows=len(z[0]),
               published_cols=list(pub.columns))
    # board's own leaderboard function
    lb_nz = L.get_leaderboard([x.copy() for x in nz], names, L.shifted_gmean, L.NON_ZERO_METRICS)
    lb_z = L.get_leaderboard([x.copy() for x in z], names, L.shifted_gmean, L.ZERO_METRICS)
    comb = L.get_separate_zero_inflated_leaderboard([x.copy() for x in nz], [x.copy() for x in z], names, L.shifted_gmean, L.NON_ZERO_METRICS, L.ZERO_METRICS)
    res["board_function_output_equals_published_numbers_exactly"] = bool((comb.astype(float).reindex(pub.index).values == pub.values).all())
    res["board_output_order_equals_published_order"] = list(comb.index) == list(pub.index)
    # unrounded values from the same data
    u_nz, X_nz = rank_matrix(nz, names, "eval_metrics/mean_weighted_sum_quantile_loss")
    u_z, X_z = rank_matrix(z, names, "eval_metrics/mean_weighted_sum_quantile_loss")
    g_nz = agg_values(nz, names, L.NON_ZERO_METRICS); g_z = agg_values(z, names, L.ZERO_METRICS)
    raw = pd.DataFrame({"MASE-6750-scaled": g_nz["eval_metrics/MASE[0.5]"], "CRPS-6750-scaled": g_nz["eval_metrics/mean_weighted_sum_quantile_loss"],
                        "Rank-6750-scaled": pd.Series(X_nz.mean(1), index=names),
                        "eval_metrics/MAE[0.5]-663-unscaled": g_z["eval_metrics/MAE[0.5]"], "CRPS-663-unscaled": g_z["eval_metrics/mean_weighted_sum_quantile_loss"],
                        "Rank-663-unscaled": pd.Series(X_z.mean(1), index=names)}).loc[pub.index]
    diff = (raw - pub).abs()
    # published values are rounded to 3 decimals by the board (format_number); half a unit in the last place is the best possible agreement
    res["max_abs_diff_by_column_vs_rounded_csv"] = diff.max().to_dict()
    res["all_within_half_rounding_unit_5e-4"] = bool((diff <= 5.0e-4 + 1e-12).all().all())
    rel = (diff / pub.abs().replace(0, np.nan))
    res["max_rel_diff_by_column_nonzero_published"] = rel.max().to_dict()
    res["rank_columns_max_abs_diff"] = {c: float(diff[c].max()) for c in ["Rank-6750-scaled", "Rank-663-unscaled"]}
    disp = raw["Rank-6750-scaled"].sort_values(kind="stable").index.tolist()
    res["displayed_order_from_unrounded_ranks"] = disp
    res["displayed_order_equals_published"] = disp == list(pub.index)
    res["displayed_top10_equal"] = disp[:10] == list(pub.index)[:10]
    # does tie order depend on model order? rerun with sorted order
    nz2, z2, n2 = load(sorted(os.listdir(f"{BOOM}/results")))
    _, X2 = rank_matrix(nz2, n2, "eval_metrics/mean_weighted_sum_quantile_loss")
    r2 = pd.Series(X2.mean(1), index=n2).loc[names]
    res["rank_change_if_model_order_sorted_alphabetically_max_abs"] = float((r2 - raw["Rank-6750-scaled"]).abs().max())
    res["tied_units_in_ranking_metric_scaled"] = int(sum(len(set(np.round(c, 12))) < len(c) for c in
        pd.concat([x.assign(model=n) for x, n in zip(nz, names)]).groupby("full_dataset_name")["eval_metrics/mean_weighted_sum_quantile_loss"].apply(lambda s: s.values)))
    # published rounded-rank ties in displayed column
    res["published_rank_column_has_exact_ties"] = bool(pub["Rank-6750-scaled"].duplicated().any())
    res["raw_table"] = raw.round(6).to_dict("index")
    res["order_used_for_ties_is_os_listdir"] = True
    res["seconds"] = time.time() - t0
    jdump(res, f"{PART}/repro.json")
    # matrices for later stages (primary: CRPS rank; secondary: MASE rank)
    _, Xm = rank_matrix(nz, names, "eval_metrics/MASE[0.5]")
    np.savez(f"{PART}/matrices.npz", units=np.array(u_nz), names=np.array(names), X=X_nz, X_mase=Xm, X_z=X_z, units_z=np.array(u_z))
    # raw scores for pool test: scaled CRPS per (model, unit) (ranks depend only on order within unit, not on scaling)
    d = [x.copy() for x in nz]
    for x, n in zip(d, names): x["model"] = n
    S = pd.concat(d).pivot(index="model", columns="full_dataset_name", values="eval_metrics/mean_weighted_sum_quantile_loss").loc[names][u_nz]
    np.save(f"{PART}/S_crps_scaled.npy", S.values)
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw_table",)}, indent=1, default=str))

def stage_clusters():
    z = np.load(f"{PART}/matrices.npz")
    cs, info = clusters(list(z["units"]))
    res = dict(props=info, definitions={})
    for k, v in cs.items():
        res["definitions"][k] = dict(n_clusters=v["n_clusters"], size_dist=size_dist(v["sizes"]), labels=v["labels"],
                                     sizes_by_label=(dict(zip(v["labels"], v["sizes"].tolist())) if v["labels"] else None))
    jdump(res, f"{PART}/clusters.json")
    print(json.dumps(res, indent=1, default=str)[:6000])

def stage_report(d, a):
    z = np.load(f"{PART}/matrices.npz")
    cs, _ = clusters(list(z["units"]))
    names = list(z["names"])
    out = {}
    for tag, key in (("crps", "X"), ("mase", "X_mase")):
        if tag == "mase" and d != "a": continue
        X = z[key]; order = list(np.argsort(X.mean(1), kind="stable")) if tag == "crps" else list(np.argsort(z["X"].mean(1), kind="stable"))
        t0 = time.time()
        nc = a.n_control if tag == "crps" else 0
        rep = vb.report_from_matrix(X, cs[d]["cl"], names, order, B_boot=a.B_boot, B_wild=a.B_wild, n_control=nc)
        rep["runtime_s"] = time.time() - t0
        rep["df_satterthwaite_pairs_icc"] = None
        # keep full pair table (45 + tier); drop nothing
        out[tag] = rep
    jdump(out, f"{PART}/report_{d}.json")
    r = out["crps"]
    print(d, r["resolution"], r["counts"], {k: v["size"] for k, v in r["leading_tier"].items()}, r["known_shift"] and r["known_shift"]["detection"], r["runtime_s"])

def stage_shift(d, a):
    """Known-shift control, one adjacent pair per process (vb._known_shift, same code report_from_matrix calls)."""
    z = np.load(f"{PART}/matrices.npz"); cs, _ = clusters(list(z["units"])); names = list(z["names"])
    order = list(np.argsort(z["X"].mean(1), kind="stable")); X = z["X"]
    rs = json.load(open(f"{PART}/report_{d}.json"))["crps"]["resolution"]
    k = a.pair; dv = X[order[k + 1]] - X[order[k]]
    t0 = time.time()
    rng = np.random.default_rng(20261111 + 100 * k + ord(d))
    kr, pp = vb._known_shift([dv], cs[d]["cl"], [rs["paper"], rs["corrected_icc"], 0.0], a.n_control, rng)
    jdump(dict(pair=k, n_control=a.n_control, deltas=dict(paper_resolution=rs["paper"], corrected_resolution=rs["corrected_icc"], zero=0.0),
               detection=kr, seconds=time.time() - t0), f"{PART}/shift_{d}_{k}.json")

def stage_pool():
    S = np.load(f"{PART}/S_crps_scaled.npy"); z = np.load(f"{PART}/matrices.npz"); names = list(z["names"])
    rule = vb.mean_rank_rule("ordinal")
    t0 = time.time()
    r = vb.pool_tests(S, rule, list(range(len(names))))
    res = dict(top10=[names[i] for i in r["top"]], removal=dict(changed=r["removal"]["changed"], of=r["removal"]["of"], entries=[names[i] for i in r["removal"]["entries"]]),
               near_copy=dict(changed=r["near_copy"]["changed"], of=r["near_copy"]["of"], entries=[names[i] for i in r["near_copy"]["entries"]]), seconds=time.time() - t0)
    # git history
    g = subprocess.run(["git", "-C", f"{OUT}/toto_repo", "log", "--format=COMMIT %h %ad", "--date=short", "--name-status", "--", "boom/results"], capture_output=True, text=True).stdout
    commits, cur = [], None
    for ln in g.splitlines():
        if ln.startswith("COMMIT"):
            _, h, dt = ln.split(); cur = dict(hash=h, date=dt, added=[], other=[]); commits.append(cur)
        elif ln.strip():
            st, p = ln.split("\t", 1)
            if p.endswith("all_results.csv"):
                (cur["added"] if st == "A" else cur["other"]).append((st, p.split("/")[2]))
    res["git_commits"] = commits
    jdump(res, f"{PART}/pool.json"); print(json.dumps(res, indent=1, default=str)[:3000])

def stage_merge():
    rp = json.load(open(f"{PART}/repro.json")); cl_ = json.load(open(f"{PART}/clusters.json")); pool = json.load(open(f"{PART}/pool.json"))
    out = dict(reproduction=rp, clusters=cl_, pool_dependence=pool, defs={})
    z = np.load(f"{PART}/matrices.npz"); X = z["X"]; Xm = z["X_mase"]; names = list(z["names"])
    order = np.argsort(X.mean(1), kind="stable"); om = np.argsort(Xm.mean(1), kind="stable")
    out["mase_rank_order"] = [names[i] for i in om]; out["mase_rank_top10_same_set_as_crps_rank_top10"] = set(om[:10]) == set(order[:10])
    out["crps_rank_order"] = [names[i] for i in order]
    for d in "abc":
        R = json.load(open(f"{PART}/report_{d}.json"))
        r = R["crps"]; adj = [p for p in r["pairs"] if p["adjacent"] and p["rank_b"] <= 10]
        shifts = [json.load(open(f"{PART}/shift_{d}_{k}.json")) for k in range(9) if os.path.exists(f"{PART}/shift_{d}_{k}.json")]
        det = None
        if len(shifts) == 9:
            det = {m: {key: float(np.mean([s["detection"][m][i] for s in shifts])) for i, key in enumerate(["at_paper_resolution", "at_corrected_resolution", "false_alarm_delta0"])} for m in ["paper", "CR2-icc", "wild"]}
            det["n_control_per_pair"] = shifts[0]["n_control"]
        dfs = [p["cr2_icc"]["df"] for p in adj]
        entry = dict(B_boot=r["B_boot"], B_wild=r["B_wild"], n_clusters=r["n_sources"], resolution=r["resolution"], counts=r["counts"],
                     tier={k: v["size"] for k, v in r["leading_tier"].items()}, tier_members=r["leading_tier"],
                     known_shift=det, cr2_icc_df_adjacent=dict(min=min(dfs), median=float(np.median(dfs)), max=max(dfs)),
                     icc_hat_adjacent=[p["icc_hat"] for p in adj], runtime_s=r["runtime_s"], top10=r["top10"])
        if "mase" in R:
            m = R["mase"]; entry["mase_rank_displayed_by_crps_order"] = dict(resolution=m["resolution"], counts=m["counts"], tier={k: v["size"] for k, v in m["leading_tier"].items()})
        out["defs"][d] = entry
    jdump(out, f"{OUT}/boom_report.json")
    for d, e in out["defs"].items():
        rs, c = e["resolution"], e["counts"]
        print(d, e["n_clusters"], f'res paper {rs["paper"]:.3f} corr {rs["corrected_icc"]:.3f} gap {rs["median_adjacent_gap"]:.3f} ratio {rs["ratio_paper"]:.2f}/{rs["ratio_corrected_icc"]:.2f} needed {rs["sources_needed_boot"]:.0f}/{rs["sources_needed_cr2"]:.0f}',
              "adj paper/icc/wild", c["paper"]["adjacent"], c["CR2-icc"]["adjacent"], c["wild"]["adjacent"], "all45 holm paper/icc/wild", c["paper"]["all_holm"], c["CR2-icc"]["all_holm"], c["wild"]["all_holm"], "all45 unc", c["paper"]["all"], c["CR2-icc"]["all"], c["wild"]["all"], "tier", e["tier"], "df", e["cr2_icc_df_adjacent"], "det", e["known_shift"])

OVERLAP = dict(
    gift_eval_datasets=28,
    dataset_card="huggingface.co/datasets/Datadog/BOOM: data 'sourced from an internal Datadog deployment monitoring pre-production systems', 2,807 series, no public dataset named as a source; GIFT-Eval appears only as a comparison point",
    properties_file="boom_properties.json holds only anonymous ids ds-<n>-<freq> with domain/type/term/frequency/num_variates; no dataset names that could match GIFT-Eval's 28 names",
    conclusion="no overlap found; BOOM is disjoint from GIFT-Eval's public datasets as far as the files and card show (the card makes no explicit disjointness claim)",
    model_name_matches=dict(exact=["chronos_bolt_base", "chronos_bolt_small", "timesfm_2_0_500m", "visionts"],
        case_or_separator=[("toto_2.0_1b", "Toto-2.0-1B"), ("toto_2.0_2.5b", "Toto-2.0-2.5B"), ("toto_2.0_22m", "Toto-2.0-22m"), ("toto_2.0_313m", "Toto-2.0-313m"), ("toto_2.0_4m", "Toto-2.0-4m"),
                           ("chronos_2", "chronos-2"), ("timesfm_2_5_200m", "TimesFM-2.5"), ("autoarima", "auto_arima"), ("autoets", "auto_ets"), ("autotheta", "auto_theta"), ("seasonalnaive", "seasonal_naive")],
        approximate_version_unclear=[("toto", "Toto_Open_Base_1.0"), ("moirai_1.1_small", "Moirai_small"), ("moirai_1.1_base", "Moirai_base"), ("moirai_1.1_large", "Moirai_large"), ("moirai_2_small", "Moirai2")],
        no_match=["time_moe_50M", "time_moe_200M", "timer (GIFT has Timer-s1, a different model)"]))

def stage_md():
    r = json.load(open(f"{OUT}/boom_report.json")); r["overlap"] = OVERLAP; jdump(r, f"{OUT}/boom_report.json")
    D = r["defs"]; rp = r["reproduction"]; pool = r["pool_dependence"]; cl = r["clusters"]["definitions"]
    f = lambda x, n=2: f"{x:.{n}f}"
    rows = []
    for d, lab in (("a", "(a) series id"), ("b", "(b) domain+type"), ("c", "(c) domain")):
        e = D[d]; rs, c = e["resolution"], e["counts"]; sd = cl[d]["size_dist"]; k = e["known_shift"]
        rows.append(f"| {lab} | {e['n_clusters']} | {sd['min']} / {sd['median']:g} / {sd['max']} | {f(rs['paper'],3)} / {f(rs['corrected_icc'],3)} | {f(rs['median_adjacent_gap'],3)} | {f(rs['ratio_paper'])} / {f(rs['ratio_corrected_icc'])} | "
                    f"{c['paper']['adjacent']} / {c['CR2-icc']['adjacent']} / {c['wild']['adjacent']} | {c['paper']['all_holm']} / {c['CR2-icc']['all_holm']} / {c['wild']['all_holm']} | {e['tier']['paper']} / {e['tier']['CR2-icc']} / {e['tier']['wild']} | "
                    + (f"{100*k['paper']['at_paper_resolution']:.0f}% / {100*k['CR2-icc']['at_corrected_resolution']:.0f}% / {100*k['wild']['at_corrected_resolution']:.0f}%" if k else "n/a") + " |")
    tab = "\n".join(rows)
    dfrows = "\n".join(f"| {lab} | {D[d]['cr2_icc_df_adjacent']['min']:.1f} / {D[d]['cr2_icc_df_adjacent']['median']:.1f} / {D[d]['cr2_icc_df_adjacent']['max']:.1f} | {min(D[d]['icc_hat_adjacent']):.2f} to {max(D[d]['icc_hat_adjacent']):.2f} | {D[d]['resolution']['sources_needed_boot']:.0f} / {D[d]['resolution']['sources_needed_cr2']:.0f} | {D[d]['counts']['paper']['all']} / {D[d]['counts']['CR2-icc']['all']} / {D[d]['counts']['wild']['all']} | {D[d]['counts']['means-t']['adjacent']} |" for d, lab in (("a", "(a)"), ("b", "(b)"), ("c", "(c)")))
    kd = "\n".join(f"| {lab} | " + " | ".join(f"{100*D[d]['known_shift'][m][key]:.0f}%" for m in ("paper", "CR2-icc", "wild") for key in ("at_paper_resolution", "at_corrected_resolution", "false_alarm_delta0")) + f" | {D[d]['known_shift']['n_control_per_pair']} |" for d, lab in (("a", "(a)"), ("b", "(b)"), ("c", "(c)")) if D[d]["known_shift"])
    git = "\n".join(f"| {c['date']} | {c['hash']} | {len(c['added'])} added, {len(c['other'])} modified |" for c in pool["git_commits"])
    md = open(f"{HERE}/BOARD_BOOM.template.md").read()
    for k, v in dict(TAB=tab, DFROWS=dfrows, KD=kd, GIT=git, REM=f"{pool['removal']['changed']} of {pool['removal']['of']}", NC=f"{pool['near_copy']['changed']} of {pool['near_copy']['of']}",
                     R6750=f"{rp['rank_columns_max_abs_diff']['Rank-6750-scaled']:.6f}", R663=f"{rp['rank_columns_max_abs_diff']['Rank-663-unscaled']:.6f}", M6750=f"{rp['max_abs_diff_by_column_vs_rounded_csv']['MASE-6750-scaled']:.6f}",
                     TIED=f"{rp['tied_units_in_ranking_metric_scaled']:,}", TIEDIFF=f"{rp['rank_change_if_model_order_sorted_alphabetically_max_abs']:.3f}",
                     CNA=str(cl['a']['n_clusters']), CNB=str(cl['b']['n_clusters']), CNC=str(cl['c']['n_clusters']),
                     SB=f"{cl['b']['size_dist']['min']} / {cl['b']['size_dist']['median']:g} / {cl['b']['size_dist']['max']}", SC=f"{cl['c']['size_dist']['min']} / {cl['c']['size_dist']['median']:g} / {cl['c']['size_dist']['max']}",
                     GAP=f"{D['a']['resolution']['median_adjacent_gap']:.3f}", NCA=str(D['a']['known_shift']['n_control_per_pair']) if D['a']['known_shift'] else "n/a",
                     NCNAMES=", ".join(pool["near_copy"]["entries"]), TOP10=", ".join(pool["top10"])).items():
        md = md.replace("{{" + k + "}}", v)
    open(f"{HERE}/BOARD_BOOM.md", "w").write(md)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("d", nargs="?")
    ap.add_argument("--B_boot", type=int, default=20000); ap.add_argument("--B_wild", type=int, default=99999); ap.add_argument("--n_control", type=int, default=0); ap.add_argument("--pair", type=int, default=0)
    a = ap.parse_args()
    {"repro": stage_repro, "clusters": stage_clusters, "pool": stage_pool, "merge": stage_merge, "md": stage_md}.get(a.stage, (lambda: stage_shift(a.d, a)) if a.stage == "shift" else (lambda: stage_report(a.d, a)))()
