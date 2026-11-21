"""Leaderboard audit of TIME (arXiv 2602.12147). Stages (each cached under boards/_time_cache):
  python audit_time.py units     -> boards/time_units.csv + reproduction data
  python audit_time.py report R  -> report_from_matrix for rule R in {mase_gmean, mase_rank, crps_gmean, crps_rank, nonoverlap...}
  python audit_time.py rest      -> dominance, pool tests, overlap setup
  python audit_time.py history   -> history replay
  python audit_time.py final     -> assemble boards/time_report.json and BOARD_TIME.md
"""
import sys, os, json, time, types, warnings, hashlib, subprocess
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
B = os.path.join(HERE, "boards"); LB = os.path.join(B, "time_lb"); RES = os.path.join(B, "time_output", "results")
HIST = os.path.join(B, "time_output_history"); CACHE = os.path.join(B, "_time_cache")
os.makedirs(CACHE, exist_ok=True)
warnings.filterwarnings("ignore")
import vernier_board as vb, vernier_stats as vs


def load_board_code():
    """Import the board's own utils.py and leaderboard.py with stubs for packages that are not installed
    (gradio, timebench, the HF download config, src.about which downloads data at import)."""
    sys.path.insert(0, LB)
    for name in ["gradio", "timebench", "timebench.evaluation", "timebench.evaluation.data", "src.hf_config", "src.about"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["timebench.evaluation.data"].Dataset = None
    sys.modules["timebench.evaluation.data"].get_dataset_settings = None
    sys.modules["timebench.evaluation.data"].load_dataset_config = None
    h = sys.modules["src.hf_config"]; h.get_datasets_root = h.get_config_root = lambda: None
    a = sys.modules["src.about"]
    a.DATASETS_DF = None; a.OVERALL_TABLE_COLUMNS = ["model", "MASE", "CRPS", "MASE_rank", "CRPS_rank"]
    a.ALL_MODELS = []; a.RESULTS_ROOT = RES; a.DATASET_DISPLAY_TO_ID = {}
    import src.utils as U, src.leaderboard as L
    # DEVIATION (paths/inputs only): the board's loader insists on predictions.npz, which were not downloaded
    # (3.9 GB). Same function minus that file requirement; the reduction code (np.nanmean of metrics) is the board's own.
    def load_time_results(root_dir, model_name, dataset_with_freq, horizon):
        d = os.path.join(root_dir, model_name, dataset_with_freq, horizon)
        mp = os.path.join(d, "metrics.npz")
        if not os.path.exists(mp):
            return None, None, None
        m = np.load(mp, allow_pickle=False)
        return {k: m[k] for k in m.files}, None, {}
    U.load_time_results = load_time_results
    return U, L


def stage_units():
    U, L = load_board_code()
    t0 = time.time()
    raw = U.get_all_datasets_results(root_dir=RES)                   # board's own reduction (one row per model, dataset_id, horizon)
    raw = U.compute_ranks(raw, groupby_cols=["dataset_id", "horizon"])   # rank columns, as about.py does
    lb = L.get_overall_leaderboard(raw, "MASE")                      # board's displayed overall table (sorted by MASE (norm.))
    lb_crps = L.get_overall_leaderboard(raw, "CRPS")
    norm = U.normalize_by_seasonal_naive(raw, "seasonal_naive", ["MASE", "CRPS"], ["dataset_id", "horizon"])
    norm = norm.rename(columns={"MASE": "MASE_norm", "CRPS": "CRPS_norm"})[["model", "dataset_id", "horizon", "MASE_norm", "CRPS_norm"]]
    out = raw.merge(norm, on=["model", "dataset_id", "horizon"], how="left")   # keeps the board's row order
    out.to_csv(os.path.join(B, "time_units.csv"), index=False)
    lb.to_csv(os.path.join(CACHE, "board_overall_mase_sorted.csv"), index=False)
    lb_crps.to_csv(os.path.join(CACHE, "board_overall_crps_sorted.csv"), index=False)
    print("units stage", round(time.time() - t0), "s", out.shape)


def load_matrices():
    d = pd.read_csv(os.path.join(B, "time_units.csv"))
    d["unit"] = d.dataset_id + "/" + d.horizon
    models = sorted(d.model.unique())
    units = sorted(d.unit.unique())
    piv = lambda c: d.pivot(index="model", columns="unit", values=c).loc[models, units].to_numpy(float)
    M = dict(MASE=piv("MASE"), CRPS=piv("CRPS"), MASE_norm=piv("MASE_norm"), CRPS_norm=piv("CRPS_norm"),
             MASE_rank=piv("MASE_rank"), CRPS_rank=piv("CRPS_rank"))
    dsets = d.drop_duplicates("unit").set_index("unit").loc[units, "dataset"].to_numpy()
    return d, models, units, dsets, M


RULES = {  # name -> (matrix builder, description)
    "mase_gmean": lambda M: np.log(M["MASE_norm"]),
    "mase_rank": lambda M: M["MASE_rank"],
    "crps_gmean": lambda M: np.log(M["CRPS_norm"]),
    "crps_rank": lambda M: M["CRPS_rank"],
}


def order_of(X):
    return list(np.argsort(X.mean(1), kind="stable"))


def stage_report(rule, subset=None):
    d, models, units, dsets, M = load_matrices()
    X = RULES[rule](M)
    keep = np.ones(len(units), bool)
    tag = rule
    if subset:
        drop = json.load(open(os.path.join(CACHE, "overlap.json")))[subset]
        keep = ~np.isin(dsets, drop)
        tag = f"{rule}__{subset}"
        X = X[:, keep]
    cl = pd.factorize(dsets[keep])[0]
    t0 = time.time()
    rep = vb.report_from_matrix(X, cl, models, order_of(X))
    rep["runtime_s"] = time.time() - t0
    rep["rule"] = rule; rep["units_used"] = int(keep.sum())
    json.dump(rep, open(os.path.join(CACHE, f"report_{tag}.json"), "w"))
    c, r = rep["counts"], rep["resolution"]
    print(tag, f"{rep['runtime_s']:.0f}s", "adj", c["paper"]["adjacent"], "all", c["paper"]["all"], "res", r["paper"], r["corrected_icc"], "gap", r["median_adjacent_gap"])


GIFT_CSV = os.path.join(os.path.dirname(HERE), "snapshot_2026-09-07", "results", "seasonal_naive", "all_results.csv")


def overlap_sets(time_datasets):
    """GIFT-Eval vs TIME dataset-name comparison. Judgement is from names and the TIME paper's appendix B
    (data providers), written down here before the rerun."""
    g = pd.read_csv(GIFT_CSV)
    gift = sorted(set(g.dataset.str.split("/").str[0]))
    definite = []   # none: no TIME dataset shares a provider with a GIFT-Eval dataset
    uncertain = {   # same topic as a GIFT-Eval dataset, different provider per TIME appendix B (not proven the same)
        "ECDC_COVID": "covid_deaths (GIFT: JHU case counts; TIME: ECDC)",
        "Australia_Solar": "solar (GIFT: Alabama solar plants; TIME: PVOutput stations in Australia)",
        "EWELD_Load": "electricity (GIFT: UCI hourly consumption; TIME: Liu et al. 2023 industrial/commercial users)",
        "epf_electricity_price": "electricity (price series, not load)",
        "SG_Weather": "jena_weather / temperature_rain (weather, different providers)",
        "azure2019_D": "bitbrains_* (cloud VM traces, different provider)",
        "azure2019_I": "bitbrains_* (cloud VM traces, different provider)",
        "azure2019_U": "bitbrains_* (cloud VM traces, different provider)",
    }
    uncertain = {k: v for k, v in uncertain.items() if k in time_datasets}
    return gift, definite, uncertain


def stage_overlap():
    d, models, units, dsets, M = load_matrices()
    gift, definite, uncertain = overlap_sets(set(dsets))
    json.dump(dict(gift=gift, time=sorted(set(dsets)), definite=definite, uncertain=uncertain,
                   drop_uncertain=sorted(uncertain), drop_definite=definite), open(os.path.join(CACHE, "overlap.json"), "w"), indent=1)
    # the subsets used by stage_report: keys must be lists
    ov = json.load(open(os.path.join(CACHE, "overlap.json")))
    ov["definite_list"] = definite
    ov["uncertain_list"] = sorted(uncertain)
    ov["uncertain_or_definite"] = sorted(set(definite) | set(uncertain))
    json.dump(ov, open(os.path.join(CACHE, "overlap.json"), "w"), indent=1)
    print("overlap", len(gift), "gift;", len(set(dsets)), "time; definite", definite, "uncertain", sorted(uncertain))


def dominance(X, order, cl):
    """Share of the variance estimate of a pair's mean difference that comes from the single largest dataset
    (sum over datasets of S_g^2, S_g = summed deviation of the dataset's units from the pair mean)."""
    top = order[:10]
    sh = []
    for i in range(10):
        for j in range(i + 1, 10):
            dv = X[top[j]] - X[top[i]]
            S = np.bincount(cl, weights=dv - dv.mean())
            sh.append(float((S ** 2).max() / (S ** 2).sum()))
    sh = np.array(sh)
    return dict(median=float(np.median(sh)), max=float(sh.max()), n_above_half=int((sh > 0.5).sum()), n_pairs=len(sh),
                n_above_third=int((sh > 1 / 3).sum()))


def stage_rest():
    d, models, units, dsets, M = load_matrices()
    cl = pd.factorize(dsets)[0]
    sn = models.index("seasonal_naive")
    out = dict(n_entries=len(models), n_units=len(units), n_datasets=len(set(dsets)),
               n_dataset_freq=int(d.dataset_id.nunique()))
    # ---- reproduction against the board's displayed table
    lb = pd.read_csv(os.path.join(CACHE, "board_overall_mase_sorted.csv"))
    lbc = pd.read_csv(os.path.join(CACHE, "board_overall_crps_sorted.csv"))
    Xm = RULES["mase_gmean"](M); om = order_of(Xm)
    gm = np.exp(Xm.mean(1))
    unrounded = {m: float(stats.gmean(M["MASE_norm"][i])) for i, m in enumerate(models)}
    rep = dict(order_matches_board=[models[i] for i in om] == list(lb.model),
               max_abs_diff_gmean_vs_scipy=float(max(abs(gm[i] - unrounded[m]) for i, m in enumerate(models))),
               max_abs_diff_rounded_value=float(max(abs(round(gm[models.index(m)], 3) - v) for m, v in zip(lb.model, lb["MASE (norm.)"]))),
               crps_order_matches_board=[models[i] for i in order_of(RULES["crps_gmean"](M))] == list(lbc.model),
               max_abs_diff_rank_cols=float(max(abs(M["MASE_rank"][models.index(m)].mean() - r) for m, r in zip(lb.model, lb["MASE_rank"]))),
               top10=list(lb.model[:10]))
    # Hart-36M shipped summaries
    hd = pd.read_csv(os.path.join(RES, "Hart-36M", "full_results.csv"))
    hd["unit"] = hd.dataset
    hm = d[d.model == "Hart-36M"].set_index("unit")
    j = hd.set_index("unit")
    common = sorted(set(j.index) & set(hm.index))
    ha = pd.read_csv(os.path.join(RES, "Hart-36M", "average_results.csv")).iloc[0]
    rep["hart"] = dict(units_in_shipped_csv=len(j), units_matched=len(common),
                       max_abs_diff_MASE=float(np.abs(j.loc[common, "eval_metrics/MASE"] - hm.loc[common, "MASE"]).max()),
                       max_abs_diff_CRPS=float(np.abs(j.loc[common, "eval_metrics/CRPS"] - hm.loc[common, "CRPS"]).max()),
                       max_abs_diff_MAE=float(np.abs(j.loc[common, "eval_metrics/MAE"] - hm.loc[common, "MAE"]).max()),
                       mean_MASE_mine=float(hm.loc[common, "MASE"].mean()), average_results_MASE=float(ha["eval_metrics/MASE"]),
                       mean_CRPS_mine=float(hm.loc[common, "CRPS"].mean()), average_results_CRPS=float(ha["eval_metrics/CRPS"]),
                       geo_mean_norm_MASE_mine=float(np.exp(Xm[models.index("Hart-36M")].mean())))
    # live table
    live = pd.read_csv(os.path.join(CACHE, "live_overall_table.csv"))
    shared = [m for m in live.model if m in models]
    lv = live.set_index("model")
    mine_round = {m: round(float(gm[models.index(m)]), 3) for m in models}
    rep["live"] = dict(entries_live=len(live), entries_pinned=len(models), only_live=[m for m in live.model if m not in models],
                       live_top10=list(live.model[:10]), pinned_top10=list(lb.model[:10]),
                       top10_identical=list(live.model[:10]) == list(lb.model[:10]),
                       order_of_shared_identical=shared == [m for m in lb.model if m in set(shared)],
                       max_abs_diff_MASE_norm_shared=float(max(abs(lv.loc[m, "MASE (norm.)"] - mine_round[m]) for m in shared)),
                       max_abs_diff_CRPS_norm_shared=float(max(abs(lv.loc[m, "CRPS (norm.)"] - round(float(np.exp(RULES["crps_gmean"](M)[models.index(m)].mean())), 3)) for m in shared)),
                       top10_by_position_after_dropping_live_only=[m for m in live.model if m in models][:10])
    out["reproduction"] = rep
    # ---- NaNs / flags
    out["nan_cells"] = int(np.isnan(M["MASE_norm"]).sum() + np.isnan(M["CRPS_norm"]).sum())
    out["ties_in_unit_MASE"] = int(sum(len(set(M["MASE"][:, k])) < len(models) for k in range(len(units))))
    # ---- dominance
    out["dominance"] = {r: dominance(RULES[r](M), order_of(RULES[r](M)), cl) for r in RULES}
    # ---- pool dependence
    pool = {}
    SN = M["MASE"][sn]; SNC = M["CRPS"][sn]
    for name, key, base in (("mase_gmean", "MASE", SN), ("crps_gmean", "CRPS", SNC)):
        others = [i for i in range(len(models)) if i != sn]
        S = M[key][others]
        rule = lambda A, base=base: np.log(A / base).mean(1)   # the board's rule: own scores and the fixed baseline only
        r = vb.pool_tests(S, rule, list(range(len(others))))
        # direct check: each entry's aggregate from the whole pool equals the aggregate from [entry, baseline] alone
        alone = np.array([np.log(S[i] / base).mean() for i in range(len(S))])
        r2 = dict(top=[models[others[i]] for i in r["top"]], removal=dict(changed=r["removal"]["changed"], of=r["removal"]["of"]),
                  near_copy=dict(changed=r["near_copy"]["changed"], of=r["near_copy"]["of"]),
                  max_abs_diff_pool_vs_alone=float(np.abs(rule(S) - alone).max()),
                  note="seasonal_naive is the fixed normaliser, not a member of the perturbed pool (removing it makes the board skip normalisation)")
        pool[name] = r2
    for name, key in (("mase_rank", "MASE"), ("crps_rank", "CRPS")):
        rule = vb.mean_rank_rule("ordinal")
        r = vb.pool_tests(M[key], rule, list(range(len(models))))
        pool[name] = dict(top=[models[i] for i in r["top"]], removal=dict(changed=r["removal"]["changed"], of=r["removal"]["of"],
                          entries=[models[i] for i in r["removal"]["entries"]]),
                          near_copy=dict(changed=r["near_copy"]["changed"], of=r["near_copy"]["of"], entries=[models[i] for i in r["near_copy"]["entries"]]),
                          rule_matches_board_rank_column=float(np.abs(rule(M[key]) - M[key.replace("MASE", "MASE").replace("CRPS", "CRPS") + "_rank"].mean(1)).max()))
    out["pool"] = pool
    json.dump(out, open(os.path.join(CACHE, "rest.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("reproduction", "dominance")}, indent=1)[:3500]); print(json.dumps(pool, indent=1)[:2500])


def git(*a):
    return subprocess.run(["git", "-C", HIST] + list(a), capture_output=True, text=True, check=True).stdout


def lfs_oids(blob_ids):
    """Read blobs through one git cat-file --batch; return {blob id: LFS sha256 oid (or 'raw:'+sha256 of content)}."""
    p = subprocess.run(["git", "-C", HIST, "cat-file", "--batch"], input=("\n".join(blob_ids) + "\n").encode(), capture_output=True, check=True)
    buf, pos, out, nraw = p.stdout, 0, {}, 0
    while pos < len(buf):
        nl = buf.index(b"\n", pos)
        oid, typ, size = buf[pos:nl].decode().split()
        body = buf[nl + 1: nl + 1 + int(size)]
        pos = nl + 1 + int(size) + 1
        txt = body.decode("utf8", "replace")
        if txt.startswith("version https://git-lfs"):
            out[oid] = [l for l in txt.splitlines() if l.startswith("oid sha256:")][0].split(":", 1)[1]
        else:
            out[oid] = "raw:" + hashlib.sha256(body).hexdigest(); nraw += 1
    out["_n_raw_not_pointer"] = nraw
    return out


def stage_history():
    d, models, units, dsets, M = load_matrices()
    t0 = time.time()
    head = git("rev-parse", "HEAD").strip()
    commits = [l.split(" ", 2) for l in git("log", "--reverse", "--format=%H %ad %s", "--date=short", "--", "results").splitlines()]
    per_commit, allblobs = [], set()
    for c, date, subj in commits:
        mp = {}   # model -> {unit: blob id}
        folders = set()
        for line in git("ls-tree", "-r", c, "--", "results").splitlines():
            meta, path = line.split("\t", 1)
            blob = meta.split()[2]
            parts = path.split("/")
            if len(parts) >= 2:
                folders.add(parts[1])
            if len(parts) == 6 and parts[5] == "metrics.npz":
                mp.setdefault(parts[1], {})["/".join(parts[2:5])] = blob
                allblobs.add(blob)
        per_commit.append((c, date, subj, mp, folders))
    ob = lfs_oids(sorted(allblobs)); nraw = ob.pop("_n_raw_not_pointer")
    # head oids vs local files (the scores in time_units.csv come from the local files)
    headmap = {m: {u: ob[b] for u, b in per_commit[-1][3][m].items()} for m in per_commit[-1][3]}
    assert per_commit[-1][0] == head
    mism = 0
    for m in headmap:
        for u, oid in headmap[m].items():
            h = hashlib.sha256(open(os.path.join(RES, m, u, "metrics.npz"), "rb").read()).hexdigest()
            mism += (h != oid)
    hashes = lambda om: hashlib.sha256("\n".join(sorted(om.values())).encode()).hexdigest()
    head_hash = {m: hashes(headmap[m]) for m in headmap}
    ui = {u: k for k, u in enumerate(units)}
    mi = {m: k for k, m in enumerate(models)}
    states, log = {k: [] for k in RULE_KEYS}, []
    tot = dict(model_states=0, scored=0, not_at_head=0, changed_vs_head=0, partial=0)
    per_model_hashes = {}
    for c, date, subj, mp, folders in per_commit:
        ents, n_un = {}, dict(not_at_head=0, changed_vs_head=0, partial=0)
        for m, bm in mp.items():
            om = {u: ob[b] for u, b in bm.items()}
            h = hashes(om)
            per_model_hashes.setdefault(m, []).append((date, h))
            tot["model_states"] += 1
            if m not in headmap:
                why = "not_at_head"
            elif len(om) != len(units) and all(headmap[m].get(u) == o for u, o in om.items()):
                why = "partial"
            elif om != headmap[m]:
                why = "changed_vs_head"
            else:
                why = None
            if why:
                n_un[why] += 1; tot[why] += 1
            else:
                tot["scored"] += 1; ents[m] = h
        log.append(dict(commit=c[:8], date=date, subject=subj, models_present=len(mp), scored=len(ents), **n_un,
                        has_seasonal_naive="seasonal_naive" in ents, folders_without_metrics=sorted(folders - set(mp))))
        for rk, (key, _) in RULE_KEYS.items():
            states[rk].append(dict(date=f"{date} {c[:8]}", units=tuple(units),
                                   entries={m: (M[key][mi[m]], ents[m]) for m in ents}))
    def gmean_rule(S, nm):
        if "seasonal_naive" not in nm:
            return None
        return np.log(S / S[nm.index("seasonal_naive")]).mean(1)
    rules = {"mase_gmean": gmean_rule, "mase_rank": vb.mean_rank_rule("ordinal"),
             "crps_gmean": gmean_rule, "crps_rank": vb.mean_rank_rule("ordinal")}
    # gmean rule on CRPS needs CRPS seasonal naive: S already holds CRPS vectors, so the same function applies
    res = {}
    for rk in RULE_KEYS:
        h = vb.history_replay(states[rk], rules[rk])
        h["events"] = [dict(from_date=e["from_date"], to_date=e["to_date"], pairs=[list(p) for p in e["pairs"]], entries_added=e["entries_added"]) for e in h["events"]]
        res[rk] = h
    changed = {m: sorted({h for _, h in v}) for m, v in per_model_hashes.items()}
    changed_models = sorted(m for m, v in changed.items() if len(v) > 1)
    out = dict(head=head, n_commits=len(commits), n_replayable_states=sum(1 for x in log if x["scored"] >= 10 and x["has_seasonal_naive"]),
               local_files_vs_head_oid_mismatches=mism, non_pointer_blobs=nraw, totals=tot,
               models_ever=sorted(per_model_hashes), models_at_head=len(headmap),
               models_whose_content_changed_while_present=changed_models,
               models_not_at_head=sorted(set(per_model_hashes) - set(headmap)),
               log=log, replay=res, runtime_s=time.time() - t0)
    json.dump(out, open(os.path.join(CACHE, "history.json"), "w"), indent=1)
    print({k: out[k] for k in ("n_commits", "n_replayable_states", "local_files_vs_head_oid_mismatches", "non_pointer_blobs", "totals", "models_whose_content_changed_while_present", "models_not_at_head")})
    for rk, h in res.items():
        print(rk, h["transitions"], h["transitions_with_reversal"], h["reversed_pairs"])


RULE_KEYS = {"mase_gmean": ("MASE", None), "mase_rank": ("MASE", None), "crps_gmean": ("CRPS", None), "crps_rank": ("CRPS", None)}


def stage_final():
    J = lambda n: json.load(open(os.path.join(CACHE, n)))
    rest, hist, ov = J("rest.json"), J("history.json"), J("overlap.json")
    reps = {r: J(f"report_{r}.json") for r in RULES}
    rep_nov = J("report_mase_gmean__uncertain_list.json")
    names = {"mase_gmean": "MASE geometric mean vs seasonal naive (displayed sort)", "mase_rank": "MASE_rank (mean rank on MASE)",
             "crps_gmean": "CRPS geometric mean vs seasonal naive", "crps_rank": "CRPS_rank (mean rank on CRPS)"}
    pooldep = {"mase_gmean": "no", "crps_gmean": "no", "mase_rank": "yes", "crps_rank": "yes"}
    f = lambda x, n=4: f"{x:.{n}g}"
    def row(tag, r, label, pool, hr, pooldependent):
        c, rs, kt = r["counts"], r["resolution"], r["known_shift"]["detection"]
        return (f"| {label} | {r['n_entries']} | {r['n_units']} | {r['n_sources']} | {pooldependent} | {f(rs['paper'])} / {f(rs['corrected_icc'])} | "
                f"{f(rs['median_adjacent_gap'])} | {f(rs['ratio_paper'],3)} / {f(rs['ratio_corrected_icc'],3)} | "
                f"{c['paper']['adjacent']} / {c['CR2-icc']['adjacent']} / {c['wild']['adjacent']} | "
                f"{c['paper']['all_holm']} / {c['CR2-icc']['all_holm']} / {c['wild']['all_holm']} | "
                f"{r['leading_tier']['paper']['size']} / {r['leading_tier']['CR2-icc']['size']} / {r['leading_tier']['wild']['size']} | "
                f"{kt['paper']['at_corrected_resolution']:.0%} / {kt['CR2-icc']['at_corrected_resolution']:.0%} / {kt['wild']['at_corrected_resolution']:.0%} | "
                f"{pool} | {hr} |")
    rows = []
    for r in RULES:
        p = rest["pool"][r]
        pool = f"{p['removal']['changed']} of {p['removal']['of']}; {p['near_copy']['changed']} of {p['near_copy']['of']}"
        h = hist["replay"][r]
        hr = f"{h['transitions_with_reversal']} of {h['transitions']} transitions, {h['reversed_pairs']} pairs"
        rows.append(row(r, reps[r], names[r], pool, hr, pooldep[r]))
    rows.append(row("nov", rep_nov, "MASE geometric mean, 8 uncertain-overlap datasets removed (sensitivity)", "not run", "not run", "no"))
    hdr = ("| Rule | Entries | Units | Sources | Depends on pool? | Resolution paper / corrected | Median adjacent gap | Ratio (resolution / gap) paper / corrected | "
           "Adjacent separated of 9: paper / CR2-icc / wild | All 45 after Holm: paper / CR2-icc / wild | Leading tier size: paper / CR2-icc / wild | "
           "Known-shift detection at corrected resolution: paper / CR2-icc / wild | Removals / near-copies that reorder the top ten | History replay reversals |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    table = hdr + "\n" + "\n".join(rows)
    r0 = reps["mase_gmean"]; rp = rest["reproduction"]; lv = rp["live"]; hc = rp["hart"]; dm = rest["dominance"]
    allc = lambda r: " / ".join(str(reps[r]["counts"][m]["all"]) for m in ("paper", "CR2-icc", "wild"))
    kt = lambda r: reps[r]["known_shift"]["detection"]
    ev = lambda k: "\n".join(f"- {e['from_date']} -> {e['to_date']}: reversed {', '.join(' vs '.join(p) for p in e['pairs'])}; arrived or changed: {', '.join(e['entries_added'])}" for e in hist["replay"][k]["events"]) or "- none"
    tot = hist["totals"]
    md = f"""# TIME ("It's TIME", arXiv 2602.12147): leaderboard audit

Files: results dataset Real-TSF/TIME-Output at commit 3bf103c (2026-10-31), leaderboard code from the space at commit d83188a. Plan: boards/PREREG_BOARDS.md. Code: audit_time.py. Numbers: boards/time_report.json, boards/time_units.csv. Statistics run with vernier_board.py and vernier_stats.py unchanged. Resolution = the smallest true difference between two neighbours the test would detect 80% of the time. All geometric-mean rules are analysed on log scale (a gap of 0.002 is about 0.2%).

## Summary

{table}

Reading it: the displayed rule orders the top ten by gaps of about {f(r0['resolution']['median_adjacent_gap'],2)} in log units, while the smallest difference it could resolve is about {f(r0['resolution']['paper'],2)} (paper) to {f(r0['resolution']['corrected_icc'],2)} (corrected), so {r0['resolution']['ratio_paper']:.1f} to {r0['resolution']['ratio_corrected_icc']:.1f} times larger than a typical neighbour gap. At 39 datasets, a typical neighbour gap would need about {r0['resolution']['sources_needed_boot']:.0f} datasets to be resolved. Under the paper rule {r0['counts']['paper']['adjacent']} of 9 adjacent pairs and {r0['counts']['paper']['all']} of 45 top-ten pairs are separated (uncorrected; {r0['counts']['paper']['all_holm']} after Holm); under CR2-icc {r0['counts']['CR2-icc']['adjacent']} of 9 and {r0['counts']['CR2-icc']['all']} of 45 ({r0['counts']['CR2-icc']['all_holm']} after Holm). Leading tier (entries not separated from the displayed leader by an uncorrected test): {r0['leading_tier']['paper']['size']} under the paper rule ({', '.join(r0['leading_tier']['paper']['members'])}), {r0['leading_tier']['CR2-icc']['size']} under CR2-icc, {r0['leading_tier']['wild']['size']} under the wild bootstrap. 
Uncorrected pairs of 45 (paper / CR2-icc / wild): MASE geometric mean {allc('mase_gmean')}; MASE_rank {allc('mase_rank')}; CRPS geometric mean {allc('crps_gmean')}; CRPS_rank {allc('crps_rank')}.

Known-shift control (resample datasets, add a shift equal to the stated resolution to an adjacent pair, 1000 repeats; false-alarm = shift of zero), MASE geometric mean: at the paper resolution the detection rates are paper {kt('mase_gmean')['paper']['at_paper_resolution']:.0%}, CR2-icc {kt('mase_gmean')['CR2-icc']['at_paper_resolution']:.0%}, wild {kt('mase_gmean')['wild']['at_paper_resolution']:.0%}; at the corrected resolution {kt('mase_gmean')['paper']['at_corrected_resolution']:.0%}, {kt('mase_gmean')['CR2-icc']['at_corrected_resolution']:.0%}, {kt('mase_gmean')['wild']['at_corrected_resolution']:.0%}. False alarms at zero shift: paper {kt('mase_gmean')['paper']['false_alarm_delta0']:.1%}, CR2-icc {kt('mase_gmean')['CR2-icc']['false_alarm_delta0']:.1%}, wild {kt('mase_gmean')['wild']['false_alarm_delta0']:.1%}. The paper-rule false-alarm rate above 5% means that rule is too eager on this board; CR2-icc keeps it near 5% but then detects the shift less often at the paper resolution (the reason the corrected resolution is larger).

## What the board shows

- {rest['n_entries']} entries are shown (all model folders, including seasonal_naive, which is the normaliser and sits last at 1.000). The board's code has no exclusion, flag or leakage column; every folder with results is shown (searched time_lb/src for leak, flag, exclude, hide).
- Default sort: MASE (norm.) ascending: for each (dataset/frequency, horizon) the mean of the MASE array (all series, windows, variates) divided by seasonal_naive's, then the geometric mean over the {rest['n_units']} units. MASE_rank and CRPS_rank are mean ranks over the same units (rank method "first"; no exact ties exist in MASE or CRPS, so row order does not matter).
- Units: {rest['n_units']} (dataset/frequency, horizon) combinations. Sources: {rest['n_datasets']} dataset names (first path component). Counting dataset and frequency together gives {rest['n_dataset_freq']} (the TIME paper says 50 datasets; its count is by dataset and frequency).
- NaN cells in the normalised tables: {rest['nan_cells']}.

## Reproduction check

- Board code run on the pinned files: the displayed order of all {rest['n_entries']} entries from my log-mean equals the board function's order: {rp['order_matches_board']} (CRPS order: {rp['crps_order_matches_board']}). Geometric mean from log-mean vs scipy gmean: max difference {rp['max_abs_diff_gmean_vs_scipy']}; rounded value difference {rp['max_abs_diff_rounded_value']}; rank columns match to {rp['max_abs_diff_rank_cols']:.1e} (display rounding).
- Top ten (pinned files): {', '.join(rp['top10'])}.
- Hart-36M shipped summaries: full_results.csv has {hc['units_in_shipped_csv']} rows, {hc['units_matched']} matched my units; max difference in MASE {hc['max_abs_diff_MASE']}, CRPS {hc['max_abs_diff_CRPS']}, MAE {hc['max_abs_diff_MAE']} (exact). average_results.csv MASE {hc['average_results_MASE']:.6f} vs my plain mean over units {hc['mean_MASE_mine']:.6f}; CRPS {hc['average_results_CRPS']:.6f} vs {hc['mean_CRPS_mine']:.6f}. That file is the plain average of raw scores, not the board's displayed number (Hart-36M's displayed MASE (norm.) is {hc['geo_mean_norm_MASE_mine']:.3f}).
- Live board: the HuggingFace space config, fetched 2026-11-11 (about 14:40 UTC), embeds a table of {lv['entries_live']} entries, {lv['entries_pinned']} on the pinned files. The extra entries are {', '.join(lv['only_live'])}, not in the pinned files; presumably added to the results repository after the pinned commit (the dataset's last-modified time was 2026-11-11 14:06 UTC; I did not fetch those files, and the space's table is whatever it computed when it last started). **Top ten does not match**: live has {', '.join(lv['live_top10'])}; pinned has {', '.join(lv['pinned_top10'])}; the difference is Simons-297M at place 8. For the {lv['entries_pinned']} shared entries, order is identical: {lv['order_of_shared_identical']}, and MASE (norm.) and CRPS (norm.) agree to the displayed 3 decimals (max difference {lv['max_abs_diff_MASE_norm_shared']}, {lv['max_abs_diff_CRPS_norm_shared']}). Rank columns differ as expected because the pool differs. Reported, not fixed.

## Single-dataset dominance

For each of the 45 top-ten pairs, the share of the variance estimate (sum over datasets of S_g squared, S_g = the dataset's summed deviation from the pair mean) coming from the largest single dataset:

| Rule | Median share | Maximum share | Pairs with one dataset above one half | Pairs above one third |
|---|---|---|---|---|
""" + "\n".join(f"| {names[r]} | {dm[r]['median']:.2f} | {dm[r]['max']:.2f} | {dm[r]['n_above_half']} of 45 | {dm[r]['n_above_third']} of 45 |" for r in RULES) + f"""

## Pool dependence

- Displayed geometric-mean rule: each entry's aggregate computed from the whole pool equals the aggregate from that entry and seasonal_naive alone (max difference {rest['pool']['mase_gmean']['max_abs_diff_pool_vs_alone']}; CRPS {rest['pool']['crps_gmean']['max_abs_diff_pool_vs_alone']}). Removing any of the 32 entries outside the top ten and adding a near-copy (score times 1+1e-6) of any of the 42 non-baseline entries reorders the top ten in {rest['pool']['mase_gmean']['removal']['changed']} and {rest['pool']['mase_gmean']['near_copy']['changed']} cases. Seasonal_naive itself is not in the perturbed pool: it is the fixed normaliser, and if it were removed the board's code skips normalisation and averages raw MASE instead.
- Mean-rank columns (pool dependent): MASE_rank: removal {rest['pool']['mase_rank']['removal']['changed']} of {rest['pool']['mase_rank']['removal']['of']}, near-copy {rest['pool']['mase_rank']['near_copy']['changed']} of {rest['pool']['mase_rank']['near_copy']['of']} (near-copies of: {', '.join(rest['pool']['mase_rank']['near_copy']['entries'])}). CRPS_rank: removal {rest['pool']['crps_rank']['removal']['changed']} of {rest['pool']['crps_rank']['removal']['of']} (removing {', '.join(rest['pool']['crps_rank']['removal']['entries']) or 'none'}), near-copy {rest['pool']['crps_rank']['near_copy']['changed']} of {rest['pool']['crps_rank']['near_copy']['of']} (near-copies of: {', '.join(rest['pool']['crps_rank']['near_copy']['entries'])}). Here the pool includes seasonal_naive as an ordinary entry, as the board's ranks do.

## History replay

The results repository has {hist['n_commits']} commits that touch results/. A state is the set of model folders with metrics.npz at that commit; a model's content hash is the hash of its sorted metrics.npz LFS identifiers (sha256 of the file; I also checked that the sha256 of all 4214 local files equals the identifiers at the head commit: {hist['local_files_vs_head_oid_mismatches']} mismatches, {hist['non_pointer_blobs']} files that were not LFS pointers). Scores come from the current files, so a model can be scored at a commit only if all {rest['n_units']} of its files at that commit are identical to the head files.

- Model-states (model present at a commit): {tot['model_states']}; scored {tot['scored']}; could not be scored {tot['not_at_head']+tot['changed_vs_head']+tot['partial']} ({tot['changed_vs_head']} content differs from the head, {tot['not_at_head']} folder name no longer exists at head, {tot['partial']} incomplete).
- Models whose content changed while present: {', '.join(hist['models_whose_content_changed_while_present'])}. Names that no longer exist at head: {', '.join(hist['models_not_at_head'])}.
- seasonal_naive itself changed on 2026-06-02 (the Crypto/D and Global_Influenza/W refresh), so the geometric-mean rule cannot be computed for the first 16 commits (2026-02-09 to 2026-06-02 before c43961bb), and no state before that has even ten scoreable entries. {hist['n_replayable_states']} of {hist['n_commits']} states are replayable, giving {hist['replay']['mase_gmean']['transitions']} transitions (2026-06-02 to 2026-10-31). The replayed pool is smaller than the board's real pool at the time wherever models could not be scored (22 of 22 present at 2026-06-02 start vs 13 scored), which matters for the mean-rank columns.

| Rule | Transitions examined | Transitions with a reversal | Reversed pairs among top-ten entries that did not change |
|---|---|---|---|
""" + "\n".join(f"| {names[r]} | {hist['replay'][r]['transitions']} | {hist['replay'][r]['transitions_with_reversal']} | {hist['replay'][r]['reversed_pairs']} |" for r in RULES) + f"""

MASE_rank events (date of previous state -> date of new state; pair order is previous order):
{ev('mase_rank')}

CRPS_rank events:
{ev('crps_rank')}

Geometric-mean rules: no reversals in either metric (as expected, no entry's aggregate depends on the others).

## Overlap with GIFT-Eval

GIFT-Eval has {len(ov['gift'])} dataset names ({', '.join(ov['gift'])}). TIME's {len(ov['time'])} datasets: {', '.join(ov['time'])}.

- Sharing a data source (names, aliases, providers named in the TIME paper's appendix B): none found. TIME was built from fresh data; its names do not match any GIFT-Eval name.
- Uncertain (same topic as a GIFT-Eval dataset, provider appears different; the TIME paper names the providers, I did not inspect raw data): """ + "; ".join(f"{k} ~ {v}" for k, v in ov["uncertain"].items()) + f""".
- Because no definite overlap exists, no overlap rerun was required. As a sensitivity, the primary report was rerun with the 8 uncertain datasets removed ({rep_nov['n_units']} units, {rep_nov['n_sources']} sources; displayed order recomputed on those units; top ten there: {', '.join(rep_nov['top10'])}): resolution {f(rep_nov['resolution']['paper'])} / {f(rep_nov['resolution']['corrected_icc'])}, median gap {f(rep_nov['resolution']['median_adjacent_gap'])}, adjacent separated {rep_nov['counts']['paper']['adjacent']} / {rep_nov['counts']['CR2-icc']['adjacent']} / {rep_nov['counts']['wild']['adjacent']}, all 45 after Holm {rep_nov['counts']['paper']['all_holm']} / {rep_nov['counts']['CR2-icc']['all_holm']} / {rep_nov['counts']['wild']['all_holm']} (paper / CR2-icc / wild).

## Deviations and things that went wrong

1. The board's loader (utils.load_time_results) refuses to read a unit unless predictions.npz exists. Predictions (3.9 GB) were not downloaded, so I ran the board's own get_all_datasets_results, compute_ranks, normalize_by_seasonal_naive and get_overall_leaderboard unchanged, after replacing only that loader by a version that reads metrics.npz alone. The repository lists predictions.npz for all 4214 units at the head commit, so the live board should have the same set. Packages not installed here (gradio, timebench) and src.about/src.hf_config (which download data at import) were replaced by empty stand-ins; none of their code is used in the reduction.
2. Row order of the board's table comes from os.listdir on this machine; it only affects rank ties, and there are none.
3. Seasonal_naive was left out of the perturbed pool in the geometric-mean pool test (reason above); in the history replay and in the mean-rank pool test it is an ordinary entry.
4. History replay: the first 16 states cannot be scored (see above), and unscoreable model-states are dropped from the replayed pool, so the mean-rank replay is not the board's table as displayed at the time.
5. Live table: pinned top ten differs from live by the four entries added later (see reproduction check).
6. vernier_board.py (shared, not modified) defines _display_order, pool_tests, history_replay and mean_rank_rule twice in a row and has one unreachable line after mean_rank_rule's return; harmless, but worth cleaning.
7. Secondary rules use the rank columns as the board computes them (ties impossible here) and are ordered by their own mean; the board's default order is the geometric mean.
8. Shared-code check required by the plan: test_vernier_board.py was run after the board analysis and before reporting (about 8 minutes; the plan says it should come first, so this is a deviation in order) and printed ALL OK (GIFT-Eval: 0 of 9 adjacent, 12 of 45, 2 after Holm, tier of 3, 28 of 121 removals, 42 of 128 near-copies, replay 141 / 18 / 14). Its paper resolution came out 6.209 against the published 6.16, inside the test's tolerance of 0.1; the test regenerates gift_mase_rank_report.json, which I restored from a copy afterwards.
"""
    open(os.path.join(HERE, "BOARD_TIME.md"), "w").write(md)
    full = dict(rest=rest, history=hist, overlap=ov, reports={**reps, "mase_gmean__uncertain_overlap_removed": rep_nov})
    json.dump(full, open(os.path.join(B, "time_report.json"), "w"))
    print("written")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "units": stage_units()
    elif st == "overlap": stage_overlap()
    elif st == "history": stage_history()
    elif st == "final": stage_final()
    elif st == "rest": stage_rest()
    elif st == "report": stage_report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
