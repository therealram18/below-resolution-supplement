"""Experiment A3, context-length part (see DESIGN.md). Run: python retest_context.py"""
import json, math, os, random, statistics, sys, time
import numpy as np
from collections import defaultdict
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
AUDIT = os.path.normpath(os.path.join(HERE, "..", "audit")); sys.path.insert(0, AUDIT)
import vernier_stats as vs
import run_audit as ra, result2_bands as rb, result2_grid as rg, critic_checks as cc
t0 = time.time()
root = rg.DEFAULT_SNAPSHOT_ROOT; warnings = []
keys = rb.load_selected_configs(rg.SELECTED_CONFIGS_PATH); strs = ["/".join(k) for k in keys]
rows, w = ra.load_all_results(root); warnings += w
canon, model_data, _ = ra.build_pipeline(rows, warnings)
groups = defaultdict(list)
for r in rows: groups[r["config_key"]].append(r)
per = {}  # model -> ctx -> list of (rank, scaled)
for m in rg.GRID_MODELS:
    per[m] = {}
    for ctx in rg.CONTEXTS:
        data = rg.load_grid_csv(os.path.join(rg.GRID_ROOT, m, rg.ctx_dir(ctx), "all_results.csv"), m)
        per[m][ctx] = [(rg.insertion_rank(data[s], groups[k]), data[s] / model_data["Seasonal_Naive"][k]["mase"]) for k, s in zip(keys, strs)]
CELLS = [(m, a) for m in rg.GRID_MODELS for a in ("rank", "gmean")]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
reps = rb.bootstrap_reps(12, random.Random(rg.PAIRED_DIFF_SEED))            # B=4000, as result2_grid.py
reps_hi = cc.gen_reps(12, 100000, random.Random(rg.PAIRED_DIFF_SEED))        # B=100000 for paper-rule Holm p-values, as critic_checks.py

# ---- Step 0 regression: paper's paired percentile bootstrap, built from result2_grid's own function ----
cells = {}
for m, a in CELLS:
    d0, d1 = per[m]["default"], per[m]["512"]
    fld = "insertion_rank" if a == "rank" else "scaled_mase"
    pd0 = [dict(insertion_rank=x[0], scaled_mase=x[1]) for x in d0]; pd1 = [dict(insertion_rank=x[0], scaled_mase=x[1]) for x in d1]
    pc = rg.paired_diff_ci(pd0, pd1, fld, reps, warnings, f"{m}/{a}")
    # B=100000 p-value, same recipe as critic_checks.py
    v0 = [x[0 if a == "rank" else 1] for x in d0]; v1 = [x[0 if a == "rank" else 1] for x in d1]
    agg = statistics.mean if a == "rank" else gm
    pb = cc.bootstrap_pvalue([agg([v1[i] for i in r]) - agg([v0[i] for i in r]) for r in reps_hi])
    d = np.array([v1[i] - v0[i] for i in range(12)]) if a == "rank" else np.log(v1) - np.log(v0)
    cells[(m, a)] = dict(model=m, agg=a, d=d, point=pc["point_delta"], paper_ci=pc["ci_2.5_97.5"], paper_excl=pc["excludes_zero"], paper_p=pb)
pct = {m: 100 * (gm([x[1] for x in per[m]["512"]]) / gm([x[1] for x in per[m]["default"]]) - 1) for m in rg.GRID_MODELS}

# ---- Step 1: new tests ----
cl12 = np.arange(12)
src = [("bizitobs" if s.startswith("bizitobs") else "ett" if s.startswith(("ett1", "ett2")) else s) for s in strs]
cl9 = np.unique(src, return_inverse=True)[1]
assert len(set(cl9)) == 9
rng = np.random.default_rng(20261111)
for c in cells.values():
    d = c["d"]
    c["signflip_p"] = vs.signflip_exact_p(d)
    t = vs.paired_t_test(d); c["t_p"] = float(t["p"]); c["t_ci"] = [float(x) for x in t["ci95"]]; c["t_mean"] = float(t["theta"])
    c["wild_p"] = vs.wild_cluster_p(d, cl12)["p"]
    c["cr2_9"] = float(vs.cr2_test(d, cl9, df="icc")["p"]); c["wcb_9"] = vs.wild_cluster_p(d, cl9)["p"]
    c["cr2_9_df"] = float(vs.cr2_test(d, cl9, df="icc")["df"])
    if c["agg"] == "gmean":
        c["pct"] = 100 * (math.exp(c["t_mean"]) - 1); c["pct_ci_t"] = [100 * (math.exp(x) - 1) for x in c["t_ci"]]
methods = {"paper": "paper_p", "signflip": "signflip_p", "t": "t_p", "wild": "wild_p", "cr2_9": "cr2_9", "wcb_9": "wcb_9"}
holm = {}
for mth, key in methods.items():
    ps = [cells[c][key] for c in CELLS]; h = vs.holm(ps)
    for c, hh in zip(CELLS, h): cells[c].setdefault("holm", {})[mth] = bool(hh)
    holm[mth] = int(h.sum())
unc = dict(paper=sum(c["paper_excl"] for c in cells.values()), **{k: sum(c[methods[k]] < 0.05 for c in cells.values()) for k in methods if k != "paper"})
# regression
pts = {m: (cells[(m, "rank")]["point"], cells[(m, "gmean")]["point"], pct[m]) for m in rg.GRID_MODELS}
expect = dict(chronos_bolt_base=(6.67, -0.0189, -2.0), moirai2_small=(4.08, 0.0494, 6.8), timesfm25=(26.58, 0.1629, 22.2), toto=(16.58, 0.1675, 20.6))
reg = dict(n_excl=unc["paper"], holm_paper=holm["paper"], excl_set=[f"{m}/{a}" for m, a in CELLS if cells[(m, a)]["paper_excl"]],
           points={m: dict(rank=round(p[0], 2), gmean=round(p[1], 4), pct=round(p[2], 1)) for m, p in pts.items()},
           points_match=all(round(pts[m][0], 2) == e[0] and round(pts[m][1], 4) == e[1] and round(pts[m][2], 1) == e[2] for m, e in expect.items()))
reg["pass"] = reg["n_excl"] == 5 and reg["holm_paper"] == 4 and reg["points_match"] and set(reg["excl_set"]) == {"moirai2_small/gmean", "timesfm25/rank", "timesfm25/gmean", "toto/rank", "toto/gmean"}
out = dict(runtime_s=time.time() - t0, regression=reg, counts_unc=unc, counts_holm=holm, B_paper_ci=4000, B_paper_holm_p=100000,
           cells=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cells[c].items()} for c in CELLS])
out["runtime_s"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "retest_context.json"), "w"), indent=1, default=float)

f = lambda x, k=3: f"{x:.{k}f}"
yn = lambda b: "excludes 0" if b else "includes 0"
rj = lambda b: "yes" if b else "no"
L = ["# Context-length experiment re-test (512 vs default context, 12 configs)", "",
     "Each cell is one model under one aggregation. Per-config difference = value at 512 minus value at default (rank places, or log of scaled MASE). The 12 configs come from 12 different datasets, so each is its own cluster. Sign-flip null: the 12 differences are independent and each symmetric about zero; p is from all 4,096 sign patterns. t-test: paired, 11 df. Wild: studentized, Rademacher, exact over 4,096 patterns (singleton clusters).", "",
     "## 0. Regression check against the submitted paper", "",
     "| quantity | paper | reproduced |", "|---|---|---|",
     f"| movements with paper interval excluding 0 (of 8) | 5 | {reg['n_excl']} ({', '.join(reg['excl_set'])}) |",
     f"| survivors after Holm, paper rule (bootstrap p, B=100,000) | 4 | {reg['holm_paper']} |",
     "| rank movement chronos / moirai / timesfm / toto | +6.67 / +4.08 / +26.58 / +16.58 | " + " / ".join(f"{reg['points'][m]['rank']:+.2f}" for m in rg.GRID_MODELS) + " |",
     "| geometric-mean movement | -0.0189 / +0.0494 / +0.1629 / +0.1675 | " + " / ".join(f"{reg['points'][m]['gmean']:+.4f}" for m in rg.GRID_MODELS) + " |",
     "| percentage change | -2.0 / +6.8 / +22.2 / +20.6 | " + " / ".join(f"{reg['points'][m]['pct']:+.1f}" for m in rg.GRID_MODELS) + " |", "",
     "This table shows whether the submitted numbers come back from the pinned files with the paper's bootstrap (B=4,000, seed 20260914).", "",
     "## Results per cell", "",
     "| cell | point (diff.) | paper 95% interval (B=4,000) | paper | sign-flip p | t-test p | t-test 95% interval | Holm: paper / sign-flip / t / wild |", "|---|---|---|---|---|---|---|---|"]
for c in CELLS:
    x = cells[c]; h = x["holm"]
    L.append(f"| {c[0]} {c[1]} | {f(x['point'],4)} | [{f(x['paper_ci'][0],4)}, {f(x['paper_ci'][1],4)}] | {yn(x['paper_excl'])} | {x['signflip_p']:.4f} | {x['t_p']:.4f} | [{f(x['t_ci'][0],4)}, {f(x['t_ci'][1],4)}] | {rj(h['paper'])} / {rj(h['signflip'])} / {rj(h['t'])} / {rj(h['wild'])} |")
L += ["", "This table shows, for each of the 8 cells, the paper's interval, the exact sign-flip and paired t results, and whether each survives Holm over 8.", "",
      "| cell | wild p (exact) | paper bootstrap p (B=100,000) |", "|---|---|---|"]
for c in CELLS: L.append(f"| {c[0]} {c[1]} | {cells[c]['wild_p']:.4f} | {cells[c]['paper_p']:.4f} |")
L += ["", "This table shows the wild studentized exact p-value and the bootstrap p-value used for the paper's Holm step.", "",
      "| model | % change in geometric mean | t interval on log scale, exponentiated | paper bootstrap interval (pct_change.json) |", "|---|---|---|---|"]
pj = json.load(open(os.path.join(AUDIT, "pct_change.json")))
for m in rg.GRID_MODELS:
    x = cells[(m, "gmean")]; L.append(f"| {m} | {x['pct']:+.1f}% | [{x['pct_ci_t'][0]:+.1f}%, {x['pct_ci_t'][1]:+.1f}%] | [{pj[m]['ci95'][0]:+.1f}%, {pj[m]['ci95'][1]:+.1f}%] |")
L += ["", "This table shows the percentage change in the geometric mean with the corrected and the paper's intervals.", "",
      "## Cells surviving (of 8)", "", "| method | uncorrected (p < 0.05 or interval excludes 0) | after Holm over 8 |", "|---|---|---|"]
names = dict(paper="paper bootstrap", signflip="exact sign-flip", t="paired t-test", wild="wild exact (12 clusters)", cr2_9="CR2 icc, 9 source clusters", wcb_9="wild exact, 9 source clusters")
for k in methods: L.append(f"| {names[k]} | {unc[k]} | {holm[k]} |")
L += ["", "This table counts survivors for each method.", "",
      "## Sensitivity: 9 source clusters (3 bizitobs configs as one, ett1 + ett2 as one)", "",
      "| cell | CR2 (icc df) p | CR2 df | wild exact p (G=9) |", "|---|---|---|---|"]
for c in CELLS: L.append(f"| {c[0]} {c[1]} | {cells[c]['cr2_9']:.4f} | {f(cells[c]['cr2_9_df'],1)} | {cells[c]['wcb_9']:.4f} |")
L += ["", "This table shows the two small-sample-corrected p-values when the 12 configs are collapsed to 9 source clusters.", "",
      "## What changes relative to the submitted paper", "",
      f"- Movements separated, uncorrected: 5 of 8 (percentile bootstrap) -> {unc['signflip']} (sign-flip), {unc['t']} (t-test), {unc['wild']} (wild exact).",
      f"- After Holm: 4 of 8 (bootstrap) -> {holm['signflip']} (sign-flip), {holm['t']} (t-test), {holm['wild']} (wild exact).",
      f"- 9-cluster sensitivity: uncorrected {unc['cr2_9']} (CR2) / {unc['wcb_9']} (wild exact); after Holm {holm['cr2_9']} / {holm['wcb_9']}."]
for m in rg.GRID_MODELS:
    x = cells[(m, "gmean")]; L.append(f"- {m} percentage change interval: [{pj[m]['ci95'][0]:+.1f}%, {pj[m]['ci95'][1]:+.1f}%] -> [{x['pct_ci_t'][0]:+.1f}%, {x['pct_ci_t'][1]:+.1f}%]")
open(os.path.join(HERE, "RETEST_CONTEXT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(dict(reg=reg, unc=unc, holm=holm, rt=out["runtime_s"]), indent=1, default=float))
