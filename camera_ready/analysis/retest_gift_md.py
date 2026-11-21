"""Writes RETEST_GIFT.md from retest_gift.json (no new computation)."""
import json, numpy as np
H = "fmts/camera_ready"
R = json.load(open(f"{H}/retest_gift.json"))
L = []; w = L.append
def tab(head, rows):
    w("| " + " | ".join(head) + " |"); w("|" + "---|" * len(head))
    for r in rows: w("| " + " | ".join(str(x) for x in r) + " |")
    w("")
M = ["P-both", "P-clu", "CR2-icc", "CR2-indep", "CR1", "WCB-Rad", "Means-t"]
RN = ["MASE_Rank", "CRPS_Rank", "gmean"]
w("# GIFT-Eval re-test under small-sample-corrected methods\n")
w("Pre-registered in DESIGN.md (Experiment A2, Revision 2 items 5 and 7). Script: `retest_gift.py` (numbers, `retest_gift.json`) and `retest_gift_md.py` (this file). Base seed 20261111. Runtime %.0f s.\n" % R["runtime_s"])
w("Deviation: see the end of this file.\n")
w("## Step 0. Regression check against the published counts\n")
w("Shows whether the paper's rule (percentile bootstrap, iid and cluster both exclude zero; B=100,000) reproduces the published counts on the displayed top-10.\n")
rows = []
for r in RN:
    g = R["regression"]["per_rule"][r]; e, o = g["expected"], g["got"]
    rows.append([r, f"{e['adj_unc']} / {o['adj_unc']}", f"{e['all_unc']} / {o['all_unc']}", f"{e['all_holm45']} / {o['all_holm45']}", "exact" if g["match"] else f"Holm differs by 1 (nearest pair {g['nearest_to_holm_threshold']['pair']}: p={g['nearest_to_holm_threshold']['p']:.5f} vs Holm threshold {g['nearest_to_holm_threshold']['thr']:.5f}, Monte-Carlo SE of p {g['nearest_to_holm_threshold']['mc_se_of_p']:.5f})"])
tab(["rule", "adjacent of 9 (published / ours)", "pairs of 45 (published / ours)", "Holm of 45 (published / ours)", "status"], rows)
w("The gmean pair 1-5 sits 3e-5 from its Holm threshold, inside the bootstrap's own noise. Every uncorrected count matches exactly. gmean pairs here use the displayed MASE_Rank order and log ratios (as `load_gift_pairs.py` provides); the paper's gmean bootstrap may have been on the ratio scale, which could account for the one-pair difference (not checked).\n")
w("## Step 1. Per-pair numbers\n")
w("All 45 pairs per rule (theta, estimated within-dataset correlation, skewness of the 28 dataset means, p-values and 95% intervals under every method) are in `retest_gift.json` under `pairs`. The dataset-means t-test weights datasets equally, so it tests a slightly different quantity than the config-weighted mean. Summary of the shape of the data:\n")
rows = []
for r in RN:
    rr = R["pairs"][r]
    rows.append([r, "%.2f - %.2f" % (min(x["icc_hat"] for x in rr), max(x["icc_hat"] for x in rr)), "%.2f" % np.median([x["icc_hat"] for x in rr]), "%.2f" % np.median([x["skew_dataset_means"] for x in rr]), "%.1f" % np.median([x["CR2-icc"]["df"] for x in rr])])
tab(["rule", "icc range over 45 pairs", "median icc", "median skewness of dataset means", "median CR2 df (icc)"], rows)
w("## Step 2. Counts of separated pairs\n")
w("Shows how many displayed-top-10 pairs each method separates at 5%, without correction and after Holm over the 45 pairs (adjacent = 9 neighbouring pairs; all = 45).\n")
for r in RN:
    w(f"**{r}**\n")
    c = R["counts"][r]
    tab(["method", "adjacent unc. (of 9)", "all unc. (of 45)", "adjacent Holm", "all Holm (of 45)", "adjacent pairs separated (uncorrected)"],
        [[m, c[m]["adj_unc"], c[m]["all_unc"], c[m]["adj_holm"], c[m]["all_holm"], ", ".join(c[m]["adj_unc_positions"]) or "none"] for m in M])
    wv = R["counts_webb_variant"][r]
    w(f"WCB with Webb weights: adjacent {wv['adj_unc']}, all {wv['all_unc']}, Holm adjacent {wv['adj_holm']}, Holm all {wv['all_holm']}.\n")
w("Pair at positions 9-10 under MASE_Rank (mean rank difference %.2f places):\n" % R["pair_9_10_MASE"]["theta"])
p = R["pair_9_10_MASE"]
tab(["method", "p-value", "separated at 5%"], [["P-iid (paper)", "%.4f" % p["P-iid_p"], p["P-iid_p"] < .05], ["P-clu (paper)", "%.4f" % p["P-cluster_p"], p["P-cluster_p"] < .05], ["P-both (paper rule)", "%.4f" % p["P-both"]["p"], p["P-both"]["excl0"]]] +
    [[m, "%.4f" % p[m]["p"], p[m]["excl0"]] for m in ("CR2-icc", "CR2-indep", "CR1", "WCB-Rad", "WCB-Webb", "Means-t")])
w("## Step 3. Resolution\n")
w("Shows the smallest gap each method could detect with 80% power, the median gap between displayed neighbours, and the number of datasets needed to resolve that gap. Paper = 2.80 x cluster-bootstrap SE (B=4000, resolution_budget.py's exact resamples). Corrected = (t_0.975 + t_0.80 at Satterthwaite df) x CR2 SE, median over the 9 adjacent pairs. Units: rank places (MASE_Rank, CRPS_Rank), log scaled-MASE (gmean). gmean uses the gmean's own displayed order, as resolution_budget.py does.\n")
rows = []
for r, lab in (("MASE_Rank", "MASE_Rank"), ("CRPS_Rank", "CRPS_Rank"), ("gmean_own", "gmean (own order)")):
    s = R["resolution"][r]["summary"]; f = "%.4g"
    rows.append([lab, f % s["paper_res"], f % s["corr_res_icc"], f % s["corr_res_indep"], f % s["median_gap"], "%.1f / %.1f" % (s["ratio_paper"], s["ratio_icc"]),
                 "%.0f (%.0fx)" % (s["Gstar_boot"], s["Gstar_boot_x"]), "%.0f (%.0fx)" % (s["Gstar_cr2"], s["Gstar_cr2_x"]), "%.3f" % s["median_ratio_se"]])
tab(["rule", "paper resolution", "corrected (df=icc)", "corrected (df=indep)", "median adjacent gap", "resolution / gap (paper / corrected)", "datasets needed, paper SE (multiple of 28)", "datasets needed, CR2 SE", "median se_cr2 / bootstrap SE"], rows)
w("Per-pair se_cr2 / bootstrap SE ratios (9 adjacent pairs, in displayed order):\n")
tab(["rule"] + [f"{i}-{i+1}" for i in range(1, 10)], [[r] + ["%.3f" % x["ratio_se"] for x in R["resolution"][r]["per_pair"]] for r in ("MASE_Rank", "CRPS_Rank", "gmean_own")])
w("## Step 4. Leading tier\n")
w("Shows which of the next 39 displayed entries are not separated from the displayed leader by uncorrected tests (leak-flagged entries pushed down, complete entries only, as resolution_budget.py). Not a confidence set for the best entry.\n")
for r in ("MASE_Rank", "CRPS_Rank", "gmean_own"):
    t = R["tiers"][r]
    w(f"**{r}** (leader: {t['tier']['P-both (paper B=4000)'][0]})\n")
    tab(["method", "tier size", "members"], [[k, len(v), ", ".join(v)] for k, v in t["tier"].items()])
w("## Step 5. Coarser sources (19 clusters)\n")
c5 = R["coarse"]
w("Shows the same counts and resolution when related datasets share one cluster. Datasets in the snapshot (28): " + ", ".join(c5["dataset_names"]) + ". Merged: m4_* (6) -> m4; ett1, ett2 -> ett; bizitobs_* (3) -> bizitobs; bitbrains_* (2) -> bitbrains. Sources (%d): %s.\n" % (c5["n_sources"], ", ".join(c5["sources"])))
for r in ("MASE_Rank", "gmean"):
    c = c5["counts"][r]
    w(f"**{r}** (19 clusters)\n")
    tab(["method", "adjacent unc.", "all unc.", "adjacent Holm", "all Holm", "adjacent separated"], [[m, c[m]["adj_unc"], c[m]["all_unc"], c[m]["adj_holm"], c[m]["all_holm"], ", ".join(c[m]["adj_unc_positions"]) or "none"] for m in c])
w("Pair 9-10 under MASE_Rank at 19 clusters, p-values: " + ", ".join(f"{k} {v:.4f}" for k, v in c5["pair_9_10_MASE"].items()) + ".\n")
rows = []
for r, lab in (("MASE_Rank", "MASE_Rank"), ("gmean_own", "gmean (own order)")):
    s = c5["resolution"][r]
    rows.append([lab, "%.4g" % s["paper_style_res"], "%.4g" % s["corr_res_icc"], "%.4g" % s["median_gap"], "%.1f" % s["ratio"], "%.0f (%.0fx of 19)" % (s["Gstar_boot"], s["Gstar_boot_x"]), "%.0f (%.0fx of 19)" % (s["Gstar_cr2"], s["Gstar_cr2_x"])])
tab(["rule", "2.80 x bootstrap SE (B=20,000)", "corrected resolution (CR2, icc df)", "median gap", "resolution / gap", "sources needed, bootstrap SE", "sources needed, CR2 SE"], rows)
w("## Step 6. Leave one dataset out\n")
w("Shows how far the corrected resolution (CR2, icc df) and the number of adjacent pairs separated at 5% (CR2-icc, uncorrected) move when each of the 28 datasets is removed in turn. For gmean this uses the gmean's own displayed order.\n")
rows = []
for r in ("MASE_Rank", "gmean_own"):
    x = R["loo"][r]; ch = [f"{d['dropped']} ({d['adj_count']})" for d in x["per_dataset"] if d["adj_count"] != x["full_adj_count_cr2icc"]]
    rows.append([r, "%.4g" % x["full_resolution"], "%.4g - %.4g" % (x["min"], x["max"]), f"{x['most_moving']['dropped']} -> {x['most_moving']['resolution']:.4g}", x["full_adj_count_cr2icc"], "; ".join(ch) or "none"])
tab(["rule", "full-data resolution", "range over 28 drops", "dataset that moves it most", "adjacent count, full data", "drops that change the adjacent count (new count)"], rows)
w("## Step 7. Stability under subsets\n")
w("A stability check, not a test of extrapolation: subsets of a fixed 28 datasets. Shows the median (over 200 random subsets) of SE_CR2 x sqrt(G') relative to its value at 28, MASE_Rank adjacent pairs. A value near 1 means SE shrinks like 1/sqrt(G') within these 28 datasets.\n")
tab(["G'", "median over 9 pairs", "range over pairs"], [[g, "%.3f" % x["median_rel"], "%.2f - %.2f" % (min(x["per_pair_rel"]), max(x["per_pair_rel"]))] for g, x in R["subsets"]["per_Gp"].items()])
# changes
C = R["counts"]; res = R["resolution"]; T = R["tiers"]
w("## What changes relative to the submitted paper\n")
ch = [
 "Counts that stay: paper rule reproduces MASE_Rank 0/9, 12/45, Holm 2/45; CRPS_Rank 0/9, 16/45, Holm 0/45; gmean 2/9, 23/45 (Holm 10 vs published 11, borderline pair 1-5).",
 f"Adjacent pairs separated, MASE_Rank: paper rule 0 -> CR2-icc {C['MASE_Rank']['CR2-icc']['adj_unc']}, WCB {C['MASE_Rank']['WCB-Rad']['adj_unc']} (pair 9-10 only), CR1 {C['MASE_Rank']['CR1']['adj_unc']}, Means-t {C['MASE_Rank']['Means-t']['adj_unc']}.",
 f"Adjacent pairs separated, gmean: paper rule 2 -> CR2-icc {C['gmean']['CR2-icc']['adj_unc']}, WCB {C['gmean']['WCB-Rad']['adj_unc']}, CR1 {C['gmean']['CR1']['adj_unc']}, Means-t {C['gmean']['Means-t']['adj_unc']}.",
 f"Adjacent pairs separated, CRPS_Rank: 0 -> 0 under every corrected method.",
 f"All 45 pairs uncorrected, MASE_Rank: 12 -> CR2-icc {C['MASE_Rank']['CR2-icc']['all_unc']}, WCB {C['MASE_Rank']['WCB-Rad']['all_unc']}, Means-t {C['MASE_Rank']['Means-t']['all_unc']}; CRPS_Rank: 16 -> CR2-icc {C['CRPS_Rank']['CR2-icc']['all_unc']}, WCB {C['CRPS_Rank']['WCB-Rad']['all_unc']}, Means-t {C['CRPS_Rank']['Means-t']['all_unc']}; gmean: 23 -> CR2-icc {C['gmean']['CR2-icc']['all_unc']}, WCB {C['gmean']['WCB-Rad']['all_unc']}, Means-t {C['gmean']['Means-t']['all_unc']}.",
 f"All 45 pairs after Holm: MASE_Rank 2 -> CR2-icc {C['MASE_Rank']['CR2-icc']['all_holm']}, WCB {C['MASE_Rank']['WCB-Rad']['all_holm']}; CRPS_Rank 0 -> 0; gmean 11 (ours 10) -> CR2-icc {C['gmean']['CR2-icc']['all_holm']}, WCB {C['gmean']['WCB-Rad']['all_holm']}.",
 "Resolution (paper -> corrected, df=icc): " + "; ".join(f"{lab} {res[k]['summary']['paper_res']:.4g} -> {res[k]['summary']['corr_res_icc']:.4g}" for k, lab in (("MASE_Rank", "MASE_Rank"), ("CRPS_Rank", "CRPS_Rank"), ("gmean_own", "gmean"))) + ".",
 "Datasets needed (multiple of 28, paper SE -> CR2 SE): " + "; ".join(f"{lab} {res[k]['summary']['Gstar_boot_x']:.0f}x -> {res[k]['summary']['Gstar_cr2_x']:.0f}x" for k, lab in (("MASE_Rank", "MASE_Rank"), ("CRPS_Rank", "CRPS_Rank"), ("gmean_own", "gmean"))) + ".",
 "Leading tier size (paper rule -> CR2-icc / WCB / Means-t): " + "; ".join(f"{k} {T[k]['size']['P-both (paper B=4000)']} -> {T[k]['size']['CR2-icc']} / {T[k]['size']['WCB-Rad']} / {T[k]['size']['Means-t']}" for k in T) + ".",
]
for c in ch: w("- " + c)
w("")
w("## Deviations from DESIGN.md\n")
for d in [
 "Step 0: gmean Holm count is 10, published 11. The pair 1-5 p-value (0.00146) is within Monte-Carlo noise of its Holm threshold (0.00143). The run was allowed to continue on that basis (the instruction said borderline pairs can flip); an earlier run stopped at this point and wrote the same numbers.",
 "gmean has two orders: counts (steps 0-2, 5) use the MASE_Rank top-10 and log ratios from `load_gift_pairs.py`; resolution, tiers, leave-one-out use gmean's own displayed order as in `resolution_budget.py`. The two top-10 lists differ (see JSON `gmean_own_order_top10`, `gmean_pairs_order_top10`).",
 "Step 3/4 data for entries outside the top-10 come from the same audit pipeline as `load_gift_pairs.py` (ties ranked 'first'), restricted to entries with all 97 configs, as resolution_budget.py. Paper-style bootstrap SEs use resolution_budget.py's own resamples (seed 20260907, B=4000, string-sorted configs) and reproduce its published numbers exactly.",
 "Step 4 'P-both (B=100k)' separation is two-sided CI exclusion; the paper-exact line uses resolution_budget.py's one-sided lower-bound test. Tier sizes match.",
 "Step 5 bootstrap SE for the 19-source resolution uses B=20,000 (own seeds), not the paper's 4000.",
 "Step 6 adjacent count is the uncorrected CR2-icc count (9 pairs), with the pair order fixed to the full-data order.",
 "Bootstrap p-values and percentile intervals use the paper's conventions (iid / dataset resampling, shared resamples across pairs, index rule from run_audit.percentile_ci); `vernier_stats.pct_bootstrap` returns no draws so was not used for them.",
]: w("- " + d)
open(f"{H}/RETEST_GIFT.md", "w").write("\n".join(L))
