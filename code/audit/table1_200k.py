#!/usr/bin/env python3
"""
Table 1 refresh at B=200,000, plus a CRPS_Rank Holm check at B=100,000.

Pure stdlib. Imports run_audit.py and critic_checks.py unchanged and reuses
their machinery instead of reimplementing any of it:
  - ra.load_all_results / ra.build_pipeline / ra.load_config_info /
    ra.merge_display / ra.sort_leak_then / ra.pair_win_rate  (run_audit.py)
  - cc.compute_pairwise_stats / cc.load_all_results_crps / cc.bootstrap_pvalue
    / cc.holm_reject  (critic_checks.py)
compute_pairwise_stats already does the expensive part once per model per
scheme (not once per pair), which is what makes B=200,000 over 45 pairs
finish in about a minute instead of ~9x longer -- see critic_checks.py's own
module docstring for why.

Two computations:
  1. Table 1 refresh. For the 9 adjacent pairs in the displayed top-10
     (MASE_Rank ordering, snapshot 2026-09-07, seed 20260907): point gap,
     independent-scheme 95% CI, clustered-scheme 95% CI, gmean point gap and
     win rate 95% CI, all at B=200,000, plus the headline both-schemes
     survival counts over all 45 top-10 pairs (needs all 45, not just the 9
     adjacent ones, so run_audit_check() below computes 45 pairs once and
     the 9-row table is a filter of that).
  2. CRPS_Rank Holm. For the 45 top-10 pairs under CRPS_Rank ordering,
     bootstrap p-values at B=100,000 (p = max over the two resampling
     schemes of each scheme's own 2*min(frac<=0, frac>=0)), then Holm at 5%
     over the 45-pair family.

Determinism: every draw goes through a random.Random(seed) with a fixed
seed (ra.SEED = 20260907 for computation 1's rank/gmean/win-rate bootstraps,
cc.SEED_MAIN = 20260907 for computation 2's CRPS bootstrap -- same value,
kept as the two upstream modules' own names rather than a third constant).
Nothing here iterates a bare Python set() before writing output -- ra's own
merge_display/sort_leak_then already sort before returning, and every loop
below walks a fixed list (top10, the 45 index pairs in a fixed order, the 9
adjacent pairs). Run twice and diff the md5 of table1_200k.json to confirm.

Usage: python3 table1_200k.py [snapshot_root]
Output: table1_200k.json, TABLE1_200K.md (in this script's directory)
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_audit as ra          # load_all_results, build_pipeline, load_config_info,
                                 # merge_display, sort_leak_then, pair_win_rate, SEED
import critic_checks as cc      # compute_pairwise_stats, load_all_results_crps,
                                 # bootstrap_pvalue, holm_reject, SEED_MAIN, B_MED, HOLM_ALPHA

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths below are for this zip's layout (code/audit/, code/selection/,
# snapshot/snapshot_2026-09-07/ all under one root) -- see README.txt.
DEFAULT_SNAPSHOT_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "snapshot", "snapshot_2026-09-07"))

B_RANK_GMEAN = 200000   # computation 1: rank/gmean CIs (measured ~62s for all 45 pairs)
B_WIN_RATE = 200000     # computation 1: win-rate CIs, 9 pairs only (measured ~2.2s/pair -> ~20s)
B_HOLM = cc.B_MED        # computation 2: CRPS_Rank bootstrap p-values (100,000, reused from critic_checks.py)
SEED = ra.SEED           # 20260907, same seed run_audit.py and critic_checks.py's SEED_MAIN use

# Critic's reference numbers (a prior independent run at B=200,000), quoted in the task brief,
# used only as a sanity check -- not asserted exactly, since a different seed gives a slightly
# different draw of the same near-zero bound (the statistic moves in steps of 1/97 =~ 0.0103).
CRITIC_REF = {
    "pair_1_2_independent": [0.00, 8.49],
    "pair_9_10_independent": [-0.05, 2.71],
    "pair_9_10_clustered": [0.13, 2.41],
}
CRITIC_EXTERNAL_CLAIM_CRPS_HOLM_SURVIVORS = 0
EXPECTED_RAW_BOTH_SCHEMES_CRPS_ALL45 = 16   # must match critic_checks.json's claim1 figure


# --------------------------------------------------------------------------
# LaTeX \phantom-aligned formatting, matching paper/main.tex's Table 1 style:
# sign is always either "$-$" or "\phantom{$-$}" so the digits line up, and
# within one column (across all 9 rows) any value with fewer integer digits
# than the column's widest gets left-padded with "\phantom{1...}" so the
# decimal points line up too. Both paddings are computed from the actual
# data, not hardcoded, so this keeps working if 200k moves a value onto a
# different digit-width than the B=4000 table happened to have.
# --------------------------------------------------------------------------

def _int_digits(x):
    return len(f"{abs(x):.0f}")


def fmt_phantom(val, decimals, int_width, needs_sign):
    """needs_sign: whether THIS column (across all 9 rows) mixes negative and
    non-negative values -- only then does a non-negative entry need
    \\phantom{$-$} to line up under the real "$-$" signs elsewhere in the same
    column (matches paper/main.tex's existing Table 1: e.g. the independent-CI
    upper-bound column is all positive there, so it never gets a sign phantom
    at all, while the lower-bound column mixes signs and does)."""
    if needs_sign:
        sign = "$-$" if val < 0 else "\\phantom{$-$}"
    else:
        sign = "$-$" if val < 0 else ""  # column has no negative entries anywhere
    s = f"{abs(val):.{decimals}f}"
    int_part = s.split(".")[0]
    pad = int_width - len(int_part)
    phantom = "\\phantom{" + "1" * pad + "}" if pad > 0 else ""
    return f"{sign}{phantom}{s}"


def latex_table1_rows(rows):
    """rows: the 9 dicts built in main(). Returns the ready-to-paste tabular
    body (one line per pair, matching \\begin{tabular}{lrrrrr} in main.tex)."""
    lo_iid = [r["independent_ci"][0] for r in rows]
    hi_iid = [r["independent_ci"][1] for r in rows]
    lo_cl = [r["clustered_ci"][0] for r in rows]
    hi_cl = [r["clustered_ci"][1] for r in rows]
    dgmean_vals = [r["delta_gmean"] for r in rows]

    lo_iid_w, hi_iid_w = max(_int_digits(v) for v in lo_iid), max(_int_digits(v) for v in hi_iid)
    lo_cl_w, hi_cl_w = max(_int_digits(v) for v in lo_cl), max(_int_digits(v) for v in hi_cl)
    lo_iid_sign, hi_iid_sign = any(v < 0 for v in lo_iid), any(v < 0 for v in hi_iid)
    lo_cl_sign, hi_cl_sign = any(v < 0 for v in lo_cl), any(v < 0 for v in hi_cl)
    dgmean_sign = any(v < 0 for v in dgmean_vals)
    pair_w = max(len(r["pair"]) for r in rows)

    lines = []
    for r in rows:
        pair = r["pair"].ljust(pair_w)
        dmase = f"{r['delta_mase_rank']:.2f}"
        iid_lo = fmt_phantom(r["independent_ci"][0], 2, lo_iid_w, lo_iid_sign)
        iid_hi = fmt_phantom(r["independent_ci"][1], 2, hi_iid_w, hi_iid_sign)
        cl_lo = fmt_phantom(r["clustered_ci"][0], 2, lo_cl_w, lo_cl_sign)
        cl_hi = fmt_phantom(r["clustered_ci"][1], 2, hi_cl_w, hi_cl_sign)
        dgmean = fmt_phantom(r["delta_gmean"], 4, 1, dgmean_sign)
        wr = f"{r['win_rate']:.2f} [{r['win_rate_ci'][0]:.2f}, {r['win_rate_ci'][1]:.2f}]"
        lines.append(
            f"{pair} & {dmase} & [{iid_lo}, {iid_hi}] & [{cl_lo}, {cl_hi}] & {dgmean} & {wr} \\\\"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAPSHOT_ROOT
    warnings = []

    # ---- shared MASE pipeline, same as run_audit.py / critic_checks.py ----
    rows, load_warnings = ra.load_all_results(root)
    warnings.extend(load_warnings)
    canonical_keys, model_data, model_agg = ra.build_pipeline(rows, warnings)
    by_folder, by_model_field, dup_model_fields = ra.load_config_info(root)
    displayed, dropped_results_only, dropped_config_only = ra.merge_display(model_agg, by_model_field)
    top10_records = ra.sort_leak_then(displayed, "MASE_Rank")[:10]
    top10 = [d["model"] for d in top10_records]

    # Self-check: reproduce audit_results.json's published top-10 before
    # trusting anything built on top of it (same pattern critic_checks.py uses).
    with open(os.path.join(OUT_DIR, "audit_results.json")) as f:
        published = json.load(f)
    published_top10 = [d["model"] for d in published["displayed_top10"]]
    assert top10 == published_top10, f"top-10 pipeline mismatch: {top10} != {published_top10}"

    # ======================================================================
    # Computation 1: Table 1 refresh at B=200,000
    # ======================================================================
    pairs_mase_200k = cc.compute_pairwise_stats(
        top10, model_data, model_agg, canonical_keys, warnings, B_RANK_GMEAN, SEED
    )  # all 45 top-10 pairs, both MASE_Rank and gmean, both schemes

    def both_schemes_counts(pairs, agg_name):
        adj = sum(1 for p in pairs if p["posB"] == p["posA"] + 1
                   and p[agg_name]["iid"]["survives"] and p[agg_name]["cluster"]["survives"])
        all45 = sum(1 for p in pairs if p[agg_name]["iid"]["survives"] and p[agg_name]["cluster"]["survives"])
        return adj, all45

    mase_adj, mase_all45 = both_schemes_counts(pairs_mase_200k, "MASE_Rank")
    gmean_adj, gmean_all45 = both_schemes_counts(pairs_mase_200k, "gmean")

    headline_counts = {
        "mase_rank_both_schemes_adjacent_of_9": mase_adj,
        "mase_rank_both_schemes_all_of_45": mase_all45,
        "gmean_both_schemes_adjacent_of_9": gmean_adj,
        "gmean_both_schemes_all_of_45": gmean_all45,
        "expected": {"mase_adjacent": 0, "mase_all45": 12, "gmean_adjacent": 2, "gmean_all45": 23},
        "unchanged_at_200k": (mase_adj, mase_all45, gmean_adj, gmean_all45) == (0, 12, 2, 23),
    }
    assert headline_counts["unchanged_at_200k"], f"headline counts moved at B=200000: {headline_counts}"

    adjacent_pairs = [p for p in pairs_mase_200k if p["posB"] == p["posA"] + 1]
    assert len(adjacent_pairs) == 9

    # Win-rate CIs, 9 adjacent pairs only (Table 1 does not show win rate for
    # the other 36 pairs), same B and same seed family as the rank/gmean CIs.
    rng_wr = random.Random(SEED)
    orig_B = ra.B
    ra.B = B_WIN_RATE
    try:
        win_by_pair = {}
        for p in adjacent_pairs:
            wr = ra.pair_win_rate(p["modelA"], p["modelB"], model_data, canonical_keys, rng_wr)
            win_by_pair[(p["posA"], p["posB"])] = wr
    finally:
        ra.B = orig_B

    table1_rows = []
    for p in adjacent_pairs:
        wr = win_by_pair[(p["posA"], p["posB"])]
        table1_rows.append({
            "pair": f"{p['posA']}-{p['posB']}",
            "modelA": p["modelA"],
            "modelB": p["modelB"],
            "delta_mase_rank": p["MASE_Rank"]["point_gap"],
            "independent_ci": p["MASE_Rank"]["iid"]["ci"],
            "independent_survives": p["MASE_Rank"]["iid"]["survives"],
            "clustered_ci": p["MASE_Rank"]["cluster"]["ci"],
            "clustered_survives": p["MASE_Rank"]["cluster"]["survives"],
            "delta_gmean": p["gmean"]["point_gap"],
            "win_rate": wr["win_rate"],
            "win_rate_ci": wr["ci"],
        })

    pair12 = next(r for r in table1_rows if r["pair"] == "1-2")
    pair910 = next(r for r in table1_rows if r["pair"] == "9-10")
    endpoint_checks = {
        "pair_1_2_independent_ours": [round(pair12["independent_ci"][0], 2), round(pair12["independent_ci"][1], 2)],
        "pair_1_2_independent_critic_ref": CRITIC_REF["pair_1_2_independent"],
        "pair_9_10_independent_ours": [round(pair910["independent_ci"][0], 2), round(pair910["independent_ci"][1], 2)],
        "pair_9_10_independent_critic_ref": CRITIC_REF["pair_9_10_independent"],
        "pair_9_10_clustered_ours": [round(pair910["clustered_ci"][0], 2), round(pair910["clustered_ci"][1], 2)],
        "pair_9_10_clustered_critic_ref": CRITIC_REF["pair_9_10_clustered"],
        "note": ("Sanity-check only, not asserted exact: the critic's run used its own seed, and "
                 "MASE_Rank moves in steps of 1/97 =~ 0.0103, so a near-zero bound can land a step "
                 "either side between runs (paper text: the pair-1-2 independent lower bound 'sits at "
                 "zero' and the pair-9-10 clustered lower bound 'holds near 0.13 across three seeds')."),
    }

    latex_body = latex_table1_rows(table1_rows)

    computation1 = {
        "seed": SEED,
        "B_rank_gmean": B_RANK_GMEAN,
        "B_win_rate": B_WIN_RATE,
        "top10_mase_rank_order": top10,
        "table1_rows": table1_rows,
        "headline_counts": headline_counts,
        "endpoint_sanity_checks": endpoint_checks,
        "latex_table1_body": latex_body,
    }

    # ======================================================================
    # Computation 2: CRPS_Rank Holm at B=100,000
    # ======================================================================
    rows_c, load_warnings_c = cc.load_all_results_crps(root)
    warnings_c = list(load_warnings_c)
    canonical_keys_c, model_data_c, model_agg_c = ra.build_pipeline(rows_c, warnings_c)
    assert len(canonical_keys_c) == ra.EXPECTED_CONFIGS
    displayed_c, _, _ = ra.merge_display(model_agg_c, by_model_field)
    top10_crps_records = ra.sort_leak_then(displayed_c, "MASE_Rank")[:10]  # "MASE_Rank" field holds CRPS_Rank here
    top10_crps = [d["model"] for d in top10_crps_records]

    pairs_crps = cc.compute_pairwise_stats(
        top10_crps, model_data_c, model_agg_c, canonical_keys_c, warnings_c, B_HOLM, cc.SEED_MAIN
    )  # all 45 pairs
    assert len(pairs_crps) == 45

    raw_both_schemes_all45 = sum(
        1 for p in pairs_crps if p["MASE_Rank"]["iid"]["survives"] and p["MASE_Rank"]["cluster"]["survives"]
    )
    assert raw_both_schemes_all45 == EXPECTED_RAW_BOTH_SCHEMES_CRPS_ALL45, (
        f"raw both-schemes CRPS count {raw_both_schemes_all45} != critic_checks.json's {EXPECTED_RAW_BOTH_SCHEMES_CRPS_ALL45}"
    )

    labeled_pvals = [
        (f"{p['posA']}-{p['posB']}", max(p["MASE_Rank"]["iid"]["p_value"], p["MASE_Rank"]["cluster"]["p_value"]))
        for p in pairs_crps
    ]
    raw_reject = {label: p <= cc.HOLM_ALPHA for label, p in labeled_pvals}
    holm_reject = cc.holm_reject(labeled_pvals, cc.HOLM_ALPHA)
    holm_survivor_count = sum(holm_reject.values())

    computation2 = {
        "seed": cc.SEED_MAIN,
        "B_holm": B_HOLM,
        "holm_alpha": cc.HOLM_ALPHA,
        "top10_crps_rank_order": top10_crps,
        "raw_significant_count_of_45": sum(raw_reject.values()),
        "raw_both_schemes_survive_all45": raw_both_schemes_all45,
        "raw_both_schemes_matches_critic_checks_json": raw_both_schemes_all45 == EXPECTED_RAW_BOTH_SCHEMES_CRPS_ALL45,
        "holm_survivor_count_of_45": holm_survivor_count,
        "critic_external_claim_holm_survivors": CRITIC_EXTERNAL_CLAIM_CRPS_HOLM_SURVIVORS,
        "matches_critic_external_claim": holm_survivor_count == CRITIC_EXTERNAL_CLAIM_CRPS_HOLM_SURVIVORS,
        "per_pair_p_both": {label: p for label, p in labeled_pvals},
        "holm_reject": holm_reject,
    }

    # ======================================================================
    # Assemble + write
    # ======================================================================
    def clean(x):
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    out = {
        "meta": {
            "snapshot_commit": ra.SNAPSHOT_COMMIT,
            "snapshot_root": os.path.abspath(root),
            "seed": SEED,
        },
        "computation1_table1_at_200k": computation1,
        "computation2_crps_holm_at_100k": computation2,
        "warnings_mase_pipeline": warnings,
        "warnings_crps_pipeline": warnings_c,
    }

    json_path = os.path.join(OUT_DIR, "table1_200k.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False, default=clean)
        f.write("\n")
    print(f"Wrote {json_path}")

    write_markdown(out, os.path.join(OUT_DIR, "TABLE1_200K.md"))
    print(f"Wrote {os.path.join(OUT_DIR, 'TABLE1_200K.md')}")

    if warnings or warnings_c:
        print(f"\n{len(warnings)} MASE-pipeline warning(s), {len(warnings_c)} CRPS-pipeline warning(s)", file=sys.stderr)


def write_markdown(out, path):
    c1 = out["computation1_table1_at_200k"]
    c2 = out["computation2_crps_holm_at_100k"]
    hc = c1["headline_counts"]
    ec = c1["endpoint_sanity_checks"]

    lines = []
    lines.append(f"# Table 1 refresh at B={c1['B_rank_gmean']:,} (snapshot commit `{out['meta']['snapshot_commit']}`)\n")
    lines.append(f"Seed {c1['seed']} for computation 1 (rank/gmean/win-rate); seed {c2['seed']} for computation 2 "
                  f"(CRPS_Rank, matches critic_checks.py's SEED_MAIN -- same value, different name upstream). "
                  "Deterministic: run twice and md5 table1_200k.json to confirm.\n")

    lines.append("## 1. The nine adjacent gaps, recomputed at B=%d (win rate at B=%d)\n" % (c1["B_rank_gmean"], c1["B_win_rate"]))
    lines.append("| Pair | Δ MASE_Rank | independent 95% | clustered 95% | Δ gmean | win rate (95%) |")
    lines.append("|---|---|---|---|---|---|")
    for r in c1["table1_rows"]:
        lines.append(
            f"| {r['pair']} | {r['delta_mase_rank']:.2f} | "
            f"[{r['independent_ci'][0]:.2f}, {r['independent_ci'][1]:.2f}] | "
            f"[{r['clustered_ci'][0]:.2f}, {r['clustered_ci'][1]:.2f}] | "
            f"{r['delta_gmean']:.4f} | {r['win_rate']:.2f} [{r['win_rate_ci'][0]:.2f}, {r['win_rate_ci'][1]:.2f}] |"
        )
    lines.append("")

    lines.append("## Headline counts, unchanged at 200k?\n")
    lines.append(f"- MASE_Rank both schemes: {hc['mase_rank_both_schemes_adjacent_of_9']} of 9 adjacent "
                 f"(paper: 0), {hc['mase_rank_both_schemes_all_of_45']} of 45 all pairs (paper: 12).")
    lines.append(f"- gmean both schemes: {hc['gmean_both_schemes_adjacent_of_9']} of 9 adjacent "
                 f"(paper: 2), {hc['gmean_both_schemes_all_of_45']} of 45 all pairs (paper: 23).")
    lines.append(f"- **Unchanged at 200k: {hc['unchanged_at_200k']}**\n")

    lines.append("## Endpoint sanity checks vs. the critic's B=200,000 run\n")
    lines.append(f"- Pair 1-2 independent: ours {ec['pair_1_2_independent_ours']}, critic {ec['pair_1_2_independent_critic_ref']}")
    lines.append(f"- Pair 9-10 independent: ours {ec['pair_9_10_independent_ours']}, critic {ec['pair_9_10_independent_critic_ref']}")
    lines.append(f"- Pair 9-10 clustered: ours {ec['pair_9_10_clustered_ours']}, critic {ec['pair_9_10_clustered_critic_ref']}")
    lines.append(f"- {ec['note']}\n")

    lines.append("## Ready-to-paste LaTeX table body\n")
    lines.append("```latex")
    lines.append(c1["latex_table1_body"])
    lines.append("```\n")

    lines.append(f"## 2. CRPS_Rank Holm correction at B={c2['B_holm']:,}\n")
    lines.append(f"- Top-10 by CRPS_Rank: {c2['top10_crps_rank_order']}")
    lines.append(f"- Raw both-schemes survivors (95% CI excludes zero under iid AND cluster): "
                 f"{c2['raw_both_schemes_survive_all45']}/45 "
                 f"(matches critic_checks.json's cached figure: {c2['raw_both_schemes_matches_critic_checks_json']})")
    lines.append(f"- Raw significant (p_both <= 0.05, uncorrected): {c2['raw_significant_count_of_45']}/45")
    lines.append(f"- **Holm-corrected survivors: {c2['holm_survivor_count_of_45']}/45** "
                 f"(critic's external claim: {c2['critic_external_claim_holm_survivors']}/45, "
                 f"matches: {c2['matches_critic_external_claim']})\n")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
