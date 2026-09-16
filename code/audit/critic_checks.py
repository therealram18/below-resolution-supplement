#!/usr/bin/env python3
"""
Checks a set of statistical claims an external critic made about the
leaderboard-audit paper (paper/main.tex), against the same pinned snapshot
run_audit.py already validated.

Pure stdlib. Reuses run_audit.py's loading/ranking pipeline by importing it
(same pattern as audit/result2_bands.py and audit/result2_grid.py: import
run_audit as ra, call ra.load_all_results/ra.build_pipeline/ra.merge_display/
ra.sort_leak_then/ra.build_position_arrays/ra.gen_resamples/
ra.agg_over_positions/ra.percentile_ci/ra.gmean_safe instead of
reimplementing any of them). For the Result-2 (context-length grid) claims,
reuses the already-computed, already-verified per-config point values
(insertion_rank / scaled_mase, which do not depend on bootstrap B) sitting in
audit/result2_grid.json and audit/result2_bands.json, rather than re-reading
grid_results_h100/*/all_results.csv a second time -- those two JSONs are
exactly what running result2_grid.py against grid_results_h100/results/
already produced, so reading them is reuse, not re-derivation. Their md5s are
recorded below for provenance, and result2_grid.py's own insertion_rank() is
imported and reused unchanged for the one place a fresh insertion-rank
computation is needed (claim 7).

Performance note: a naive per-pair bootstrap (call ra.pair_resampling_bands
per pair) redoes each of the 10 top-10 models' resampled aggregate B times
for every one of the 45 pairs it appears in (9x redundant work). Instead,
compute_pairwise_stats() below precomputes, once per model per scheme, the
array of B resampled aggregate values, then gets every pair's gap array by a
cheap elementwise subtraction. Same reps (same cache, same RNG draws), ~9x
less work -- this is what makes B=100000 over 45 pairs, and B=200000 with 3
seeds, finish in a couple of minutes instead of ~25.

Determinism: every random draw goes through a random.Random(seed) with an
explicit fixed seed; every set() is sorted before being iterated, printed, or
serialized (Python's set/str-hash iteration order is not guaranteed stable
across runs). Run twice and diff the md5 of critic_checks.json to confirm.

Usage: python3 critic_checks.py [snapshot_root]
Output: critic_checks.json, CRITIC_CHECKS.md (in this script's directory)
"""
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_audit as ra                 # load_all_results, build_pipeline, merge_display, sort_leak_then,
                                        # build_position_arrays, gen_resamples, agg_over_positions,
                                        # percentile_ci, gmean_safe, load_config_info
import result2_bands as rb             # bootstrap_reps, resolve_target_dirs
import result2_grid as r2g             # insertion_rank, GRID_MODELS, GRID_TO_BANDS_LABEL, PAIRED_DIFF_SEED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths below are for this zip's layout (code/audit/, code/selection/,
# snapshot/snapshot_2026-09-07/ all under one root) -- see README.txt.
DEFAULT_SNAPSHOT_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "snapshot", "snapshot_2026-09-07"))
RESULT2_BANDS_PATH = os.path.join(OUT_DIR, "result2_bands.json")
RESULT2_GRID_PATH = os.path.join(OUT_DIR, "result2_grid.json")

SEED_MAIN = 20260907           # same seed run_audit.py uses -- claim 1's "B=4000 seed 20260907 like run_audit"
SEEDS_200K = [20260907, 20260908, 20260909]   # 3 fixed seeds for claim 2's B=200000 stability check
PAIRED_DIFF_SEED = r2g.PAIRED_DIFF_SEED       # 20260914, matches the paper's primary Result-2 test
LOO_SEED_BASE = PAIRED_DIFF_SEED              # leave-one-out reps are keyed off this + dropped config name

B_SMALL = 4000
B_MED = 100000
B_LARGE = 200000
HOLM_ALPHA = 0.05


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# --------------------------------------------------------------------------
# Generic bootstrap helpers
# --------------------------------------------------------------------------

def bootstrap_pvalue(gaps):
    """p = 2*min(frac(diff<=0), frac(diff>=0)), capped at 1 -- as specified
    in the task brief for the Holm-correction claims."""
    n = len(gaps)
    frac_le0 = sum(1 for g in gaps if g <= 0) / n
    frac_ge0 = sum(1 for g in gaps if g >= 0) / n
    return min(1.0, 2 * min(frac_le0, frac_ge0))


def holm_reject(labeled_pvals, alpha=HOLM_ALPHA):
    """labeled_pvals: list of (label, p). Returns dict label -> rejected(bool),
    standard Holm-Bonferroni step-down within this one family."""
    m = len(labeled_pvals)
    order = sorted(range(m), key=lambda i: labeled_pvals[i][1])
    reject = {label: False for label, _ in labeled_pvals}
    stop = False
    for rank, idx in enumerate(order):
        if stop:
            break
        label, p = labeled_pvals[idx]
        if p <= alpha / (m - rank):
            reject[label] = True
        else:
            stop = True
    return reject


def gen_reps(n, b, rng):
    """Flat with-replacement resampling of n items, b replicates -- used for
    the Result-2 12-config paired bootstraps, where (per result2_bands.py's
    own documented finding) all 12 configs come from distinct base datasets,
    so there is no separate cluster scheme to worry about."""
    return [rng.choices(range(n), k=n) for _ in range(b)]


# --------------------------------------------------------------------------
# CRPS loading (mirrors ra.load_all_results exactly, only the value column
# differs -- kept as a separate function because ra.load_all_results has no
# column parameter and run_audit.py must not be modified). The output rows
# store the CRPS value under the "mase" key on purpose, so ra.build_pipeline
# -- which only ever reads r["mase"] generically -- ranks on CRPS unchanged.
# --------------------------------------------------------------------------

CRPS_COLUMN = "eval_metrics/mean_weighted_sum_quantile_loss"


def load_all_results_crps(root):
    rows = []
    warnings = []
    results_dir = os.path.join(root, "results")
    for entry in sorted(os.listdir(results_dir)):
        subdir = os.path.join(results_dir, entry)
        csv_path = os.path.join(subdir, "all_results.csv")
        if not os.path.isdir(subdir) or not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                model = r["model"]
                dataset_raw = r["dataset"]
                try:
                    val = float(r[CRPS_COLUMN])
                except (TypeError, ValueError):
                    warnings.append(f"non-numeric CRPS in {csv_path}: {r.get(CRPS_COLUMN)!r}")
                    continue
                parts = dataset_raw.split("/")
                if len(parts) != 3:
                    warnings.append(f"dataset field does not split into 3 parts: {dataset_raw!r} ({csv_path})")
                    continue
                base_dataset, frequency, term_length = parts
                if val <= 0 or math.isnan(val):
                    warnings.append(f"non-positive/NaN CRPS={val} for model={model} dataset={dataset_raw}")
                rows.append({
                    "model": model,
                    "dataset_raw": dataset_raw,
                    "config_key": (base_dataset, frequency, term_length),
                    "mase": val,               # CRPS value, stored under "mase" so ra.build_pipeline works as-is
                    "source_dir": entry,
                })
    rows.sort(key=lambda r: (r["model"], r["dataset_raw"]))
    return rows, warnings


# --------------------------------------------------------------------------
# Rank-method variant of ra.build_pipeline (claim 8: 'first' vs 'average')
# --------------------------------------------------------------------------

def build_pipeline_rank_method(rows, warnings, method):
    """Same as ra.build_pipeline, except the per-config rank assignment uses
    pandas-style rank(method=method): 'first' (stable, sequential -- what
    ra.build_pipeline already does) or 'average' (tied values share the mean
    of the rank positions they occupy). Only elements that are part of a tied
    group get a different rank under 'average'; everyone else is unaffected
    (their position among the sorted values doesn't change just because two
    OTHER entries tied)."""
    baseline = {}
    for r in rows:
        if r["model"] == "Seasonal_Naive":
            if r["config_key"] in baseline:
                warnings.append(f"duplicate Seasonal_Naive row for config {r['config_key']}")
            baseline[r["config_key"]] = r["mase"]
    canonical_keys = sorted(baseline.keys())

    groups = defaultdict(list)
    for r in rows:
        groups[r["config_key"]].append(r)

    model_data = defaultdict(dict)
    for key, group_rows in groups.items():
        base_val = baseline.get(key)
        order = sorted(range(len(group_rows)), key=lambda i: group_rows[i]["mase"])
        rank_of_idx = {}
        if method == "first":
            for pos, idx in enumerate(order):
                rank_of_idx[idx] = pos + 1
        elif method == "average":
            n = len(order)
            i = 0
            while i < n:
                j = i
                while j + 1 < n and group_rows[order[j + 1]]["mase"] == group_rows[order[i]]["mase"]:
                    j += 1
                avg_rank = ((i + 1) + (j + 1)) / 2.0
                for k in range(i, j + 1):
                    rank_of_idx[order[k]] = avg_rank
                i = j + 1
        else:
            raise ValueError(method)
        for idx, r in enumerate(group_rows):
            norm = None
            if base_val is None or base_val <= 0:
                warnings.append(f"no usable Seasonal_Naive baseline for config {key}; norm undefined for model {r['model']}")
            else:
                norm = r["mase"] / base_val
            model_data[r["model"]][key] = {"mase": r["mase"], "rank": rank_of_idx[idx], "norm": norm}

    model_agg = {}
    for model, cfgs in model_data.items():
        ranks = [c["rank"] for c in cfgs.values()]
        norms = [c["norm"] for c in cfgs.values()]
        model_agg[model] = {
            "MASE_Rank": statistics.mean(ranks),
            "gmean": ra.gmean_safe(norms, warnings, f"model={model}"),
            "num_configs": len(cfgs),
        }
    return canonical_keys, model_data, model_agg


# --------------------------------------------------------------------------
# Precomputed pairwise bootstrap: rank/gmean CI + p-value for every (i<j)
# pair among a given ordered top-10 list, at a given B and seed. See module
# docstring for why this precomputes per-model instead of per-pair.
# --------------------------------------------------------------------------

def compute_pairwise_stats(top10_models, model_data, model_agg, canonical_keys, warnings, b, seed, pairs=None):
    orig_B = ra.B
    ra.B = b
    try:
        rng = random.Random(seed)
        gen_cache = {}     # (scheme, tuple(usable)) -> list of B position-lists
        pos_arrays = {}
        for m in top10_models:
            ranks, norms, _, present = ra.build_position_arrays(m, model_data, canonical_keys)
            pos_arrays[m] = {"rank": ranks, "gmean": norms, "usable": present}
            if len(present) != ra.EXPECTED_CONFIGS:
                warnings.append(f"{m}: only {len(present)}/{ra.EXPECTED_CONFIGS} configs usable in compute_pairwise_stats")

        agg_cache = {}      # (model, scheme, kind) -> list of B aggregated values

        def get_agg(model, scheme, kind):
            key = (model, scheme, kind)
            if key in agg_cache:
                return agg_cache[key]
            usable = pos_arrays[model]["usable"]
            cache_key = (scheme, tuple(usable))
            if cache_key not in gen_cache:
                gen_cache[cache_key] = ra.gen_resamples(usable, scheme, canonical_keys, rng)
            reps = gen_cache[cache_key]
            arr = pos_arrays[model][kind]
            vals = [ra.agg_over_positions(arr, positions, kind, warnings, f"{model}/{kind}/{scheme}") for positions in reps]
            agg_cache[key] = vals
            return vals

        n = len(top10_models)
        index_pairs = pairs if pairs is not None else [(i, j) for i in range(n) for j in range(i + 1, n)]

        results = []
        for i, j in index_pairs:
            mA, mB = top10_models[i], top10_models[j]
            pair_res = {"posA": i + 1, "posB": j + 1, "modelA": mA, "modelB": mB}
            for agg_name, kind in [("MASE_Rank", "rank"), ("gmean", "gmean")]:
                point_gap = model_agg[mB][agg_name] - model_agg[mA][agg_name]
                sub = {"point_gap": point_gap}
                for scheme in ("iid", "cluster"):
                    vA = get_agg(mA, scheme, kind)
                    vB = get_agg(mB, scheme, kind)
                    gaps = [b_val - a_val for a_val, b_val in zip(vA, vB)]
                    lo, hi = ra.percentile_ci(gaps)
                    sub[scheme] = {
                        "ci": [lo, hi],
                        "survives": not (lo <= 0 <= hi),
                        "p_value": bootstrap_pvalue(gaps),
                    }
                pair_res[agg_name] = sub
            results.append(pair_res)
        return results
    finally:
        ra.B = orig_B


def find_pair(pair_results, model_a, model_b):
    for p in pair_results:
        if {p["modelA"], p["modelB"]} == {model_a, model_b}:
            return p
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAPSHOT_ROOT
    warnings = []
    out = {"meta": {
        "snapshot_commit": ra.SNAPSHOT_COMMIT,
        "snapshot_root": os.path.abspath(root),
        "seed_main": SEED_MAIN,
        "seeds_200k": SEEDS_200K,
        "paired_diff_seed": PAIRED_DIFF_SEED,
        "B_small": B_SMALL, "B_med": B_MED, "B_large": B_LARGE,
        "holm_alpha": HOLM_ALPHA,
    }}

    # ---- shared MASE pipeline (same as run_audit.py / result2_bands.py) ----
    rows, load_warnings = ra.load_all_results(root)
    warnings.extend(load_warnings)
    canonical_keys, model_data, model_agg = ra.build_pipeline(rows, warnings)
    by_folder, by_model_field, dup_model_fields = ra.load_config_info(root)
    displayed, dropped_results_only, dropped_config_only = ra.merge_display(model_agg, by_model_field)
    top10_mase_records = ra.sort_leak_then(displayed, "MASE_Rank")[:10]
    top10_mase = [d["model"] for d in top10_mase_records]

    # Self-check: reproduce audit_results.json's already-published top-10
    # exactly, before trusting anything built on top of this pipeline.
    with open(os.path.join(OUT_DIR, "audit_results.json")) as f:
        published = json.load(f)
    published_top10 = [d["model"] for d in published["displayed_top10"]]
    assert top10_mase == published_top10, f"pipeline mismatch: {top10_mase} != {published_top10}"
    for mine, pub in zip(top10_mase_records, published["displayed_top10"]):
        assert abs(mine["MASE_Rank"] - pub["MASE_Rank"]) < 1e-9
        assert abs(mine["gmean"] - pub["gmean"]) < 1e-9

    # Self-check: build_pipeline_rank_method(method='first') must reproduce
    # ra.build_pipeline's model_agg exactly (needed before trusting the
    # 'average' variant used in claim 8).
    _, _, model_agg_first_check = build_pipeline_rank_method(rows, [], "first")
    for m in model_agg:
        assert abs(model_agg_first_check[m]["MASE_Rank"] - model_agg[m]["MASE_Rank"]) < 1e-9, m
        assert abs(model_agg_first_check[m]["gmean"] - model_agg[m]["gmean"]) < 1e-9, m

    # ==================================================================
    # Claim 1: CRPS_Rank audit
    # ==================================================================
    rows_c, load_warnings_c = load_all_results_crps(root)
    warnings_c = list(load_warnings_c)
    canonical_keys_c, model_data_c, model_agg_c = ra.build_pipeline(rows_c, warnings_c)
    assert len(canonical_keys_c) == ra.EXPECTED_CONFIGS
    displayed_c, dropped_results_only_c, dropped_config_only_c = ra.merge_display(model_agg_c, by_model_field)
    top10_crps_records = ra.sort_leak_then(displayed_c, "MASE_Rank")[:10]   # "MASE_Rank" field here holds CRPS_Rank
    top10_crps = [d["model"] for d in top10_crps_records]

    same_set = set(top10_mase) == set(top10_crps)
    caststar_pos = top10_crps.index("CastStar") + 1 if "CastStar" in top10_crps else None
    timehedge_pos = top10_crps.index("TIMEHEDGE") + 1 if "TIMEHEDGE" in top10_crps else None

    pairs_crps_b4000 = compute_pairwise_stats(top10_crps, model_data_c, model_agg_c, canonical_keys_c, warnings_c, B_SMALL, SEED_MAIN)
    pairs_crps_b100000 = compute_pairwise_stats(top10_crps, model_data_c, model_agg_c, canonical_keys_c, warnings_c, B_MED, SEED_MAIN)

    def both_schemes_counts(pair_results, agg_name="MASE_Rank"):
        adj = sum(1 for p in pair_results if p["posB"] == p["posA"] + 1 and p[agg_name]["iid"]["survives"] and p[agg_name]["cluster"]["survives"])
        all45 = sum(1 for p in pair_results if p[agg_name]["iid"]["survives"] and p[agg_name]["cluster"]["survives"])
        return adj, all45

    crps_both_adj_b4000, crps_both_all_b4000 = both_schemes_counts(pairs_crps_b4000)
    crps_both_adj_b100000, crps_both_all_b100000 = both_schemes_counts(pairs_crps_b100000)

    ls_pair_b100000 = find_pair(pairs_crps_b100000, "LS-MoE", "LS-Agent")
    stride_exaone_pair_b100000 = find_pair(pairs_crps_b100000, "STRIDE w/ Synapse", "EXAONE-Forecast-Agent")

    claim1 = {
        "top10_crps_rank": top10_crps,
        "top10_mase_rank": top10_mase,
        "same_top10_set": same_set,
        "caststar_position_under_crps": caststar_pos,
        "timehedge_position_under_crps": timehedge_pos,
        "caststar_4th_claim_matches": caststar_pos == 4,
        "timehedge_6th_claim_matches": timehedge_pos == 6,
        "both_schemes_survive_adjacent_B4000": crps_both_adj_b4000,
        "both_schemes_survive_all45_B4000": crps_both_all_b4000,
        "both_schemes_survive_adjacent_B100000": crps_both_adj_b100000,
        "both_schemes_survive_all45_B100000": crps_both_all_b100000,
        "exactly_1_of_9_both_schemes_at_B100000": crps_both_adj_b100000 == 1,
        "ls_moe_vs_ls_agent_pair_B100000": ls_pair_b100000,
        "stride_vs_exaone_pair_B100000": stride_exaone_pair_b100000,
        "all_45_pairs_B4000": pairs_crps_b4000,
        "all_45_pairs_B100000": pairs_crps_b100000,
    }

    # ==================================================================
    # Claim 2: B=200,000 stability of the two MASE_Rank single-scheme
    # survivors -- pair 1-2 (iid) and pair 9-10 (cluster).
    # ==================================================================
    claim2_per_seed = []
    for seed in SEEDS_200K:
        pairs = compute_pairwise_stats(top10_mase, model_data, model_agg, canonical_keys, warnings, B_LARGE, seed, pairs=[(0, 1), (8, 9)])
        p12 = find_pair(pairs, top10_mase[0], top10_mase[1])
        p910 = find_pair(pairs, top10_mase[8], top10_mase[9])
        claim2_per_seed.append({
            "seed": seed,
            "pair_1_2": p12,
            "pair_9_10": p910,
        })

    claim2 = {
        "paper_B4000_pair_1_2_iid_lower_bound": 0.04,
        "paper_B4000_pair_9_10_cluster_lower_bound": 0.14,
        "critic_B200000_pair_1_2_iid_lower_bound_claim": 0.0,
        "per_seed": claim2_per_seed,
        "all_seeds_pair_1_2_iid_lower_bounds": [s["pair_1_2"]["MASE_Rank"]["iid"]["ci"][0] for s in claim2_per_seed],
        "all_seeds_pair_9_10_cluster_lower_bounds": [s["pair_9_10"]["MASE_Rank"]["cluster"]["ci"][0] for s in claim2_per_seed],
        "pair_1_2_iid_survives_all_seeds": all(s["pair_1_2"]["MASE_Rank"]["iid"]["survives"] for s in claim2_per_seed),
        "pair_9_10_cluster_survives_all_seeds": all(s["pair_9_10"]["MASE_Rank"]["cluster"]["survives"] for s in claim2_per_seed),
    }

    # ==================================================================
    # Claim 3: Holm correction
    # ==================================================================
    # (a)+(b): MASE_Rank and gmean, 45 top-10 pairs, both-schemes p = max(p_iid, p_cluster)
    pairs_mase_b100000 = compute_pairwise_stats(top10_mase, model_data, model_agg, canonical_keys, warnings, B_MED, SEED_MAIN)

    def holm_family(pair_results, agg_name):
        labeled = []
        for p in pair_results:
            label = f"{p['posA']}-{p['posB']}"
            p_both = max(p[agg_name]["iid"]["p_value"], p[agg_name]["cluster"]["p_value"])
            labeled.append((label, p_both))
        raw_reject = {label: p <= HOLM_ALPHA for label, p in labeled}
        holm = holm_reject(labeled, HOLM_ALPHA)
        return {
            "n_pairs": len(labeled),
            "raw_significant_count": sum(raw_reject.values()),
            "holm_significant_count": sum(holm.values()),
            "per_pair_p_both": {label: p for label, p in labeled},
            "raw_reject": raw_reject,
            "holm_reject": holm,
        }

    claim3a = holm_family(pairs_mase_b100000, "MASE_Rank")
    claim3b = holm_family(pairs_mase_b100000, "gmean")

    # (c): Result-2's 8 movements, paired bootstrap over the 12 configs, B=100000
    with open(RESULT2_GRID_PATH) as f:
        result2_grid = json.load(f)
    with open(RESULT2_BANDS_PATH) as f:
        result2_bands = json.load(f)
    result2_grid_md5 = md5_file(RESULT2_GRID_PATH)
    result2_bands_md5 = md5_file(RESULT2_BANDS_PATH)

    def paired_diff_stats(vals_default, vals_512, is_rank, reps, warn_list, context):
        def agg(vals):
            return statistics.mean(vals) if is_rank else ra.gmean_safe(vals, warn_list, context)
        point = agg(vals_512) - agg(vals_default)
        diffs = []
        for rep in reps:
            d_def = agg([vals_default[i] for i in rep])
            d_512 = agg([vals_512[i] for i in rep])
            diffs.append(d_512 - d_def)
        lo, hi = ra.percentile_ci(diffs)
        return {
            "point_delta": point,
            "ci_2.5_97.5": [lo, hi],
            "excludes_zero": not (lo <= 0 <= hi),
            "p_value": bootstrap_pvalue(diffs),
        }

    reps_12_b100000 = gen_reps(r2g.N_SELECTED, B_MED, random.Random(PAIRED_DIFF_SEED))
    result2_movements = []
    for model in r2g.GRID_MODELS:
        mct = result2_grid["model_context_table"][model]
        per_default = mct["default"]["per_config"]
        per_512 = mct["512"]["per_config"]
        for agg_name, field, is_rank in [("mean_insertion_rank", "insertion_rank", True), ("gmean_scaled_mase", "scaled_mase", False)]:
            vals_default = [p[field] for p in per_default]
            vals_512 = [p[field] for p in per_512]
            stats = paired_diff_stats(vals_default, vals_512, is_rank, reps_12_b100000, warnings, f"{model}/{agg_name}/B100000")
            result2_movements.append({"model": model, "aggregation": agg_name, **stats})

    labeled_c = [(f"{m['model']}/{m['aggregation']}", m["p_value"]) for m in result2_movements]
    raw_reject_c = {label: p <= HOLM_ALPHA for label, p in labeled_c}
    holm_c = holm_reject(labeled_c, HOLM_ALPHA)
    claim3c = {
        "n_movements": len(labeled_c),
        "raw_significant_count": sum(raw_reject_c.values()),
        "holm_significant_count": sum(holm_c.values()),
        "movements": result2_movements,
        "raw_reject": raw_reject_c,
        "holm_reject": holm_c,
        "moirai_gmean_p_value": next(m["p_value"] for m in result2_movements if m["model"] == "moirai2_small" and m["aggregation"] == "gmean_scaled_mase"),
        "moirai_gmean_holm_significant": holm_c["moirai2_small/gmean_scaled_mase"],
        "result2_grid_json_md5": result2_grid_md5,
        "result2_bands_json_md5": result2_bands_md5,
    }

    claim3 = {"a_mase_rank_top10_pairs": claim3a, "b_gmean_top10_pairs": claim3b, "c_result2_movements": claim3c}

    # ==================================================================
    # Claim 4: leave-one-out on the PRIMARY paired interval, Moirai gmean
    # ==================================================================
    moirai_mct = result2_grid["model_context_table"]["moirai2_small"]
    moirai_default = moirai_mct["default"]["per_config"]
    moirai_512 = moirai_mct["512"]["per_config"]
    n_cfg = len(moirai_default)
    assert n_cfg == r2g.N_SELECTED

    loo_results = []
    for drop_idx in range(n_cfg):
        dropped_config = moirai_default[drop_idx]["config"]
        keep = [i for i in range(n_cfg) if i != drop_idx]
        vals_default_11 = [moirai_default[i]["scaled_mase"] for i in keep]
        vals_512_11 = [moirai_512[i]["scaled_mase"] for i in keep]
        reps11 = gen_reps(n_cfg - 1, B_SMALL, random.Random(f"{LOO_SEED_BASE}:{dropped_config}"))
        stats = paired_diff_stats(vals_default_11, vals_512_11, False, reps11, warnings, f"moirai_loo_drop_{dropped_config}")
        loo_results.append({"dropped_config": dropped_config, **stats})

    highlight_configs = ["bizitobs_service/10S/short", "ett2/15T/long", "loop_seattle/5T/long"]
    # "Primary" = paper's own Table 2 value: B=4000, seed 20260914, straight from result2_grid.json
    # (not the B=100000 re-estimate computed above for claim 3c -- close to it, but the paper's
    # own reported interval is what claim 4 asks to leave-one-out against).
    moirai_grid_delta = result2_grid["deltas_vs_bands"][r2g.GRID_MODELS.index("moirai2_small")]["gmean_scaled_mase"]
    claim4 = {
        "primary_full12_ci": moirai_grid_delta["paired_ci"],
        "primary_full12_ci_source": "result2_grid.json deltas_vs_bands[moirai2_small].gmean_scaled_mase.paired_ci (B=4000, seed 20260914, same as paper Table 2)",
        "leave_one_out": loo_results,
        "n_drops_crossing_zero": sum(1 for r in loo_results if not r["excludes_zero"]),
        "highlighted_drops": {c: next(r for r in loo_results if r["dropped_config"] == c) for c in highlight_configs},
        "highlighted_all_cross_zero": all(not next(r for r in loo_results if r["dropped_config"] == c)["excludes_zero"] for c in highlight_configs),
    }

    # ==================================================================
    # Claim 5: leakage push-down relevance
    # ==================================================================
    top12_raw = sorted(displayed, key=lambda d: d["MASE_Rank"])[:12]
    claim5 = {
        "top12_raw_mase_rank": [
            {"model": d["model"], "MASE_Rank": d["MASE_Rank"], "testdata_leakage": d["testdata_leakage"]}
            for d in top12_raw
        ],
        "all_unflagged": all(d["testdata_leakage"] != "Yes" for d in top12_raw),
    }

    # ==================================================================
    # Claim 6: gmean first-place gap, EXAONE vs STRIDE, B=100000 both schemes
    # ==================================================================
    exaone_stride_pair = find_pair(pairs_mase_b100000, "STRIDE w/ Synapse", "EXAONE-Forecast-Agent")
    claim6 = {
        "pair": exaone_stride_pair,
        "iid_includes_zero": exaone_stride_pair["gmean"]["iid"]["ci"][0] <= 0 <= exaone_stride_pair["gmean"]["iid"]["ci"][1],
        "cluster_includes_zero": exaone_stride_pair["gmean"]["cluster"]["ci"][0] <= 0 <= exaone_stride_pair["gmean"]["cluster"]["ci"][1],
    }

    # ==================================================================
    # Claim 7: Toto like-for-like baseline
    # ==================================================================
    resolved_dirs = rb.resolve_target_dirs(root)
    toto_folder = resolved_dirs["Toto-Open-Base-1.0"]
    selected_keys = rb.load_selected_configs(os.path.normpath(os.path.join(OUT_DIR, "..", "selection", "selected_configs.json")))
    selected_strs = ["/".join(k) for k in selected_keys]

    groups = defaultdict(list)
    for r in rows:
        groups[r["config_key"]].append(r)

    toto_default_by_cfg = {p["config"]: p["mase"] for p in result2_grid["model_context_table"]["toto"]["default"]["per_config"]}

    like_for_like_rows = []
    for key, cfg_str in zip(selected_keys, selected_strs):
        board_mase = model_data[toto_folder][key]["mase"]
        our_mase = toto_default_by_cfg[cfg_str]
        pool_minus_toto = [r for r in groups[key] if r["model"] != toto_folder]
        board_ins_rank = r2g.insertion_rank(board_mase, pool_minus_toto)
        our_ins_rank = r2g.insertion_rank(our_mase, pool_minus_toto)
        like_for_like_rows.append({
            "config": cfg_str, "board_mase": board_mase, "our_default_mase": our_mase,
            "board_insertion_rank_like_for_like": board_ins_rank,
            "our_insertion_rank_like_for_like": our_ins_rank,
        })
    board_avg_ll = statistics.mean(r["board_insertion_rank_like_for_like"] for r in like_for_like_rows)
    our_avg_ll = statistics.mean(r["our_insertion_rank_like_for_like"] for r in like_for_like_rows)

    # num_variates for the 12 selected configs, from Toto's own snapshot row (a per-dataset
    # property, constant across models -- read directly since ra.load_all_results doesn't
    # carry the num_variates column).
    with open(os.path.join(root, "results", toto_folder, "all_results.csv"), newline="") as f:
        toto_csv_rows = {r["dataset"]: r for r in csv.DictReader(f)}
    num_variates_by_cfg = {cfg: int(toto_csv_rows[cfg]["num_variates"]) for cfg in selected_strs}

    toto_repro = result2_grid["reproduction_check"]["toto"]
    over5pct = sorted(p["config"] for p in toto_repro["per_config"] if p["relative_diff"] > 0.05)
    multivariate_cfgs = sorted(cfg for cfg, nv in num_variates_by_cfg.items() if nv > 1)
    critic_claimed_over5pct = sorted(["loop_seattle/5T/long", "bizitobs_application/10S/medium", "bizitobs_l2c/H/long", "jena_weather/10T/medium"])

    board_baseline_rank = result2_bands["target_models"][[t["label"] for t in result2_bands["target_models"]].index("Toto-Open-Base-1.0")]["aggregation_A_mean_insertion_rank"]["point"]
    at512_rank = result2_grid["model_context_table"]["toto"]["512"]["avg_insertion_rank"]
    movement_vs_board_baseline = at512_rank - board_baseline_rank

    claim7 = {
        "a_like_for_like": {
            "per_config": like_for_like_rows,
            "board_avg_like_for_like_insertion_rank": board_avg_ll,
            "our_avg_like_for_like_insertion_rank": our_avg_ll,
            "gap_like_for_like": our_avg_ll - board_avg_ll,
            "original_mismatched_gap": toto_repro["our_default_avg_insertion_rank"] - toto_repro["published_avg_insertion_rank_12cfg"],
            "critic_claimed_gap": 12.00,
        },
        "b_multivariate_check": {
            "over_5pct_reproduction_error_configs": over5pct,
            "critic_claimed_over5pct_configs": critic_claimed_over5pct,
            "over5pct_matches_critic_list": over5pct == critic_claimed_over5pct,
            "num_variates_by_config": num_variates_by_cfg,
            "multivariate_configs_gt1": multivariate_cfgs,
            "over5pct_set_equals_multivariate_set": set(over5pct) == set(multivariate_cfgs),
        },
        "c_board_baseline_movement": {
            "board_published_avg_insertion_rank_12cfg": board_baseline_rank,
            "at_512_avg_insertion_rank": at512_rank,
            "movement_vs_board_baseline": movement_vs_board_baseline,
            "paper_movement_vs_our_default": toto_repro["our_default_avg_insertion_rank"],
            "paper_reported_delta": result2_grid["deltas_vs_bands"][r2g.GRID_MODELS.index("toto")]["mean_insertion_rank"]["delta"],
            "critic_claimed_movement": 28.58,
        },
    }

    # ==================================================================
    # Claim 8: Falcon-X aggregation swing under rank method 'average'
    # ==================================================================
    canonical_keys_avg, model_data_avg, model_agg_avg = build_pipeline_rank_method(rows, warnings, "average")
    displayed_avg, _, _ = ra.merge_display(model_agg_avg, by_model_field)
    # gmean is unaffected by rank method -- sanity check before using it unchanged.
    displayed_by_model = {d["model"]: d for d in displayed}
    for d in displayed_avg:
        assert abs(d["gmean"] - displayed_by_model[d["model"]]["gmean"]) < 1e-9

    posA_avg_of = {d["model"]: i + 1 for i, d in enumerate(ra.sort_leak_then(displayed_avg, "MASE_Rank"))}
    posB_of = {d["model"]: i + 1 for i, d in enumerate(ra.sort_leak_then(displayed_avg, "gmean"))}
    swings_avg = sorted(
        ({"model": m, "rank_MASE_Rank_avg": posA_avg_of[m], "rank_gmean": posB_of[m], "abs_diff": abs(posA_avg_of[m] - posB_of[m])}
         for m in posA_avg_of),
        key=lambda d: -d["abs_diff"],
    )

    falcon_x_avg = next(s for s in swings_avg if s["model"] == "Falcon-X")
    falcon_x_first = next(s for s in published["aggregation_disagreement"]["top3_rank_swings"] if s["model"] == "Falcon-X")

    # `groups` (config_key -> list of rows) was already built above for claim 7; reuse it.
    n_configs_with_ties = sum(
        1 for key in canonical_keys
        if len({r["mase"] for r in groups[key]}) != len(groups[key])
    )

    claim8 = {
        "falcon_x_swing_method_first": falcon_x_first,
        "falcon_x_swing_method_average": falcon_x_avg,
        "twenty_one_becomes_twenty": falcon_x_avg["abs_diff"] == 20 and falcon_x_first["abs_diff"] == 21,
        "top3_swings_method_average": swings_avg[:3],
        "n_configs_with_at_least_one_mase_tie": n_configs_with_ties,
        "n_configs_total": len(canonical_keys),
    }

    # ==================================================================
    # Assemble + write
    # ==================================================================
    out.update({
        "claim1_crps_rank_audit": claim1,
        "claim2_b200000_stability": claim2,
        "claim3_holm_correction": claim3,
        "claim4_leave_one_out_moirai_gmean": claim4,
        "claim5_leakage_pushdown_relevance": claim5,
        "claim6_gmean_first_place_gap": claim6,
        "claim7_toto_like_for_like": claim7,
        "claim8_falcon_x_rank_method": claim8,
        "warnings_mase_pipeline": warnings,
        "warnings_crps_pipeline": warnings_c,
    })

    def clean(x):
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    json_path = os.path.join(OUT_DIR, "critic_checks.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False, default=clean)
        f.write("\n")
    print(f"Wrote {json_path}")

    write_markdown(out, os.path.join(OUT_DIR, "CRITIC_CHECKS.md"))
    print(f"Wrote {os.path.join(OUT_DIR, 'CRITIC_CHECKS.md')}")

    if warnings or warnings_c:
        print(f"\n{len(warnings)} MASE-pipeline warning(s), {len(warnings_c)} CRPS-pipeline warning(s)", file=sys.stderr)


def write_markdown(out, path):
    lines = []
    lines.append("# Critic claim checks\n")
    lines.append(f"Snapshot commit `{out['meta']['snapshot_commit']}`. All bootstraps use fixed seeds "
                  f"(main {out['meta']['seed_main']}; B=200000 stability check {out['meta']['seeds_200k']}; "
                  f"Result-2 paired {out['meta']['paired_diff_seed']}), so this script is deterministic: "
                  "run twice and md5 critic_checks.json to confirm.\n")

    c1 = out["claim1_crps_rank_audit"]
    lines.append("## Claim 1: CRPS_Rank audit\n")
    lines.append(f"- Same top-10 SET as MASE_Rank: **{c1['same_top10_set']}**. CRPS_Rank top-10: {c1['top10_crps_rank']}")
    lines.append(f"- CastStar position under CRPS_Rank: {c1['caststar_position_under_crps']} (critic claims 4th: {c1['caststar_4th_claim_matches']})")
    lines.append(f"- TIMEHEDGE position under CRPS_Rank: {c1['timehedge_position_under_crps']} (critic claims 6th: {c1['timehedge_6th_claim_matches']})")
    lines.append(f"- Both-schemes adjacent survivors at B=4000: {c1['both_schemes_survive_adjacent_B4000']}/9; at B=100000: {c1['both_schemes_survive_adjacent_B100000']}/9")
    lines.append(f"- Both-schemes all-45 survivors at B=4000: {c1['both_schemes_survive_all45_B4000']}/45; at B=100000: {c1['both_schemes_survive_all45_B100000']}/45")
    lm = c1["ls_moe_vs_ls_agent_pair_B100000"]
    lines.append(f"- LS-MoE vs LS-Agent (B=100000): point_gap={lm['MASE_Rank']['point_gap']:.4f}, "
                  f"iid CI={lm['MASE_Rank']['iid']['ci']}, cluster CI={lm['MASE_Rank']['cluster']['ci']}, "
                  f"survives both: {lm['MASE_Rank']['iid']['survives'] and lm['MASE_Rank']['cluster']['survives']}")
    se = c1["stride_vs_exaone_pair_B100000"]
    lines.append(f"- STRIDE vs EXAONE (CRPS_Rank, B=100000): point_gap={se['MASE_Rank']['point_gap']:.4f}, "
                  f"iid CI={se['MASE_Rank']['iid']['ci']} (survives={se['MASE_Rank']['iid']['survives']}), "
                  f"cluster CI={se['MASE_Rank']['cluster']['ci']} (survives={se['MASE_Rank']['cluster']['survives']})\n")

    c2 = out["claim2_b200000_stability"]
    lines.append("## Claim 2: B=200000 stability\n")
    lines.append(f"- Paper (B=4000): pair 1-2 iid lower bound {c2['paper_B4000_pair_1_2_iid_lower_bound']}, "
                  f"pair 9-10 cluster lower bound {c2['paper_B4000_pair_9_10_cluster_lower_bound']}")
    for s in c2["per_seed"]:
        p12 = s["pair_1_2"]["MASE_Rank"]["iid"]
        p910 = s["pair_9_10"]["MASE_Rank"]["cluster"]
        lines.append(f"- seed {s['seed']}: pair 1-2 iid CI={p12['ci']} (survives={p12['survives']}); "
                      f"pair 9-10 cluster CI={p910['ci']} (survives={p910['survives']})")
    lines.append(f"- pair 1-2 iid survives at all 3 seeds: {c2['pair_1_2_iid_survives_all_seeds']}")
    lines.append(f"- pair 9-10 cluster survives at all 3 seeds: {c2['pair_9_10_cluster_survives_all_seeds']}\n")

    c3 = out["claim3_holm_correction"]
    a, b, cc = c3["a_mase_rank_top10_pairs"], c3["b_gmean_top10_pairs"], c3["c_result2_movements"]
    lines.append("## Claim 3: Holm correction\n")
    lines.append(f"- (a) MASE_Rank, 45 pairs: raw {a['raw_significant_count']}/45 -> Holm {a['holm_significant_count']}/45 (critic: 12/45 -> 2/45)")
    lines.append(f"- (b) gmean, 45 pairs: raw {b['raw_significant_count']}/45 -> Holm {b['holm_significant_count']}/45 (critic: 23/45 -> 10/45)")
    lines.append(f"- (c) Result-2, 8 movements: raw {cc['raw_significant_count']}/8 -> Holm {cc['holm_significant_count']}/8 (critic: 5/8 -> 4/8)")
    lines.append(f"  - Moirai gmean p-value: {cc['moirai_gmean_p_value']:.5f} (critic: approx 0.016); Holm-significant: {cc['moirai_gmean_holm_significant']}\n")

    c4 = out["claim4_leave_one_out_moirai_gmean"]
    lines.append("## Claim 4: Leave-one-out, Moirai gmean (primary paired interval)\n")
    lines.append(f"- Full-12 CI: {c4['primary_full12_ci']['ci_2.5_97.5']} (excludes zero: {c4['primary_full12_ci']['excludes_zero']})")
    lines.append(f"- Drops that cross zero: {c4['n_drops_crossing_zero']}/12")
    for name, key in [("bizitobs_service", "bizitobs_service/10S/short"), ("ett2", "ett2/15T/long"), ("loop_seattle", "loop_seattle/5T/long")]:
        r = c4["highlighted_drops"][key]
        lines.append(f"  - drop {name}: CI={r['ci_2.5_97.5']}, excludes zero: {r['excludes_zero']}")
    lines.append(f"- All 3 highlighted drops cross zero: {c4['highlighted_all_cross_zero']}\n")

    c5 = out["claim5_leakage_pushdown_relevance"]
    lines.append("## Claim 5: Leakage push-down relevance\n")
    lines.append(f"- Top-12 by raw MASE_Rank all unflagged: **{c5['all_unflagged']}**")
    lines.append(f"  - {[(d['model'], d['testdata_leakage']) for d in c5['top12_raw_mase_rank']]}\n")

    c6 = out["claim6_gmean_first_place_gap"]
    lines.append("## Claim 6: gmean first-place gap (EXAONE vs STRIDE)\n")
    p = c6["pair"]["gmean"]
    lines.append(f"- point_gap={c6['pair']['gmean']['point_gap']:.4f}, iid CI={p['iid']['ci']} (includes zero: {c6['iid_includes_zero']}), "
                  f"cluster CI={p['cluster']['ci']} (includes zero: {c6['cluster_includes_zero']})\n")

    c7 = out["claim7_toto_like_for_like"]
    lines.append("## Claim 7: Toto like-for-like baseline\n")
    a7 = c7["a_like_for_like"]
    lines.append(f"- (a) Original mismatched gap (insertion-rank ours vs within-pool board): {a7['original_mismatched_gap']:.2f}")
    lines.append(f"  - Like-for-like (both insertion ranks into pool-minus-Toto): board avg {a7['board_avg_like_for_like_insertion_rank']:.2f}, "
                  f"our avg {a7['our_avg_like_for_like_insertion_rank']:.2f}, gap {a7['gap_like_for_like']:.2f} (critic: {a7['critic_claimed_gap']})")
    b7 = c7["b_multivariate_check"]
    lines.append(f"- (b) >5% reproduction-error configs: {b7['over_5pct_reproduction_error_configs']}")
    lines.append(f"  - matches critic's list exactly: {b7['over5pct_matches_critic_list']}")
    lines.append(f"  - num_variates per config: {b7['num_variates_by_config']}")
    lines.append(f"  - multivariate configs (num_variates>1): {b7['multivariate_configs_gt1']}")
    lines.append(f"  - >5%-error set equals multivariate set: {b7['over5pct_set_equals_multivariate_set']}")
    c7c = c7["c_board_baseline_movement"]
    lines.append(f"- (c) Movement using board's published per-config values as baseline: {c7c['movement_vs_board_baseline']:.2f} "
                  f"(paper's movement vs our own default-context baseline: {c7c['paper_reported_delta']:.2f}; critic claims approx {c7c['critic_claimed_movement']})\n")

    c8 = out["claim8_falcon_x_rank_method"]
    lines.append("## Claim 8: Falcon-X swing under rank method 'average'\n")
    lines.append(f"- method='first' (paper): {c8['falcon_x_swing_method_first']}")
    lines.append(f"- method='average': {c8['falcon_x_swing_method_average']}")
    lines.append(f"- 21 becomes 20 under 'average': **{c8['twenty_one_becomes_twenty']}**")
    lines.append(f"- configs with at least one MASE tie: {c8['n_configs_with_at_least_one_mase_tie']}/{c8['n_configs_total']}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
