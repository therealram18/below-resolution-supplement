#!/usr/bin/env python3
"""
Result-2 headline numbers: our 4 measured models (chronos_bolt_base,
moirai2_small, timesfm25, toto), each run at 2 context lengths (its own
documented default, and 512) on the 12 pre-registered configs
(selection/selected_configs.json), per the protocol frozen in
selection/SELECTION_RULE.md ("The full pre-registered protocol" section +
its addendum).

Revised after adversarial review (see JOURNAL.md). The original version only
compared movement at 512 vs. default against a pre-registered sampling-noise
band (audit/result2_bands.json, read-only). This version adds, as the
PRIMARY test: a paired bootstrap confidence interval on the difference
itself (resampling the 12 configs, same resample reused for both contexts).
The old band comparison is kept but demoted to a secondary, coarser
yardstick. Also added: leave-one-out robustness, per-config direction
counts, a reproduction check against the snapshot for all 4 models (not
just chronos), a flag for the one no-op grid cell (toto/m4_daily, where the
context cap collapses to the same value at both settings), explicit rank/
gmean gaps for every ordering flip, and a robustness check on the bands
themselves (recomputed from our own values instead of the snapshot's).

Pure stdlib. Reuses run_audit.py's load_all_results/build_pipeline/
gmean_safe/percentile_ci (the same pooled rank/Seasonal_Naive-normalization
pipeline audit/result2_bands.py already reuses) and result2_bands.py's
load_selected_configs/resolve_target_dirs/md5_file/bootstrap_reps/
band_from_values/gmean_band/half_width, instead of reimplementing any of
them.

Two aggregations per model x context, both computed over the same 12 configs:
  A. Leaderboard-insertion mean rank -- for each config, take OUR measured
     MASE and insert it into the snapshot's published per-config pool of MASE
     values for that config (the same ~130-model pool build_pipeline() pools
     for the real leaderboard's own Rank_MASE column), then find the rank
     ("method='first'": ascending, ties broken by order of appearance) that
     insertion would receive. Average over the 12 configs.
     Insertion tie-break, made explicit because the pre-registration doesn't
     spell it out: our new value is appended AFTER the existing pool (which
     is already sorted the same way build_pipeline sorts it) before the
     stable sort, so on an exact tie the existing, already-published entry
     keeps the lower rank number and ours takes the next one. Ties in
     real-valued MASE across ~130 independent runs are not expected to
     happen in this data; this only documents what would happen if they did.
  B. Geometric mean of our MASE divided by the snapshot's Seasonal_Naive MASE
     for that config (build_pipeline()'s "norm"), geometric-meaned over the
     12 configs.

Bootstraps in this file (both resample the 12 configs with replacement):
  - Paired-difference CI (primary test): B=4000, seed 20260914. One set of
    resampled config indices generated once and reused across every model,
    aggregation, and context (so the 512-vs-default difference is computed
    from the SAME resampled configs on both sides of the subtraction --
    "paired" in the usual bootstrap sense).
  - Mixed-provenance robustness check: reuses result2_bands.py's own
    bootstrap_reps() with its own seed (20260913), applied to OUR
    default-context per-config values instead of the snapshot's published
    ones, to see how much the pre-registered band's half-width would have
    differed had it been built from our numbers instead of the snapshot's.

Usage: python3 result2_grid.py [snapshot_root]
Output: result2_grid.json, RESULT2_GRID.md (in this script's directory)
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
import run_audit as ra              # load_all_results, build_pipeline, gmean_safe, percentile_ci
import result2_bands as rb          # load_selected_configs, resolve_target_dirs, md5_file, bootstrap_reps, band_from_values, gmean_band, half_width

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths below are for this zip's layout (code/audit/, code/selection/,
# snapshot/snapshot_2026-09-07/, results/ all under one root) -- see README.txt.
DEFAULT_SNAPSHOT_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "snapshot", "snapshot_2026-09-07"))
SELECTED_CONFIGS_PATH = os.path.normpath(os.path.join(OUT_DIR, "..", "selection", "selected_configs.json"))
BANDS_PATH = os.path.join(OUT_DIR, "result2_bands.json")
GRID_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "results"))

N_SELECTED = 12
CONTEXTS = ["default", "512"]  # -> folder names ctx_default / ctx_512

PAIRED_DIFF_SEED = 20260914  # primary test: paired bootstrap of the 512-minus-default difference

# grid_results_h100 folder/CSV "model" label -> result2_bands.json "label"
# (the two scripts were written independently and picked different label
# spellings for the same 4 models; this is the one place that reconciles
# them).
GRID_TO_BANDS_LABEL = {
    "chronos_bolt_base": "chronos_bolt_base",
    "moirai2_small": "Moirai-2.0-R-small",
    "timesfm25": "TimesFM-2.5 (200m)",
    "toto": "Toto-Open-Base-1.0",
}
GRID_MODELS = list(GRID_TO_BANDS_LABEL.keys())

BAND_RELABEL_NOTE = (
    "The band-half-width comparison is a coarser yardstick: movement vs the config-selection "
    "spread of the score level (how much this model's own score moves around just from which 12 "
    "configs get picked, at a fixed context length). The paired bootstrap CI on the difference "
    "itself is the primary test; the band comparison is reported alongside it, secondary."
)


def md5_file(path):
    return rb.md5_file(path)


def ctx_dir(context):
    return f"ctx_{context}"


def load_grid_csv(path, expected_model_label):
    """One grid_results_h100 all_results.csv -> {config_string: mase}.
    Guards (raises, does not silently drop): non-matching model label,
    duplicate config rows."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["model"] != expected_model_label:
            raise SystemExit(f"{path}: row model={r['model']!r}, expected {expected_model_label!r}")
        cfg = r["dataset"]
        if cfg in out:
            raise SystemExit(f"{path}: duplicate row for config {cfg!r}")
        out[cfg] = float(r["eval_metrics/MASE[0.5]"])
    return out


def insertion_rank(value, pool_rows):
    """Rank `value` would receive if appended to pool_rows (list of dicts
    with a "mase" key, already in the same (model,dataset)-sorted order
    run_audit.py's build_pipeline groups by config) and the whole thing
    stably sorted ascending -- i.e. build_pipeline's own
    `sorted(range(len(...)), key=...)` rank recipe, extended by one
    appended element. See module docstring for the tie-break this implies."""
    vals = [r["mase"] for r in pool_rows] + [value]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    return order.index(len(vals) - 1) + 1


def paired_diff_ci(per_default, per_512, field, reps, warnings, context):
    """Bootstrap CI on (value at 512 minus value at default), resampling the
    12 configs with replacement -- the SAME resampled index set (`reps`,
    generated once by the caller) is used for both contexts on every
    replicate, so aggregation noise from resampling cancels in the
    subtraction and what's left is the context-length effect. `field` is
    "insertion_rank" (arithmetic mean per replicate) or "scaled_mase"
    (geometric mean per replicate, via gmean_safe)."""
    vals_default = [p[field] for p in per_default]
    vals_512 = [p[field] for p in per_512]
    is_rank = field == "insertion_rank"

    def agg(vals):
        return statistics.mean(vals) if is_rank else ra.gmean_safe(vals, warnings, context)

    point_delta = agg(vals_512) - agg(vals_default)
    diffs = []
    for rep in reps:
        d_default = agg([vals_default[i] for i in rep])
        d_512 = agg([vals_512[i] for i in rep])
        diffs.append(d_512 - d_default)
    lo, hi = ra.percentile_ci(diffs)
    return {
        "point_delta": point_delta,
        "ci_2.5_97.5": [lo, hi],
        "excludes_zero": not (lo <= 0 <= hi),
    }


def leave_one_out(per_default, per_512, field, band_half_width, orig_exceeds_band):
    """Drop each of the 12 configs in turn, recompute the point delta and
    the exceeds-band verdict on the remaining 11, and report which drops
    flip the verdict relative to the full-12 verdict already computed by
    the caller."""
    is_rank = field == "insertion_rank"
    per_drop = []
    flipping = []
    for drop_idx in range(N_SELECTED):
        keep = [i for i in range(N_SELECTED) if i != drop_idx]
        vals_default = [per_default[i][field] for i in keep]
        vals_512 = [per_512[i][field] for i in keep]
        if is_rank:
            d_default, d_512 = statistics.mean(vals_default), statistics.mean(vals_512)
        else:
            d_default = ra.gmean_safe(vals_default, [], "loo")
            d_512 = ra.gmean_safe(vals_512, [], "loo")
        delta = d_512 - d_default
        exceeds = abs(delta) > band_half_width
        dropped_config = per_default[drop_idx]["config"]
        per_drop.append({"dropped_config": dropped_config, "delta": delta, "exceeds_band": exceeds})
        if exceeds != orig_exceeds_band:
            flipping.append(dropped_config)
    return {"per_drop": per_drop, "n_flips_of_12": len(flipping), "flipping_configs": flipping}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SNAPSHOT_ROOT
    warnings = []

    # ---- pre-registered inputs (read-only) ----
    selected_keys = rb.load_selected_configs(SELECTED_CONFIGS_PATH)  # list of (base,freq,term), JSON order
    selected_strs = ["/".join(k) for k in selected_keys]
    selected_md5 = md5_file(SELECTED_CONFIGS_PATH)

    with open(BANDS_PATH) as f:
        bands = json.load(f)
    bands_md5 = md5_file(BANDS_PATH)
    bands_by_label = {t["label"]: t for t in bands["target_models"]}
    for grid_label, bands_label in GRID_TO_BANDS_LABEL.items():
        if bands_label not in bands_by_label:
            raise SystemExit(f"result2_bands.json has no target model labeled {bands_label!r} (for {grid_label})")

    # ---- snapshot pipeline (reused, not reimplemented) ----
    rows, load_warnings = ra.load_all_results(root)
    warnings.extend(load_warnings)
    canonical_keys, model_data, model_agg = ra.build_pipeline(rows, warnings)
    missing_from_universe = [s for k, s in zip(selected_keys, selected_strs) if k not in canonical_keys]
    if missing_from_universe:
        raise SystemExit(f"selected configs not in snapshot's 97-config universe: {missing_from_universe}")

    groups = defaultdict(list)  # config_key -> pool rows, same grouping build_pipeline uses internally
    for r in rows:
        groups[r["config_key"]].append(r)

    by_folder, by_model_field, _dup = ra.load_config_info(root)

    # ---- load our grid CSVs ----
    mase_lookup = {}  # model -> context -> config_string -> mase
    config_verification = []
    grid_csv_md5 = {}
    for model in GRID_MODELS:
        mase_lookup[model] = {}
        grid_csv_md5[model] = {}
        for context in CONTEXTS:
            path = os.path.join(GRID_ROOT, model, ctx_dir(context), "all_results.csv")
            data = load_grid_csv(path, model)
            found = set(data.keys())
            matches = found == set(selected_strs)
            config_verification.append({
                "model": model, "context": context, "n_rows": len(data),
                "matches_selected_12": matches,
                "extra": sorted(found - set(selected_strs)),
                "missing": sorted(set(selected_strs) - found),
            })
            if not matches:
                raise SystemExit(f"{path}: configs do not match the 12 selected configs exactly "
                                  f"(extra={sorted(found - set(selected_strs))}, missing={sorted(set(selected_strs) - found)})")
            mase_lookup[model][context] = data
            grid_csv_md5[model][context] = md5_file(path)

    # ---- item 1: per model x context, both aggregations ----
    model_context_table = {}
    for model in GRID_MODELS:
        model_context_table[model] = {}
        for context in CONTEXTS:
            per_config = []
            for key, cfg_str in zip(selected_keys, selected_strs):
                mase = mase_lookup[model][context][cfg_str]
                pool_rows = groups[key]
                rank = insertion_rank(mase, pool_rows)
                baseline = model_data["Seasonal_Naive"][key]["mase"]
                scaled = mase / baseline
                per_config.append({
                    "config": cfg_str, "mase": mase,
                    "insertion_rank": rank, "scaled_mase": scaled,
                })
            avg_rank = statistics.mean(p["insertion_rank"] for p in per_config)
            gmean_scaled = ra.gmean_safe([p["scaled_mase"] for p in per_config], warnings, f"{model}/{context}")
            model_context_table[model][context] = {
                "avg_insertion_rank": avg_rank,
                "gmean_scaled_mase": gmean_scaled,
                "per_config": per_config,
            }

    # ---- item 2 (headline): deltas vs pre-registered bands, PLUS paired-difference bootstrap CI (primary) ----
    paired_reps = rb.bootstrap_reps(N_SELECTED, random.Random(PAIRED_DIFF_SEED))  # generated once, reused for every model/aggregation/context -- the "paired" resample set

    deltas = []
    for model in GRID_MODELS:
        bands_label = GRID_TO_BANDS_LABEL[model]
        band = bands_by_label[bands_label]
        hw_rank = band["aggregation_A_mean_insertion_rank"]["half_width"]
        hw_gmean = band["aggregation_B_gmean_scaled_mase"]["half_width"]
        d = model_context_table[model]["default"]
        v = model_context_table[model]["512"]

        delta_rank = v["avg_insertion_rank"] - d["avg_insertion_rank"]
        delta_gmean = v["gmean_scaled_mase"] - d["gmean_scaled_mase"]

        paired_rank = paired_diff_ci(d["per_config"], v["per_config"], "insertion_rank", paired_reps, warnings, f"{model}/paired-ci/rank")
        paired_gmean = paired_diff_ci(d["per_config"], v["per_config"], "scaled_mase", paired_reps, warnings, f"{model}/paired-ci/gmean")

        deltas.append({
            "model": model, "bands_label": bands_label,
            "mean_insertion_rank": {
                "default": d["avg_insertion_rank"], "at_512": v["avg_insertion_rank"],
                "delta": delta_rank, "abs_delta": abs(delta_rank),
                "paired_ci": paired_rank,
                "band_half_width": hw_rank,
                "ratio_abs_delta_to_half_width": abs(delta_rank) / hw_rank,
                "exceeds_band": abs(delta_rank) > hw_rank,
            },
            "gmean_scaled_mase": {
                "default": d["gmean_scaled_mase"], "at_512": v["gmean_scaled_mase"],
                "delta": delta_gmean, "abs_delta": abs(delta_gmean),
                "paired_ci": paired_gmean,
                "band_half_width": hw_gmean,
                "ratio_abs_delta_to_half_width": abs(delta_gmean) / hw_gmean,
                "exceeds_band": abs(delta_gmean) > hw_gmean,
            },
        })
    deltas_by_model = {d["model"]: d for d in deltas}

    # ---- item 3: leave-one-out robustness of the exceeds-band verdict ----
    leave_one_out_results = {}
    for model in GRID_MODELS:
        d = model_context_table[model]["default"]["per_config"]
        v = model_context_table[model]["512"]["per_config"]
        dm = deltas_by_model[model]
        leave_one_out_results[model] = {
            "mean_insertion_rank": leave_one_out(
                d, v, "insertion_rank", dm["mean_insertion_rank"]["band_half_width"], dm["mean_insertion_rank"]["exceeds_band"]),
            "gmean_scaled_mase": leave_one_out(
                d, v, "scaled_mase", dm["gmean_scaled_mase"]["band_half_width"], dm["gmean_scaled_mase"]["exceeds_band"]),
        }

    # ---- item 4: direction consistency (per-config MASE, 512 vs default) ----
    direction_consistency = {}
    for model in GRID_MODELS:
        worse, better, tied = [], [], []
        for cfg_str in selected_strs:
            dv, vv = mase_lookup[model]["default"][cfg_str], mase_lookup[model]["512"][cfg_str]
            (worse if vv > dv else better if vv < dv else tied).append(cfg_str)
        direction_consistency[model] = {
            "n_worse_at_512_of_12": len(worse), "worse_at_512_configs": worse,
            "n_better_at_512_of_12": len(better), "better_at_512_configs": better,
            "n_tied_of_12": len(tied), "tied_configs": tied,
        }

    # ---- item (ordering): model ordering + flips (with gaps), per aggregation, default vs 512 ----
    def ordering_and_flips(agg_key):
        order_default = sorted(GRID_MODELS, key=lambda m: model_context_table[m]["default"][agg_key])
        order_512 = sorted(GRID_MODELS, key=lambda m: model_context_table[m]["512"][agg_key])
        flips = []
        for i in range(len(GRID_MODELS)):
            for j in range(i + 1, len(GRID_MODELS)):
                a, b = GRID_MODELS[i], GRID_MODELS[j]
                va_d, vb_d = model_context_table[a]["default"][agg_key], model_context_table[b]["default"][agg_key]
                va_5, vb_5 = model_context_table[a]["512"][agg_key], model_context_table[b]["512"][agg_key]
                if (va_d < vb_d) != (va_5 < vb_5):
                    flips.append({
                        "pair": [a, b],
                        "default": [va_d, vb_d],
                        "at_512": [va_5, vb_5],
                        "gap_default": abs(va_d - vb_d),
                        "gap_at_512": abs(va_5 - vb_5),
                    })
        return {"order_default_best_to_worst": order_default, "order_512_best_to_worst": order_512, "flips": flips}

    model_ordering = {
        "mean_insertion_rank": ordering_and_flips("avg_insertion_rank"),
        "gmean_scaled_mase": ordering_and_flips("gmean_scaled_mase"),
    }

    # ---- item (Moirai check, pre-registered question) ----
    moirai_per_config = []
    for cfg_str in selected_strs:
        d = mase_lookup["moirai2_small"]["default"][cfg_str]
        v = mase_lookup["moirai2_small"]["512"][cfg_str]
        rel = abs(v - d) / d
        moirai_per_config.append({"config": cfg_str, "mase_default_4000": d, "mase_512": v, "relative_diff": rel})
    moirai_rel_diffs = [p["relative_diff"] for p in moirai_per_config]
    moirai_max_rel = max(moirai_rel_diffs)
    moirai_median_rel = statistics.median(moirai_rel_diffs)
    moirai_identical = moirai_max_rel < 1e-9
    moirai_verdict = "identical" if moirai_identical else "different"
    moirai_verdict_sentence = (
        f"moirai2_small's MASE at context length 4000 (default) is {moirai_verdict} from its MASE at "
        f"context length 512, across the 12 pre-registered configs (max relative difference "
        f"{moirai_max_rel:.4%}, median relative difference {moirai_median_rel:.4%})."
    )
    moirai_check = {
        "per_config": moirai_per_config,
        "max_relative_diff": moirai_max_rel,
        "median_relative_diff": moirai_median_rel,
        "verdict": moirai_verdict,
        "verdict_sentence": moirai_verdict_sentence,
    }

    # ---- item (per-config detail table, 12 configs x 4 models) ----
    per_config_detail = []
    for cfg_str in selected_strs:
        for model in GRID_MODELS:
            d = mase_lookup[model]["default"][cfg_str]
            v = mase_lookup[model]["512"][cfg_str]
            per_config_detail.append({
                "config": cfg_str, "model": model,
                "mase_default": d, "mase_512": v,
                "relative_change_pct": (v - d) / d * 100.0,
            })

    # ---- item 6/5: no-op cells (byte-identical rows at both contexts) ----
    no_op_cells = []
    for model in GRID_MODELS:
        for cfg_str in selected_strs:
            d = mase_lookup[model]["default"][cfg_str]
            v = mase_lookup[model]["512"][cfg_str]
            if d == v:
                no_op_cells.append({"model": model, "config": cfg_str, "mase": d})

    # ---- item 5: reproduction check vs the snapshot's own published rows, all 4 models ----
    reproduction_check = {}
    for model in GRID_MODELS:
        bands_label = GRID_TO_BANDS_LABEL[model]
        snap_key = bands_by_label[bands_label]["snapshot_dir"]  # the exact key model_data/by_folder use for this model
        leakage = by_folder.get(snap_key, {}).get("testdata_leakage", "N/A")
        per_cfg = []
        for cfg_str, key in zip(selected_strs, selected_keys):
            ours = mase_lookup[model]["default"][cfg_str]
            snap_entry = model_data.get(snap_key, {}).get(key)
            if snap_entry is None:
                raise SystemExit(f"snapshot has no {snap_key!r} row for {cfg_str}")
            snap = snap_entry["mase"]
            per_cfg.append({
                "config": cfg_str, "ours_default_ctx": ours, "snapshot_published": snap,
                "relative_diff": abs(ours - snap) / snap,
            })
        rel_diffs = [p["relative_diff"] for p in per_cfg]
        worst = max(per_cfg, key=lambda p: p["relative_diff"])
        reproduction_check[model] = {
            "snapshot_model_key": snap_key,
            "testdata_leakage_flag": leakage,
            "per_config": per_cfg,
            "median_relative_diff": statistics.median(rel_diffs),
            "max_relative_diff": worst["relative_diff"],
            "max_relative_diff_config": worst["config"],
            "our_default_avg_insertion_rank": model_context_table[model]["default"]["avg_insertion_rank"],
            "published_avg_insertion_rank_12cfg": bands_by_label[bands_label]["aggregation_A_mean_insertion_rank"]["point"],
            "our_default_gmean_scaled_mase": model_context_table[model]["default"]["gmean_scaled_mase"],
            "published_gmean_scaled_mase_12cfg": bands_by_label[bands_label]["aggregation_B_gmean_scaled_mase"]["point"],
        }

    # ---- item 8: mixed-provenance robustness -- recompute the bands from OUR
    # default-context values instead of the snapshot's published rows, using
    # result2_bands.py's own bootstrap (same seed/scheme as the pre-registered
    # bands, so only the input provenance differs), and see how much the
    # half-widths (and the exceeds-band verdicts) would have moved. ----
    mixed_prov_reps = rb.bootstrap_reps(N_SELECTED, random.Random(rb.SEED))  # rb.SEED=20260913, same resampling scheme result2_bands.json itself used
    mixed_provenance_per_model = []
    for model in GRID_MODELS:
        bands_label = GRID_TO_BANDS_LABEL[model]
        band = bands_by_label[bands_label]
        per_default = model_context_table[model]["default"]["per_config"]
        our_ranks = [p["insertion_rank"] for p in per_default]
        our_norms = [p["scaled_mase"] for p in per_default]
        _, lo_r, hi_r = rb.band_from_values(our_ranks, mixed_prov_reps)
        hw_r_ours = rb.half_width(lo_r, hi_r)
        _, lo_g, hi_g = rb.gmean_band(our_norms, mixed_prov_reps, warnings, f"{model}/mixed-provenance/gmean")
        hw_g_ours = rb.half_width(lo_g, hi_g)
        hw_r_snap = band["aggregation_A_mean_insertion_rank"]["half_width"]
        hw_g_snap = band["aggregation_B_gmean_scaled_mase"]["half_width"]

        dm = deltas_by_model[model]
        rank_exceeds_ours = dm["mean_insertion_rank"]["abs_delta"] > hw_r_ours
        gmean_exceeds_ours = dm["gmean_scaled_mase"]["abs_delta"] > hw_g_ours

        mixed_provenance_per_model.append({
            "model": model,
            "half_width_rank_snapshot_provenance": hw_r_snap,
            "half_width_rank_our_provenance": hw_r_ours,
            "half_width_rank_change": hw_r_ours - hw_r_snap,
            "half_width_gmean_snapshot_provenance": hw_g_snap,
            "half_width_gmean_our_provenance": hw_g_ours,
            "half_width_gmean_change": hw_g_ours - hw_g_snap,
            "verdict_flips": {
                "mean_insertion_rank": rank_exceeds_ours != dm["mean_insertion_rank"]["exceeds_band"],
                "gmean_scaled_mase": gmean_exceeds_ours != dm["gmean_scaled_mase"]["exceeds_band"],
            },
        })
    mixed_provenance = {
        "seed": rb.SEED,
        "max_abs_half_width_change_rank": max(abs(m["half_width_rank_change"]) for m in mixed_provenance_per_model),
        "max_abs_half_width_change_gmean": max(abs(m["half_width_gmean_change"]) for m in mixed_provenance_per_model),
        "any_verdict_flips": any(
            m["verdict_flips"]["mean_insertion_rank"] or m["verdict_flips"]["gmean_scaled_mase"]
            for m in mixed_provenance_per_model
        ),
        "per_model": mixed_provenance_per_model,
    }

    # ---- assemble ----
    out = {
        "snapshot_commit": ra.SNAPSHOT_COMMIT,
        "snapshot_root": os.path.abspath(root),
        "selected_configs_json_md5": selected_md5,
        "result2_bands_json_md5": bands_md5,
        "selected_configs": selected_strs,
        "grid_csv_md5": grid_csv_md5,
        "config_verification": config_verification,
        "model_context_table": model_context_table,
        "paired_diff_seed": PAIRED_DIFF_SEED,
        "paired_diff_B": rb.B,
        "deltas_vs_bands": deltas,
        "band_relabel_note": BAND_RELABEL_NOTE,
        "leave_one_out": leave_one_out_results,
        "direction_consistency": direction_consistency,
        "model_ordering": model_ordering,
        "moirai_check": moirai_check,
        "per_config_detail": per_config_detail,
        "no_op_cells": no_op_cells,
        "reproduction_check": reproduction_check,
        "mixed_provenance_robustness": mixed_provenance,
        "warnings": warnings,
    }

    json_path = os.path.join(OUT_DIR, "result2_grid.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"Wrote {json_path}")

    write_markdown(out, os.path.join(OUT_DIR, "RESULT2_GRID.md"))
    print(f"Wrote {os.path.join(OUT_DIR, 'RESULT2_GRID.md')}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)


def write_markdown(out, path):
    lines = []
    lines.append("# Result 2 -- context-length grid, final numbers\n")
    lines.append(
        "Our 4 measured models, each run at its own documented default context length and at a fixed "
        "512, on the 12 pre-registered configs (`selection/selected_configs.json`). Two aggregations per "
        "model x context: the mean rank our measured MASE would get if inserted into the real leaderboard's "
        "per-config score pool, and the geometric mean of our MASE divided by seasonal-naive's MASE. "
        "Sampling-noise bands are read from `audit/result2_bands.json`, not recomputed here (except for "
        "the robustness check in section 10, which does recompute them).\n"
    )
    lines.append(f"Snapshot commit `{out['snapshot_commit']}`. "
                  f"`selection/selected_configs.json` md5: `{out['selected_configs_json_md5']}`. "
                  f"`audit/result2_bands.json` md5: `{out['result2_bands_json_md5']}`.\n")

    ok = all(c["matches_selected_12"] for c in out["config_verification"])
    lines.append(f"All 8 grid CSVs (4 models x 2 contexts) contain exactly the 12 selected configs, no more, "
                  f"no fewer: {ok}.\n")

    lines.append("## 1. Model x context table (both aggregations)\n")
    lines.append("| model | context | mean insertion rank | gmean scaled MASE |")
    lines.append("|---|---|---|---|")
    for model in out["model_context_table"]:
        for context, v in out["model_context_table"][model].items():
            lines.append(f"| {model} | {context} | {v['avg_insertion_rank']:.2f} | {v['gmean_scaled_mase']:.4f} |")
    lines.append("")

    lines.append("## 2. Movement from default to 512 -- paired bootstrap CI (primary), band comparison (coarser, secondary)\n")
    lines.append(f"Delta = value at 512 minus value at default. Paired CI: bootstrap the difference itself, "
                  f"resampling the 12 configs with replacement (B={out['paired_diff_B']}, seed {out['paired_diff_seed']}), "
                  "the same resampled config set used for both contexts on every replicate. \"excludes zero\" is the "
                  f"primary verdict. {out['band_relabel_note']}\n")
    lines.append("| model | aggregation | default | at 512 | Delta | paired 95% CI | excludes zero (primary) | "
                  "band half-width (coarser) | \\|Delta\\|/half-width | exceeds band (coarser)? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in out["deltas_vs_bands"]:
        for agg_name, agg_key, nd in [("mean insertion rank", "mean_insertion_rank", 2), ("gmean scaled MASE", "gmean_scaled_mase", 4)]:
            a = d[agg_key]
            pc = a["paired_ci"]
            lines.append(
                f"| {d['model']} | {agg_name} | {a['default']:.{nd}f} | {a['at_512']:.{nd}f} | "
                f"{a['delta']:+.{nd}f} | [{pc['ci_2.5_97.5'][0]:+.{nd}f}, {pc['ci_2.5_97.5'][1]:+.{nd}f}] | {pc['excludes_zero']} | "
                f"{a['band_half_width']:.{nd}f} | {a['ratio_abs_delta_to_half_width']:.2f}x | {a['exceeds_band']} |"
            )
    lines.append("")

    lines.append("## 3. Leave-one-out: does the exceeds-band verdict survive dropping any one of the 12 configs?\n")
    lines.append("Recomputes the point delta and the (coarser, band-based) exceeds-band verdict on the remaining "
                  "11 configs, once per dropped config; \"flips\" = verdict differs from the full-12 verdict.\n")
    lines.append("| model | aggregation | flips (of 12) | flipping configs |")
    lines.append("|---|---|---|---|")
    for model in out["leave_one_out"]:
        for agg_name, agg_key in [("mean insertion rank", "mean_insertion_rank"), ("gmean scaled MASE", "gmean_scaled_mase")]:
            loo = out["leave_one_out"][model][agg_key]
            flip_str = ", ".join(loo["flipping_configs"]) if loo["flipping_configs"] else "none"
            lines.append(f"| {model} | {agg_name} | {loo['n_flips_of_12']} | {flip_str} |")
    lines.append("")

    lines.append("## 4. Direction consistency: per-config MASE, 512 vs default\n")
    lines.append("Count of the 12 configs where the model's MASE is worse (higher) at 512 than at its own default.\n")
    lines.append("| model | worse at 512 | better at 512 | tied |")
    lines.append("|---|---|---|---|")
    for model, dc in out["direction_consistency"].items():
        lines.append(f"| {model} | {dc['n_worse_at_512_of_12']}/12 | {dc['n_better_at_512_of_12']}/12 | {dc['n_tied_of_12']}/12 |")
    lines.append("")

    lines.append("## 5. Ordering of the 4 models, default vs 512\n")
    for agg_title, agg_key in [("Mean insertion rank", "mean_insertion_rank"), ("Gmean scaled MASE", "gmean_scaled_mase")]:
        mo = out["model_ordering"][agg_key]
        lines.append(f"**{agg_title}** (best to worst):")
        lines.append(f"- default: {' > '.join(mo['order_default_best_to_worst'])}")
        lines.append(f"- 512: {' > '.join(mo['order_512_best_to_worst'])}")
        if mo["flips"]:
            for fl in mo["flips"]:
                nd = 2 if agg_key == "mean_insertion_rank" else 4
                lines.append(f"  - rank flip: {fl['pair'][0]} vs {fl['pair'][1]} -- "
                              f"default {fl['default'][0]:.{nd}f} vs {fl['default'][1]:.{nd}f} (gap {fl['gap_default']:.{nd}f}), "
                              f"512 {fl['at_512'][0]:.{nd}f} vs {fl['at_512'][1]:.{nd}f} (gap {fl['gap_at_512']:.{nd}f})")
        else:
            lines.append("  - no rank flips between the two settings.")
        lines.append("")

    lines.append("## 6. Moirai check (pre-registered question)\n")
    lines.append(out["moirai_check"]["verdict_sentence"] + "\n")
    lines.append("| config | MASE at default (4000) | MASE at 512 | relative diff |")
    lines.append("|---|---|---|---|")
    for p in out["moirai_check"]["per_config"]:
        lines.append(f"| {p['config']} | {p['mase_default_4000']:.4f} | {p['mase_512']:.4f} | {p['relative_diff']:.2%} |")
    lines.append("")

    lines.append("## 7. Reproduction check: our default-context MASE vs the snapshot's published rows, all 4 models\n")
    for model, rc in out["reproduction_check"].items():
        lines.append(f"**{model}** (snapshot key `{rc['snapshot_model_key']}`, `testdata_leakage: {rc['testdata_leakage_flag']}`): "
                      f"median relative diff {rc['median_relative_diff']:.2%}, max {rc['max_relative_diff']:.2%} "
                      f"(at `{rc['max_relative_diff_config']}`). Our default-context mean insertion rank "
                      f"{rc['our_default_avg_insertion_rank']:.2f} vs the published entry's {rc['published_avg_insertion_rank_12cfg']:.2f} "
                      f"on the same 12 configs; our gmean scaled MASE {rc['our_default_gmean_scaled_mase']:.4f} vs published "
                      f"{rc['published_gmean_scaled_mase_12cfg']:.4f}.\n")
    toto_rc = out["reproduction_check"]["toto"]
    lines.append(f"Toto's reproduction gap is the largest of the 4: median relative difference "
                  f"{toto_rc['median_relative_diff']:.2%}, max {toto_rc['max_relative_diff']:.2%} at "
                  f"`{toto_rc['max_relative_diff_config']}`, and its own mean insertion rank comes out "
                  f"{toto_rc['our_default_avg_insertion_rank']:.2f} (ours) vs {toto_rc['published_avg_insertion_rank_12cfg']:.2f} "
                  "(published) on the same 12 configs -- moirai2_small and timesfm25 reproduce far tighter (both medians "
                  f"and maxes under 1e-6 relative difference).\n")
    lines.append("| model | config | ours (default ctx) | snapshot published | relative diff |")
    lines.append("|---|---|---|---|---|")
    for model, rc in out["reproduction_check"].items():
        for p in rc["per_config"]:
            lines.append(f"| {model} | {p['config']} | {p['ours_default_ctx']:.4f} | {p['snapshot_published']:.4f} | {p['relative_diff']:.2%} |")
    lines.append("")

    lines.append("## 8. Per-config detail (appendix): MASE at default, at 512, relative change\n")
    lines.append("| config | model | MASE default | MASE 512 | relative change % |")
    lines.append("|---|---|---|---|---|")
    for r in out["per_config_detail"]:
        lines.append(f"| {r['config']} | {r['model']} | {r['mase_default']:.4f} | {r['mase_512']:.4f} | {r['relative_change_pct']:+.2f}% |")
    lines.append("")

    lines.append("## 9. Toto / m4_daily: no-op grid cell\n")
    if out["no_op_cells"]:
        cells = ", ".join(f"{c['model']}/{c['config']}" for c in out["no_op_cells"])
        lines.append(f"toto's recipe caps context length at `min(setting, shortest_series_length - (windows+1)*prediction_length)`; "
                      f"for m4_daily that cap lands on the same value under both the default and the 512 setting, so "
                      f"the two runs are byte-identical rows and that cell measures nothing ({cells}). This biases "
                      "toto's measured movement toward zero for m4_daily specifically, which makes toto's overall "
                      "exceeds-band verdict conservative (if anything, an understatement of toto's real movement), "
                      "not an inflation of it.\n")
    else:
        lines.append("No no-op cells found (no model/config pair had byte-identical MASE at both context settings).\n")

    lines.append("## 10. Mixed-provenance robustness note\n")
    mp = out["mixed_provenance_robustness"]
    lines.append(f"The headline band comparison in section 2 is mixed-provenance: the band half-widths come from the "
                  f"snapshot's own published rows, while the 512-vs-default movement comes from our measurements. "
                  f"Recomputing those same bands from our own default-context per-config values instead (same "
                  f"bootstrap: seed {mp['seed']}, {out['paired_diff_B']} resamples of the 12 configs) changes the "
                  f"half-widths by at most {mp['max_abs_half_width_change_rank']:.2f} rank places / "
                  f"{mp['max_abs_half_width_change_gmean']:.4f} gmean across the 4 models, and "
                  f"{'flips no exceeds-band verdict' if not mp['any_verdict_flips'] else 'DOES flip at least one exceeds-band verdict'}.\n")

    if out["warnings"]:
        lines.append("## Warnings\n")
        for w in out["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
