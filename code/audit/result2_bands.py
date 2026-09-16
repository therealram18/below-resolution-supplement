#!/usr/bin/env python3
"""
Result-2 pre-registered sampling-noise reference bands, computed against the
same pinned GIFT-Eval snapshot audit/run_audit.py already validated. No model
runs, no GPU -- everything here reads the snapshot's own published CSVs.

Reuses run_audit.py's row loading and rank/gmean pipeline (build_pipeline)
instead of reimplementing the pandas rank(method='first') + Seasonal_Naive
normalization logic a second time -- that function already pools every
model's published MASE per config and assigns the same insertion rank the
live app's Rank_MASE column would show.

Question this answers: later we will run 4 models at two context lengths and
ask whether the movement from the context switch is bigger than ordinary
sampling noise across dataset-configs. This script computes that "ordinary
sampling noise" yardstick -- a bootstrap band -- using ONLY the 12
pre-registered configs (selection/selected_configs.json), per the
pre-registered decision that the noise band must come from the same 12
configs the real comparison will use, not the full 97-config universe (which
would understate the noise).

Two aggregations, matching the leaderboard's own two summary columns:
  A. mean leaderboard-insertion rank over the 12 configs (each config's rank
     is this model's position when all ~130 models' published MASE on that
     config are sorted ascending, method='first' i.e. stable tie-break by
     row order -- exactly build_pipeline()'s "rank" field).
  B. geometric mean of Seasonal_Naive-scaled MASE over the 12 configs
     (build_pipeline()'s "norm" field, i.e. MASE / Seasonal_Naive MASE for
     that same config).

Bootstrap: resample the 12 configs with replacement, B=4000, seed=20260913.
"Paired" means one set of B resampled 12-index draws is generated once and
reused for every model and both aggregations (so aggregation A and B see the
same resampled config set on a given replicate, and models are comparable
replicate-for-replicate) -- fixed generation order off one seeded RNG makes
the whole thing deterministic.

Pre-registration note baked into the output: the 12 selected configs come
from 12 distinct base datasets (verified below), so resampling configs
with replacement and resampling dataset-clusters with replacement are the
same operation here -- unlike the 97-config universe, where several configs
share a base dataset. No separate iid/cluster scheme is needed.

Usage: python3 result2_bands.py [snapshot_root]
Output: result2_bands.json, RESULT2_BANDS.md (in this script's directory)
"""
import hashlib
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_audit as ra  # reuse load_all_results / build_pipeline / gmean_safe / percentile_ci

SEED = 20260913
B = 4000
N_SELECTED = 12
SNAPSHOT_COMMIT = ra.SNAPSHOT_COMMIT

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths below are for this zip's layout (code/audit/, code/selection/,
# snapshot/snapshot_2026-09-07/ all under one root) -- see README.txt.
DEFAULT_ROOT = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "snapshot", "snapshot_2026-09-07"))
SELECTED_CONFIGS_PATH = os.path.normpath(os.path.join(OUT_DIR, "..", "selection", "selected_configs.json"))

# The 4 models of interest, identified by a substring of their HuggingFace
# model_link (verified unique across all snapshot dirs -- see resolve below),
# not by guessing the folder-name spelling convention.
TARGET_LINK_NEEDLES = {
    "chronos_bolt_base": "chronos-bolt-base",
    "Moirai-2.0-R-small": "moirai-2.0-r-small",
    "TimesFM-2.5 (200m)": "timesfm-2.5-200m",
    "Toto-Open-Base-1.0": "toto-open-base-1.0",
}


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_selected_configs(path):
    with open(path) as f:
        sel = json.load(f)
    keys = [(c["base_dataset"], c["frequency"], c["term"]) for c in sel["selected"]]
    if len(keys) != N_SELECTED:
        raise SystemExit(f"expected {N_SELECTED} selected configs, found {len(keys)} in {path}")
    return keys


def resolve_target_dirs(root):
    """Match each wanted model to its snapshot folder via config.json's
    model_link (a HuggingFace URL), and error loudly on 0 or >1 matches
    instead of guessing."""
    results_dir = os.path.join(root, "results")
    hits = {label: [] for label in TARGET_LINK_NEEDLES}
    for entry in sorted(os.listdir(results_dir)):
        cfg_path = os.path.join(results_dir, entry, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            link = (json.load(f).get("model_link") or "").lower()
        for label, needle in TARGET_LINK_NEEDLES.items():
            if needle in link:
                hits[label].append(entry)
    resolved = {}
    for label, dirs in hits.items():
        if len(dirs) != 1:
            raise SystemExit(f"expected exactly 1 snapshot dir matching {label!r} (needle {TARGET_LINK_NEEDLES[label]!r}), found {dirs}")
        resolved[label] = dirs[0]
    return resolved


def base_datasets_distinct(keys):
    bases = [k[0] for k in keys]
    return len(set(bases)) == len(bases)


def per_model_arrays(model_folder, model_data, keys, warnings):
    """rank[]/norm[] aligned to `keys` (the 12 selected configs) for one
    model, using build_pipeline's pooled rank/Seasonal_Naive-normalized
    values. Reports (not silently drops) any missing config."""
    # model_data is keyed by the CSV's "model" column value, which is not
    # always the folder name -- read it straight off row data via config.json.
    cfgs = model_data.get(model_folder, {})
    ranks, norms, missing = [], [], []
    for key in keys:
        c = cfgs.get(key)
        if c is None:
            missing.append("/".join(key))
            continue
        ranks.append(c["rank"])
        norms.append(c["norm"])
    if missing:
        warnings.append(f"{model_folder}: missing {len(missing)}/{N_SELECTED} selected configs: {missing}")
    return ranks, norms, missing


def bootstrap_reps(n, rng):
    """B replicates, each a list of n indices in [0, n) drawn with
    replacement -- generated once off the seeded RNG and reused for every
    model/aggregation (the 'paired' resample set)."""
    return [rng.choices(range(n), k=n) for _ in range(B)]


def band_from_values(values, reps):
    """values: list aligned to the original 12-slot index order (may be
    shorter than 12 if some configs were missing -- reps indices are then
    reinterpreted against the *available* value list, see caller)."""
    point_ranks_mean = statistics.mean(values)
    boot = [statistics.mean([values[i] for i in rep]) for rep in reps]
    lo, hi = ra.percentile_ci(boot)
    return point_ranks_mean, lo, hi


def gmean_band(norms, reps, warnings, context):
    point = ra.gmean_safe(norms, warnings, context)
    boot = [ra.gmean_safe([norms[i] for i in rep], warnings, context + "/bootstrap") for rep in reps]
    lo, hi = ra.percentile_ci(boot)
    return point, lo, hi


def half_width(lo, hi):
    return (hi - lo) / 2.0


def compute_model_bands(label, folder, model_data, keys, reps, warnings):
    ranks, norms, missing = per_model_arrays(folder, model_data, keys, warnings)
    n_used = len(ranks)
    if n_used == 0:
        raise SystemExit(f"{label} ({folder}): 0/{N_SELECTED} selected configs present, cannot compute bands")
    # Reps were generated for indices [0, N_SELECTED); if this model is
    # missing configs, rebuild reps over the reduced index range so every
    # drawn index is valid -- and say so explicitly (no silent drop).
    local_reps = reps if n_used == N_SELECTED else bootstrap_reps(n_used, random.Random((SEED, folder)))

    rank_point, rank_lo, rank_hi = band_from_values(ranks, local_reps)
    gmean_point, gmean_lo, gmean_hi = gmean_band(norms, local_reps, warnings, f"{label}/gmean")

    return {
        "label": label,
        "snapshot_dir": folder,
        "n_configs_used": n_used,
        "missing_configs": missing,
        "aggregation_A_mean_insertion_rank": {
            "point": rank_point,
            "ci_2.5_97.5": [rank_lo, rank_hi],
            "half_width": half_width(rank_lo, rank_hi),
        },
        "aggregation_B_gmean_scaled_mase": {
            "point": gmean_point,
            "ci_2.5_97.5": [gmean_lo, gmean_hi],
            "half_width": half_width(gmean_lo, gmean_hi),
        },
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    warnings = []

    selected_md5 = md5_file(SELECTED_CONFIGS_PATH)
    keys = load_selected_configs(SELECTED_CONFIGS_PATH)
    clusters_coincide = base_datasets_distinct(keys)
    if not clusters_coincide:
        warnings.append("12 selected configs do NOT all come from distinct base datasets -- resampling configs and resampling clusters would differ")

    resolved_dirs = resolve_target_dirs(root)

    rows, load_warnings = ra.load_all_results(root)
    warnings.extend(load_warnings)
    canonical_keys, model_data, model_agg = ra.build_pipeline(rows, warnings)
    missing_from_universe = [("/".join(k)) for k in keys if k not in canonical_keys]
    if missing_from_universe:
        raise SystemExit(f"selected configs not found in snapshot's 97-config universe: {missing_from_universe}")

    rng = random.Random(SEED)
    reps = bootstrap_reps(N_SELECTED, rng)

    target_results = []
    for label, folder in resolved_dirs.items():
        target_results.append(compute_model_bands(label, folder, model_data, keys, reps, warnings))

    # Reference context: leaderboard's displayed top-3 (Test Leak. then
    # MASE_Rank, same sort as run_audit.py), on the same 12 configs.
    by_folder, by_model_field, dup_model_fields = ra.load_config_info(root)
    displayed, dropped_results_only, dropped_config_only = ra.merge_display(model_agg, by_model_field)
    top3 = ra.sort_leak_then(displayed, "MASE_Rank")[:3]
    reference_results = []
    for d in top3:
        model_name = d["model"]  # this is model_data's key (CSV "model" column value)
        reference_results.append(compute_model_bands(model_name, model_name, model_data, keys, reps, warnings))

    out = {
        "seed": SEED,
        "bootstrap_B": B,
        "n_selected_configs": N_SELECTED,
        "selected_configs_json_md5": selected_md5,
        "snapshot_commit": SNAPSHOT_COMMIT,
        "snapshot_root": os.path.abspath(root),
        "selected_configs": ["/".join(k) for k in keys],
        "resampling_note": (
            "The 12 selected configs come from 12 distinct base datasets, so resampling "
            "configs with replacement and resampling dataset-clusters with replacement are "
            "the same operation here -- no separate iid/cluster scheme is used (contrast "
            "with the 97-config universe in run_audit.py, where several configs share a base "
            "dataset)." if clusters_coincide else
            "WARNING: base datasets are NOT all distinct among the 12 selected configs -- "
            "config-resampling and cluster-resampling would differ; only config-resampling was run."
        ),
        "target_models": target_results,
        "reference_top3_displayed": reference_results,
        "warnings": warnings,
    }

    json_path = os.path.join(OUT_DIR, "result2_bands.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"Wrote {json_path}")

    write_markdown(out, os.path.join(OUT_DIR, "RESULT2_BANDS.md"))
    print(f"Wrote {os.path.join(OUT_DIR, 'RESULT2_BANDS.md')}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)


def write_markdown(out, path):
    lines = []
    lines.append("# Result 2 -- sampling-noise reference bands\n")
    lines.append(
        "This is the noise yardstick for Result 2: how much a model's score moves around just from "
        "which 12 dataset-configs happen to get picked, before we ever change context length. It comes "
        "only from the pre-registered 12 configs (not the full 97), because that is the same set the "
        "context-length comparison will use.\n"
    )
    lines.append(f"Snapshot commit `{out['snapshot_commit']}`. Bootstrap: seed {out['seed']}, "
                  f"{out['bootstrap_B']} resamples of the 12 configs, 95% band from the 2.5/97.5 percentiles.\n")
    lines.append(f"`selection/selected_configs.json` md5: `{out['selected_configs_json_md5']}`.\n")
    lines.append(out["resampling_note"] + "\n")

    def table(title, results):
        lines.append(f"## {title}\n")
        lines.append("| model | snapshot dir | n configs | mean rank (point) | rank 95% band | rank half-width | gmean scaled MASE (point) | gmean 95% band | gmean half-width |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in results:
            a = r["aggregation_A_mean_insertion_rank"]
            b = r["aggregation_B_gmean_scaled_mase"]
            lines.append(
                f"| {r['label']} | {r['snapshot_dir']} | {r['n_configs_used']}/12 | "
                f"{a['point']:.2f} | [{a['ci_2.5_97.5'][0]:.2f}, {a['ci_2.5_97.5'][1]:.2f}] | {a['half_width']:.2f} | "
                f"{b['point']:.4f} | [{b['ci_2.5_97.5'][0]:.4f}, {b['ci_2.5_97.5'][1]:.4f}] | {b['half_width']:.4f} |"
            )
            if r["missing_configs"]:
                lines.append(f"  - **{r['label']} missing configs (not silently dropped):** {r['missing_configs']}")
        lines.append("")

    table("The 4 target models", out["target_models"])
    table("Reference only: leaderboard's displayed top-3 (sanity context, not part of the noise yardstick)", out["reference_top3_displayed"])

    if out["warnings"]:
        lines.append("## Warnings\n")
        for w in out["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
