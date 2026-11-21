"""Per-config differences for the displayed GIFT-Eval top-10 pairs (snapshot 2026-09-07).

top10_pair_diffs(rule) -> (names, {(i, j): d}, cl) with 0-based i < j over `names`.
Sign convention: d = (B's value) - (A's value) where A = names[i] is the higher-placed
entry, so positive d means the higher-placed entry is better:
  'MASE_Rank': d = rank_B - rank_A (per-config rank by MASE; lower rank is better)
  'CRPS_Rank': same, ranks by weighted quantile loss, top-10 in CRPS_Rank's own order
  'gmean'    : d = log(MASE_B/SN) - log(MASE_A/SN), SN = Seasonal_Naive; pairs/order as MASE_Rank
cl = integer label of each config's base dataset (sorted dataset names -> 0..G-1);
configs are in sorted (dataset, freq, term) order. Deterministic; results are cached.
"""
import os
import sys
import numpy as np

AUDIT = "fmts/audit"
ROOT = "fmts/snapshot_2026-09-07"
sys.path.insert(0, AUDIT)
import run_audit as ra          # noqa: E402
import critic_checks as cc      # noqa: E402

_cache = {}


def _pipeline(crps):
    rows, w = (cc.load_all_results_crps if crps else ra.load_all_results)(ROOT)
    keys, md, agg = ra.build_pipeline(rows, w)
    _, by_field, _ = ra.load_config_info(ROOT)
    disp, _, _ = ra.merge_display(agg, by_field)
    top = [r["model"] for r in ra.sort_leak_then(disp, "MASE_Rank")[:10]]
    return keys, md, top


def top10_pair_diffs(rule):
    if rule in _cache:
        return _cache[rule]
    assert rule in ("MASE_Rank", "CRPS_Rank", "gmean")
    keys, md, top = _pipeline(rule == "CRPS_Rank")
    if rule == "gmean":
        top = _pipeline(False)[2]
    names = [k[0] for k in keys]
    _, cl = np.unique(names, return_inverse=True)
    field = "norm" if rule == "gmean" else "rank"
    def arr(m):
        v = np.array([md[m][k][field] for k in keys], float)
        return np.log(v) if rule == "gmean" else v
    diffs = {(i, j): arr(top[j]) - arr(top[i]) for i in range(10) for j in range(i + 1, 10)}
    _cache[rule] = (top, diffs, cl)
    return _cache[rule]
