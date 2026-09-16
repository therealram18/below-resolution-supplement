#!/usr/bin/env python3
"""
Compute EFFECTIVE context length (past time steps a model actually saw,
after its truncation rule) for every (model, config, setting) cell of the
4-model x 12-config x 2-setting grid.

CPU only, no model runs. Reuses harness/run_grid.py's own
resolve_storage_name() and per-model constants/truncation logic, and
gift_eval.data.Dataset's own prediction_length/windows properties -- nothing
here re-derives those rules, it only reads the values off the real objects
and replicates the *truncation arithmetic* each builder performs on a
context array (arr[-N:], min(N, available), etc.), which needs no GPU.

Available history per (series, window) comes straight out of gluonts split
semantics used by Dataset.test_data: for a series of length L, with
prediction_length P and windows W, generate_instances(distance=P) yields W
non-overlapping windows whose *input* length is
    available(i) = L - P * (W - i),   i = 0 .. W-1
(earliest window has the least history, last window the most). Confirmed by
reading gluonts.dataset.split's offset/distance semantics against
Dataset.test_data's call (offset=-P*W, distance=P).

Deterministic: pure arithmetic over on-disk series lengths, no sampling, no
model calls. Run twice -> identical file -> identical md5.
"""
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMTS_ROOT = HERE.parent
HARNESS = FMTS_ROOT / "harness"

sys.path.insert(0, str(HARNESS / "gift-eval" / "src"))
sys.path.insert(0, str(HARNESS))
os.environ["GIFT_EVAL"] = str(HARNESS / "data")

import numpy as np
import pyarrow.compute as pc

import run_grid  # harness/run_grid.py -- resolve_storage_name, per-model constants, to_univariate_for
from gift_eval.data import Dataset

SELECTED_CONFIGS_PATH = FMTS_ROOT / "selection" / "selected_configs.json"
OUT_CSV = HERE / "effective_context.csv"
OUT_MD = HERE / "EFFECTIVE_CONTEXT.md"

MODELS = ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto"]
SETTINGS = ["default", "512"]


def series_lengths(ds: Dataset) -> np.ndarray:
    """Per-series target length, one value per raw series (pre to_univariate
    split) -- exact same pyarrow path Dataset._min_series_length uses,
    just without collapsing to a single min()."""
    col = ds.hf_dataset.data.column("target")
    if ds.hf_dataset[0]["target"].ndim > 1:
        lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(col, 0, 1)))
    else:
        lengths = pc.list_value_length(col)
    return lengths.to_numpy()


def available_history(lengths: np.ndarray, P: int, W: int) -> np.ndarray:
    """available(s, i) = L_s - P*(W-i) for i=0..W-1, flattened over all
    series x windows. Matches gluonts generate_instances(distance=P) with
    the offset=-P*W split Dataset.test_data uses."""
    i = np.arange(W)
    # shape (n_series, W)
    return lengths[:, None] - P * (W - i)[None, :]


def toto_context_length(target_default: int, min_len: int, W: int, P: int) -> int:
    """Exact formula from harness/run_grid.py build_toto()."""
    max_available = min_len - (W + 1) * P
    return min(target_default, max_available) if max_available > 0 else target_default


def effective_for_model(model: str, setting: str, avail: np.ndarray, target_dim: int,
                         min_len: int, W: int, P: int):
    """Returns (nominal, eff_array)."""
    if model == "chronos_bolt_base":
        nominal = run_grid.CHRONOS_BOLT_BASE_DEFAULT_CONTEXT if setting == "default" else 512
        eff = np.minimum(avail, nominal)
    elif model == "moirai2_small":
        nominal = run_grid.MOIRAI2_SMALL_DEFAULT_CONTEXT if setting == "default" else 512
        eff = np.minimum(avail, nominal)
    elif model == "timesfm25":
        nominal = run_grid.TIMESFM25_MAX_CONTEXT_CAP if setting == "default" else 512
        eff = np.minimum(avail, nominal)
    elif model == "toto":
        target_default = run_grid.TOTO_DEFAULT_CONTEXT if setting == "default" else 512
        nominal = target_default
        ctx_len = toto_context_length(target_default, min_len, W, P)
        eff = np.minimum(avail, ctx_len)
    else:
        raise ValueError(model)
    return nominal, eff


def main():
    selected = json.loads(SELECTED_CONFIGS_PATH.read_text())["selected"]
    rows = []
    per_config_cache = {}

    for entry in selected:
        base, freq, term = entry["base_dataset"], entry["frequency"], entry["term"]
        dataset_name = f"{base}/{freq}"
        storage_name = run_grid.resolve_storage_name(dataset_name)
        ds = Dataset(name=storage_name, term=term, to_univariate=False)

        P = ds.prediction_length
        W = ds.windows
        min_len = ds._min_series_length
        target_dim = ds.target_dim
        lengths = series_lengths(ds)
        assert lengths.min() == min_len, (entry["config"], lengths.min(), min_len)

        avail = available_history(lengths, P, W).ravel()
        n_raw = avail.size  # n_series * W, before any to_univariate split

        per_config_cache[entry["config"]] = dict(
            P=P, W=W, min_len=min_len, target_dim=target_dim, avail=avail, n_raw=n_raw
        )

        for model in MODELS:
            to_univ = run_grid.to_univariate_for(model, target_dim)
            n_series_windows = n_raw * (target_dim if to_univ else 1)
            for setting in SETTINGS:
                nominal, eff = effective_for_model(model, setting, avail, target_dim, min_len, W, P)
                rows.append(dict(
                    model=model,
                    config=entry["config"],
                    setting=setting,
                    nominal=int(nominal),
                    eff_min=int(eff.min()),
                    eff_median=float(np.median(eff)),
                    eff_max=int(eff.max()),
                    n_series_windows=int(n_series_windows),
                ))

    # ---- write CSV (deterministic: iterate selected order, then MODELS, then SETTINGS) ----
    import csv
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "setting", "nominal",
                                                "eff_min", "eff_median", "eff_max", "n_series_windows"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()

    # ---- key questions ----
    default_rows = [r for r in rows if r["setting"] == "default"]
    below_nominal = [r for r in default_rows if r["eff_median"] < r["nominal"]]

    named_check = {
        "us_births/W/short": [r for r in default_rows if r["config"] == "us_births/W/short"],
        "bizitobs_l2c/H/long": [r for r in default_rows if r["config"] == "bizitobs_l2c/H/long"],
        "bizitobs_application/10S/medium_timesfm25": [
            r for r in default_rows
            if r["config"] == "bizitobs_application/10S/medium" and r["model"] == "timesfm25"
        ],
    }

    toto_m4 = [r for r in rows if r["model"] == "toto" and r["config"] == "m4_daily/D/short"]

    setting512_rows = [r for r in rows if r["setting"] == "512"]
    weaker_than_512 = [r for r in setting512_rows if r["eff_median"] < 512]
    weaker_than_512_maxbased = [r for r in setting512_rows if r["eff_max"] < 512]

    return dict(
        rows=rows, md5=md5, below_nominal=below_nominal, named_check=named_check,
        toto_m4=toto_m4, weaker_than_512=weaker_than_512,
        weaker_than_512_maxbased=weaker_than_512_maxbased,
        per_config_cache=per_config_cache,
    )


if __name__ == "__main__":
    result = main()
    print("md5:", result["md5"])
    print("rows:", len(result["rows"]))
    print("below_nominal (default, eff_median<nominal):", len(result["below_nominal"]))
    for r in result["below_nominal"]:
        print("  ", r["model"], r["config"], "eff_median=", r["eff_median"], "nominal=", r["nominal"])
    print("named checks:")
    for k, v in result["named_check"].items():
        for r in v:
            print("  ", k, "->", r["model"], r["config"], "eff_median=", r["eff_median"], "nominal=", r["nominal"])
    print("toto m4_daily:", result["toto_m4"])
    print("weaker_than_512 (eff_median<512):", result["weaker_than_512"])
    print("weaker_than_512 (eff_max<512):", result["weaker_than_512_maxbased"])
