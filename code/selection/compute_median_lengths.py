#!/usr/bin/env python3
"""Computes the median series length for each of the 97 GIFT-Eval configs,
by reading the raw downloaded data files in harness/data/ (arrow files; the
'target' column holds each series). This is the input the v2 eligibility
filter (a config needs a median length >= 1024 points to make the
"default context length" vs "512" comparison mean anything) is built from.

Needs pyarrow, which is not in the plain stdlib. Run with the interpreter
that has it:
  harness/envs/chronos_bolt_base/bin/python3 selection/compute_median_lengths.py

Writes selection/config_median_lengths.json: config -> {median_length,
n_series, data_dir}. select_configs.py (v2) only reads that JSON, so it
stays stdlib-only and deterministic.
"""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve().parent
SNAPSHOT_CSV = HERE.parent / "snapshot_2026-09-07" / "results" / "seasonal_naive" / "all_results.csv"
DATA_ROOT = HERE.parent / "harness" / "data"
OUTPUT_JSON = HERE / "config_median_lengths.json"
# NOTE (repo-layout-only): SNAPSHOT_CSV and DATA_ROOT above assume the
# original repo layout (this file's parent's siblings snapshot_2026-09-07/
# and harness/data/ under papers/fmts/), not the supplementary zip's layout
# (snapshot/snapshot_2026-09-07/, and no downloaded GIFT-Eval dataset shipped
# at all -- see code/harness/README.md for the download command). Not meant
# to be rerun from this zip; its output, config_median_lengths.json, ships
# already computed and is what select_configs.py actually reads.

# Config's base-dataset name (as it appears in the results CSV) -> the
# directory name actually used on disk under harness/data/. Most names
# match directly; a few differ because the raw download keeps the source
# dataset's original name (its "_with_missing" suffix, or its original
# capitalization). This is GIFT-Eval's own mapping -- read from
# harness/gift-eval/notebooks/*.ipynb (the `pretty_names` dict every
# submission notebook uses to turn a storage name back into the name the
# results CSV uses) -- and independently confirmed against the actual
# directory listing under harness/data/.
BASE_DIR_MAP = {
    "bitbrains_fast_storage": "bitbrains_fast_storage",
    "bitbrains_rnd": "bitbrains_rnd",
    "bizitobs_application": "bizitobs_application",
    "bizitobs_l2c": "bizitobs_l2c",
    "bizitobs_service": "bizitobs_service",
    "car_parts": "car_parts_with_missing",
    "covid_deaths": "covid_deaths",
    "electricity": "electricity",
    "ett1": "ett1",
    "ett2": "ett2",
    "hierarchical_sales": "hierarchical_sales",
    "hospital": "hospital",
    "jena_weather": "jena_weather",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m4_daily": "m4_daily",
    "m4_hourly": "m4_hourly",
    "m4_monthly": "m4_monthly",
    "m4_quarterly": "m4_quarterly",
    "m4_weekly": "m4_weekly",
    "m4_yearly": "m4_yearly",
    "m_dense": "M_DENSE",
    "restaurant": "restaurant",
    "saugeen": "saugeenday",
    "solar": "solar",
    "sz_taxi": "SZ_TAXI",
    "temperature_rain": "temperature_rain_with_missing",
    "us_births": "us_births",
}

# Base datasets whose directory on disk is split into one sub-folder per
# frequency (e.g. harness/data/solar/W/data-00000-of-00001.arrow). Every
# other base dataset has exactly one frequency and stores its one arrow
# file directly under the base directory.
FREQ_SUBDIR_BASES = {
    "bitbrains_fast_storage", "bitbrains_rnd", "bizitobs_l2c", "electricity",
    "ett1", "ett2", "hierarchical_sales", "jena_weather", "kdd_cup_2018",
    "loop_seattle", "m_dense", "saugeen", "solar", "sz_taxi", "us_births",
}


def load_configs(csv_path: Path):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    configs = [row["dataset"] for row in rows]
    assert len(configs) == 97, f"expected 97 configs, got {len(configs)} in {csv_path}"
    assert len(set(configs)) == 97, "duplicate config names found"
    return configs


def resolve_data_file(base_dataset: str, frequency: str):
    """Returns (arrow_file_path, data_dir_str) for one (base, frequency) pair."""
    if base_dataset not in BASE_DIR_MAP:
        raise KeyError(f"no directory mapping for base dataset {base_dataset!r}")
    dir_name = BASE_DIR_MAP[base_dataset]
    data_dir = f"{dir_name}/{frequency}" if base_dataset in FREQ_SUBDIR_BASES else dir_name
    return DATA_ROOT / data_dir / "data-00000-of-00001.arrow", data_dir


def series_lengths(arrow_path: Path):
    """Length (number of time points) of every series' target in one file.
    Handles both univariate (target: list<float>) and multivariate
    (target: fixed_size_list<list<float>>[num_variates]) layouts -- for
    multivariate rows every variate has the same length, so we read
    variate 0, matching how harness/gift-eval/src/gift_eval/analysis
    computes per-series length."""
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.open_stream(source).read_all()
    col = table.column("target").combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        col = pc.list_element(col, 0)
    return pc.list_value_length(col).to_numpy()


def main():
    configs = load_configs(SNAPSHOT_CSV)
    csv_md5 = hashlib.md5(SNAPSHOT_CSV.read_bytes()).hexdigest()

    pair_cache = {}  # (base_dataset, frequency) -> {median_length, n_series, data_dir}
    missing = []
    result = {}
    for config in configs:
        base_dataset, frequency, term = config.split("/")
        pair = (base_dataset, frequency)
        if pair not in pair_cache:
            try:
                arrow_path, data_dir = resolve_data_file(base_dataset, frequency)
            except KeyError as e:
                missing.append(f"{config}: {e}")
                continue
            if not arrow_path.exists():
                missing.append(f"{config}: no data file at {arrow_path}")
                continue
            lengths = series_lengths(arrow_path)
            pair_cache[pair] = {
                "median_length": float(statistics.median(lengths.tolist())),
                "n_series": int(len(lengths)),
                "data_dir": data_dir,
            }
        result[config] = pair_cache[pair]

    if missing:
        print("STOP -- data missing on disk for these configs (download may still be running):")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    assert len(result) == 97, f"resolved {len(result)}/97 configs"

    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    out_md5 = hashlib.md5(OUTPUT_JSON.read_bytes()).hexdigest()

    print(f"input CSV: {SNAPSHOT_CSV} (md5={csv_md5})")
    print(f"resolved {len(pair_cache)} distinct (base dataset, frequency) data files "
          f"covering all {len(result)} configs")
    print(f"wrote {OUTPUT_JSON} (md5={out_md5})")


if __name__ == "__main__":
    main()
