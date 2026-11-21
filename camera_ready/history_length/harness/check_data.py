#!/usr/bin/env python3
"""Check c of the GPU round build: every eligible configuration loads through
the same code path run_grid.py uses, without a model.

For each config "base/freq/term" in selection/config_median_lengths.json with
median length >= 1024: resolve_storage_name, Dataset(to_univariate=False),
then for both the univariate-split rule (Chronos-Bolt, Moirai-2.0,
TimesFM-2.5) and Toto's rule the Dataset run_grid.py builds, the first test
input and label, dataset_config_name (must equal the config name), and the
dataset_properties.json lookups run_grid.py makes for the csv row.

Run with a python that has gift-eval installed, from this folder:
  GIFT_EVAL=<data> HARNESS_GIFT_EVAL_REPO=<gift-eval clone> <python> check_data.py
Prints CHECK_DATA_OK n=<n> or one CHECK_DATA_FAIL line per failure.
"""
import json
import sys
import time
from pathlib import Path

import run_grid
from run_grid import (
    DATASET_PROPERTIES_PATH,
    dataset_config_name,
    pretty_key,
    resolve_storage_name,
    to_univariate_for,
)

FMTS = Path(__file__).resolve().parents[3]


def check(config, properties, Dataset):
    dataset_name, term = config.rsplit("/", 1)
    storage = resolve_storage_name(dataset_name)
    probe = Dataset(name=storage, term=term, to_univariate=False)
    out = {"storage": storage, "target_dim": probe.target_dim}
    for model in ("chronos_bolt_base", "toto"):
        to_uni = to_univariate_for(model, probe.target_dim)
        ds = Dataset(name=storage, term=term, to_univariate=to_uni)
        first_input, first_label = next(iter(ds.test_data))
        assert len(first_label["target"].shape) >= 1
        out[f"{model}_to_univariate"] = to_uni
    ds_config = dataset_config_name(dataset_name, term, properties)
    if ds_config != config:
        raise AssertionError(f"dataset column would be {ds_config!r}, expected {config!r}")
    props = properties[pretty_key(dataset_name)]
    out["domain"], out["num_variates"] = props["domain"], props["num_variates"]
    return out


def main():
    from gift_eval.data import Dataset

    medians = json.loads((FMTS / "selection" / "config_median_lengths.json").read_text())
    eligible = sorted(c for c, v in medians.items() if v["median_length"] >= 1024)
    properties = json.loads(DATASET_PROPERTIES_PATH.read_text())
    renamed = sorted({c.split("/")[0] for c in eligible if resolve_storage_name(c.rsplit("/", 1)[0]).split("/")[0] != c.split("/")[0]})
    print(f"data folder (GIFT_EVAL): {run_grid.os.environ['GIFT_EVAL']}")
    print(f"eligible configs: {len(eligible)}; base names that differ from their folder: {renamed}")
    fails = 0
    t0 = time.time()
    for config in eligible:
        try:
            r = check(config, properties, Dataset)
            print(f"OK   {config:42s} -> {r['storage']:34s} dim={r['target_dim']}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"CHECK_DATA_FAIL {config}: {type(exc).__name__}: {exc}")
    print(f"elapsed {time.time() - t0:.1f}s")
    if fails:
        print(f"CHECK_DATA_FAILED n_fail={fails} of {len(eligible)}")
        return 1
    print(f"CHECK_DATA_OK n={len(eligible)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
