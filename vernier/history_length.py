"""Effect on MASE of giving a model 512 steps of history instead of its documented length, from model runs.

Each variant folder holds ctx_default/ and ctx_512/ results. Per unit, d = log(MASE at 512 / MASE at the
documented length); the effect is exp(mean d) - 1, tested by exact sign flips of the sources' totals.
"""
import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .board import MASE
from .stats import holm_adjusted, sign_flip_interval, sign_flip_p, source_totals

FAMILIES = ("bizitobs", "bitbrains", "ett")  # datasets of one collection, counted as one source
ELIGIBLE_MEDIAN_LENGTH = 1024  # a configuration is eligible when its median series length is at least this


class Effect(NamedTuple):
    variant: str
    n_units: int
    n_sources: int
    effect: float  # exp(mean d) - 1: +0.05 means MASE 5% higher at 512
    low: float
    high: float
    p: float
    p_holm: float


def source(unit: str) -> str:
    """Return the source of a unit: its base dataset, with each family of datasets merged into one."""
    base = unit.split("/")[0]
    return next((family for family in FAMILIES if base.startswith(family)), base)


def eligible_units(path):
    """Return the units of a JSON file of each unit's median_length that are long enough to be eligible."""
    lengths = json.loads(Path(path).read_text())
    return [unit for unit, info in lengths.items() if info["median_length"] >= ELIGIBLE_MEDIAN_LENGTH]


def _mase(folder):
    """Return {unit: MASE} from a folder's all_results.csv."""
    with open(folder / "all_results.csv", newline="") as file:
        return {row["dataset"]: float(row[MASE]) for row in csv.DictReader(file)}


def _documented_length(folder, unit):
    """Return the history length the documented run gave the model on `unit`."""
    meta = folder / "ctx_default" / "per_config" / (unit.replace("/", "__") + ".run_meta.json")
    return json.loads(meta.read_text())["context_length_used"]


def differences(folder, units):
    """Return the `units` one variant folder has results on and d = log(MASE at 512 / MASE at the documented length)."""
    documented, at_512 = _mase(folder / "ctx_default"), _mase(folder / "ctx_512")
    for unit in set(documented) - set(at_512):
        if _documented_length(folder, unit) <= 512:  # the 512 run would repeat the documented one, so none was made
            at_512[unit] = documented[unit]
    used = [unit for unit in sorted(documented) if unit in at_512 and unit in units]
    log_ratios = np.log([at_512[unit] / documented[unit] for unit in used])
    if not np.isfinite(log_ratios).all():
        raise ValueError(f"{folder}: a log ratio of MASE is not finite")
    return used, log_ratios


def effects(runs_dir, variants, units):
    """Return the Effect of 512 steps for each variant with results on all `units`, Holm-adjusted over them."""
    rows, p_values = [], []
    for variant in variants:
        used, log_ratios = differences(Path(runs_dir) / variant, units)
        if len(used) < len(units):
            continue
        sources = [source(unit) for unit in used]
        effect = np.expm1(log_ratios.mean())
        low, high = np.expm1(sign_flip_interval(log_ratios, sources))
        rows.append((variant, len(used), len(set(sources)), effect, low, high))
        p_values.append(sign_flip_p(source_totals(log_ratios, sources)[0]))
    return [Effect(*row, p, p_holm) for row, p, p_holm in zip(rows, p_values, holm_adjusted(p_values))]
