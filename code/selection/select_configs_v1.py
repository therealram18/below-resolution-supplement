#!/usr/bin/env python3
"""Pre-registered selection of 12 GIFT-Eval dataset-configs for the
protocol-injected-variance experiment (Result 2).

Rule, in plain language (full text in SELECTION_RULE.md):
  1. Read the 97 config names (e.g. "m4_weekly/W/short") from a model's
     all_results.csv in the pinned snapshot. Config names only -- no
     performance numbers are read or used.
  2. Split each config into (base_dataset, frequency, term). There are
     three terms: short, medium, long.
  3. Split 12 slots across the three terms in proportion to how many of
     the 97 configs belong to each term, using largest-remainder rounding
     (ties broken alphabetically by term name).
  4. Process terms in the fixed order short, medium, long. For each term,
     shuffle that term's configs with random.Random(20260912), then walk
     the shuffled list twice:
       pass 1: take a config only if its base dataset has not been used
                by ANY term yet, AND its frequency has not been used
                within this term yet.
       pass 2 (only if slots remain): same, but drop the frequency
                condition -- base dataset must still be unused.
     This guarantees 12 distinct base datasets across the whole
     selection, one config per base dataset.

No model results are read anywhere in this script. Deterministic: same
seed, same input file, same output every run.
"""
import csv
import hashlib
import json
import random
from pathlib import Path

SEED = 20260912
TOTAL_SLOTS = 12
TERM_ORDER = ["short", "medium", "long"]

HERE = Path(__file__).resolve().parent
SNAPSHOT_CSV = HERE.parent / "snapshot_2026-09-07" / "results" / "seasonal_naive" / "all_results.csv"
OUTPUT_JSON = HERE / "selected_configs.json"
# NOTE (repo-layout-only): SNAPSHOT_CSV above assumes the original repo layout
# (this file's parent's sibling snapshot_2026-09-07/ under papers/fmts/), not
# the supplementary zip's reorganized layout (snapshot/snapshot_2026-09-07/).
# Superseded by select_configs.py (v2) -- kept only as the pre-registration
# record of the earlier rule; not meant to be rerun (OUTPUT_JSON also still
# names the *current* selected_configs.json, i.e. rerunning this would
# overwrite the v2 output it was superseded by, not just miss the snapshot).


def load_configs(csv_path: Path):
    """Read the 'dataset' column (config names) from a results CSV. Names only."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    configs = [row["dataset"] for row in rows]
    assert len(configs) == 97, f"expected 97 configs, got {len(configs)} in {csv_path}"
    assert len(set(configs)) == 97, "duplicate config names found"
    return configs


def parse_config(config: str):
    parts = config.split("/")
    assert len(parts) == 3, f"unexpected config format: {config!r}"
    base_dataset, frequency, term = parts
    assert term in TERM_ORDER, f"unexpected term {term!r} in {config!r}"
    return base_dataset, frequency, term


def largest_remainder_allocation(counts: dict, total_slots: int):
    """Hamilton's largest-remainder method. Ties broken alphabetically by key."""
    universe = sum(counts.values())
    quotas = {t: counts[t] * total_slots / universe for t in counts}
    floors = {t: int(quotas[t]) for t in counts}
    remainders = {t: quotas[t] - floors[t] for t in counts}
    remaining = total_slots - sum(floors.values())
    order = sorted(counts, key=lambda t: (-remainders[t], t))
    allocation = dict(floors)
    for t in order[:remaining]:
        allocation[t] += 1
    return allocation, quotas, remainders


def select_for_term(term_configs, slots_needed, used_base_datasets):
    """term_configs: list of (config, base_dataset, frequency) for one term."""
    rng = random.Random(SEED)
    shuffled = list(term_configs)
    rng.shuffle(shuffled)

    selected = []
    used_frequencies = set()
    remaining = [c for c in shuffled]

    # pass 1: new base dataset AND new frequency within this term
    still_remaining = []
    for item in remaining:
        config, base_dataset, frequency = item
        if len(selected) >= slots_needed:
            still_remaining.append(item)
            continue
        if base_dataset not in used_base_datasets and frequency not in used_frequencies:
            selected.append(item)
            used_base_datasets.add(base_dataset)
            used_frequencies.add(frequency)
        else:
            still_remaining.append(item)

    # pass 2: relax the frequency preference, base dataset still must be new
    if len(selected) < slots_needed:
        next_remaining = []
        for item in still_remaining:
            config, base_dataset, frequency = item
            if len(selected) >= slots_needed:
                next_remaining.append(item)
                continue
            if base_dataset not in used_base_datasets:
                selected.append(item)
                used_base_datasets.add(base_dataset)
                used_frequencies.add(frequency)
            else:
                next_remaining.append(item)
        still_remaining = next_remaining

    assert len(selected) == slots_needed, (
        f"term ran out of eligible base datasets: got {len(selected)}/{slots_needed}"
    )
    return selected


def main():
    configs = load_configs(SNAPSHOT_CSV)
    parsed = [(c,) + parse_config(c) for c in configs]  # (config, base_dataset, frequency, term)

    by_term = {t: [] for t in TERM_ORDER}
    for config, base_dataset, frequency, term in parsed:
        by_term[term].append((config, base_dataset, frequency))

    counts = {t: len(by_term[t]) for t in TERM_ORDER}
    allocation, quotas, remainders = largest_remainder_allocation(counts, TOTAL_SLOTS)

    used_base_datasets = set()
    selected_all = []
    for term in TERM_ORDER:
        picked = select_for_term(by_term[term], allocation[term], used_base_datasets)
        for config, base_dataset, frequency in picked:
            selected_all.append(
                {
                    "config": config,
                    "base_dataset": base_dataset,
                    "frequency": frequency,
                    "term": term,
                }
            )

    assert len(selected_all) == TOTAL_SLOTS
    assert len({s["base_dataset"] for s in selected_all}) == TOTAL_SLOTS

    # deterministic, readable ordering in the output file
    selected_all.sort(key=lambda s: (TERM_ORDER.index(s["term"]), s["config"]))

    result = {
        "seed": SEED,
        "rule_version": 1,
        "selected": selected_all,
        "term_allocation": {
            "counts_of_97": counts,
            "quota": {t: quotas[t] for t in TERM_ORDER},
            "remainder": {t: remainders[t] for t in TERM_ORDER},
            "slots": allocation,
        },
        "universe_size": 97,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    print("Term allocation (of 97 configs -> 12 slots):")
    for t in TERM_ORDER:
        print(
            f"  {t:7s} count={counts[t]:2d}  quota={quotas[t]:.4f}  "
            f"remainder={remainders[t]:.4f}  slots={allocation[t]}"
        )
    print()
    print("Selected 12 configs:")
    for s in selected_all:
        print(f"  {s['term']:6s} {s['config']:28s} base={s['base_dataset']:20s} freq={s['frequency']}")
    print()
    md5 = hashlib.md5(OUTPUT_JSON.read_bytes()).hexdigest()
    print(f"wrote {OUTPUT_JSON} (md5={md5})")


if __name__ == "__main__":
    main()
