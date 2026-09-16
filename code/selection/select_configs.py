#!/usr/bin/env python3
"""Pre-registered selection of 12 GIFT-Eval dataset-configs for the
protocol-injected-variance experiment (Result 2). Rule version 2 (full
text in SELECTION_RULE.md; what changed from v1 and why is in that file's
"Revision history" section).

Rule, in plain language:
  1. Read the 97 config names (e.g. "m4_weekly/W/short") from a model's
     all_results.csv in the pinned snapshot. Config names only -- no
     performance numbers are read or used.
  2. ELIGIBILITY FILTER: a config only qualifies if the underlying data's
     median series length is >= 1024 points, read from
     config_median_lengths.json (itself computed from the downloaded data,
     see compute_median_lengths.py). Below that, the two context-length
     settings we test (a model's own default, and 512) would show the
     model an identical history, so the config could not measure anything.
     This drops 27 of the 97 configs.
  3. Split each surviving config into (base_dataset, frequency, term).
     There are three terms: short, medium, long.
  4. Split 12 slots across the three terms in proportion to how many
     ELIGIBLE configs belong to each term (not all 97), using
     largest-remainder rounding (ties broken alphabetically by term name).
  5. Process terms in the fixed order short, medium, long. For each term,
     sort that term's eligible configs alphabetically (so the result does
     not depend on the CSV's row order), then shuffle with
     random.Random(20260912), then walk the shuffled list up to three
     times:
       pass 1: take a config only if its base dataset has not been used
                by ANY term yet, AND its domain (Energy, Transport, etc.)
                has not been used by ANY term yet, AND its frequency has
                not been used within this term yet.
       pass 2 (only if slots remain): drop the domain condition.
       pass 3 (only if slots remain): also drop the frequency condition.
     The base-dataset condition is never dropped -- it guarantees 12
     distinct base datasets across the whole selection.

No model results are read anywhere in this script. Deterministic: same
seed, same two input files (the CSV and config_median_lengths.json), same
output every run -- verified by running twice and comparing md5.
"""
import csv
import hashlib
import json
import random
from pathlib import Path

SEED = 20260912
TOTAL_SLOTS = 12
TERM_ORDER = ["short", "medium", "long"]
MIN_MEDIAN_LENGTH = 1024

HERE = Path(__file__).resolve().parent
SNAPSHOT_CSV = HERE.parent / "snapshot_2026-09-07" / "results" / "seasonal_naive" / "all_results.csv"
MEDIAN_LENGTHS_JSON = HERE / "config_median_lengths.json"
DOMAIN_PROPERTIES_JSON = HERE.parent / "harness" / "gift-eval" / "notebooks" / "dataset_properties.json"
# NOTE (repo-layout-only): SNAPSHOT_CSV and DOMAIN_PROPERTIES_JSON above assume
# the original repo layout (this file's parent's siblings snapshot_2026-09-07/
# and harness/gift-eval/ under papers/fmts/), not the supplementary zip's
# reorganized layout (snapshot/snapshot_2026-09-07/, and no gift-eval clone
# shipped at all). Left unpatched on purpose -- per README.txt this script is
# documentation-only in the zip regardless, since DOMAIN_PROPERTIES_JSON's
# source (the GIFT-Eval repo clone) isn't shipped either way.
OUTPUT_JSON = HERE / "selected_configs.json"


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


def select_for_term(term_configs, slots_needed, used_base_datasets, used_domains_global, domain_of):
    """term_configs: list of (config, base_dataset, frequency) for one term,
    already filtered to eligible configs. Mutates used_base_datasets and
    used_domains_global in place as picks are made."""
    # sort BEFORE shuffling: makes the result independent of the CSV's row
    # order, since random.Random(SEED).shuffle() is only deterministic
    # given a fixed starting order
    sorted_configs = sorted(term_configs)
    rng = random.Random(SEED)
    shuffled = list(sorted_configs)
    rng.shuffle(shuffled)

    selected = []
    used_frequencies_term = set()

    def run_pass(candidates, require_new_domain, require_new_frequency):
        still_remaining = []
        for item in candidates:
            config, base_dataset, frequency = item
            if len(selected) >= slots_needed:
                still_remaining.append(item)
                continue
            if base_dataset in used_base_datasets:
                still_remaining.append(item)
                continue
            if require_new_domain and domain_of[base_dataset] in used_domains_global:
                still_remaining.append(item)
                continue
            if require_new_frequency and frequency in used_frequencies_term:
                still_remaining.append(item)
                continue
            selected.append(item)
            used_base_datasets.add(base_dataset)
            used_domains_global.add(domain_of[base_dataset])
            used_frequencies_term.add(frequency)
        return still_remaining

    # pass 1: new base dataset (hard rule) AND new domain globally AND new frequency in this term
    remaining = run_pass(shuffled, require_new_domain=True, require_new_frequency=True)
    # pass 2: drop the domain condition
    if len(selected) < slots_needed:
        remaining = run_pass(remaining, require_new_domain=False, require_new_frequency=True)
    # pass 3: also drop the frequency condition
    if len(selected) < slots_needed:
        remaining = run_pass(remaining, require_new_domain=False, require_new_frequency=False)

    assert len(selected) == slots_needed, (
        f"term ran out of eligible base datasets: got {len(selected)}/{slots_needed}"
    )
    return selected


def main():
    configs = load_configs(SNAPSHOT_CSV)
    csv_md5 = hashlib.md5(SNAPSHOT_CSV.read_bytes()).hexdigest()

    median_lengths = json.loads(MEDIAN_LENGTHS_JSON.read_text())
    median_lengths_md5 = hashlib.md5(MEDIAN_LENGTHS_JSON.read_bytes()).hexdigest()
    assert set(median_lengths) == set(configs), (
        "config_median_lengths.json does not cover exactly the 97 configs in the snapshot CSV -- "
        "re-run compute_median_lengths.py"
    )

    domain_properties = json.loads(DOMAIN_PROPERTIES_JSON.read_text())
    domain_of = {base: info["domain"] for base, info in domain_properties.items()}

    parsed = [(c,) + parse_config(c) for c in configs]  # (config, base_dataset, frequency, term)
    for config, base_dataset, frequency, term in parsed:
        assert base_dataset in domain_of, f"no domain label for base dataset {base_dataset!r}"

    eligible = [p for p in parsed if median_lengths[p[0]]["median_length"] >= MIN_MEDIAN_LENGTH]
    ineligible = [p for p in parsed if median_lengths[p[0]]["median_length"] < MIN_MEDIAN_LENGTH]

    by_term_all = {t: [] for t in TERM_ORDER}
    by_term_eligible = {t: [] for t in TERM_ORDER}
    for config, base_dataset, frequency, term in parsed:
        by_term_all[term].append(config)
    for config, base_dataset, frequency, term in eligible:
        by_term_eligible[term].append((config, base_dataset, frequency))

    counts_of_97 = {t: len(by_term_all[t]) for t in TERM_ORDER}
    eligible_counts = {t: len(by_term_eligible[t]) for t in TERM_ORDER}
    allocation, quotas, remainders = largest_remainder_allocation(eligible_counts, TOTAL_SLOTS)

    used_base_datasets = set()
    used_domains_global = set()
    selected_all = []
    for term in TERM_ORDER:
        picked = select_for_term(
            by_term_eligible[term], allocation[term], used_base_datasets, used_domains_global, domain_of
        )
        for config, base_dataset, frequency in picked:
            selected_all.append(
                {
                    "config": config,
                    "base_dataset": base_dataset,
                    "frequency": frequency,
                    "term": term,
                    "domain": domain_of[base_dataset],
                    "median_length": median_lengths[config]["median_length"],
                }
            )

    assert len(selected_all) == TOTAL_SLOTS
    assert len({s["base_dataset"] for s in selected_all}) == TOTAL_SLOTS

    # deterministic, readable ordering in the output file
    selected_all.sort(key=lambda s: (TERM_ORDER.index(s["term"]), s["config"]))

    domain_composition = {}
    for s in selected_all:
        domain_composition[s["domain"]] = domain_composition.get(s["domain"], 0) + 1

    # a domain has zero eligible configs if no eligible config anywhere maps to it
    eligible_domains = {domain_of[p[1]] for p in eligible}
    domains_with_zero_eligible = sorted(set(domain_of.values()) - eligible_domains)

    result = {
        "seed": SEED,
        "rule_version": 2,
        "eligibility_filter": {
            "min_median_length": MIN_MEDIAN_LENGTH,
            "eligible_of_97": len(eligible),
            "ineligible_of_97": len(ineligible),
        },
        "selected": selected_all,
        "term_allocation": {
            "counts_of_97": counts_of_97,
            "eligible_counts": eligible_counts,
            "quota": {t: quotas[t] for t in TERM_ORDER},
            "remainder": {t: remainders[t] for t in TERM_ORDER},
            "slots": allocation,
        },
        "domain_composition": domain_composition,
        "domains_with_zero_eligible_configs": domains_with_zero_eligible,
        "universe_size": 97,
        "input_csv_md5": csv_md5,
        "config_median_lengths_json_md5": median_lengths_md5,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    print("Eligibility filter: median series length >= 1024")
    print(f"  eligible: {len(eligible)}/97   ineligible: {len(ineligible)}/97")
    print()
    print("Term allocation (of ELIGIBLE configs -> 12 slots):")
    for t in TERM_ORDER:
        print(
            f"  {t:7s} total_of_97={counts_of_97[t]:2d}  eligible={eligible_counts[t]:2d}  "
            f"quota={quotas[t]:.4f}  remainder={remainders[t]:.4f}  slots={allocation[t]}"
        )
    print()
    print("Selected 12 configs:")
    for s in selected_all:
        print(
            f"  {s['term']:6s} {s['config']:28s} base={s['base_dataset']:20s} "
            f"freq={s['frequency']:5s} domain={s['domain']:12s} median_len={s['median_length']:.0f}"
        )
    print()
    print("Domain composition:", domain_composition)
    print("Domains with zero eligible configs:", domains_with_zero_eligible or "none")
    print()
    md5 = hashlib.md5(OUTPUT_JSON.read_bytes()).hexdigest()
    print(f"input CSV md5: {csv_md5}")
    print(f"config_median_lengths.json md5: {median_lengths_md5}")
    print(f"wrote {OUTPUT_JSON} (md5={md5})")


if __name__ == "__main__":
    main()
