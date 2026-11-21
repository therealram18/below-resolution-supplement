#!/usr/bin/env python3
"""Write the GPU round's job files (one JSON object per line) to jobs/.

Job lists of DESIGN_ROUND3.md Revision 1:
  gate.jsonl          Chronos-Bolt, Moirai-2.0, TimesFM-2.5 x {default, 512} x the 12 first-round configs (72)
  g3_seed42.jsonl     Toto setting A, seed 42 x {default, 512} x the 12, without the 512 run of
                      m4_daily/D/short (both lengths give Toto the same 79 steps; d_c = 0 by rule) (23)
  g4_seeds.jsonl      the same for seeds 43 and 44 (46)
  g1_det.jsonl        Chronos-Bolt, Moirai-2.0, TimesFM-2.5 x {default, 512} x the 58 other eligible configs (348)
  g1_toto.jsonl       Toto setting A, seed 42 x {default, 512} x the 58 (116)
  g3_fallback.jsonl   Toto setting A_tf32off, seed 42, default length, the 12 (12)
  g2.jsonl            timesfm3 and timestral209m x {default, 512} x the 12 (48)
A Toto 512 run is left out where Toto's maximal history (shortest series minus
the evaluation windows) is at most 512 steps; on the 70 that is m4_daily/D/short
only (asserted).

Eligible = median series length >= 1024 in selection/config_median_lengths.json
(rule version 2, DESIGN_ROUND3.md G1). The 12 first-round configs come from
selection/selected_configs.json ("selected"[*]["config"] = "base/freq/term").

Each job: id, model, env, toto_setting, seed, context, dataset ("base/freq"),
term, weight (estimated seconds), exclusive (every Toto sampling job),
timeout_min (240 for Toto, 90 for every other model; Revision 1).

Weight: the first round's 8 runs of one configuration (4 models x 2 lengths)
took about 48 + series x variates x windows x (0.07 + 0.00028 x prediction
length) seconds in all; one run is an eighth of that, and a Toto sampling run
is counted 4 times that. series, variates, windows and prediction length come
from gift_eval.data.Dataset on the local data, loaded the way run_grid.py
loads it (resolve_storage_name, to_univariate=False).

Run with a python that has gift-eval installed, e.g.
  ../../../harness/envs/chronos_bolt_base/bin/python make_jobs.py
GIFT_EVAL (the data folder) defaults to harness/data here if it exists,
else to the first round's harness/data.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMTS = HERE.parents[2]  # papers/fmts (HERE = papers/fmts/camera_ready/gpu_round/harness)

if "GIFT_EVAL" not in os.environ:
    local = HERE / "data"
    os.environ["GIFT_EVAL"] = str(local if local.exists() else FMTS / "harness" / "data")

from result_paths import base_variant, config_stem  # noqa: E402

FIRST_ROUND_MODELS = ["chronos_bolt_base", "moirai2_small", "timesfm25"]
G2_MODELS = ["timesfm3", "timestral209m"]
CONTEXTS = ["default", "512"]
# Which virtual environment (harness/envs/<env>) runs each model. The G2
# entries come from g2_models.G2_ENVS (written by the G2 author): TimesFM-3
# runs in the timesfm25 environment (timesfm==3.0.2 ships the timesfm3
# module), Timestral-209M in its own `timestral` environment, which
# runpod_setup.sh does not build (recipe: requirements/timestral.txt).
MODEL_ENV = {
    "chronos_bolt_base": "chronos_bolt_base",
    "moirai2_small": "moirai2_small",
    "timesfm25": "timesfm25",
    "toto": "toto",
}
try:
    from g2_models import G2_ENVS
except ImportError:
    raise SystemExit("make_jobs.py needs g2_models.py (G2_ENVS) next to it for the G2 jobs' environments")
MODEL_ENV.update(G2_ENVS)
TOTO_SAMPLING_FACTOR = 4
TIMEOUT_MIN_TOTO = 240
TIMEOUT_MIN_OTHER = 90
TOTO_512_SKIPPED = {"m4_daily/D/short"}  # expected result of the rule below, asserted
BYTES_PER_VALUE = 4  # float32


def split_config(config: str):
    dataset, term = config.rsplit("/", 1)
    return dataset, term


def load_config_info(configs):
    """series, variates, windows, prediction length (and Toto's maximal
    context) per config, via run_grid.py's own name mapping and Dataset."""
    from run_grid import resolve_storage_name
    from gift_eval.data import Dataset

    info = {}
    for config in configs:
        dataset, term = split_config(config)
        storage = resolve_storage_name(dataset)
        probe = Dataset(name=storage, term=term, to_univariate=False)
        series = len(probe.hf_dataset)
        variates = probe.target_dim
        windows = probe.windows
        pl = probe.prediction_length
        min_len = probe._min_series_length
        info[config] = {
            "storage_name": storage,
            "series": series,
            "variates": variates,
            "windows": windows,
            "prediction_length": pl,
            "min_series_length": int(min_len),
            "toto_maximal_context": int(min_len - (windows + 1) * pl),
            # The notebook's wrapper keeps every forecast in a list: 256
            # samples x prediction length x variates per series and window.
            "toto_A_samples_ram_gb": series * windows * pl * variates * 256 * BYTES_PER_VALUE / 1e9,
        }
    return info


def weight_seconds(ci, toto_sampling: bool) -> float:
    eight_runs = 48 + ci["series"] * ci["variates"] * ci["windows"] * (0.07 + 0.00028 * ci["prediction_length"])
    w = eight_runs / 8
    return round(w * TOTO_SAMPLING_FACTOR if toto_sampling else w, 1)


def make_job(model, context, config, ci, toto_setting=None, seed=None):
    dataset, term = split_config(config)
    sampling = model == "toto" and toto_setting not in (None, "point")
    variant = base_variant(model, toto_setting or "point", seed if seed is not None else 42)
    return {
        "id": f"{variant}__ctx_{context}__{config_stem(dataset, term)}",
        "model": model,
        "env": MODEL_ENV[model],
        "toto_setting": toto_setting if model == "toto" else None,
        "seed": seed if sampling else None,
        "context": context,
        "dataset": dataset,
        "term": term,
        "weight": weight_seconds(ci, sampling),
        "exclusive": bool(sampling),
        "timeout_min": TIMEOUT_MIN_TOTO if model == "toto" else TIMEOUT_MIN_OTHER,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selection-dir", default=str(FMTS / "selection"))
    p.add_argument("--jobs-dir", default=str(HERE / "jobs"))
    args = p.parse_args()

    sel = Path(args.selection_dir)
    medians = json.loads((sel / "config_median_lengths.json").read_text())
    first12 = [e["config"] for e in json.loads((sel / "selected_configs.json").read_text())["selected"]]

    eligible = sorted(c for c, v in medians.items() if v["median_length"] >= 1024)
    terms = Counter(c.rsplit("/", 1)[1] for c in eligible)
    bases = {c.split("/")[0] for c in eligible}
    assert len(medians) == 97, len(medians)
    assert len(eligible) == 70, len(eligible)
    assert terms == Counter({"short": 28, "medium": 21, "long": 21}), terms
    assert len(bases) == 18, sorted(bases)
    assert len(first12) == 12 and len(set(first12)) == 12, first12
    assert set(first12) <= set(eligible), sorted(set(first12) - set(eligible))
    other58 = [c for c in eligible if c not in set(first12)]
    assert len(other58) == 58, len(other58)

    info = load_config_info(eligible)
    for c in eligible:  # the selection's data_dir must be the folder run_grid.py opens
        assert medians[c]["data_dir"] == info[c]["storage_name"], (c, medians[c]["data_dir"], info[c]["storage_name"])

    # Toto at 512 gives the same history as at the documented length when the
    # maximal history is at most 512 steps: that 512 run is not made.
    toto_512_skip = {c for c in eligible if info[c]["toto_maximal_context"] <= 512}
    assert toto_512_skip == TOTO_512_SKIPPED, sorted(toto_512_skip)

    def toto_contexts(config):
        return [c for c in CONTEXTS if not (c == "512" and config in toto_512_skip)]

    files = {"gate": [], "g3_seed42": [], "g4_seeds": [], "g1_det": [], "g1_toto": [], "g3_fallback": [], "g2": []}
    for config in first12:
        for model in FIRST_ROUND_MODELS:
            for ctx in CONTEXTS:
                files["gate"].append(make_job(model, ctx, config, info[config]))
        for ctx in toto_contexts(config):
            files["g3_seed42"].append(make_job("toto", ctx, config, info[config], "A", 42))
        for seed in (43, 44):
            for ctx in toto_contexts(config):
                files["g4_seeds"].append(make_job("toto", ctx, config, info[config], "A", seed))
        files["g3_fallback"].append(make_job("toto", "default", config, info[config], "A_tf32off", 42))
        for model in G2_MODELS:
            for ctx in CONTEXTS:
                files["g2"].append(make_job(model, ctx, config, info[config]))
    for config in other58:
        for ctx in CONTEXTS:
            for model in FIRST_ROUND_MODELS:
                files["g1_det"].append(make_job(model, ctx, config, info[config]))
        for ctx in toto_contexts(config):
            files["g1_toto"].append(make_job("toto", ctx, config, info[config], "A", 42))

    expected = {"gate": 72, "g3_seed42": 23, "g4_seeds": 46, "g1_det": 348, "g1_toto": 116, "g3_fallback": 12, "g2": 48}
    for name, n in expected.items():
        assert len(files[name]) == n, (name, len(files[name]))
    all_ids = [j["id"] for jobs in files.values() for j in jobs]
    assert len(all_ids) == len(set(all_ids)), "duplicate job ids"
    assert all(j["exclusive"] == (j["model"] == "toto") for jobs in files.values() for j in jobs)
    assert all(j["timeout_min"] == (240 if j["model"] == "toto" else 90) for jobs in files.values() for j in jobs)

    jobs_dir = Path(args.jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for name, jobs in files.items():
        tmp = jobs_dir / f".{name}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        tmp.replace(jobs_dir / f"{name}.jsonl")
    for old in ("g3g4.jsonl", "g1.jsonl"):  # job lists before Revision 1
        if (jobs_dir / old).exists():
            (jobs_dir / old).unlink()
            print(f"REMOVED old job file {old}")
    (jobs_dir / "config_info.json").write_text(json.dumps(info, indent=1, sort_keys=True))

    print("ASSERTIONS OK: 97 configs, 70 eligible (28 short / 21 medium / 21 long), 18 base datasets, the 12 inside, 58 others; "
          f"Toto 512 left out on {sorted(toto_512_skip)}")
    for name, jobs in files.items():
        tot = sum(j["weight"] for j in jobs)
        print(f"JOBS {name:12s} n={len(jobs):4d} exclusive={sum(j['exclusive'] for j in jobs):4d} weight_total={tot / 3600:8.2f} h")
    print(f"JOBS total n={len(all_ids)}")
    every = sorted((j for jobs in files.values() for j in jobs), key=lambda j: -j["weight"])
    print("TEN HEAVIEST:")
    for j in every[:10]:
        print(f"  {j['weight']:9.1f} s  {j['id']}")
    ram = sorted(info.items(), key=lambda kv: -kv[1]["toto_A_samples_ram_gb"])
    print("LARGEST TOTO-A FORECAST LISTS (RAM held by the notebook wrapper, float32):")
    for c, ci in ram[:5]:
        print(f"  {ci['toto_A_samples_ram_gb']:8.1f} GB  {c}")
    short_ctx = [c for c in eligible if info[c]["toto_maximal_context"] < 4096]
    print(f"TOTO MAXIMAL CONTEXT < 4096 on {len(short_ctx)} configs; < 512 on "
          f"{sum(info[c]['toto_maximal_context'] < 512 for c in eligible)}; <= 0 on "
          f"{sum(info[c]['toto_maximal_context'] <= 0 for c in eligible)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
