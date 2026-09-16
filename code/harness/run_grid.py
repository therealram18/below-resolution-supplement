#!/usr/bin/env python3
"""
Evaluate one open time-series foundation model on a list of GIFT-Eval
dataset-configs, using GIFT-Eval's own Dataset class and gluonts
evaluate_model(), so numbers land in the same all_results.csv shape as
https://huggingface.co/spaces/Salesforce/GIFT-Eval.

BUILD NOTE: this script does not run automatically. It is meant to be
launched, per model, from that model's own venv (harness/envs/<model>/bin/
python), only after the pre-registration gate has cleared. See
harness/README.md for the exact commands and env layout.

Per-config output:
  - one row appended to <out>/<model>/ctx_<context>/all_results.csv, with the
    exact column set GIFT-Eval's sample results file uses.
  - a sidecar <out>/<model>/ctx_<context>/run_meta.json recording wall time,
    context length actually used, model revision, and package versions for
    every config plus the whole run.

Usage:
  envs/chronos_bolt_base/bin/python run_grid.py \\
      --model chronos_bolt_base --configs configs/selected_configs.json \\
      --context default --out results
"""
import argparse
import csv
import json
import os
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIFT_EVAL_SRC = HERE / "gift-eval" / "src"
GIFT_EVAL_NOTEBOOKS = HERE / "gift-eval" / "notebooks"
TOTO_REPO = HERE / "vendor" / "toto"

sys.path.insert(0, str(GIFT_EVAL_SRC))
os.environ.setdefault("GIFT_EVAL", str(HERE / "data"))

# dataset_properties.json ships inside gift-eval's notebooks/ dir and is the
# source of truth for domain / num_variates / frequency per dataset -- every
# gift-eval reference notebook (naive.ipynb, chronos.ipynb, moirai2.ipynb,
# timesfm2p5.ipynb) loads it from exactly this path.
DATASET_PROPERTIES_PATH = GIFT_EVAL_NOTEBOOKS / "dataset_properties.json"

# Same typo/rename table every gift-eval reference notebook hardcodes, to go
# from a GIFT_EVAL directory name to the "pretty" name used in the leaderboard
# dataset column and as the key into dataset_properties.json.
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Base dataset name (as it appears in configs/*.json, e.g. "m_dense") -> the
# directory name gift_eval.data.Dataset actually needs to find it under
# harness/data/. Same table as selection/compute_median_lengths.py's
# BASE_DIR_MAP, which reads harness/gift-eval/notebooks/*.ipynb -- every
# submission notebook's `all_datasets` list is made of these literal
# on-disk names (e.g. "M_DENSE/H", "LOOP_SEATTLE/5T") passed straight into
# Dataset(name=...) -- and is confirmed against the actual data/ listing.
STORAGE_DIR_MAP = {
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

# Base datasets whose storage directory is split into one sub-folder per
# frequency (e.g. data/solar/10T/...). Every other base dataset has exactly
# one frequency and its arrow file sits directly under the base directory --
# same set as compute_median_lengths.py's FREQ_SUBDIR_BASES, confirmed
# against every gift-eval notebook's all_datasets string (e.g. plain
# "bizitobs_application", "m4_daily" with no frequency suffix there).
FREQ_SUBDIR_BASES = {
    "bitbrains_fast_storage", "bitbrains_rnd", "bizitobs_l2c", "electricity",
    "ett1", "ett2", "hierarchical_sales", "jena_weather", "kdd_cup_2018",
    "loop_seattle", "m_dense", "saugeen", "solar", "sz_taxi", "us_births",
}


def resolve_storage_name(dataset_name: str) -> str:
    """Map a leaderboard-config dataset string from configs/*.json (e.g.
    "m_dense/H", "m4_daily/D", "solar/10T") to the name
    gift_eval.data.Dataset needs -- the literal on-disk directory under
    harness/data/, with a frequency sub-path only for datasets that have
    one. This is exactly what every gift-eval reference notebook passes to
    Dataset(name=...) (see harness/gift-eval/notebooks/*.ipynb,
    `all_datasets`), so results stay comparable to the leaderboard."""
    parts = dataset_name.split("/")
    base = parts[0].lower()
    freq = parts[1] if len(parts) > 1 else None
    if base not in STORAGE_DIR_MAP:
        raise KeyError(f"no on-disk storage-dir mapping for base dataset {parts[0]!r} (from config {dataset_name!r})")
    dir_name = STORAGE_DIR_MAP[base]
    if base in FREQ_SUBDIR_BASES:
        if freq is None:
            raise ValueError(f"{dataset_name!r} needs a frequency sub-path but none was given")
        return f"{dir_name}/{freq}"
    return dir_name


CSV_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]

# Per-model "default" context length, each sourced from where GIFT-Eval's own
# reference notebook / the checkpoint's published config gets it -- see
# harness/README.md "Context length per model" for the full writeup,
# including the one real discrepancy we found (Moirai 2.0).
CHRONOS_BOLT_BASE_DEFAULT_CONTEXT = 2048
# Source: huggingface.co/amazon/chronos-bolt-base/raw/main/config.json ->
# chronos_config.context_length. This is baked into the checkpoint; Chronos-
# Bolt's predict()/embed() always truncates to the rightmost
# self.model_context_length values and this cannot be overridden by any
# kwarg (verified against chronos_bolt.py source). Forcing "512" therefore
# means truncating the input series ourselves before calling pipeline.predict.

MOIRAI2_SMALL_DEFAULT_CONTEXT = 4000
# Source: gift-eval/notebooks/moirai2.ipynb hardcodes context_length=4000 in
# the Moirai2Forecast constructor -- that is the number that produces the
# leaderboard's published Moirai-2.0-R-small rows, so we replicate it for
# "default" here even though it disagrees with the checkpoint's own
# huggingface.co/Salesforce/moirai-2.0-R-small/raw/main/config.json, which
# reports "max_seq_len": 512. We could not resolve this gap from public
# documentation (uni2ts's own getting-started notebook uses yet another value,
# context_length=1680, for a different dataset). Flagging rather than
# guessing -- see harness/README.md.

TIMESFM25_MAX_CONTEXT_CAP = 15360
# Source: gift-eval/notebooks/timesfm2p5.ipynb computes, per predict() batch,
# max_context = min(15360, ceil(longest_series_in_batch / patch) * patch) and
# recompiles the model for that shape every batch. TimesFM 2.5's own model
# card / GitHub README advertises "up to 16k context, up from 2048" (16384);
# gift-eval's actual recipe caps at 15360, so that's what we replicate for
# "default" comparability. There is no single fixed "default context" for
# this model -- it's however much history each series has, patch-rounded, up
# to the cap.

TOTO_DEFAULT_CONTEXT = 4096
# Source: gift-eval/notebooks/toto.ipynb DEFAULT_CONTEXT_LENGTH = 4096,
# further capped per-dataset by get_maximal_context_length() (context can
# never exceed the shortest series' available history minus the eval
# windows). Toto's own config.json (huggingface.co/Datadog/Toto-Open-Base-1.0)
# has no max_seq_len-style field, and the model card advertises "variable
# context lengths" -- so unlike Chronos-Bolt and Moirai 2.0, there's no
# architectural cap to report here, 4096 is just gift-eval's practical choice.


def pretty_key(dataset_name: str) -> str:
    key = dataset_name.split("/")[0].lower()
    return PRETTY_NAMES.get(key, key)


def dataset_config_name(dataset_name: str, term: str, properties: dict) -> str:
    """Reproduce the exact `f"{dataset_name}/{freq}/{term}"` naming every
    gift-eval reference notebook uses for the leaderboard's `dataset` column."""
    ds_key = pretty_key(dataset_name)
    if "/" in dataset_name:
        ds_freq = dataset_name.split("/")[1]
    else:
        ds_freq = properties[ds_key]["frequency"]
    return f"{ds_key}/{ds_freq}/{term}"


def _json_default(o):
    """json.dumps default= for numpy scalar types (e.g. int64) that sneak
    into run_meta via context-length computations."""
    return o.item() if hasattr(o, "item") else str(o)


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hf_revision(repo_id: str) -> str:
    """Best-effort commit sha for a public HF model repo. Never needs a
    token (public repos); falls back to 'unknown' rather than failing the
    run if the network call doesn't work."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception as exc:  # noqa: BLE001 -- best-effort metadata only
        return f"unknown ({exc.__class__.__name__})"


# --------------------------------------------------------------------------
# Per-model predictor builders. Each returns (predictor, context_used,
# model_revision, predict_batch_size). Imports are local to the function so
# this file only requires whichever one package is installed in the active
# venv (see harness/envs/<model>/).
# --------------------------------------------------------------------------


def to_univariate_for(model: str, target_dim: int) -> bool:
    if model == "toto":
        # gift-eval/notebooks/toto.ipynb always passes to_univariate=False --
        # Toto is evaluated natively multivariate.
        return False
    # chronos.ipynb, moirai2.ipynb and timesfm2p5.ipynb all use this same
    # rule: split multivariate series into one univariate series per
    # dimension before evaluating. (Moirai2's own notebook markdown claims
    # native multivariate support, but its code does this split anyway and
    # hardcodes target_dim=1 in the predictor -- we replicate the code, not
    # the comment, since the code is what produced the leaderboard numbers.)
    return target_dim != 1


def build_chronos_bolt_base(dataset, context_mode: str, num_samples: int = 20):
    import torch
    from chronos import BaseChronosPipeline, ForecastType
    from gluonts.itertools import batcher
    from gluonts.model.forecast import QuantileForecast, SampleForecast

    model_path = "amazon/chronos-bolt-base"
    context_length = (
        None if context_mode == "default" else int(context_mode)
    )  # None = let the checkpoint use its own fixed 2048, no client truncation

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    class ChronosPredictor:
        def __init__(self):
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_path, device_map=device
            )
            self.prediction_length = dataset.prediction_length
            self.num_samples = num_samples
            # gluonts' evaluate_model does not forward its batch_size into
            # predict(), so the cap must live here. CHRONOS_BATCH_SIZE exists
            # because the largest configs OOM an 80GB H100 at 1024 with the
            # default 2048 context. Batch size never changes forecasts.
            self.max_batch_size = int(os.environ.get("CHRONOS_BATCH_SIZE", "1024"))

        def predict(self, test_data_input, batch_size: int = 1024):
            batch_size = min(batch_size, self.max_batch_size)
            pipeline = self.pipeline
            predict_kwargs = (
                {"num_samples": self.num_samples}
                if pipeline.forecast_type == ForecastType.SAMPLES
                else {}
            )
            for batch in batcher(test_data_input, batch_size=batch_size):
                context = []
                for entry in batch:
                    arr = entry["target"]
                    if context_length is not None:
                        arr = arr[-context_length:]
                    context.append(torch.tensor(arr))
                out = pipeline.predict(
                    context,
                    prediction_length=self.prediction_length,
                    **predict_kwargs,
                ).numpy()
                for item, ts in zip(out, batch):
                    start = ts["start"] + len(ts["target"])
                    if pipeline.forecast_type == ForecastType.SAMPLES:
                        yield SampleForecast(samples=item, start_date=start)
                    else:
                        yield QuantileForecast(
                            forecast_arrays=item,
                            forecast_keys=list(map(str, pipeline.quantiles)),
                            start_date=start,
                        )

    used_context = context_length or CHRONOS_BOLT_BASE_DEFAULT_CONTEXT
    # CHRONOS_BATCH_SIZE: memory relief for the largest configs (e.g.
    # electricity/H/long, loop_seattle/5T/long OOM an 80GB H100 at the
    # default 1024 with full-length series). Batch size does not change
    # forecasts, only how many series are processed per GPU call.
    chronos_bs = int(os.environ.get("CHRONOS_BATCH_SIZE", "1024"))
    return ChronosPredictor(), used_context, hf_revision(model_path), chronos_bs


def build_moirai2_small(dataset, context_mode: str):
    import numpy as np
    import torch
    from gluonts.itertools import batcher
    from gluonts.model.forecast import QuantileForecast
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model_path = "Salesforce/moirai-2.0-R-small"
    context_length = (
        MOIRAI2_SMALL_DEFAULT_CONTEXT if context_mode == "default" else int(context_mode)
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    quantile_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    class MoiraiQuantilePredictor:
        def __init__(self):
            self.model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained(model_path),
                prediction_length=dataset.prediction_length,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            ).to(device)
            self.batch_size = 512

        def predict(self, test_data_input):
            for batch in batcher(test_data_input, batch_size=self.batch_size):
                batch = list(batch)
                past_target = [entry["target"] for entry in batch]
                forecasts = self.model.predict(past_target)
                for item, ts in zip(forecasts, batch):
                    start = ts["start"] + len(ts["target"])
                    yield QuantileForecast(
                        item_id=ts["item_id"],
                        forecast_arrays=item,
                        start_date=start,
                        forecast_keys=list(map(str, quantile_levels)),
                    )

    return MoiraiQuantilePredictor(), context_length, hf_revision(model_path), 512


def build_timesfm25(dataset, context_mode: str):
    import numpy as np
    from gluonts.itertools import batcher
    from gluonts.model.forecast import QuantileForecast
    from timesfm import configs
    from timesfm.timesfm_2p5 import timesfm_2p5_torch

    model_path = "google/timesfm-2.5-200m-pytorch"
    forced_context = None if context_mode == "default" else int(context_mode)
    quantiles = [i / 10.0 for i in range(1, 10)]

    tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(model_path)
    used_context = {"value": None}  # populated on first compiled batch

    class TimesFmPredictor:
        def __init__(self):
            self.prediction_length = dataset.prediction_length

        def predict(self, test_data_input, batch_size: int = 1024):
            for batch in batcher(test_data_input, batch_size=batch_size):
                batch = list(batch)
                context = [np.asarray(entry["target"]) for entry in batch]
                if forced_context is not None:
                    context = [arr[-forced_context:] for arr in context]
                    target = forced_context
                else:
                    target = max(arr.shape[0] for arr in context)
                # round up to a multiple of the model's patch size, same as
                # gift-eval/notebooks/timesfm2p5.ipynb
                patch = tfm.model.p
                max_context = ((target + patch - 1) // patch) * patch
                max_context = min(TIMESFM25_MAX_CONTEXT_CAP, max_context)
                used_context["value"] = max_context
                tfm.compile(
                    forecast_config=configs.ForecastConfig(
                        max_context=max_context,
                        max_horizon=1024,
                        infer_is_positive=True,
                        use_continuous_quantile_head=True,
                        fix_quantile_crossing=True,
                        force_flip_invariance=True,
                        return_backcast=False,
                        normalize_inputs=True,
                        per_core_batch_size=128,
                    ),
                )
                _, full_preds = tfm.forecast(
                    horizon=self.prediction_length, inputs=context
                )
                full_preds = full_preds[:, : self.prediction_length, 1:]
                full_preds = full_preds.transpose((0, 2, 1))
                for item, ts in zip(full_preds, batch):
                    start = ts["start"] + len(ts["target"])
                    yield QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, quantiles)),
                        start_date=start,
                    )

    predictor = TimesFmPredictor()
    # context actually used is data-dependent (see used_context, filled in
    # during predict()); report the requested cap here and the sidecar
    # records the real per-batch value once predict() has run.
    nominal_context = forced_context or TIMESFM25_MAX_CONTEXT_CAP
    return predictor, nominal_context, hf_revision(model_path), 1024


def build_toto(dataset, context_mode: str):
    sys.path.insert(0, str(TOTO_REPO))
    from toto.inference.gluonts_predictor import Multivariate, TotoPredictor
    from toto.model.toto import Toto

    model_path = "Datadog/Toto-Open-Base-1.0"
    target_default = TOTO_DEFAULT_CONTEXT if context_mode == "default" else int(context_mode)

    # gift-eval/notebooks/toto.ipynb: context can never exceed the shortest
    # series' available history minus the eval windows it needs to hold out.
    shortest = dataset._min_series_length
    max_available = shortest - (dataset.windows + 1) * dataset.prediction_length
    context_length = min(target_default, max_available) if max_available > 0 else target_default

    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = Toto.from_pretrained(model_path).to(device).eval()

    predictor = TotoPredictor.create_for_eval(
        model=model,
        prediction_length=dataset.prediction_length,
        context_length=context_length,
        mode=Multivariate(batch_size=16),
        samples_per_batch=256,
    )
    return predictor, context_length, hf_revision(model_path), 256


MODEL_BUILDERS = {
    "chronos_bolt_base": build_chronos_bolt_base,
    "moirai2_small": build_moirai2_small,
    "timesfm25": build_timesfm25,
    "toto": build_toto,
}

MODEL_PACKAGES = {
    "chronos_bolt_base": ["chronos-forecasting", "torch", "gluonts"],
    "moirai2_small": ["uni2ts", "torch", "gluonts"],
    "timesfm25": ["timesfm", "torch", "gluonts"],
    "toto": ["torch", "gluonts"],  # toto-ts is vendored, not pip-metadata-visible unless installed -e
}


def run(args):
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]

    configs_list = json.loads(Path(args.configs).read_text())
    properties = json.loads(DATASET_PROPERTIES_PATH.read_text())

    out_dir = Path(args.out) / args.model / f"ctx_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"
    meta_path = out_dir / "run_meta.json"

    new_file = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(CSV_COLUMNS)

    run_meta = {
        "model": args.model,
        "context_mode": args.context,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_versions": package_versions(MODEL_PACKAGES[args.model]),
        "gift_eval_commit": os.environ.get("GIFT_EVAL_COMMIT", "unknown"),
        "configs": [],
    }

    for entry in configs_list:
        dataset_name, term = entry["dataset"], entry["term"]
        ds_config = None
        t0 = time.time()
        record = {"dataset": dataset_name, "term": term}
        try:
            storage_name = resolve_storage_name(dataset_name)
            probe = Dataset(name=storage_name, term=term, to_univariate=False)
            to_univariate = to_univariate_for(args.model, probe.target_dim)
            dataset = Dataset(name=storage_name, term=term, to_univariate=to_univariate)
            ds_config = dataset_config_name(dataset_name, term, properties)

            predictor, context_used, revision, batch_size = MODEL_BUILDERS[args.model](
                dataset, args.context
            )
            season_length = get_seasonality(dataset.freq)

            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            writer.writerow(
                [
                    ds_config,
                    args.model,
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    properties[pretty_key(dataset_name)]["domain"],
                    properties[pretty_key(dataset_name)]["num_variates"],
                ]
            )
            csv_file.flush()

            record.update(
                {
                    "ds_config": ds_config,
                    "status": "ok",
                    "context_length_used": context_used,
                    "model_revision": revision,
                    "wall_time_seconds": time.time() - t0,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            record.update(
                {
                    "ds_config": ds_config,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_time_seconds": time.time() - t0,
                }
            )
            print(f"[FAILED] {dataset_name}/{term}: {type(exc).__name__}: {exc}", file=sys.stderr)

        run_meta["configs"].append(record)
        meta_path.write_text(json.dumps(run_meta, indent=2, default=_json_default))
        print(f"[{record['status']}] {ds_config or dataset_name + '/' + term} -> {csv_path}")

    csv_file.close()
    run_meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_path.write_text(json.dumps(run_meta, indent=2, default=_json_default))
    return any(c["status"] == "failed" for c in run_meta["configs"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    p.add_argument("--configs", required=True, help="Path to a JSON list of {\"dataset\": ..., \"term\": ...}")
    p.add_argument("--context", required=True, choices=["default", "512"])
    p.add_argument("--out", required=True, help="Output directory root")
    return p.parse_args()


if __name__ == "__main__":
    had_failures = run(parse_args())
    if had_failures:
        sys.exit(1)
