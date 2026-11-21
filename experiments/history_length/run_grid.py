#!/usr/bin/env python3
"""
Evaluate one open time-series foundation model on a list of GIFT-Eval
dataset-configs, using GIFT-Eval's own Dataset class and gluonts
evaluate_model(), so numbers land in the same all_results.csv shape as
https://huggingface.co/spaces/Salesforce/GIFT-Eval.

BUILD NOTE: this script does not run automatically. It is meant to be
launched, per model, from that model's own venv (harness/envs/<model>/bin/
python), only after the pre-registration gate has cleared. See
record/submitted/harness/README.md for the exact commands and env layout.

Per-config output (GPU round: one pair of files per run, so several runs of
the same model and context can write at the same time):
  - <out>/<variant>/ctx_<context>/per_config/<dataset with "/" -> "__">__<term>.csv
    one header line plus one row, the exact column set GIFT-Eval's sample
    results file uses (the `dataset` column is "base/freq/term").
  - the matching ...__<term>.run_meta.json: wall time, context length used,
    model revision, package versions, GPU, Toto sampling settings.
  Both are written to a temporary name and renamed (meta first, csv last), so
  a csv that exists is complete and means the run is done. A failed run
  writes only ...__<term>.failed.run_meta.json. merge_results.py builds
  <out>/<variant>/ctx_<context>/all_results.csv from the per-config files.

<variant> is the model name, except Toto's sampling settings
(--toto-setting A, A_tf32off, A_nocache, A_bs16), which write to
toto__<setting>__seed<k>. --toto-setting point (default) is the first
round's Toto behaviour and writes to plain `toto`. With TOTO_SMOKE_SAMPLES
set, every variant gets the suffix __SMOKE.

Usage:
  envs/chronos_bolt_base/bin/python run_grid.py \\
      --model chronos_bolt_base --configs configs/selected_configs.json \\
      --context default --out results
  envs/toto/bin/python run_grid.py --model toto --toto-setting A --seed 42 \\
      --configs <one-config.json> --context default --out results
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import socket
import sys
import time
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path

from result_paths import (
    TOTO_SAMPLING_SETTINGS,
    TOTO_SETTINGS,
    atomic_write_text,
    result_csv,
    result_meta,
    smoke_samples,
    variant_name,
)

HERE = Path(__file__).resolve().parent
# Value at process start, before this module or anything it loads imports
# torch (Revision 1: required for Toto's sampling settings on CUDA).
CUBLAS_AT_START = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
# GPU round: the gift-eval and Toto clones default to the same places as in
# the first round (harness/gift-eval, harness/vendor/toto). The two overrides
# exist only so this copy can run on the Mac against the first round's clones
# without copying them; the pod does not set them.
GIFT_EVAL_REPO = Path(os.environ.get("HARNESS_GIFT_EVAL_REPO", str(HERE / "gift-eval")))
GIFT_EVAL_SRC = GIFT_EVAL_REPO / "src"
GIFT_EVAL_NOTEBOOKS = GIFT_EVAL_REPO / "notebooks"
TOTO_REPO = Path(os.environ.get("HARNESS_TOTO_REPO", str(HERE / "vendor" / "toto")))

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
# harness/data/. Same table as code/selection/compute_median_lengths.py's
# BASE_DIR_MAP (in the repository's history), which reads harness/gift-eval/notebooks/*.ipynb -- every
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
# same set as compute_median_lengths.py's FREQ_SUBDIR_BASES (in the repository's history), confirmed
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
# record/submitted/harness/README.md "Context length per model" for the full writeup,
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
# guessing -- see record/submitted/harness/README.md.

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
    if model in G2_TO_UNIVARIATE:  # GPU round, G2 models (see g2_models.py)
        return bool(G2_TO_UNIVARIATE[model]) and target_dim != 1
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


# --------------------------------------------------------------------------
# GPU round: Toto with sampling (DESIGN_ROUND3.md section G3, "setting A").
#
# build_toto above is the first round's behaviour and is kept unchanged
# (--toto-setting point). It passes no sample count: gluonts' evaluate_model
# calls predictor.predict(test_data.input) with only the dataset, so
# TotoPredictor.predict runs with num_samples=None and use_kv_cache=False
# and returns one point forecast (the mean) per series.
#
# The board entry's notebook (gift-eval/notebooks/toto.ipynb) instead wraps
# the predictor in TOTOModelPredictorWrapper, whose predict() passes
# num_samples and use_kv_cache. Everything below copies the notebook: the
# wrapper class, get_maximal_context_length, calculate_optimal_batch_size,
# the per-task seeding, determinism and matmul-precision calls.
# DESIGN_ROUND3.md Revision 1: torch.compile is left out (the notebook
# compiles a wrapper whose forward is never called: TotoSampleForecastGenerator
# uses prediction_net.model, the uncompiled inner module); at 512 steps the
# history is min(512, maximal length); CUBLAS_WORKSPACE_CONFIG=:4096:8 must be
# in the environment when the process starts (dispatch.py sets it).
# --------------------------------------------------------------------------

TOTO_NOTEBOOK_NUM_SAMPLES = 256  # toto.ipynb: num_samples = 256
TOTO_FIXED_BATCH_SIZE = 16  # setting A_bs16
CUBLAS_REQUIRED = ":4096:8"  # toto.ipynb, set before torch is imported


def toto_get_total_gpu_memory():
    """toto.ipynb get_total_gpu_memory: total VRAM of the current card in MB."""
    import torch

    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)


def toto_calculate_optimal_batch_size(
    model,
    target_dim,
    prediction_length,
    context_length,
    use_kv_cache,
    num_samples,
    safety_factor=0.01,
):
    """toto.ipynb calculate_optimal_batch_size, copied line for line."""
    try:
        model_width = model.model.embed_dim
        model_depth = model.model.num_layers
        model_param_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)
        base_memory_per_sample = (model_width * model_depth * 4) / (1024 * 1024)
        io_memory = (target_dim * (context_length + prediction_length) * 4) / (
            1024 * 1024
        )
        kv_memory = 0
        if use_kv_cache:
            kv_memory = (model_depth * model_width * 2 * context_length * 4) / (
                1024 * 1024
            )
        mem_per_sample_mb = base_memory_per_sample + io_memory + kv_memory
        mem_per_batch_mb = mem_per_sample_mb * target_dim * num_samples
        gpu_mem = toto_get_total_gpu_memory()
        cuda_reserved_mb = 1024
        available_memory = (
            gpu_mem - model_param_memory_mb - cuda_reserved_mb
        ) * safety_factor
        max_batch_size = max(
            1, int(available_memory / (mem_per_batch_mb / num_samples))
        )
        max_batch_size = min(16, max_batch_size)
        return max_batch_size
    except RuntimeError as e:
        print(f"Error calculating optimal batch size: {e}")
        return 1


def toto_get_maximal_context_length(dataset):
    """toto.ipynb get_maximal_context_length: shortest series minus the
    test windows and the one validation window."""
    shortest_series_in_dataset = dataset._min_series_length
    total_prediction_windows = dataset.windows + 1
    return shortest_series_in_dataset - total_prediction_windows * dataset.prediction_length


class TotoNotebookPredictor:
    """toto.ipynb TOTOModelPredictorWrapper: predict() passes num_samples and
    use_kv_cache to TotoPredictor.predict, and on the first call halves
    samples_per_batch after each CUDA out-of-memory error (starting from
    num_samples), rebuilding the TotoPredictor each time. Only addition:
    run_meta() reports the values finally used."""

    def __init__(self, model, prediction_length, context_length, mode, num_samples=128, use_kv_cache=True):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.mode = mode
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = num_samples
        self.model = model
        self._adjusted = False
        self._oom_halvings = 0
        self._initialize_predictor()

    def _initialize_predictor(self):
        from toto.inference.gluonts_predictor import TotoPredictor

        self.predictor = TotoPredictor.create_for_eval(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            mode=self.mode,
            samples_per_batch=self.samples_per_batch,
        )

    def predict(self, gluonts_test_data):
        import torch

        if not self._adjusted:
            print("Initializing predictor with samples_per_batch =", self.samples_per_batch)
            while self.samples_per_batch >= 1:
                try:
                    print(
                        f"Attempting prediction with samples_per_batch = {self.samples_per_batch} "
                        f"and context_length = {self.context_length}"
                    )
                    predictions = list(
                        self.predictor.predict(
                            gluonts_test_data,
                            use_kv_cache=self.use_kv_cache,
                            num_samples=self.num_samples,
                        )
                    )
                    self._adjusted = True
                    return predictions
                except RuntimeError as e:
                    print("Caught exception during prediction:", e)
                    if "CUDA out of memory" in str(e):
                        if self.samples_per_batch > 1:
                            print(
                                f"Out of memory with samples_per_batch = {self.samples_per_batch}. Reducing batch size."
                            )
                            self.samples_per_batch = self.samples_per_batch // 2
                            self._oom_halvings += 1
                            torch.cuda.empty_cache()
                        else:
                            print("OOM at minimal batch size. Cannot proceed with this context length and sample count.")
                            raise e
                        self._initialize_predictor()
                    else:
                        raise e
        return self.predictor.predict(
            gluonts_test_data,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )

    def run_meta(self):
        return {
            "samples_per_batch_final": self.samples_per_batch,
            "oom_halvings": self._oom_halvings,
        }


def build_toto_sampling(dataset, context_mode: str, setting: str, seed: int, num_samples: int):
    """Toto with sampling. setting A = toto.ipynb's settings (Revision 1);
    A_tf32off = A without torch.set_float32_matmul_precision("high");
    A_nocache = A with use_kv_cache=False; A_bs16 = A with the Multivariate
    batch size fixed at 16 (the last two are no longer in the job lists).
    Returns (predictor, context_used, revision, evaluate_model batch size,
    meta dict)."""
    if setting not in TOTO_SAMPLING_SETTINGS:
        raise ValueError(f"not a Toto sampling setting: {setting!r}")
    sys.path.insert(0, str(TOTO_REPO))
    import numpy as np
    import torch
    from toto.inference.gluonts_predictor import Multivariate
    from toto.model.toto import Toto

    if torch.cuda.is_available() and CUBLAS_AT_START != CUBLAS_REQUIRED:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG was {CUBLAS_AT_START!r} when this process started; "
            f"Toto's sampling settings on CUDA need {CUBLAS_REQUIRED!r} in the environment "
            "before torch is imported (dispatch.py sets it)."
        )

    model_path = "Datadog/Toto-Open-Base-1.0"
    use_kv_cache = setting != "A_nocache"

    # toto.ipynb evaluate_tasks: load, move to cuda (else cpu), eval.
    # torch.compile omitted (Revision 1).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Toto.from_pretrained(model_path)
    model = model.to(device)
    model = model.eval()

    # toto.ipynb evaluate_dataset_with_model: per-task setup, done after the
    # model is loaded and before the dataset and predictor are built. The
    # seed is set once here; an out-of-memory retry does not set it again.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if setting != "A_tf32off":
        torch.set_float32_matmul_precision("high")

    min_context_length = model.model.patch_embed.stride
    maximal_context = toto_get_maximal_context_length(dataset)
    if context_mode == "default":
        # toto.ipynb (pad_short_series=False)
        context_for_batch_size = min(TOTO_DEFAULT_CONTEXT, maximal_context)
        # toto.ipynb try_prediction_with_config floors it at the patch
        # stride after the batch size has been computed.
        context_length = max(context_for_batch_size, min_context_length)
        context_rule = "min(4096, get_maximal_context_length), then max(., patch stride) (toto.ipynb)"
    else:
        # Revision 1: min(512, the notebook's maximal length); the batch size
        # below uses the same length.
        context_for_batch_size = min(int(context_mode), maximal_context)
        context_length = context_for_batch_size
        context_rule = "min(512, get_maximal_context_length) (Revision 1)"

    if setting == "A_bs16":
        batch_size = TOTO_FIXED_BATCH_SIZE
        batch_size_source = "fixed 16 (setting A_bs16)"
    elif torch.cuda.is_available():
        batch_size = toto_calculate_optimal_batch_size(
            model=model,
            target_dim=dataset.target_dim,
            prediction_length=dataset.prediction_length,
            context_length=context_for_batch_size,
            use_kv_cache=use_kv_cache,
            num_samples=num_samples,
        )
        batch_size_source = "toto.ipynb calculate_optimal_batch_size"
    else:
        # No card to size from. The notebook's function returns 1 when its
        # GPU-memory query fails; used here only for CPU smoke tests.
        batch_size = 1
        batch_size_source = "cpu: no GPU to size from, notebook's fallback value 1"

    predictor = TotoNotebookPredictor(
        model=model,
        prediction_length=dataset.prediction_length,
        context_length=context_length,
        mode=Multivariate(batch_size=batch_size),
        num_samples=num_samples,
        use_kv_cache=use_kv_cache,
    )
    meta = {
        "toto_setting": setting,
        "seed": seed,
        "num_samples": num_samples,
        "samples_per_batch_initial": num_samples,
        "use_kv_cache": use_kv_cache,
        "toto_batch_size": batch_size,
        "toto_batch_size_source": batch_size_source,
        "torch_compile": "omitted",
        "min_context_length_patch_stride": int(min_context_length),
        "maximal_context_length": int(maximal_context),
        "context_rule": context_rule,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "toto_device": device,
    }
    # evaluate_model batch size = num_samples, as toto.ipynb passes.
    return predictor, context_length, hf_revision(model_path), num_samples, meta


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

# GPU round, G2: TimesFM-3 and Timestral-209M live in g2_models.py (written
# separately). Absent or unimportable in this venv -> those names are not
# offered. Its optional tables: G2_PACKAGES (versions recorded in run_meta)
# and G2_TO_UNIVARIATE (False = evaluate multivariate datasets jointly, as
# both board notebooks do; read by to_univariate_for).
G2_TO_UNIVARIATE = {}
try:
    from g2_models import G2_BUILDERS

    MODEL_BUILDERS.update(G2_BUILDERS)
except ImportError:
    pass
else:
    try:
        from g2_models import G2_PACKAGES, G2_TO_UNIVARIATE
    except ImportError:
        G2_PACKAGES = {}
    MODEL_PACKAGES.update(G2_PACKAGES)


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

    toto_sampling = args.model == "toto" and args.toto_setting in TOTO_SAMPLING_SETTINGS
    variant = variant_name(args.model, args.toto_setting, args.seed)
    smoke = smoke_samples()
    num_samples = smoke if smoke is not None else TOTO_NOTEBOOK_NUM_SAMPLES

    torch_version, gpu_name, torch_cuda = device_info()
    if toto_sampling and gpu_name != "cpu" and CUBLAS_AT_START != CUBLAS_REQUIRED:
        raise SystemExit(
            f"run_grid.py: CUBLAS_WORKSPACE_CONFIG was {CUBLAS_AT_START!r} when this process started. "
            f"Toto's sampling setting {args.toto_setting} on a GPU needs CUBLAS_WORKSPACE_CONFIG={CUBLAS_REQUIRED} "
            "in the environment before torch is imported (dispatch.py sets it). Nothing was run."
        )
    xformers_installed = importlib.util.find_spec("xformers") is not None
    common_meta = {
        "model": args.model,
        "variant": variant,
        "context_mode": args.context,
        "toto_setting": args.toto_setting if args.model == "toto" else None,
        "seed": args.seed if toto_sampling else None,
        "toto_smoke_samples": smoke,
        "package_versions": package_versions(MODEL_PACKAGES.get(args.model, ["torch", "gluonts"])),
        "gift_eval_commit": os.environ.get("GIFT_EVAL_COMMIT", "unknown"),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda,
        "gpu_name": gpu_name,
        "cublas_workspace_config": CUBLAS_AT_START,
        "xformers_installed": xformers_installed,
        "run_grid_sha256": file_sha256(Path(__file__)),
        "g2_models_sha256": file_sha256(HERE / "g2_models.py"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "chronos_batch_size_env": os.environ.get("CHRONOS_BATCH_SIZE"),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
    }

    had_failures = False
    for entry in configs_list:
        dataset_name, term = entry["dataset"], entry["term"]
        csv_path = result_csv(args.out, variant, args.context, dataset_name, term)
        meta_path = result_meta(args.out, variant, args.context, dataset_name, term)
        ds_config = None
        t0 = time.time()
        record = dict(common_meta)
        record.update(
            {
                "dataset": dataset_name,
                "term": term,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        cuda_reset_peak()
        try:
            if args.model == "toto" and xformers_installed:
                # Revision 1: the first round's Toto ran without xFormers.
                raise RuntimeError("xFormers is importable in this environment; Toto runs must not have it")
            storage_name = resolve_storage_name(dataset_name)
            probe = Dataset(name=storage_name, term=term, to_univariate=False)
            to_univariate = to_univariate_for(args.model, probe.target_dim)
            dataset = Dataset(name=storage_name, term=term, to_univariate=to_univariate)
            ds_config = dataset_config_name(dataset_name, term, properties)

            extra_meta = {}
            if toto_sampling:
                predictor, context_used, revision, batch_size, extra_meta = build_toto_sampling(
                    dataset, args.context, args.toto_setting, args.seed, num_samples
                )
            else:
                built = MODEL_BUILDERS[args.model](dataset, args.context)
                predictor, context_used, revision, batch_size = built[:4]
                if len(built) > 4 and isinstance(built[4], dict):
                    extra_meta = dict(built[4])
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
            # Revision 1: a missing or non-finite MASE[0.5] is a failed run.
            try:
                mase = float(res["MASE[0.5]"][0])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(f"MASE[0.5] missing from the evaluation result ({type(exc).__name__}: {exc})")
            if not math.isfinite(mase):
                raise RuntimeError(f"MASE[0.5] is not finite: {mase!r}")

            row = [
                ds_config,
                variant,
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
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(CSV_COLUMNS)
            writer.writerow(row)

            if callable(getattr(predictor, "run_meta", None)):
                extra_meta.update(predictor.run_meta())
            record.update(extra_meta)
            record.update(
                {
                    "ds_config": ds_config,
                    "status": "ok",
                    "context_length_used": context_used,
                    "model_revision": revision,
                    "evaluate_batch_size": batch_size,
                    "peak_gpu_memory_gb": cuda_peak_gb(),
                    "float32_matmul_precision": matmul_precision(),
                    "deterministic_algorithms": deterministic_enabled(),
                    "wall_time_seconds": time.time() - t0,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            # meta first, csv last: the csv's existence means both are complete
            atomic_write_text(meta_path, json.dumps(record, indent=2, default=_json_default))
            atomic_write_text(csv_path, buf.getvalue())
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            had_failures = True
            record.update(
                {
                    "ds_config": ds_config,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "peak_gpu_memory_gb": cuda_peak_gb(),
                    "float32_matmul_precision": matmul_precision(),
                    "deterministic_algorithms": deterministic_enabled(),
                    "wall_time_seconds": time.time() - t0,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            failed_meta = meta_path.with_name(meta_path.name.replace(".run_meta.json", ".failed.run_meta.json"))
            atomic_write_text(failed_meta, json.dumps(record, indent=2, default=_json_default))
            print(f"[FAILED] {dataset_name}/{term}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

        print(f"[{record['status']}] {ds_config or dataset_name + '/' + term} -> {csv_path}")

    return had_failures


def device_info():
    """(torch version, GPU name or "cpu", torch's CUDA build version) for run_meta.json."""
    try:
        import torch
    except ImportError:
        return None, "unknown (no torch)", None
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001 -- metadata only
        gpu = f"unknown ({type(exc).__name__})"
    return torch.__version__, gpu, torch.version.cuda


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _torch_or_none():
    try:
        import torch

        return torch
    except ImportError:
        return None


def cuda_reset_peak():
    torch = _torch_or_none()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_gb():
    """Peak GPU memory allocated by this process since the last reset, in GB."""
    torch = _torch_or_none()
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def matmul_precision():
    torch = _torch_or_none()
    return torch.get_float32_matmul_precision() if torch is not None else None


def deterministic_enabled():
    torch = _torch_or_none()
    return torch.are_deterministic_algorithms_enabled() if torch is not None else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    p.add_argument("--configs", required=True, help="Path to a JSON list of {\"dataset\": ..., \"term\": ...}")
    p.add_argument("--context", required=True, choices=["default", "512"])
    p.add_argument("--out", required=True, help="Output directory root")
    p.add_argument(
        "--toto-setting",
        default="point",
        choices=list(TOTO_SETTINGS),
        help="Toto only. point = first round (one point forecast, folder `toto`); "
        "A = gift-eval toto.ipynb settings as corrected in DESIGN_ROUND3.md Revision 1 "
        "(256 samples, KV cache, seeded, deterministic, matmul precision high); "
        "A_tf32off = A without the matmul-precision line; "
        "A_nocache = A without KV cache; A_bs16 = A with batch size 16 (last two not in the job lists). "
        "Sampling settings on a GPU need CUBLAS_WORKSPACE_CONFIG=:4096:8 at process start. "
        "Sampling settings write to toto__<setting>__seed<k>.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for Toto's sampling settings (ignored otherwise)")
    args = p.parse_args()
    if args.model != "toto" and args.toto_setting != "point":
        p.error("--toto-setting applies to --model toto only")
    return args


if __name__ == "__main__":
    had_failures = run(parse_args())
    if had_failures:
        sys.exit(1)
