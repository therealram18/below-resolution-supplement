"""
G2 models for the GPU round (record/history_length/DESIGN_ROUND3.md, section G2):
TimesFM-3 (google/timesfm-3.0-pytorch) and Timestral-209M (mistralai/Timestral-209M).

run_grid.py picks these up through its hook:

    from g2_models import G2_BUILDERS, G2_PACKAGES, G2_TO_UNIVARIATE
    MODEL_BUILDERS.update(G2_BUILDERS)
    MODEL_PACKAGES.update(G2_PACKAGES)          # run() reads MODEL_PACKAGES[args.model]
    # and in to_univariate_for(): if model in G2_TO_UNIVARIATE: return False
    # (both board notebooks build the dataset with to_univariate=False; the
    #  builders below refuse a dataset whose variates were split apart)

Each builder has the first-round contract (harness/run_grid.py, build_*):
    builder(dataset, context_mode) -> (predictor, context_used, model_revision, predict_batch_size)
with context_mode "default" (the documented length) or "512" (the last 512 steps).
predict_batch_size is what run_grid passes to gluonts' evaluate_model as batch_size.

Every setting below is a named constant with its source. Sources:
  [TFM3-NB cell N]  gift-eval notebooks/timesfm3.ipynb, cell N (identical at gift-eval
                    9e88bb6 and at gift-eval main on 2026-11-19; last changed in
                    gift-eval commit c786558e, 2026-08-31). This is the code_link of the
                    board entry results/TimesFM-3/config.json.
  [TFM3-PKG]        timesfm==3.0.2 (PyPI), module timesfm3/torch/, which is what the
                    notebook imports (`from timesfm3 import TimesFM3Evaluator, ModelConfig`).
  [TST-CARD]        huggingface.co/mistralai/Timestral-209M README (model card), revision
                    9874d252.
  [TST-README]      github.com/mistralai/timestral README at commit a534bdbd (the repository
                    the model card links to).
  [TST-CODE]        github.com/mistralai/timestral src/timestral/forecaster.py at a534bdbd.
  [TST-NB cell N]   gift-eval notebooks/timestral.ipynb, cell N (gift-eval commit 42504946,
                    2026-10-16). This is the code_link of the board entry
                    results/Timestral-209M/config.json at the 2026-11-13 pin; it states the
                    same protocol as the card and README and fixes the batching.
Nothing here was tuned against a result.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

GIFT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Source: run_grid.py metrics, MeanWeightedSumQuantileLoss(quantile_levels=[0.1, ..., 0.9]);
# MASE[0.5] needs 0.5. Same list in [TFM3-NB cell 2/4] and [TST-NB cell 3].

FORCED_CONTEXT = 512
# Source: DESIGN_ROUND3.md "Lengths": "512" gives every model the last 512 steps.

QUANTILE_MATCH_TOL = 1e-9
# Source: [TST-NB cell 4] matches levels with abs(lv - q) < 1e-9. A needed level that is not
# among the model's levels raises; nothing is interpolated.


# --------------------------------------------------------------------------
# TimesFM-3
# --------------------------------------------------------------------------

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"
# Source: [TFM3-NB cell 3] ModelConfig(checkpoint_path=os.getenv("TIMESFM3_CHECKPOINT",
# "google/timesfm-3.0-pytorch")).

TIMESFM3_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
# Hub commit of google/timesfm-3.0-pytorch on 2026-11-19 (last change 2026-09-02). Pinned so
# the pod reads the same files as the local build. The weights were uploaded on 2026-08-24
# (commit 6455822f); every later commit edits README, LICENSE or card metadata only.

TIMESFM3_INPUT_PATCH_LENGTH = 32  # Source: [TFM3-NB cell 3] input_patch_length=32
TIMESFM3_OUTPUT_PATCH_LENGTH = 64  # Source: [TFM3-NB cell 3] output_patch_length=64
TIMESFM3_INITIAL_PER_CORE_BATCH_SIZE = 64
# Source: [TFM3-NB cell 3] per_core_batch_size=64. The predictor overwrites it with its batch
# size on every predict() call ([TFM3-NB cell 4]), as below.
TIMESFM3_USE_VARIATE_ATTENTION = True  # Source: [TFM3-NB cell 3] use_variate_attention=True
TIMESFM3_USE_SDPA = True  # Source: [TFM3-NB cell 3] use_sdpa=True

TIMESFM3_BATCH_SIZE_MULTIVARIATE = 16
TIMESFM3_BATCH_SIZE_UNIVARIATE = 64
# Source: [TFM3-NB cell 6] batch_size = 16 if num_variates > 1 else 64, where num_variates is
# dataset_properties.json's num_variates. On all 12 first-round configurations that number
# equals gift_eval Dataset.target_dim (checked 2026-11-19), which is what is used here. The same
# number is the predictor's batch and evaluate_model's batch_size in the notebook.

TIMESFM3_PREDICT_KWARGS = dict(
    return_quantiles=True,
    use_symmetric_averaging=True,
    make_positive=True,
    sort_quantiles=True,
)
# Source: [TFM3-NB cell 4] forecaster.predict_batch(..., return_quantiles=True,
# use_symmetric_averaging=True, make_positive=True, sort_quantiles=True).

TIMESFM3_TO_UNIVARIATE = False
# Source: [TFM3-NB cell 6] Dataset(name=ds_name, term=term, to_univariate=False); [TFM3-NB
# cell 0] "cross-variate attention", "Variate chunking (<= 32 variates per forward pass)".

TIMESFM3_PASS_PAST_COVARIATES = True
# Source: [TFM3-NB cell 4] past_feat_dynamic_real, when present, is passed as
# past_only_covariates.

TIMESFM3_DOCUMENTED_CONTEXT = 15360
# Source: [TFM3-NB cell 4] passes each series' full target with no truncation. The package
# then caps it: [TFM3-PKG] timesfm3_forecaster.py _MAX_CONTEXT_LENGTH = 15360, and per batch
# uses min(ceil(longest history in the batch / 32) * 32, 15360) steps, keeping the most recent
# (_Query.format: targets[:, -context_len:]); shorter series in the batch are left-padded and
# masked. So "documented" = full history, at most the last 15,360 steps. The same value in
# timesfm 3.0.0 and 3.0.1. This constant is only reported; the cap is applied by the package.


# --------------------------------------------------------------------------
# Timestral-209M
# --------------------------------------------------------------------------

TIMESTRAL_CHECKPOINT = "mistralai/Timestral-209M"
# Source: [TST-CARD] usage, TimestralForecaster.from_pretrained("mistralai/Timestral-209M");
# [TST-NB cell 2] CHECKPOINT = f"mistralai/{MODEL_NAME}" with SIZE = "209M".

TIMESTRAL_REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
# Hub commit of mistralai/Timestral-209M on 2026-11-19 (last change 2026-10-15, before the board
# notebook of 2026-10-16). Pinned so the pod reads the same files as the local build.

TIMESTRAL_CODE_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
# github.com/mistralai/timestral, the repository the model card links to ([TST-CARD]: "We're
# releasing the source code in our GitHub repository"). Its only commit on 2026-11-19
# ("Timestral release", 2026-10-15). Installed as
# "timestral @ git+https://github.com/mistralai/timestral.git@<this commit>".

TIMESTRAL_DOCUMENTED_CONTEXT = 32768
# Source: [TST-CARD] "It accepts up to 32,768 historical time steps"; "Context length: up to
# 32,768 time steps (1,024 patches)". [TST-README] "The context is capped at the trained length
# (32768 steps); the most recent steps are kept." [TST-CODE] max_context defaults to
# config.context_length (32768 in the checkpoint's config.json) and keeps rows[..., -max_context:].
# [TST-NB cell 0] "The full history is the context; the model keeps the most recent 32768 steps."
# The predictor reads the value from the loaded forecaster (fc.max_context) as the notebook does,
# and checks it equals this constant.

TIMESTRAL_MODEL_QUANTILE_LEVELS = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)
# Source: [TST-CARD] "Outputs: 21 quantiles (0.01, 0.05, 0.1, 0.15, ..., 0.85, 0.9, 0.95,
# 0.99)"; checkpoint config.json quantile_knots (the same 21). All nine GIFT levels and the
# median are among them, so the mapping is a selection, not an interpolation. The predictor
# reads the levels from the loaded forecaster and checks they equal this tuple.

TIMESTRAL_TO_UNIVARIATE = False
# Source: [TST-CARD] "Inputs: univariate or multivariate target variates"; [TST-README] "The
# variates of a series are forecast jointly"; [TST-NB cell 0] "Multivariate datasets are
# forecast jointly (to_univariate=False)".

TIMESTRAL_PASS_PAST_COVARIATES = True
# Source: [TST-README] Benchmarks table: GIFT-Eval, "past_feat_dynamic_real as past covariates";
# [TST-NB cell 5] use_past_covariates=bool(ds.past_feat_dynamic_real_dim).

TIMESTRAL_MAX_ROWS = 1024
TIMESTRAL_MAX_TOKENS = 2 ** 24
# Source: [TST-NB cell 2] "Batch budget: max rows (series x (variates + covariates)) and max
# rows x (context + horizon) steps." Series are sorted by length and grouped within this budget,
# left-padded with NaN ([TST-NB cell 4]).

TIMESTRAL_EVAL_BATCH_SIZE = 512
# Source: [TST-NB cell 5] evaluate_model(..., batch_size=512, ...).

TIMESTRAL_MEAN_IS_MEDIAN = True
# Source: [TST-NB cell 4] forecast_keys = the nine levels + ["mean"], "mean" being the median.
# gluonts returns the median for a missing "mean" key anyway; this only silences its warning.

# Forecaster defaults, all left as is. Source: [TST-README] "evaluated zero-shot ... with the
# forecaster defaults"; [TST-NB cell 0] "The forecaster is used with its default settings";
# [TST-NB cell 6] from_pretrained(CHECKPOINT, device=DEVICE). The defaults in [TST-CODE] are
# per_shot_horizon_patches=32, quantile_real_cap_k=1e4, extreme_value_threshold=1e30,
# symmetric_averaging=True, clip_nonnegative=True. They are not passed, only listed here.


# --------------------------------------------------------------------------
# Pure helpers (no torch; covered by test_g2_models.py, which is in the repository's history)
# --------------------------------------------------------------------------


def context_limit(context_mode: str):
    """None for "default" (no client-side truncation), else the forced number of steps."""
    if context_mode == "default":
        return None
    n = int(context_mode)
    if n != FORCED_CONTEXT:
        raise ValueError(f"context_mode must be 'default' or '{FORCED_CONTEXT}', got {context_mode!r}")
    return n


def last_steps(arr, n):
    """The last n steps along the time axis (the last axis); the whole array when n is None."""
    if arr is None or n is None:
        return arr
    return arr[..., -n:]


def select_quantile_indices(model_levels, wanted=GIFT_QUANTILE_LEVELS, tol=QUANTILE_MATCH_TOL):
    """Index into model_levels for every wanted level. Raises if a wanted level is missing:
    the model's quantiles are selected, never interpolated."""
    levels = [float(x) for x in model_levels]
    out = []
    for q in wanted:
        hits = [i for i, lv in enumerate(levels) if abs(lv - q) < tol]
        if not hits:
            raise ValueError(
                f"quantile level {q} is not among the model's levels {tuple(levels)}; "
                "refusing to interpolate"
            )
        out.append(hits[0])
    return out


def quantiles_to_forecast_array(q, q_index, univariate_target: bool, append_median_as_mean: bool = False):
    """Model quantiles -> gluonts QuantileForecast.forecast_arrays.

    q: (H, Q) for a univariate output or (V, H, Q) for a multivariate one (Q = model levels).
    Returns (K, H) for a univariate target, else (K, H, V); K = len(q_index) (+1 with the mean).
    """
    import numpy as np

    q = np.asarray(q)[..., q_index]
    if append_median_as_mean:
        median_pos = GIFT_QUANTILE_LEVELS.index(0.5)
        q = np.concatenate([q, q[..., median_pos, None]], axis=-1)
    if q.ndim == 2:
        return q.T
    arr = q.transpose(2, 1, 0)
    if univariate_target:
        if arr.shape[-1] != 1:
            raise ValueError(f"univariate target but {arr.shape[-1]} output variates")
        arr = arr[..., 0]
    return arr


def forecast_keys(append_mean: bool = False):
    keys = [str(q) for q in GIFT_QUANTILE_LEVELS]
    return keys + ["mean"] if append_mean else keys


def timesfm3_batch_size(target_dim: int) -> int:
    return TIMESFM3_BATCH_SIZE_MULTIVARIATE if target_dim > 1 else TIMESFM3_BATCH_SIZE_UNIVARIATE


def timestral_batches(lengths, n_rows: int, prediction_length: int,
                      max_rows: int = TIMESTRAL_MAX_ROWS, max_tokens: int = TIMESTRAL_MAX_TOKENS):
    """[TST-NB cell 4] grouping: indices sorted by history length, each group grown from its
    shortest series while the padded size fits the budget. Returns a list of index lists."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order):
            length = lengths[order[end]] + prediction_length
            if (end + 1 - start) * n_rows > max_rows or (end + 1 - start) * n_rows * length > max_tokens:
                break
            end += 1
        groups.append(order[start:end])
        start = end
    return groups


def left_pad_nan(x, length: int):
    """[TST-NB cell 4] left_pad: NaN in front up to `length` steps (x is (V, T))."""
    import numpy as np

    pad = length - x.shape[-1]
    if pad == 0:
        return x
    return np.concatenate([np.full((x.shape[0], pad), np.nan, dtype=np.float32), x], axis=-1)


def check_not_split(dataset, model: str) -> None:
    """Both notebooks evaluate multivariate datasets jointly (to_univariate=False). If the
    caller split the variates, the setting is not the documented one: stop."""
    import numpy as np

    if dataset.target_dim <= 1:
        return
    first = next(iter(dataset.test_data.input))
    if np.ndim(first["target"]) == 1:
        raise RuntimeError(
            f"{model}: dataset has {dataset.target_dim} target variates but was built with "
            "to_univariate=True. The board notebook evaluates it jointly; run_grid.py must use "
            "to_univariate=False for this model (G2_TO_UNIVARIATE)."
        )


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

_LOADED = {}  # one loaded model per process; loading never changes a forecast


def build_timesfm3(dataset, context_mode: str):
    import numpy as np
    from gluonts.itertools import batcher
    from gluonts.model.forecast import QuantileForecast
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    forced = context_limit(context_mode)
    check_not_split(dataset, "timesfm3")

    key = ("timesfm3", TIMESFM3_CHECKPOINT, TIMESFM3_REVISION)
    if key not in _LOADED:
        config = ModelConfig(
            checkpoint_path=TIMESFM3_CHECKPOINT,
            input_patch_length=TIMESFM3_INPUT_PATCH_LENGTH,
            output_patch_length=TIMESFM3_OUTPUT_PATCH_LENGTH,
            per_core_batch_size=TIMESFM3_INITIAL_PER_CORE_BATCH_SIZE,
            use_variate_attention=TIMESFM3_USE_VARIATE_ATTENTION,
            use_sdpa=TIMESFM3_USE_SDPA,
            revision=TIMESFM3_REVISION,
        )
        _LOADED[key] = TimesFM3Evaluator(config)
    forecaster = _LOADED[key]
    q_index = select_quantile_indices(forecaster.config.quantiles)
    batch_size = timesfm3_batch_size(dataset.target_dim)

    class TimesFm3Predictor:
        """[TFM3-NB cell 4], plus truncation to the last 512 steps for --context 512."""

        def __init__(self):
            self.prediction_length = dataset.prediction_length
            self.batch_size = batch_size
            self.longest_history_given = 0

        def run_meta(self):
            # Read by the GPU-round run_grid.py if present; ignored otherwise.
            return {
                "g2_checkpoint": f"{TIMESFM3_CHECKPOINT}@{TIMESFM3_REVISION}",
                "g2_to_univariate": TIMESFM3_TO_UNIVARIATE,
                "g2_batch_size": self.batch_size,
                "g2_longest_history_given": self.longest_history_given,
                "g2_package_context_cap": TIMESFM3_DOCUMENTED_CONTEXT,
            }

        def predict(self, test_data_input, batch_size: int = None):
            if batch_size is None:
                batch_size = self.batch_size
            forecaster.config.per_core_batch_size = batch_size
            forecasts = []
            for batch in batcher(test_data_input, batch_size=batch_size):
                contexts = [
                    last_steps(np.array(entry["target"], dtype=np.float32), forced) for entry in batch
                ]
                self.longest_history_given = max(
                    [self.longest_history_given] + [c.shape[-1] for c in contexts]
                )
                past_covs = [
                    last_steps(np.array(entry["past_feat_dynamic_real"], dtype=np.float32), forced)
                    if (TIMESFM3_PASS_PAST_COVARIATES and "past_feat_dynamic_real" in entry
                        and entry["past_feat_dynamic_real"] is not None)
                    else None
                    for entry in batch
                ]
                outputs = list(
                    forecaster.predict_batch(
                        contexts=contexts,
                        horizon=self.prediction_length,
                        past_only_covariates=past_covs,
                        **TIMESFM3_PREDICT_KWARGS,
                    )
                )
                for out, entry in zip(outputs, batch):
                    target_len = np.asarray(entry["target"]).shape[-1]
                    q = out.quantiles[..., : self.prediction_length, :]
                    forecasts.append(
                        QuantileForecast(
                            forecast_arrays=quantiles_to_forecast_array(
                                q, q_index, univariate_target=np.ndim(entry["target"]) == 1
                            ),
                            forecast_keys=forecast_keys(),
                            start_date=entry["start"] + target_len,
                        )
                    )
            return forecasts

    used_context = forced or TIMESFM3_DOCUMENTED_CONTEXT
    return TimesFm3Predictor(), used_context, f"{TIMESFM3_CHECKPOINT}@{TIMESFM3_REVISION}", batch_size


def build_timestral209m(dataset, context_mode: str):
    import numpy as np
    import torch
    from gluonts.model.forecast import QuantileForecast
    from timestral import TimestralForecaster

    forced = context_limit(context_mode)
    check_not_split(dataset, "timestral209m")

    device = "cuda" if torch.cuda.is_available() else "cpu"  # [TST-NB cell 2] DEVICE
    key = ("timestral", TIMESTRAL_CHECKPOINT, TIMESTRAL_REVISION, device)
    if key not in _LOADED:
        _LOADED[key] = TimestralForecaster.from_pretrained(
            TIMESTRAL_CHECKPOINT, device=device, revision=TIMESTRAL_REVISION
        )
    fc = _LOADED[key]
    if tuple(fc.quantile_levels) != TIMESTRAL_MODEL_QUANTILE_LEVELS:
        raise RuntimeError(f"checkpoint quantile levels {fc.quantile_levels} differ from the model card's")
    if fc.max_context != TIMESTRAL_DOCUMENTED_CONTEXT:
        raise RuntimeError(f"checkpoint max_context {fc.max_context} differs from the card's {TIMESTRAL_DOCUMENTED_CONTEXT}")
    q_index = select_quantile_indices(fc.quantile_levels)
    use_past = TIMESTRAL_PASS_PAST_COVARIATES and bool(dataset.past_feat_dynamic_real_dim)
    max_ctx = forced or fc.max_context

    class TimestralPredictor:
        """[TST-NB cell 4]; for --context 512 the notebook's `-max_ctx:` slice uses 512."""

        def __init__(self):
            self.prediction_length = dataset.prediction_length
            self.longest_history_given = 0
            self.n_batches = 0

        def run_meta(self):
            # Read by the GPU-round run_grid.py if present; ignored otherwise.
            return {
                "g2_checkpoint": f"{TIMESTRAL_CHECKPOINT}@{TIMESTRAL_REVISION}",
                "g2_code_commit": TIMESTRAL_CODE_COMMIT,
                "g2_to_univariate": TIMESTRAL_TO_UNIVARIATE,
                "g2_past_covariates": use_past,
                "g2_device": device,
                "g2_longest_history_given": self.longest_history_given,
                "g2_model_batches": self.n_batches,
            }

        def predict(self, test_data_input):
            entries = list(test_data_input)
            targets, covs = [], []
            for e in entries:
                t = np.asarray(e["target"], dtype=np.float32)
                targets.append((t[None] if t.ndim == 1 else t)[:, -max_ctx:])  # (V, T)
                c = e.get("past_feat_dynamic_real") if use_past else None
                if c is not None:
                    c = np.asarray(c, dtype=np.float32)
                    c = (c[None] if c.ndim == 1 else c)[:, -max_ctx:]
                covs.append(c)
            n_rows = targets[0].shape[0] + (covs[0].shape[0] if covs[0] is not None else 0)
            self.longest_history_given = max([self.longest_history_given] + [x.shape[-1] for x in targets])

            outputs = [None] * len(entries)
            groups = timestral_batches([x.shape[-1] for x in targets], n_rows, self.prediction_length)
            self.n_batches += len(groups)
            for idx in groups:
                self._predict_batch(idx, targets, covs, outputs)

            forecasts = []
            for e, q in zip(entries, outputs):
                target = np.asarray(e["target"])
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=quantiles_to_forecast_array(
                            q, list(range(len(GIFT_QUANTILE_LEVELS))),
                            univariate_target=target.ndim == 1,
                            append_median_as_mean=TIMESTRAL_MEAN_IS_MEDIAN,
                        ),
                        forecast_keys=forecast_keys(append_mean=TIMESTRAL_MEAN_IS_MEDIAN),
                        start_date=e["start"] + target.shape[-1],
                        item_id=e.get("item_id"),
                    )
                )
            return forecasts

        def _predict_batch(self, idx, targets, covs, outputs):
            try:
                q = self._run(idx, targets, covs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                half = len(idx) // 2
                self._predict_batch(idx[:half], targets, covs, outputs)
                self._predict_batch(idx[half:], targets, covs, outputs)
                return
            for j, i in enumerate(idx):
                outputs[i] = q[j]

        def _run(self, idx, targets, covs):
            length = max(targets[i].shape[-1] for i in idx)
            ctx = np.stack([left_pad_nan(targets[i], length) for i in idx])
            past = None
            if covs[idx[0]] is not None:
                past = np.stack([left_pad_nan(covs[i], length) for i in idx])
            f = fc.forecast(ctx, self.prediction_length, past_covariates=past)
            return f.quantiles[..., q_index].float().cpu().numpy()  # (B, V, H, 9)

    used_context = forced or TIMESTRAL_DOCUMENTED_CONTEXT
    return TimestralPredictor(), used_context, f"{TIMESTRAL_CHECKPOINT}@{TIMESTRAL_REVISION}", TIMESTRAL_EVAL_BATCH_SIZE


G2_BUILDERS = {
    "timesfm3": build_timesfm3,
    "timestral209m": build_timestral209m,
}

G2_ENVS = {
    # timesfm==3.0.2 (the first-round timesfm25 environment's pin) ships the timesfm3 module
    # the notebook imports (TimesFM3Evaluator, ModelConfig): checked by import on 2026-11-19.
    "timesfm3": "timesfm25",
    "timestral209m": "timestral",
}

G2_PACKAGES = {
    "timesfm3": ["timesfm", "torch", "gluonts"],
    "timestral209m": ["timestral", "torch", "gluonts"],
}

G2_TO_UNIVARIATE = {
    "timesfm3": TIMESFM3_TO_UNIVARIATE,
    "timestral209m": TIMESTRAL_TO_UNIVARIATE,
}
