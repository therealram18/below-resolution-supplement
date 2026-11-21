"""
Tests for g2_models.py. Run with pytest or plain python:

    <env>/bin/python -m pytest harness/test_g2_models.py      # or
    <env>/bin/python harness/test_g2_models.py

The first group needs only numpy (no torch, no GPU, no download). The last two tests run the
real predictors on a toy multivariate input; they are skipped unless the model package is
importable in the current env and the checkpoint is already in the Hugging Face cache (they set
HF_HUB_OFFLINE=1, so they never download).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_models as g2  # noqa: E402


class Skip(Exception):
    pass


def _skip(msg):
    try:
        import pytest

        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


# ---------------------------------------------------------------- pure helpers


def test_context_limit():
    assert g2.context_limit("default") is None
    assert g2.context_limit("512") == 512
    for bad in ("256", "1024"):
        try:
            g2.context_limit(bad)
        except ValueError:
            continue
        raise AssertionError(f"context_limit({bad!r}) should raise")


def test_last_steps_truncates_time_axis_only():
    uni = np.arange(1000, dtype=np.float32)
    multi = np.arange(3 * 1000, dtype=np.float32).reshape(3, 1000)
    assert g2.last_steps(uni, None) is uni
    out = g2.last_steps(uni, 512)
    assert out.shape == (512,) and out[0] == 488 and out[-1] == 999
    out = g2.last_steps(multi, 512)
    assert out.shape == (3, 512)
    assert np.array_equal(out, multi[:, -512:])
    short = np.arange(100, dtype=np.float32)
    assert g2.last_steps(short, 512).shape == (100,)  # shorter history is left as is
    assert g2.last_steps(None, 512) is None


def test_quantile_mapping_timestral_selects_without_interpolation():
    idx = g2.select_quantile_indices(g2.TIMESTRAL_MODEL_QUANTILE_LEVELS)
    assert idx == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert [g2.TIMESTRAL_MODEL_QUANTILE_LEVELS[i] for i in idx] == g2.GIFT_QUANTILE_LEVELS
    assert len(g2.TIMESTRAL_MODEL_QUANTILE_LEVELS) == 21


def test_quantile_mapping_timesfm3_is_identity():
    assert g2.select_quantile_indices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) == list(range(9))


def test_quantile_mapping_refuses_missing_level():
    try:
        g2.select_quantile_indices([0.05, 0.25, 0.5, 0.75, 0.95])
    except ValueError as exc:
        assert "refusing to interpolate" in str(exc)
        return
    raise AssertionError("a missing level must raise")


def test_forecast_array_shapes_and_values():
    H, V = 7, 3
    levels = g2.TIMESTRAL_MODEL_QUANTILE_LEVELS
    idx = g2.select_quantile_indices(levels)
    # value = level * 100 + variate, so every cell says where it came from
    q_multi = np.array([[[lv * 100 + v for lv in levels] for _ in range(H)] for v in range(V)])  # (V, H, 21)
    arr = g2.quantiles_to_forecast_array(q_multi, idx, univariate_target=False)
    assert arr.shape == (9, H, V)
    assert np.allclose(arr[4, :, 2], 50 + 2)  # 0.5 row, third variate
    arr = g2.quantiles_to_forecast_array(q_multi, idx, univariate_target=False, append_median_as_mean=True)
    assert arr.shape == (10, H, V) and np.array_equal(arr[9], arr[4])
    q_uni_v1 = q_multi[:1]  # (1, H, 21): Timestral output for a univariate target
    arr = g2.quantiles_to_forecast_array(q_uni_v1, idx, univariate_target=True, append_median_as_mean=True)
    assert arr.shape == (10, H)
    q_uni_2d = q_multi[0][:, :9]  # (H, 9): TimesFM-3 output for a univariate target
    arr = g2.quantiles_to_forecast_array(q_uni_2d, list(range(9)), univariate_target=True)
    assert arr.shape == (9, H)
    assert len(g2.forecast_keys()) == 9 and g2.forecast_keys(append_mean=True)[-1] == "mean"


def test_timesfm3_batch_size_rule():
    assert g2.timesfm3_batch_size(1) == 64
    assert g2.timesfm3_batch_size(2) == 16
    assert g2.timesfm3_batch_size(21) == 16


def test_timestral_batches_sorted_and_within_budget():
    lengths = [900, 100, 500, 300, 700]
    groups = g2.timestral_batches(lengths, n_rows=1, prediction_length=8)
    flat = [i for grp in groups for i in grp]
    assert flat == [1, 3, 2, 4, 0]  # sorted by length
    assert sorted(flat) == list(range(5))
    groups = g2.timestral_batches([10] * 5000, n_rows=1, prediction_length=8)
    assert all(len(grp) <= g2.TIMESTRAL_MAX_ROWS for grp in groups)
    groups = g2.timestral_batches([30000] * 20, n_rows=37, prediction_length=600)
    for grp in groups:
        assert len(grp) == 1 or len(grp) * 37 * (30000 + 600) <= g2.TIMESTRAL_MAX_TOKENS


def test_left_pad_nan():
    x = np.ones((2, 5), dtype=np.float32)
    out = g2.left_pad_nan(x, 8)
    assert out.shape == (2, 8) and np.isnan(out[:, :3]).all() and (out[:, 3:] == 1).all()
    assert g2.left_pad_nan(x, 5) is x


class _FakeTestData:
    def __init__(self, entries):
        self.input = entries


class _FakeDataset:
    def __init__(self, target_dim, entries, prediction_length=8, pfdr_dim=0):
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.past_feat_dynamic_real_dim = pfdr_dim
        self.test_data = _FakeTestData(entries)


def _toy_entries(n_series, n_var, length, pfdr_dim=0, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n_series):
        t = np.sin(np.arange(length) / 6.0) * 10 + 50 + rng.standard_normal(length).cumsum() * 0.1
        target = np.stack([t * (v + 1) for v in range(n_var)]) if n_var > 1 else t
        e = {"target": target.astype(np.float32), "start": pd.Period("2020-01-01 00:00", freq="h"),
             "item_id": f"toy{i}"}
        if pfdr_dim:
            e["past_feat_dynamic_real"] = rng.standard_normal((pfdr_dim, length)).astype(np.float32)
        entries.append(e)
    return entries


def test_check_not_split():
    joint = _FakeDataset(3, _toy_entries(1, 3, 50))
    g2.check_not_split(joint, "x")  # multivariate kept together: fine
    split = _FakeDataset(3, _toy_entries(3, 1, 50))
    try:
        g2.check_not_split(split, "x")
    except RuntimeError as exc:
        assert "to_univariate=False" in str(exc)
        return
    raise AssertionError("a split multivariate dataset must be refused")


def test_settings_constants():
    assert g2.TIMESTRAL_DOCUMENTED_CONTEXT == 32768
    assert g2.TIMESFM3_DOCUMENTED_CONTEXT == 15360
    assert g2.FORCED_CONTEXT == 512
    assert set(g2.G2_BUILDERS) == {"timesfm3", "timestral209m"}
    assert g2.G2_ENVS == {"timesfm3": "timesfm25", "timestral209m": "timestral"}
    assert g2.G2_TO_UNIVARIATE == {"timesfm3": False, "timestral209m": False}


# ------------------------------------------------- real predictors (optional)


def _run_toy(builder_name, package):
    try:
        __import__(package)
    except Exception as exc:  # noqa: BLE001
        _skip(f"{package} not importable here: {exc}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    H, V, T = 8, 3, 600
    ds = _FakeDataset(V, _toy_entries(2, V, T, pfdr_dim=2), prediction_length=H, pfdr_dim=2)
    builder = g2.G2_BUILDERS[builder_name]
    try:
        pred, ctx, rev, bs = builder(ds, "512")
    except Exception as exc:  # noqa: BLE001
        if "offline" in str(exc).lower() or "cache" in str(exc).lower() or "LocalEntryNotFound" in type(exc).__name__:
            _skip(f"checkpoint not in the local cache: {exc}")
        raise
    assert ctx == 512
    fcs = list(pred.predict(ds.test_data.input))
    assert len(fcs) == 2
    for f in fcs:
        assert f.forecast_array.shape[1:] == (H, V), f.forecast_array.shape
        assert np.isfinite(f.forecast_array).all()
        med = f.quantile(0.5)
        assert med.shape == (H, V)
    assert pred.run_meta()["g2_longest_history_given"] == 512
    # univariate toy through the same predictor type
    ds1 = _FakeDataset(1, _toy_entries(2, 1, 300), prediction_length=H)
    pred1, ctx1, _, _ = builder(ds1, "default")
    fcs1 = list(pred1.predict(ds1.test_data.input))
    assert fcs1[0].forecast_array.shape[1:] == (H,)
    assert pred1.run_meta()["g2_longest_history_given"] == 300


def test_timesfm3_predictor_toy_multivariate():
    _run_toy("timesfm3", "timesfm3")


def test_timestral_predictor_toy_multivariate():
    _run_toy("timestral209m", "timestral")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}")
        except Skip as exc:
            print(f"SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
