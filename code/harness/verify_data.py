"""Data-loading verification only. Loads a few GIFT-Eval dataset-configs via
gift_eval.data.Dataset and prints their shape. No model, no inference.

Run with the `data` env:
    envs/data/bin/python verify_data.py
"""
import os
import sys
import time

# Point gift_eval.data.Dataset at the local download instead of relying on a
# .env file (keeps this script portable, no dotenv discovery surprises).
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("GIFT_EVAL", os.path.join(HERE, "data"))
sys.path.insert(0, os.path.join(HERE, "gift-eval", "src"))

from gift_eval.data import Dataset  # noqa: E402

# One short-term univariate, one medium/long-term, one multivariate.
CHECKS = [
    ("m4_weekly", "short"),
    ("bizitobs_l2c/H", "medium"),
    ("ett1/H", "long"),
]

for name, term in CHECKS:
    t0 = time.time()
    ds = Dataset(name=name, term=term, to_univariate=False)
    n_series = sum(1 for _ in ds.test_data.input)
    elapsed = time.time() - t0
    print(f"--- {name} / {term} ---")
    print(f"  freq:              {ds.freq}")
    print(f"  prediction_length: {ds.prediction_length}")
    print(f"  target_dim:        {ds.target_dim}")
    print(f"  windows:           {ds.windows}")
    print(f"  test series count: {n_series}")
    print(f"  loaded in:         {elapsed:.2f}s")
