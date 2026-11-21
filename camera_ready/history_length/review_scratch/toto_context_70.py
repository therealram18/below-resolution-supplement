"""Review scratch (round 3): for each of the 70 eligible configurations, compute
what history Toto gets under the notebook's rule at the documented length (4096)
and at 512, and under the first round's rule (run_grid.py lines 445-449).

Notebook rule (toto.ipynb cell 5): ctx = min(L, shortest - (windows+1)*pred);
then ctx = max(ctx, patch_stride=64).
First-round rule (run_grid.py 448): ctx = min(L, max_avail) if max_avail > 0 else L.

Also counts the number of forecast points per configuration (series x variates x
windows x horizon), used to argue sampling noise. Reads data only; no model.
Run: harness/envs/toto/bin/python review_scratch/toto_context_70.py
"""
import json
import os
import sys
from pathlib import Path

R = Path("fmts")
sys.path.insert(0, str(R / "harness" / "gift-eval" / "src"))
sys.path.insert(0, str(R / "harness"))
os.environ.setdefault("GIFT_EVAL", str(R / "harness" / "data"))

from gift_eval.data import Dataset  # noqa: E402
from run_grid import resolve_storage_name  # noqa: E402

med = json.loads((R / "selection" / "config_median_lengths.json").read_text())
elig = sorted(k for k, v in med.items() if v["median_length"] >= 1024)
assert len(elig) == 70

STRIDE = 64
rows = []
for cfg in elig:
    base, freq, term = cfg.split("/")
    storage = resolve_storage_name(f"{base}/{freq}")
    ds = Dataset(name=storage, term=term, to_univariate=False)
    shortest = ds._min_series_length
    pred = ds.prediction_length
    w = ds.windows
    max_avail = shortest - (w + 1) * pred
    nb = {}
    fr = {}
    for L in (4096, 512):
        c = min(L, max_avail)
        nb[L] = max(c, STRIDE)
        fr[L] = min(L, max_avail) if max_avail > 0 else L
    n_series = len(ds.hf_dataset)
    rows.append(
        dict(
            config=cfg,
            shortest=int(shortest),
            pred=int(pred),
            windows=int(w),
            target_dim=int(ds.target_dim),
            n_series=int(n_series),
            max_avail=int(max_avail),
            nb_doc=int(nb[4096]),
            nb_512=int(nb[512]),
            fr_doc=int(fr[4096]),
            fr_512=int(fr[512]),
            same_history_notebook=bool(nb[4096] == nb[512]),
            notebook_differs_from_first_round=bool((nb[4096] != fr[4096]) or (nb[512] != fr[512])),
            forecast_points=int(n_series * ds.target_dim * w * pred),
        )
    )
    print(rows[-1], flush=True)

out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(rows, indent=1))
same = [r["config"] for r in rows if r["same_history_notebook"]]
diff = [r["config"] for r in rows if r["notebook_differs_from_first_round"]]
print("configs where Toto gets the same history at both lengths:", len(same), same)
print("configs where notebook rule differs from first-round rule:", len(diff), diff)
