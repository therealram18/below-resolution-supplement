"""Review scratch (round 3): compute S, the spread of the board's displayed top ten,
on the 70 eligible configurations, the 58 new ones and the first 12, from the
2026-09-07 snapshot's published rows only (no new run). Same definitions as
poolcheck/resolution_budget.py lines 41 and 55-60 (top ten by mean MASE rank
over all 97 configurations, leakage entries sorted last; S = max/min - 1 of the
geometric mean of MASE / Seasonal Naive MASE over the subset).
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/spread_S.py
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

R = Path("fmts")
SNAP = R / "snapshot_2026-09-07" / "results"
MASE = "eval_metrics/MASE[0.5]"
rows = defaultdict(dict)
cfg = {}
for root, dirs, files in os.walk(SNAP):
    if "all_results.csv" in files:
        for r in csv.DictReader(open(os.path.join(root, "all_results.csv"))):
            try:
                rows[r["model"]][r["dataset"]] = float(r[MASE])
            except (KeyError, ValueError):
                pass
    if "config.json" in files:
        try:
            j = json.load(open(os.path.join(root, "config.json")))
            cfg[j["model"]] = j
        except (ValueError, KeyError):
            pass
configs = sorted({k for m in rows for k in rows[m]})
C = len(configs)
names = sorted(rows)
Rk = np.full((len(names), C), np.nan)
for j, c in enumerate(configs):
    ids = [i for i, m in enumerate(names) if c in rows[m]]
    Rk[ids, j] = rankdata([rows[names[i]][c] for i in ids], method="average")
elig = [i for i, m in enumerate(names) if m in cfg and len(rows[m]) == C]
E = [names[i] for i in elig]
Rm = Rk[elig]
mase = np.array([[rows[m][c] for c in configs] for m in E])
L = np.log(mase / mase[E.index("Seasonal_Naive")])
leak = np.array([cfg[m].get("testdata_leakage", "No") == "Yes" for m in E])
T = list(np.lexsort((Rm.mean(1), leak)))[:10]
print("displayed top ten:", [E[i] for i in T])

med = json.loads((R / "selection" / "config_median_lengths.json").read_text())
s70 = sorted(k for k, v in med.items() if v["median_length"] >= 1024)
s12 = sorted(s["config"] for s in json.loads((R / "selection" / "selected_configs.json").read_text())["selected"])
s58 = sorted(set(s70) - set(s12))
assert set(s12) <= set(s70) and len(s58) == 58

out = {}
for label, sub in (("70", s70), ("58", s58), ("12", s12)):
    si = [configs.index(c) for c in sub]
    g = np.exp(L[:, si].mean(1))[T]
    S = g.max() / g.min() - 1
    out[label] = {"S_pct": round(100 * S, 2), "ln1pS": float(np.log1p(S)),
                  "best": E[T[int(g.argmin())]], "worst": E[T[int(g.argmax())]]}
    print(label, out[label])

# S with one base dataset dropped (how much S itself moves with the configuration set)
bases = sorted({c.split("/")[0] for c in s70})
loo = []
for b in bases:
    sub = [c for c in s70 if c.split("/")[0] != b]
    si = [configs.index(c) for c in sub]
    g = np.exp(L[:, si].mean(1))[T]
    loo.append((b, round(100 * (g.max() / g.min() - 1), 2)))
print("S on 70 with one base dataset left out (min, max):", min(loo, key=lambda x: x[1]), max(loo, key=lambda x: x[1]))
out["loo_70"] = loo
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
