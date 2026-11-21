"""Review scratch (round 3): rough size of the sampling noise in Toto's geometric
mean of scaled MASE on the 12 first-round configurations when the median forecast
is the median of 256 samples (DESIGN_ROUND3.md G3 criterion: within 1% of 0.7699).

Argument (no model run). The median of n = 256 draws has standard deviation
about 1.2533 * sigma / sqrt(n) = 0.078 sigma for a bell-shaped predictive law with
spread sigma (0.085 sigma for a t law with 3 df). If the forecast is about as good
as its stated spread, the mean absolute error is about 0.8 sigma, so each forecast
point's absolute error moves by about 0.098 of its size, in a random direction.
Averaging over a configuration divides this by the square root of the number of
independent forecast values. Two bounds:
  pessimistic: all steps of one forecast window move together (n_eff = windows x series x variates)
  optimistic : every forecast step is independent (n_eff = all forecast points)
The geometric mean over 12 configurations then has standard deviation
sqrt(sum of squares) / 12 (in log terms), and the difference between two
independent 256-sample runs (ours and the board's) has sqrt(2) times that.
Inputs: review_scratch/toto_context_70.json (instance counts from the GIFT-Eval data).
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/noise_12.py
"""
import json
from math import erf, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = Path("fmts")
rows = {r["config"]: r for r in json.loads((HERE / "toto_context_70.json").read_text())}
s12 = sorted(s["config"] for s in json.loads((R / "selection" / "selected_configs.json").read_text())["selected"])
K = 0.098
out = {}
for label, key in (("pessimistic", "inst"), ("optimistic", "points")):
    ss = 0.0
    per = {}
    for c in s12:
        r = rows[c]
        inst = r["n_series"] * r["target_dim"] * r["windows"]
        n = inst if key == "inst" else r["forecast_points"]
        sd = K / sqrt(n)
        per[c] = round(100 * sd, 2)
        ss += sd ** 2
    one = sqrt(ss) / 12
    diff = sqrt(2) * one
    p_miss = 1 - erf((0.01 / diff) / sqrt(2))   # P(|N(0, diff)| > 1%)
    out[label] = {"per_config_sd_pct": per, "gmean_sd_one_run_pct": round(100 * one, 2),
                  "gmean_sd_difference_pct": round(100 * diff, 2),
                  "chance_a_correct_setting_misses_1pct": round(100 * p_miss, 1)}
    print(label, {k: v for k, v in out[label].items() if k != "per_config_sd_pct"})
    print("   largest per-configuration SDs (%):", sorted(per.items(), key=lambda kv: -kv[1])[:4])
(HERE / "noise_12.json").write_text(json.dumps(out, indent=1))
