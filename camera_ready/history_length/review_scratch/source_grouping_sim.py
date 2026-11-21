"""Review scratch (round 3): the 70 configurations come from 18 base datasets but
fewer sources: ett1/ett2 are one collection, bizitobs_application/service/l2c one,
bitbrains_fast_storage/rnd one (REVIEW_ROUND2.md line 226 counted 19 sources on the
full board). If datasets from one source share their effect, a sign-flip test that
treats the 18 base datasets as independent units is no longer exact.

Symmetric null (so only the grouping is under study): each SOURCE draws one effect
from a symmetric heavy-tailed law (t with 3 df), every configuration of that source
gets it plus a small configuration term. Test T1 (two-sided 5%, exact sum
statistic) with 18 base-dataset groups and with 14 source groups.
Also reports the first round's d_c for the three bizitobs datasets.
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/source_grouping_sim.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from level_sim_t1t2 import D, MODELS, sign_flip_counts  # noqa: E402

R = Path("fmts")
med = json.loads((R / "selection" / "config_median_lengths.json").read_text())
s70 = sorted(k for k, v in med.items() if v["median_length"] >= 1024)
SOURCE = {"ett1": "ett", "ett2": "ett", "bizitobs_application": "bizitobs", "bizitobs_service": "bizitobs",
          "bizitobs_l2c": "bizitobs", "bitbrains_fast_storage": "bitbrains", "bitbrains_rnd": "bitbrains"}
base = [c.split("/")[0] for c in s70]
src = [SOURCE.get(b, b) for b in base]
bases = sorted(set(base)); srcs = sorted(set(src))
gb = np.array([bases.index(b) for b in base]); gs = np.array([srcs.index(s) for s in src])
REPS = 20000
rng = np.random.default_rng(20261122)
res = {}
for share_cfg in (0.0, 0.3):
    rej18 = rej14 = 0
    for _ in range(REPS):
        e_src = rng.standard_t(3, size=len(srcs))
        d = e_src[gs] + share_cfg * rng.standard_t(3, size=70)
        s18 = np.bincount(gb, weights=d, minlength=len(bases))
        s14 = np.bincount(gs, weights=d, minlength=len(srcs))
        rej18 += sign_flip_counts(s18)[0] <= 0.05
        rej14 += sign_flip_counts(s14)[0] <= 0.05
    res[f"config_term_{share_cfg}"] = {"T1_18_groups_pct": 100 * rej18 / REPS, "T1_14_sources_pct": 100 * rej14 / REPS}
    print(share_cfg, res[f"config_term_{share_cfg}"])
sizes14 = {s: int((np.array(src) == s).sum()) for s in srcs}
print("source sizes:", sizes14)
# first-round d_c (as percent change) for the three bizitobs base datasets
import csv
def pct(model):
    def load(ctx):
        p = R / "grid_results_h100" / "results" / model / f"ctx_{ctx}" / "all_results.csv"
        return {r["dataset"]: float(r["eval_metrics/MASE[0.5]"]) for r in csv.DictReader(open(p))}
    a, b = load("default"), load("512")
    return {c: round(100 * (b[c] / a[c] - 1), 1) for c in sorted(a)}
fr = {m: pct(m) for m in MODELS}
for m in MODELS:
    ranked = sorted(fr[m].items(), key=lambda kv: -abs(kv[1]))[:3]
    print(m, "three largest |change| of 12:", ranked)
Path(__file__).with_suffix(".json").write_text(json.dumps({"reps": REPS, "results": res, "source_sizes": sizes14,
                                                          "first_round_pct": fr}, indent=1))
