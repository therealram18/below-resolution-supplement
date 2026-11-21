"""Review scratch (round 3): does the exact sign-flip test on base-dataset sums
(DESIGN_ROUND3.md T1/T2) hold its level in the real layout of the 70
configurations (18 base datasets of sizes 1..7), when the true mean of d_c is
zero but d_c is skewed and heavy-tailed like the first round's values?

Null data are synthetic or resampled from the first round's 12 d_c values per
model (grid_results_h100/results/<model>/ctx_{default,512}/all_results.csv),
centred so that the mean of the resampling distribution is exactly 0.

Within a base dataset, three dependence modes:
  indep  : each configuration draws its own value
  shared : one draw per base dataset, every configuration gets it
  half   : d_c = (a_g + b_c) / sqrt(2), a_g and b_c both drawn
For each scenario: rejection rate of T1 (two-sided, p <= 0.05) split by the sign
of the observed sum, rejection rate of the one-sided upper test at 2.5% (the
form of T2 at its null boundary), and the Pratt signed-rank sign-flip test on
base-dataset means (two-sided 5%).
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/level_sim_t1t2.py
"""
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

R = Path("fmts")
sys.path.insert(0, str(R / "camera_ready"))
from exact_signflip import _signed_sums  # noqa: E402  (imported, not edited)

REPS = 20000
SEED = 20261119

# ---- layout of the 70 ----
med = json.loads((R / "selection" / "config_median_lengths.json").read_text())
s70 = sorted(k for k, v in med.items() if v["median_length"] >= 1024)
bases = sorted({c.split("/")[0] for c in s70})
SIZES = np.array([sum(1 for c in s70 if c.split("/")[0] == b) for b in bases])
assert SIZES.sum() == 70 and len(SIZES) == 18
GID = np.repeat(np.arange(18), SIZES)

# ---- first-round d_c ----
def first_round_d(model):
    def load(ctx):
        p = R / "grid_results_h100" / "results" / model / f"ctx_{ctx}" / "all_results.csv"
        return {r["dataset"]: float(r["eval_metrics/MASE[0.5]"]) for r in csv.DictReader(open(p))}
    a, b = load("default"), load("512")
    return np.array([np.log(b[c] / a[c]) for c in sorted(a)])

MODELS = ["timesfm25", "toto", "moirai2_small", "chronos_bolt_base"]
D = {m: first_round_d(m) for m in MODELS}
D["pooled48"] = np.concatenate([D[m] for m in MODELS])


def sign_flip_counts(s):
    """exact counts over all 2^G sign patterns: (#|sum|>=|obs|, #sum>=obs) / 2^G."""
    s = np.asarray(s, float)
    h = len(s) // 2
    a = _signed_sums(s[:h])
    b = np.sort(_signed_sums(s[h:]))
    obs = s.sum()
    tol = 1e-9 * max(1.0, abs(obs))
    thr = abs(obs) - tol
    if thr <= 0:
        p2 = 1.0
    else:
        hi = len(b) - np.searchsorted(b, thr - a, side="left")
        lo = np.searchsorted(b, -thr - a, side="right")
        p2 = float((hi + lo).sum() / (len(a) * len(b)))
    up = len(b) - np.searchsorted(b, obs - tol - a, side="left")
    p_up = float(up.sum() / (len(a) * len(b)))
    return p2, p_up


def pratt_two_sided(m):
    r = stats.rankdata(np.abs(m), method="average")
    return sign_flip_counts(np.sign(m) * r)[0]


def draw(rng, pool, mode):
    if mode == "indep":
        return rng.choice(pool, size=70)
    if mode == "shared":
        return rng.choice(pool, size=18)[GID]
    if mode == "half":
        return (rng.choice(pool, size=18)[GID] + rng.choice(pool, size=70)) / np.sqrt(2)
    raise ValueError(mode)


def run_scenario(args):
    name, pool, mode, seed = args
    rng = np.random.default_rng(seed)
    rej_pos = rej_neg = rej_up = rej_pratt = 0
    for _ in range(REPS):
        d = draw(rng, pool, mode)
        s = np.bincount(GID, weights=d, minlength=18)
        p2, pu = sign_flip_counts(s)
        if p2 <= 0.05:
            if s.sum() > 0:
                rej_pos += 1
            else:
                rej_neg += 1
        if pu <= 0.025:
            rej_up += 1
        if pratt_two_sided(s / SIZES) <= 0.05:
            rej_pratt += 1
    return dict(scenario=name, mode=mode, T1_two_sided=100 * (rej_pos + rej_neg) / REPS,
                T1_reject_positive=100 * rej_pos / REPS, T1_reject_negative=100 * rej_neg / REPS,
                upper_one_sided_2p5=100 * rej_up / REPS, pratt_two_sided=100 * rej_pratt / REPS,
                skew_of_pool=float(stats.skew(pool)))


def main():
    jobs = []
    k = 0
    # symmetric heavy-tailed null (sanity: exact test must be at or under 5%)
    rng0 = np.random.default_rng(SEED)
    sym = rng0.standard_t(3, size=200000) * 0.1
    sym = np.concatenate([sym, -sym])
    for mode in ("indep", "shared", "half"):
        jobs.append(("symmetric_t3", sym, mode, SEED + k)); k += 1
    for name in MODELS + ["pooled48"]:
        pool = D[name] - D[name].mean()           # centred: true mean exactly 0
        for mode in ("indep", "shared", "half"):
            jobs.append((f"centred_{name}", pool, mode, SEED + k)); k += 1
        neg = -pool                                # mirror image: skew the other way
        jobs.append((f"mirrored_{name}", neg, "shared", SEED + k)); k += 1
    with Pool(min(10, os.cpu_count() or 2)) as p:
        res = p.map(run_scenario, jobs)
    out = {"reps": REPS, "seed": SEED, "group_sizes": SIZES.tolist(),
           "first_round_d": {m: D[m].round(4).tolist() for m in MODELS}, "results": res}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(f"{'scenario':28s} {'mode':6s} {'T1%':>6s} {'+rej':>6s} {'-rej':>6s} {'up2.5':>6s} {'pratt':>6s} {'skew':>6s}")
    for r in res:
        print(f"{r['scenario']:28s} {r['mode']:6s} {r['T1_two_sided']:6.2f} {r['T1_reject_positive']:6.2f} "
              f"{r['T1_reject_negative']:6.2f} {r['upper_one_sided_2p5']:6.2f} {r['pratt_two_sided']:6.2f} {r['skew_of_pool']:6.2f}")


if __name__ == "__main__":
    main()
