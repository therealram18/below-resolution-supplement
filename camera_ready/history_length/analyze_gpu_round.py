"""Analysis of the camera-ready GPU round, as fixed in DESIGN_ROUND3.md (with its Revision 1) before any run.

Usage:
    python analyze_gpu_round.py --results <results root> --out <dir> [--analyse-despite-machine-check]

Reads <results root>/<variant>/ctx_<default|512>/all_results.csv (same columns as the first round) for the
variants in ALL_VARIANTS, and, when present, the per-run records
<results root>/<variant>/ctx_<default|512>/per_config/*.run_meta.json. Writes <dir>/GPU_ROUND_RESULTS.json
and <dir>/GPU_ROUND_RESULTS.md. Every part whose input files are absent is reported as NOT RUN.

Reference inputs (read only): the first round's results (grid_results_h100), the board snapshot of
2026-09-07, the November pin (Timestral-209M only), the selection files and DESIGN_ROUND3.md. Their paths
and md5 sums are written into the JSON.

Groups of record are sources (Revision 1): the three bizitobs datasets are one source, the two ETT datasets
one, the two bitbrains datasets one, every other base dataset its own source. Every test is also computed on
base datasets and printed beside, not used for any claim.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # imports below come from camera_ready/, which this script must not write to

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

GR = Path(__file__).resolve().parent
CR = GR.parent
R = CR.parent
sys.path.insert(0, str(CR))
from exact_signflip import TOL, _signed_sums, exact_sum_p  # noqa: E402
import vernier_stats as vs  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Fixed settings (DESIGN_ROUND3.md and its Revision 1)
# ---------------------------------------------------------------------------------------------
MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"
LENGTHS = ("default", "512")                      # folder names ctx_default and ctx_512
LENGTH_WORD = {"default": "documented", "512": "512"}

DETERMINISTIC = ("chronos_bolt_base", "moirai2_small", "timesfm25")
TOTO_A = "toto__A__seed42"
G1_MODELS = DETERMINISTIC + (TOTO_A,)
TOTO_SEEDS = (("42", "toto__A__seed42"), ("43", "toto__A__seed43"), ("44", "toto__A__seed44"))
TOTO_FALLBACK = "toto__A_tf32off__seed42"
TOTO_VARIANTS = tuple(v for _, v in TOTO_SEEDS) + (TOTO_FALLBACK,)
G2_MODELS = ("timesfm3", "timestral209m")
ALL_VARIANTS = DETERMINISTIC + TOTO_VARIANTS + G2_MODELS

LABEL = {
    "chronos_bolt_base": "Chronos-Bolt",
    "moirai2_small": "Moirai-2.0",
    "timesfm25": "TimesFM-2.5",
    "toto__A__seed42": "Toto, setting A, seed 42",
    "toto__A__seed43": "Toto, setting A, seed 43",
    "toto__A__seed44": "Toto, setting A, seed 44",
    "toto__A_tf32off__seed42": "Toto, setting A with the faster matrix products off, seed 42",
    "timesfm3": "TimesFM-3",
    "timestral209m": "Timestral-209M",
}

# Sources (Revision 1): base datasets not listed here are their own source.
SOURCE_OF = {
    "bizitobs_application": "bizitobs", "bizitobs_l2c": "bizitobs", "bizitobs_service": "bizitobs",
    "ett1": "ett", "ett2": "ett",
    "bitbrains_fast_storage": "bitbrains", "bitbrains_rnd": "bitbrains",
}
SOURCE_SIZES_70 = [12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1]
N_SOURCES_12 = 9
GROUPINGS = ("source", "base_dataset")
GROUP_WORD = {"source": "sources", "base_dataset": "base datasets"}
RECORD_GROUPING = "source"

# Toto: both lengths give Toto the same 79 steps on this configuration; d_c = 0 by rule, no 512 run.
M4_RULE_CONFIG = "m4_daily/D/short"

MC_PASS_BELOW = 1e-4    # machine check: largest relative difference below this = PASS
MC_STOP_ABOVE = 1e-3    # above this (or any d_c changing sign) = STOP; in between = CONTINUE
REPRO_TOL = 0.01        # G3 reading (a) and G2 reproduction criterion: within 1%
ALPHA_T1 = 0.05
ALPHA_T2 = 0.025
CI_LEVEL = 0.05         # 95% interval
LEVEL_DRAWS = 20000     # level check (Revision 1): draws and seed
LEVEL_SEED = 20261119
PAPER_TOTO_BOARD = 0.7699
PAPER_TOTO_FIRST_ROUND = 0.8126
DESIGN_S_PCT = {"eligible_70": 11.76, "new_58": 12.71, "first_12": 9.34}
DESIGN_S_LOO_70 = (8.51, 13.22)
DESIGN_OVERLAP_58 = (43, 20)    # of the 58: from a base dataset among the 12; same series as one of the 12

SNAP = R / "snapshot_2026-09-07" / "results"
FIRST_ROUND = R / "grid_results_h100" / "results"
SN_FILE = SNAP / "seasonal_naive" / "all_results.csv"
SN_PIN_FILE = CR / "boards" / "pin_2026-11-13" / "gift" / "results" / "Seasonal_Naive" / "all_results.csv"
PUBLISHED = {
    "toto": SNAP / "Toto_Open_Base_1.0" / "all_results.csv",
    "timesfm3": SNAP / "TimesFM-3" / "all_results.csv",
    "timestral209m": CR / "boards" / "pin_2026-11-13" / "gift" / "results" / "Timestral-209M" / "all_results.csv",
}
PUBLISHED_WHERE = {
    "toto": "Toto_Open_Base_1.0 in the 2026-09-07 snapshot",
    "timesfm3": "TimesFM-3 in the 2026-09-07 snapshot",
    "timestral209m": "Timestral-209M at the November pin, GIFT-Eval commit 42beef3e",
}
FIRST_ROUND_MODEL = {"chronos_bolt_base": "chronos_bolt_base", "moirai2_small": "moirai2_small",
                     "timesfm25": "timesfm25", **{v: "toto" for v in TOTO_VARIANTS}}
# Displayed top ten of the 2026-09-07 snapshot as documented in answers_part_a.md N11 and retest_gift.json
# ("gmean_pairs_order_top10"). The script recomputes it from the snapshot (poolcheck/resolution_budget.py
# lines 41 and 55-60) and stops if the two differ.
TOP10_DOCUMENTED = (
    "STRIDE w/ Synapse", "EXAONE-Forecast-Agent", "LS-MoE", "LS-Agent", "Falcon-Agent",
    "CastStar", "limix_moe", "TIMEHEDGE", "RegLab-AH-1", "RacineCast-1-NC",
)
SELECTED_FILE = R / "selection" / "selected_configs.json"
MEDIANS_FILE = R / "selection" / "config_median_lengths.json"
DESIGN_FILE = GR / "DESIGN_ROUND3.md"
IST = timezone(timedelta(hours=5, minutes=30))

RULES = [
    "Groups of record are sources (DESIGN_ROUND3.md, Revision 1). The same tests on base datasets are printed "
    "beside them and are not used for any claim.",
    "A configuration enters a model's analysis only if both lengths are present; the ones that drop out are "
    "named, and S for that model is computed on the configurations that remain.",
    "Toto (every setting-A variant): for m4_daily/D/short the 512 value is the documented value by rule "
    "(d_c = 0), whenever the variant's 512 results file exists. A 512 row for it, if present, is replaced.",
    "A MASE value that is missing, not finite, or not positive counts as a missing configuration (a failed run), "
    "and so does a configuration listed twice in one file with different values; both are named under "
    "files_read -> problems.",
    "T1: Holm over a fixed family (four models in G1, two in G2); a model with no result enters with p = 1. "
    "T2: Holm over the same family; a model that did not pass T1 after Holm with a positive effect, or has no "
    "result, enters with p = 1. In G1 each of the 70, the 58 and the 12 has its own Holm correction.",
    "Machine check: largest relative difference below 1e-4 is PASS; from 1e-4 to 1e-3 is CONTINUE; above 1e-3, "
    "or any d_c of the three models with a different sign (positive, zero, negative) than in the first round, is "
    "STOP. Only STOP holds the other parts (unless --analyse-despite-machine-check is given). The outcome is "
    "computed on the values present; missing values are counted and named.",
    "A reproduction check (G3, G2) with configurations missing is computed on the configurations present and "
    "marked INCOMPLETE instead of PASS or FAIL; G3 reading (b) is NOT RUN if a seed's file is absent.",
    "Level check: the model's d_c on the configurations it used, centred to mean zero, drawn with replacement "
    "into the fixed source layout (14 sources of the 70 for G1, 9 sources of the 12 for G2), one value shared "
    "inside a source or one value per configuration; a fresh generator with the design's seed for each model "
    "and each way.",
]


# ---------------------------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------------------------
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Reader:
    """Reads MASE[0.5] per configuration and keeps a record of every file read (path, md5, problems)."""

    def __init__(self):
        self.files = {}

    def mase(self, path, expect_model=None):
        path = Path(path)
        if not path.is_file():
            return None
        vals, problems, models, conflicts = {}, [], set(), set()
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                c, raw = row.get("dataset"), row.get(MASE_COL)
                models.add(row.get("model"))
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    v = float("nan")
                if not (math.isfinite(v) and v > 0):
                    problems.append(f"{c}: MASE value {raw!r} is missing, not finite or not positive; "
                                    "counted as a missing configuration")
                    continue
                if c in vals and vals[c] != v:
                    conflicts.add(c)
                    problems.append(f"{c}: appears twice with different MASE values ({vals[c]!r}, {v!r}); "
                                    "counted as a missing configuration")
                elif c in vals:
                    problems.append(f"{c}: appears twice with the same MASE value; used once")
                vals[c] = v
        for c in conflicts:
            vals.pop(c, None)
        self.files[str(path)] = {"md5": md5(path), "configurations_read": len(vals),
                                 "model_column": sorted(m for m in models if m is not None),
                                 "problems": problems}
        if expect_model is not None and models != {expect_model}:
            raise SystemExit(f"{path}: expected model {expect_model!r}, found {sorted(models)}")
        return vals

    def json(self, path):
        path = Path(path)
        self.files[str(path)] = {"md5": md5(path)}
        with open(path) as f:
            return json.load(f)


def results_path(root, variant, length):
    return Path(root) / variant / f"ctx_{length}" / "all_results.csv"


def displayed_top10(snapshot):
    """The board's displayed top ten, as poolcheck/resolution_budget.py computes it (lines 13-41): mean MASE rank
    over all configurations among the entries with a config.json and every configuration, leakage entries
    last. Returns (names, digest of the files read, number of entries)."""
    rows, cfg, digests = defaultdict(dict), {}, []
    for root, dirs, files in sorted(os.walk(snapshot)):
        if "all_results.csv" in files:
            p = os.path.join(root, "all_results.csv")
            digests.append(f"{os.path.relpath(p, snapshot)}:{md5(p)}")
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        rows[r["model"]][r["dataset"]] = (float(r[MASE_COL]), float(r[CRPS_COL]))
                    except (KeyError, ValueError):
                        pass
        if "config.json" in files:
            p = os.path.join(root, "config.json")
            digests.append(f"{os.path.relpath(p, snapshot)}:{md5(p)}")
            try:
                with open(p) as f:
                    j = json.load(f)
                cfg[j["model"]] = j
            except (ValueError, KeyError):
                pass
    configs = sorted({k for m in rows for k in rows[m]})
    names = sorted(rows)
    Rk = np.full((len(names), len(configs)), np.nan)
    for j, c in enumerate(configs):
        ids = [i for i, m in enumerate(names) if c in rows[m]]
        Rk[ids, j] = rankdata([rows[names[i]][c][0] for i in ids], method="average")
    elig = [i for i, m in enumerate(names) if m in cfg and len(rows[m]) == len(configs)]
    E = [names[i] for i in elig]
    Rm = Rk[elig]
    leak = np.array([cfg[m].get("testdata_leakage", "No") == "Yes" for m in E])
    T = list(np.lexsort((Rm.mean(1), leak)))[:10]
    digest = hashlib.md5("\n".join(sorted(digests)).encode()).hexdigest()
    return [E[i] for i in T], digest, len(names)


def design_freeze(path):
    """The freeze sentence of DESIGN_ROUND3.md and the freeze time in UTC (None if not found)."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None, None
    m = re.search(r"\*\*Freeze\*\*:\s*(.+?)(?:\n\s*\n|\Z)", text, flags=re.S)
    if not m:
        return None, None
    sentence = " ".join(m.group(1).split())
    t = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{1,2}:\d{2}) IST", sentence)
    when = (datetime.strptime(f"{t.group(1)} {t.group(2)}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            .astimezone(timezone.utc) if t else None)
    return sentence, when


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def base_of(config):
    return config.split("/")[0]


def source_of_base(base):
    return SOURCE_OF.get(base, base)


def group_of(config, grouping):
    b = base_of(config)
    return b if grouping == "base_dataset" else source_of_base(b)


def gmean_scaled(vals, sn, configs):
    """Geometric mean over configs of vals[c] / sn[c]."""
    return float(math.exp(np.mean([math.log(vals[c] / sn[c]) for c in configs])))


def two_sided_sum_p(x):
    """Exact two-sided sign-flip p on group sums x: share of the 2^G sign patterns w with
    |sum w_g x_g| >= |sum x_g|. This is exact_sum_p of camera_ready/exact_signflip.py, capped at 1:
    when the observed sum is not exactly zero but below rounding level (about 1e-16 of the half sums),
    exact_sum_p counts the patterns with a + b = 0 twice and can return slightly more than 1; the true
    value there is 1. For any other input the cap changes nothing."""
    return min(1.0, exact_sum_p(x))


def one_sided_sum_p(x):
    """Exact one-sided sign-flip p on group sums x: share of the 2^G sign patterns w with
    sum w_g x_g >= sum x_g. Same meet-in-the-middle method and tolerance as exact_sum_p."""
    x = np.asarray(x, float)
    h = len(x) // 2
    a, b = _signed_sums(x[:h]), np.sort(_signed_sums(x[h:]))
    obs = x.sum()
    thr = obs - TOL * abs(obs)
    cnt = len(b) - np.searchsorted(b, thr - a, side="left")
    return float(cnt.sum() / (len(a) * len(b)))


def holm_adjusted(p):
    """Holm-adjusted p-values: the k-th smallest p times (m - k + 1), made non-decreasing, capped at 1.
    A hypothesis is rejected at level alpha when its adjusted p is <= alpha."""
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def group_sums(configs, d, grouping):
    """Groups (sorted names), the sum of d over each group's configurations, and the group sizes."""
    names = sorted({group_of(c, grouping) for c in configs})
    idx = np.array([names.index(group_of(c, grouping)) for c in configs])
    s = np.bincount(idx, weights=np.asarray(d, float), minlength=len(names))
    n = np.bincount(idx, minlength=len(names)).astype(float)
    return names, s, n


def signflip_interval(s, n, level=CI_LEVEL):
    """Set of shifts theta whose two-sided sign-flip p for d - theta is at least `level`, where the
    statistic uses the group sums s_g - n_g * theta.

    For one sign pattern w, with a = sum w_g s_g, b = sum w_g n_g, S = sum s_g, N = sum n_g, the pattern
    counts at theta when |a - b theta| >= |S - N theta|. The difference of squares factors as
    [(a - S) + (N - b) theta] * [(a + S) - (N + b) theta], so for every pattern other than all-plus and
    all-minus (which count for every theta) the pattern counts exactly on the closed interval between
    (S - a) / (N - b) and (S + a) / (N + b). Both are fractions with positive denominators whose mediant
    is S / N, the estimate, so every such interval contains the estimate. p(theta) is therefore the share
    of these intervals that contain theta: it never increases as theta moves away from the estimate,
    the set is an interval, and its upper end is the k-th largest right end, k = ceil(level * 2^G)
    (lower end likewise). Returns (lower, upper, k); an end is +-inf when fewer than k patterns are finite.
    """
    s = np.asarray(s, float)
    n = np.asarray(n, float)
    G = len(s)
    if G > 24:
        raise ValueError("too many groups for full enumeration")
    a, b = np.zeros(1), np.zeros(1)
    for sv, nv in zip(s, n):
        a, b = np.concatenate([a + sv, a - sv]), np.concatenate([b + nv, b - nv])
    S, N = s.sum(), n.sum()
    const = np.abs(b) == N          # all-plus and all-minus (b is a sum of integers, exact in floats)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = (S - a) / (N - b)
        r2 = (S + a) / (N + b)
    lo = np.where(const, -np.inf, np.minimum(r1, r2))
    hi = np.where(const, np.inf, np.maximum(r1, r2))
    k = int(math.ceil(level * 2 ** G - 1e-9))
    upper = float(np.sort(hi)[::-1][k - 1])
    lower = float(np.sort(lo)[k - 1])
    return lower, upper, k


def signed_rank_p(means):
    """Signed-rank sign-flip test on group means (camera_ready/verify_signedrank_level.py): rank |x| with
    average ranks, zeros keep their rank and get sign zero, exact_sum_p on sign x rank."""
    x = np.asarray(means, float)
    sr = np.sign(x) * rankdata(np.abs(x), method="average")
    return two_sided_sum_p(sr), sr


def top10_spread(top10, sn, configs):
    """S: for each of the ten entries, the geometric mean over configs of its MASE divided by Seasonal
    Naive's MASE; S = largest / smallest - 1."""
    g = {name: gmean_scaled(vals, sn, configs) for name, vals in top10.items()}
    return float(max(g.values()) / min(g.values()) - 1), g


def spread_block(top10, sn, configs):
    """S on configs, with the range of S when one base dataset is left out."""
    S, g = top10_spread(top10, sn, configs)
    out = {"value": S, "pct": 100 * S, "log_1_plus_S": math.log1p(S), "configurations": len(configs),
           "top10_gmean_scaled_mase": g}
    bases = sorted({base_of(c) for c in configs})
    if len(bases) >= 2:
        loo = {b: 100 * top10_spread(top10, sn, [c for c in configs if base_of(c) != b])[0] for b in bases}
        lo_b, hi_b = min(loo, key=loo.get), max(loo, key=loo.get)
        out["leave_one_base_dataset_out_pct"] = loo
        out["loo_smallest"] = {"left_out": lo_b, "pct": loo[lo_b]}
        out["loo_largest"] = {"left_out": hi_b, "pct": loo[hi_b]}
    return out


def pct(theta):
    """exp(theta) - 1 in percent; None for an unbounded end."""
    if theta is None or not math.isfinite(theta):
        return None
    return 100 * (math.exp(theta) - 1)


def paired_logs(doc, l512, configs):
    """d_c = ln(MASE at 512 / MASE at documented) for configs present at both lengths; the others are
    returned as dropped, with the lengths at which they are missing."""
    used, d, dropped = [], [], []
    for c in configs:
        miss = [LENGTH_WORD[L] for L, vals in (("default", doc), ("512", l512)) if vals is None or c not in vals]
        if miss:
            dropped.append({"config": c, "missing_at": miss})
            continue
        used.append(c)
        d.append(math.log(l512[c] / doc[c]))
    return used, np.array(d, float), dropped


def leave_one_out(used, d, grouping):
    labels = np.array([group_of(c, grouping) for c in used])
    names = sorted(set(labels))
    if len(names) < 2:
        return {"status": "NOT RUN", "reason": f"needs at least two {GROUP_WORD[grouping]}"}
    loo = {g: pct(float(d[labels != g].mean())) for g in names}
    lo_g, hi_g = min(loo, key=loo.get), max(loo, key=loo.get)
    return {"effect_pct_without": loo, "smallest": {"left_out": lo_g, "effect_pct": loo[lo_g]},
            "largest": {"left_out": hi_g, "effect_pct": loo[hi_g]}}


def grouped_block(used, d, grouping):
    """T1 p, inverted interval, CR2 interval and signed-rank test with one grouping."""
    names, s, n = group_sums(used, d, grouping)
    G = len(names)
    p1 = two_sided_sum_p(s)
    out = {
        "grouping": grouping,
        "used_for_claims": grouping == RECORD_GROUPING,
        "n_groups": G,
        "T1": {
            "statistic": f"sum over {GROUP_WORD[grouping]} of s_g, s_g = sum of d_c over the group's configurations",
            "group_sums": dict(zip(names, map(float, s))),
            "group_sizes": dict(zip(names, map(int, n))),
            "n_sign_patterns": 2 ** G,
            "p_two_sided": p1,
            "sign_patterns_counted": int(round(p1 * 2 ** G)),
        },
    }
    lo, hi, k = signflip_interval(s, n)
    out["interval_signflip_95"] = {
        "theta_low": lo if math.isfinite(lo) else None,
        "theta_high": hi if math.isfinite(hi) else None,
        "pct_low": pct(lo), "pct_high": pct(hi),
        "unbounded": not (math.isfinite(lo) and math.isfinite(hi)),
        "sign_patterns_needed": k,
    }
    if G >= 2:
        cl = [group_of(c, grouping) for c in used]
        r = vs.cr2_test(d, cl, df="icc")
        out["interval_cr2_95"] = {
            "method": "vernier_stats.cr2_test(d, group, df='icc'): CR2 standard error, Satterthwaite df",
            "theta": float(r["theta"]), "se": float(r["se"]), "df": float(r["df"]),
            "icc": float(vs.icc_hat(d, cl)), "p_two_sided": float(r["p"]),
            "theta_low": float(r["ci95"][0]), "theta_high": float(r["ci95"][1]),
            "pct_low": pct(r["ci95"][0]), "pct_high": pct(r["ci95"][1]),
        }
        means = s / n
        p_sr, sr = signed_rank_p(means)
        out["signed_rank"] = {"group_means": dict(zip(names, map(float, means))),
                              "signed_ranks": dict(zip(names, map(float, sr))), "p_two_sided": p_sr}
    else:
        reason = f"needs at least two {GROUP_WORD[grouping]}"
        out["interval_cr2_95"] = {"status": "NOT RUN", "reason": reason}
        out["signed_rank"] = {"status": "NOT RUN", "reason": reason}
    return out


def effect_block(used, d, top10, sn):
    """Every number reported for one model on one set of configurations (tests decided in family_tests)."""
    mean_d = float(d.mean())
    out = {
        "n_configurations": len(used),
        "sources": sorted({group_of(c, "source") for c in used}),
        "base_datasets": sorted({base_of(c) for c in used}),
        "d_c": {c: float(x) for c, x in zip(used, d)},
        "mean_d": mean_d,
        "effect_pct": pct(mean_d),
    }
    out["n_sources"], out["n_base_datasets"] = len(out["sources"]), len(out["base_datasets"])
    for g in GROUPINGS:
        out[f"by_{g}"] = grouped_block(used, d, g)
    out["leave_one_out"] = {g: leave_one_out(used, d, g) for g in GROUPINGS}
    out["S"] = spread_block(top10, sn, used)
    return out


def family_tests(blocks, family, grouping):
    """T1 with Holm over `family` (a model with no result enters with p = 1), then T2 for the models whose
    T1 rejects after Holm with a positive effect, with Holm over the whole family (every other model enters
    with p = 1). Results go into block['by_<grouping>']."""
    key = f"by_{grouping}"
    p1 = [blocks[m][key]["T1"]["p_two_sided"] if blocks.get(m) else 1.0 for m in family]
    adj1 = holm_adjusted(p1)
    p2 = []
    missing = [x for x in family if not blocks.get(x)]
    for m, pa in zip(family, adj1):
        b = blocks.get(m)
        if not b:
            p2.append(1.0)
            continue
        gb = b[key]
        gb["T1"].update({"holm_family": list(family), "holm_family_missing_entered_as_p_1": missing,
                         "p_holm": float(pa), "rejects_at_5pct_after_holm": bool(pa <= ALPHA_T1)})
        if pa <= ALPHA_T1 and b["mean_d"] > 0:
            names, s, n = group_sums(list(b["d_c"]), list(b["d_c"].values()), grouping)
            shift = b["S"]["log_1_plus_S"]
            x = s - n * shift
            p = one_sided_sum_p(x)
            lo = gb["interval_signflip_95"]["theta_low"]
            gb["T2"] = {"computed": True, "null": "mean of d_c is at most ln(1 + S)",
                        "ln_1_plus_S": shift, "S_pct": b["S"]["pct"],
                        "shifted_group_totals": dict(zip(names, map(float, x))),
                        "p_one_sided": p, "sign_patterns_counted": int(round(p * 2 ** len(names))),
                        "n_sign_patterns": 2 ** len(names),
                        "interval_low_above_ln_1_plus_S": bool(lo is not None and lo > shift)}
            p2.append(p)
        else:
            why = "T1 does not reject after Holm" if pa > ALPHA_T1 else "the effect is not positive"
            gb["T2"] = {"computed": False, "reason": why, "enters_holm_with_p": 1.0}
            p2.append(1.0)
    adj2 = holm_adjusted(p2)
    for m, pa in zip(family, adj2):
        b = blocks.get(m)
        if not b:
            continue
        t2 = b[key]["T2"]
        t2.update({"holm_family": list(family), "p_holm": float(pa),
                   "rejects_at_2_5pct_after_holm": bool(t2["computed"] and pa <= ALPHA_T2)})


def level_check(d, sizes, direction, draws=LEVEL_DRAWS, seed=LEVEL_SEED):
    """Revision 1 level check (port of review_scratch/level_sim_t1t2.py): d centred to mean zero, drawn with
    replacement into groups of the given sizes, one value shared inside a group or one value per
    configuration; share of draws where T1 (two-sided, 5%) claims an effect, split by the sign of the sum."""
    pool = np.asarray(d, float) - float(np.mean(d))
    sizes = np.asarray(sizes, int)
    G, N = len(sizes), int(sizes.sum())
    gid = np.repeat(np.arange(G), sizes)
    out = {"draws": draws, "seed": seed, "group_sizes": sizes.tolist(), "pool_size": len(pool),
           "observed_direction": "positive" if direction > 0 else "negative" if direction < 0 else "zero"}
    for mode in ("shared", "independent"):
        rng = np.random.default_rng(seed)
        pos = neg = 0
        for _ in range(draws):
            x = rng.choice(pool, size=G)[gid] if mode == "shared" else rng.choice(pool, size=N)
            s = np.bincount(gid, weights=x, minlength=G)
            if two_sided_sum_p(s) <= ALPHA_T1:
                if s.sum() > 0:
                    pos += 1
                else:
                    neg += 1
        r = {"two_sided_pct": 100 * (pos + neg) / draws, "positive_pct": 100 * pos / draws,
             "negative_pct": 100 * neg / draws}
        r["in_direction_of_observed_effect_pct"] = (r["positive_pct"] if direction > 0 else
                                                    r["negative_pct"] if direction < 0 else None)
        out[mode] = r
    return out


# ---------------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------------
def not_run(reason, **extra):
    return {"status": "NOT RUN", "reason": reason, **extra}


def compare_to_published(ours, pub, sn, configs):
    """Reproduction check on `configs` at the documented length (G3 reading (a), G2)."""
    used = [c for c in configs if c in ours]
    missing = [c for c in configs if c not in ours]
    if not used:
        return not_run("no configuration of the 12 is in the file", missing=missing)
    g_ours, g_pub = gmean_scaled(ours, sn, used), gmean_scaled(pub, sn, used)
    gap = g_ours / g_pub - 1
    within = abs(gap) <= REPRO_TOL
    verdict = ("PASS" if within else "FAIL") if not missing else f"INCOMPLETE ({len(used)} of {len(configs)} configurations)"
    return {
        "status": "RUN",
        "configurations_used": used, "missing": missing,
        "gmean_scaled_ours": g_ours, "gmean_scaled_published_same_configs": g_pub,
        "relative_gap": gap, "relative_gap_pct": 100 * gap,
        "within_1pct": bool(within), "verdict": verdict,
        "per_configuration": [{"config": c, "ours": ours[c], "published": pub[c],
                               "relative_difference": ours[c] / pub[c] - 1} for c in used],
    }


def machine_check(res, first, sel12):
    rows, present = [], False
    for m in DETERMINISTIC:
        for L in LENGTHS:
            new, old = res[m][L], first[m][L]
            row = {"model": m, "length": LENGTH_WORD[L]}
            if new is None:
                row.update({"file_present": False, "n_compared": 0, "missing": list(sel12)})
                rows.append(row)
                continue
            present = True
            per = [{"config": c, "new": new[c], "first_round": old[c], "relative_difference": abs(new[c] / old[c] - 1)}
                   for c in sel12 if c in new]
            diffs = [x["relative_difference"] for x in per]
            row.update({"file_present": True, "n_compared": len(per), "missing": [c for c in sel12 if c not in new],
                        "largest_relative_difference": max(diffs) if diffs else None,
                        "count_at_or_above_1e-4": sum(x >= MC_PASS_BELOW for x in diffs),
                        "count_above_1e-3": sum(x > MC_STOP_ABOVE for x in diffs), "per_configuration": per})
            rows.append(row)
    if not present:
        return not_run("none of the six input files (three models, two lengths) is present", rows=rows)
    n_total = len(DETERMINISTIC) * len(LENGTHS) * len(sel12)
    n_cmp = sum(r["n_compared"] for r in rows)
    if n_cmp == 0:
        return not_run("the files present hold none of the 12 configurations", rows=rows)
    # d_c of each deterministic model on the 12, new against the first round
    signs = []
    for m in DETERMINISTIC:
        used_new, d_new, _ = paired_logs(res[m]["default"], res[m]["512"], sel12)
        _, d_old, _ = paired_logs(first[m]["default"], first[m]["512"], sel12)
        old = dict(zip(sel12, d_old))
        for c, x in zip(used_new, d_new):
            if np.sign(x) != np.sign(old[c]):
                signs.append({"model": m, "config": c, "d_new": float(x), "d_first_round": float(old[c])})
    largest = max(r["largest_relative_difference"] for r in rows if r.get("largest_relative_difference") is not None)
    if largest > MC_STOP_ABOVE or signs:
        verdict = "STOP"
    elif largest >= MC_PASS_BELOW:
        verdict = "CONTINUE"
    else:
        verdict = "PASS"
    worst = max(((r["model"], r["length"], x) for r in rows for x in r.get("per_configuration", [])),
                key=lambda t: t[2]["relative_difference"])
    return {"status": "RUN", "values_expected": n_total, "values_compared": n_cmp,
            "values_missing": n_total - n_cmp, "complete": n_cmp == n_total,
            "largest_relative_difference": largest,
            "largest_at": {"model": worst[0], "length": worst[1], "config": worst[2]["config"]},
            "count_at_or_above_1e-4": sum(r.get("count_at_or_above_1e-4", 0) for r in rows),
            "count_above_1e-3": sum(r.get("count_above_1e-3", 0) for r in rows),
            "d_c_sign_changes": signs, "thresholds": {"pass_below": MC_PASS_BELOW, "stop_above": MC_STOP_ABOVE},
            "verdict": verdict, "rows": rows}


def part_g3(res, ref, sel12):
    fr = compare_to_published(ref["toto_first_round_doc"], ref["pub"]["toto"], ref["sn"], sel12)
    fr["source"] = str(FIRST_ROUND / "toto" / "ctx_default" / "all_results.csv")
    fr["equals_paper_0.8126_to_4_decimals"] = round(fr["gmean_scaled_ours"], 4) == PAPER_TOTO_FIRST_ROUND
    out = {"reference": {
        "board_gmean_scaled_on_12": ref["toto_board_12"],
        "board_value_equals_paper_0.7699_to_4_decimals": round(ref["toto_board_12"], 4) == PAPER_TOTO_BOARD,
        "first_round_point_setting": fr,
    }}
    # reading (a): within 1%
    a = {}
    for v in (TOTO_A, TOTO_FALLBACK):
        vals = res[v]["default"]
        a[v] = (compare_to_published(vals, ref["pub"]["toto"], ref["sn"], sel12) if vals is not None
                else not_run("input file absent", file=str(results_path(ref["root"], v, "default"))))
    out["reading_a_within_1pct"] = a
    # reading (b): within noise of the three seeds
    seeds = {s: res[v]["default"] for s, v in TOTO_SEEDS}
    absent = [s for s, vals in seeds.items() if vals is None]
    if absent:
        out["reading_b_within_noise"] = not_run(f"the documented-length file of seed(s) {', '.join(absent)} is absent")
    else:
        common = [c for c in sel12 if all(c in vals for vals in seeds.values())]
        if not common:
            out["reading_b_within_noise"] = not_run("no configuration of the 12 is present for all three seeds")
        else:
            g = {s: gmean_scaled(vals, ref["sn"], common) for s, vals in seeds.items()}
            board = gmean_scaled(ref["pub"]["toto"], ref["sn"], common)
            lo, hi = min(g.values()), max(g.values())
            half = (hi - lo) / 2
            inside = lo - half <= board <= hi + half
            missing = [c for c in sel12 if c not in common]
            out["reading_b_within_noise"] = {
                "status": "RUN", "configurations_used": common, "missing": missing,
                "gmean_scaled_by_seed": g, "range": [lo, hi], "widened_range": [lo - half, hi + half],
                "board_gmean_scaled_same_configs": board, "inside": bool(inside),
                "verdict": ("PASS" if inside else "FAIL") if not missing else
                f"INCOMPLETE ({len(common)} of {len(sel12)} configurations)"}
    ran = [x for x in list(a.values()) + [out["reading_b_within_noise"]] if x["status"] == "RUN"]
    out["status"] = "RUN" if ran else "NOT RUN"
    if not ran:
        out["reason"] = "no Toto setting A file at the documented length is present"
    return out


def part_g4(res, ref, sel12):
    seeds = {}
    for seed, v in TOTO_SEEDS:
        doc, l512 = res[v]["default"], res[v]["512"]
        if doc is None and l512 is None:
            seeds[seed] = not_run("input files absent", variant=v)
            continue
        e = {"status": "RUN", "variant": v}
        for L, vals in (("default", doc), ("512", l512)):
            used = [c for c in sel12 if vals is not None and c in vals]
            e[f"gmean_scaled_{LENGTH_WORD[L]}"] = gmean_scaled(vals, ref["sn"], used) if used else None
            e[f"configurations_{LENGTH_WORD[L]}"] = len(used)
        used, d, dropped = paired_logs(doc, l512, sel12)
        e["configurations_both_lengths"] = len(used)
        e["dropped"] = dropped
        e["effect_pct"] = pct(float(d.mean())) if len(used) else None
        seeds[seed] = e
    ran = {k: x for k, x in seeds.items() if x["status"] == "RUN"}
    out = {"seeds": seeds, "value_of_record": "seed 42"}
    if not ran:
        out.update(status="NOT RUN", reason="none of the three seeds' files is present")
        return out
    out["status"] = "RUN"
    rng = {}
    for key in ("gmean_scaled_documented", "gmean_scaled_512", "effect_pct"):
        xs = [x[key] for x in ran.values() if x[key] is not None]
        rng[key] = {"seeds_with_value": [k for k, x in ran.items() if x[key] is not None],
                    "min": min(xs) if xs else None, "max": max(xs) if xs else None,
                    "range": (max(xs) - min(xs)) if len(xs) >= 2 else None}
    out["range_over_seeds"] = rng
    sets = [tuple(c for c in sel12 if c not in {y["config"] for y in x["dropped"]}) for x in ran.values()]
    out["same_configurations_for_every_seed"] = len(set(sets)) <= 1
    return out


def blocks_on(res, ref, models, configs):
    """Per-model effect blocks on configs; returns (blocks of models with a result, entries for all)."""
    blocks, per_model = {}, {}
    for m in models:
        doc, l512 = res[m]["default"], res[m]["512"]
        if doc is None and l512 is None:
            per_model[m] = not_run("input files absent at both lengths")
            continue
        used, d, dropped = paired_logs(doc, l512, configs)
        if not used:
            per_model[m] = not_run("no configuration has both lengths", dropped=dropped)
            continue
        b = effect_block(used, d, ref["top10"], ref["sn"])
        b.update(status="RUN", dropped=dropped, configurations_in_set=len(configs))
        blocks[m] = b
        per_model[m] = b
    for g in GROUPINGS:
        if blocks:
            family_tests(blocks, list(models), g)
    return blocks, per_model


def run_family(res, ref, models, subsets):
    """Effect blocks and tests for `models` on each named subset, plus the same on the configurations that
    every model in the family finished."""
    out = {}
    for name, configs in subsets.items():
        blocks, per_model = blocks_on(res, ref, models, configs)
        entry = {"status": "RUN" if blocks else "NOT RUN", "models": per_model}
        if not blocks:
            entry["reason"] = "no model has a configuration with both lengths"
        elif len(blocks) < len(models):
            entry["common_configurations"] = not_run("not every model of the family has a result on this set")
        else:
            common = [c for c in configs if all(c in b["d_c"] for b in blocks.values())]
            if not common:
                entry["common_configurations"] = not_run("no configuration was finished by every model")
            else:
                cb, cpm = blocks_on(res, ref, models, common)
                entry["common_configurations"] = {
                    "status": "RUN", "configurations": common, "n_configurations": len(common),
                    "same_as_every_model_used": all(len(b["d_c"]) == len(common) for b in blocks.values()),
                    "models": cpm}
        out[name] = entry
    return out


def part_g1(res, ref, subsets):
    present = [m for m in G1_MODELS if res[m]["default"] is not None or res[m]["512"] is not None]
    if not present:
        return not_run("none of the four models' files is present")
    out = run_family(res, ref, G1_MODELS, subsets)
    out["status"] = "RUN"
    out["analysis_of_record"] = "eligible_70"
    sizes = ref["source_sizes_70"]
    for m, b in out["eligible_70"]["models"].items():
        if b.get("status") == "RUN":
            b["level_check"] = level_check(list(b["d_c"].values()), sizes, np.sign(b["mean_d"]),
                                           draws=ref["level_draws"])
    rec, new = out["eligible_70"]["models"], out["new_58"]["models"]
    conf = {}
    for m in G1_MODELS:
        r70 = rec.get(m, {}).get("by_source", {}).get("T1", {})
        r58 = new.get(m, {}).get("by_source", {}).get("T1", {})
        conf[m] = {"T1_rejects_on_70": r70.get("rejects_at_5pct_after_holm"),
                   "T1_rejects_on_58": r58.get("rejects_at_5pct_after_holm"),
                   "effect_pct_70": rec.get(m, {}).get("effect_pct"), "effect_pct_58": new.get(m, {}).get("effect_pct")}
    out["confirmation_on_58"] = {"per_model": conf, "overlap": ref["overlap_58"],
                                 "rule": "The paper may say the effect is confirmed on configurations that did not "
                                         "suggest it only if T1 also rejects on the 58 (Holm over the four models)."}
    return out


def part_g2(res, ref, sel12):
    repro = {}
    for m in G2_MODELS:
        doc = res[m]["default"]
        repro[m] = (compare_to_published(doc, ref["pub"][m], ref["sn"], sel12) if doc is not None
                    else not_run("input file at the documented length absent"))
        repro[m]["published_rows"] = PUBLISHED_WHERE[m]
        repro[m]["board_gmean_scaled_on_12"] = ref["board_12"][m]
    present = [m for m in G2_MODELS if res[m]["default"] is not None or res[m]["512"] is not None]
    out = {"reproduction": repro}
    if not present:
        out.update(status="NOT RUN", reason="neither model's files are present")
        return out
    out["status"] = "RUN"
    fam = run_family(res, ref, G2_MODELS, {"first_12": sel12})["first_12"]
    for m, b in fam["models"].items():
        if b.get("status") == "RUN":
            b["level_check"] = level_check(list(b["d_c"].values()), ref["source_sizes_12"], np.sign(b["mean_d"]),
                                           draws=ref["level_draws"])
    out["tests_on_12"] = fam
    return out


# ---------------------------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------------------------
NOT_RECORDED = "not recorded"
META_KEYS = {
    "card": ("gpu_name", "gpu", "card", "device_name"),
    "torch_version": ("torch_version",),
    "cuda_version": ("cuda_version", "torch_cuda_version"),
    "hf_revision": ("model_revision", "hf_revision", "revision"),
    "matmul_precision": ("float32_matmul_precision", "matmul_precision"),
    "deterministic": ("deterministic_algorithms", "use_deterministic_algorithms", "deterministic"),
    "xformers_installed": ("xformers_installed",),
    "samples_per_batch_final": ("samples_per_batch_final", "samples_per_batch"),
    "toto_batch_size": ("toto_batch_size",),
    "evaluate_batch_size": ("evaluate_batch_size",),
    "chronos_batch_size_env": ("chronos_batch_size_env",),
    "history_used": ("context_length_used",),
    "run_grid_checksum": ("run_grid_sha256", "run_grid_md5", "run_grid_checksum"),
}


def _get(rec, keys):
    for k in keys:
        if isinstance(rec, dict) and rec.get(k) is not None:
            return rec[k]
    return NOT_RECORDED


def _counts(values):
    return dict(Counter(json.dumps(v) if not isinstance(v, str) else v for v in values))


def _parse_utc(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def run_records(root, variant, first_revisions, freeze_utc):
    """Summary of <root>/<variant>/ctx_*/per_config/*.run_meta.json, read defensively."""
    files = sorted(Path(root).glob(f"{variant}/ctx_*/per_config/*.run_meta.json"))
    if not files:
        return None
    recs, unreadable = [], []
    for p in files:
        try:
            with open(p) as f:
                j = json.load(f)
            if not isinstance(j, dict):
                raise ValueError("not an object")
            j["_file"] = str(p)
            j["_failed_file"] = p.name.endswith(".failed.run_meta.json")
            recs.append(j)
        except (OSError, ValueError) as exc:
            unreadable.append(f"{p}: {type(exc).__name__}")
    out = {"files": len(files), "unreadable": unreadable,
           "runs_ok": sum(1 for r in recs if r.get("status") == "ok" and not r["_failed_file"]),
           "runs_failed": sum(1 for r in recs if r["_failed_file"] or r.get("status") == "failed")}
    vals = {k: [_get(r, keys) for r in recs] for k, keys in META_KEYS.items()}
    vals["torch_version"] = [v if v != NOT_RECORDED else _get(r.get("package_versions") or {}, ("torch",))
                             for v, r in zip(vals["torch_version"], recs)]
    for k in ("card", "torch_version", "cuda_version", "hf_revision", "run_grid_checksum"):
        out[k] = _counts(vals[k])
    recorded = {v for v in vals["hf_revision"] if v != NOT_RECORDED}
    out["first_round_hf_revisions"] = sorted(first_revisions) if first_revisions is not None else None
    out["hf_revision_differs_from_first_round"] = (None if first_revisions is None else
                                                   bool(recorded - set(first_revisions)))
    if variant.startswith("toto"):
        for k in ("matmul_precision", "deterministic", "xformers_installed", "samples_per_batch_final",
                  "toto_batch_size", "history_used"):
            out[k] = _counts(vals[k])
        out["xformers_installed_flag"] = any(v is True or str(v).lower() == "true" for v in vals["xformers_installed"])
    else:
        out["history_used"] = _counts(vals["history_used"])
    out["evaluate_batch_size"] = _counts(vals["evaluate_batch_size"])
    if variant == "chronos_bolt_base":
        out["chronos_batch_size_env"] = _counts(vals["chronos_batch_size_env"])
    peaks = defaultdict(list)
    for r in recs:
        for k, v in r.items():
            if ("peak" in k.lower() or "max_memory" in k.lower()) and isinstance(v, (int, float)) and not isinstance(v, bool):
                peaks[k].append(v)
    out["peak_memory"] = {k: {"max": max(v), "runs": len(v)} for k, v in peaks.items()} or NOT_RECORDED
    walls = [r.get("wall_time_seconds") for r in recs]
    nums = [w for w in walls if isinstance(w, (int, float)) and not isinstance(w, bool)]
    out["wall_time_seconds_total"] = float(sum(nums)) if nums else NOT_RECORDED
    out["wall_time_hours_total"] = float(sum(nums)) / 3600 if nums else NOT_RECORDED
    out["runs_without_wall_time"] = len(walls) - len(nums)
    late = []
    if freeze_utc is not None:
        for r in recs:
            t = _parse_utc(r.get("finished_at"))
            if t is not None and t > freeze_utc:
                late.append({"file": r["_file"], "finished_at": r.get("finished_at")})
    out["runs_finished_after_freeze"] = late
    return out


def first_round_revisions(reader, model):
    revs = set()
    for L in LENGTHS:
        p = FIRST_ROUND / model / f"ctx_{L}" / "run_meta.json"
        if p.is_file():
            j = reader.json(p)
            for c in j.get("configs", []):
                if c.get("model_revision"):
                    revs.add(c["model_revision"])
    return revs


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def load_reference(reader):
    sel12 = sorted(x["config"] for x in reader.json(SELECTED_FILE)["selected"])
    med = reader.json(MEDIANS_FILE)
    elig = sorted(c for c, v in med.items() if v["median_length"] >= 1024)
    new58 = [c for c in elig if c not in sel12]
    assert len(sel12) == 12 and all(c in elig for c in sel12), "selection files are not as expected"
    assert len(elig) == 70 and len({base_of(c) for c in elig}) == 18, "eligible set is not 70 / 18"
    src70 = Counter(group_of(c, "source") for c in elig)
    assert sorted(src70.values(), reverse=True) == SOURCE_SIZES_70, f"source sizes on the 70: {src70}"
    src12 = Counter(group_of(c, "source") for c in sel12)
    assert len(src12) == N_SOURCES_12, f"sources on the 12: {src12}"
    bases12, dirs12 = {base_of(c) for c in sel12}, {med[c]["data_dir"] for c in sel12}
    overlap = (sum(base_of(c) in bases12 for c in new58), sum(med[c]["data_dir"] in dirs12 for c in new58))
    assert overlap == DESIGN_OVERLAP_58, f"overlap of the 58 with the 12: {overlap}"

    names, snap_digest, n_entries = displayed_top10(SNAP)
    if list(names) != list(TOP10_DOCUMENTED):
        raise SystemExit(f"top ten recomputed from the snapshot {names} differs from N11 {list(TOP10_DOCUMENTED)}")
    sn = reader.mase(SN_FILE, expect_model="Seasonal_Naive")
    folders = {}
    for root, dirs, files in os.walk(SNAP):
        if "all_results.csv" in files:
            with open(os.path.join(root, "all_results.csv"), newline="") as f:
                first = next(csv.DictReader(f), {})
            folders[first.get("model")] = Path(root)
    top10 = {name: reader.mase(folders[name] / "all_results.csv", expect_model=name) for name in names}
    pub = {k: reader.mase(p) for k, p in PUBLISHED.items()}
    first = {m: {L: reader.mase(FIRST_ROUND / m / f"ctx_{L}" / "all_results.csv") for L in LENGTHS}
             for m in ("chronos_bolt_base", "moirai2_small", "timesfm25", "toto")}
    for name, vals in [("Seasonal Naive", sn), *top10.items(), *pub.items()]:
        missing = [c for c in elig if vals is None or c not in vals]
        assert not missing, f"reference rows for {name} lack {missing[:3]}"
    for m in first:
        for L in LENGTHS:
            assert all(c in first[m][L] for c in sel12), f"first-round {m} ctx_{L} lacks configurations"
    sn_pin = reader.mase(SN_PIN_FILE, expect_model="Seasonal_Naive")
    sn_pin_diff = max(abs(sn_pin[c] / sn[c] - 1) for c in sel12)
    assert sn_pin_diff < 1e-12, "Seasonal Naive differs between the snapshot and the November pin"
    S_sets = {"eligible_70": elig, "new_58": new58, "first_12": sel12}
    S_ref = {k: spread_block(top10, sn, v) for k, v in S_sets.items()}
    freeze_sentence, freeze_utc = design_freeze(DESIGN_FILE)
    return {
        "sel12": sel12, "eligible70": elig, "new58": new58, "sn": sn, "top10": top10, "pub": pub, "first": first,
        "top10_names": names, "snapshot_digest": snap_digest, "snapshot_entries": n_entries,
        "source_sizes_70": [src70[k] for k in sorted(src70)], "source_sizes_12": [src12[k] for k in sorted(src12)],
        "sources_70": dict(sorted(src70.items())), "sources_12": dict(sorted(src12.items())),
        "overlap_58": {"from_a_base_dataset_among_the_12": overlap[0], "same_series_as_one_of_the_12": overlap[1],
                       "sentence": f"Of the 58, {overlap[0]} come from a base dataset that is already among the 12 "
                                   f"and {overlap[1]} use the same series as one of the 12 at another forecast horizon."},
        "toto_board_12": gmean_scaled(pub["toto"], sn, sel12),
        "toto_first_round_doc": first["toto"]["default"],
        "board_12": {m: gmean_scaled(pub[m], sn, sel12) for m in G2_MODELS},
        "S_ref": S_ref, "sn_pin_max_relative_difference_on_12": sn_pin_diff,
        "freeze_sentence": freeze_sentence, "freeze_utc": freeze_utc,
        "first_round_revisions": {m: first_round_revisions(reader, m)
                                  for m in ("chronos_bolt_base", "moirai2_small", "timesfm25", "toto")},
    }


def apply_m4_rule(res):
    """Toto variants: the 512 value of m4_daily/D/short is the documented value (d_c = 0), when the variant's
    512 file exists."""
    notes = {}
    for v in TOTO_VARIANTS:
        doc, l512 = res[v]["default"], res[v]["512"]
        if l512 is None:
            notes[v] = "not applied: the 512 results file is absent"
        elif doc is None or M4_RULE_CONFIG not in doc:
            if M4_RULE_CONFIG in l512:
                del l512[M4_RULE_CONFIG]
                notes[v] = "documented value missing; the 512 row present was removed (the configuration is missing)"
            else:
                notes[v] = "not applied: the documented value is missing (the configuration is missing)"
        else:
            had = M4_RULE_CONFIG in l512
            l512[M4_RULE_CONFIG] = doc[M4_RULE_CONFIG]
            notes[v] = "applied" + ("; a 512 row was present and was replaced" if had else "")
    return notes


def analyse(results_root, force=False, level_draws=LEVEL_DRAWS):
    results_root = Path(results_root).resolve()
    reader = Reader()
    ref = load_reference(reader)
    ref["root"] = str(results_root)
    ref["level_draws"] = level_draws
    sel12 = ref["sel12"]
    res = {v: {L: reader.mase(results_path(results_root, v, L)) for L in LENGTHS} for v in ALL_VARIANTS}
    m4_notes = apply_m4_rule(res)
    first3 = {m: ref["first"][m] for m in DETERMINISTIC}

    out = {
        "written_by": "camera_ready/gpu_round/analyze_gpu_round.py",
        "design": {"file": str(DESIGN_FILE), "md5": md5(DESIGN_FILE) if DESIGN_FILE.is_file() else None,
                   "freeze_sentence": ref["freeze_sentence"],
                   "freeze_utc": ref["freeze_utc"].strftime("%Y-%m-%dT%H:%M:%SZ") if ref["freeze_utc"] else None},
        "results_root": str(results_root),
        "result_files": {v: {LENGTH_WORD[L]: {"path": str(results_path(results_root, v, L)),
                                              "present": res[v][L] is not None,
                                              "configurations": None if res[v][L] is None else len(res[v][L])}
                             for L in LENGTHS} for v in ALL_VARIANTS},
        "m4_daily_rule": m4_notes,
        "definitions": {
            "configuration": "one GIFT-Eval dataset/frequency/term; its base dataset is the part before the first slash",
            "source": "bizitobs_application, bizitobs_l2c, bizitobs_service -> bizitobs; ett1, ett2 -> ett; "
                      "bitbrains_fast_storage, bitbrains_rnd -> bitbrains; every other base dataset is its own source",
            "score": MASE_COL,
            "d_c": "ln(MASE at 512 / MASE at the documented length), configuration c",
            "effect_pct": "100 * (exp(mean of d_c) - 1)",
            "gmean_scaled": "geometric mean over configurations of MASE / Seasonal Naive MASE (2026-09-07 snapshot)",
            "S": "for each of the displayed top ten of the 2026-09-07 snapshot, gmean_scaled on the model's "
                 "configurations; S = largest / smallest - 1",
        },
        "rules_applied": RULES,
        "reference": {
            "selected_12": sel12, "eligible_70_count": len(ref["eligible70"]),
            "eligible_70_base_datasets": len({base_of(c) for c in ref["eligible70"]}),
            "new_58_count": len(ref["new58"]),
            "sources_70": ref["sources_70"], "sources_12": ref["sources_12"],
            "overlap_58": ref["overlap_58"],
            "top10": ref["top10_names"], "top10_equals_N11": True,
            "snapshot_files_digest": ref["snapshot_digest"], "snapshot_entries": ref["snapshot_entries"],
            "S": {k: {"pct": v["pct"], "equals_design_to_2_decimals": round(v["pct"], 2) == DESIGN_S_PCT[k],
                      "loo_smallest": v.get("loo_smallest"), "loo_largest": v.get("loo_largest"),
                      "top10_gmeans": v["top10_gmean_scaled_mase"]} for k, v in ref["S_ref"].items()},
            "S_70_loo_equals_design": (round(ref["S_ref"]["eligible_70"]["loo_smallest"]["pct"], 2),
                                       round(ref["S_ref"]["eligible_70"]["loo_largest"]["pct"], 2)) == DESIGN_S_LOO_70,
            "toto_board_gmean_on_12": ref["toto_board_12"],
            "timesfm3_board_gmean_on_12": ref["board_12"]["timesfm3"],
            "timestral209m_board_gmean_on_12": ref["board_12"]["timestral209m"],
            "seasonal_naive_pin_vs_snapshot_max_relative_difference_on_12": ref["sn_pin_max_relative_difference_on_12"],
        },
    }
    mc = machine_check(res, first3, sel12)
    out["machine_check"] = mc
    v = mc.get("verdict")
    hold = v == "STOP" and not force
    out["machine_check_gate"] = (
        "The machine check says STOP. DESIGN_ROUND3.md: stop and check versions, weights and data first; nothing "
        "else is analysed." if hold else
        "The machine check says STOP; the other parts were computed because --analyse-despite-machine-check was "
        "given, and they do not count." if v == "STOP" else
        "The machine check passed." if v == "PASS" else
        "The machine check says CONTINUE: the other parts are computed. DESIGN_ROUND3.md asks for one "
        "configuration to be run twice on one card to show that repeat runs agree." if v == "CONTINUE" else
        "The machine check was not run; any part below that is computed counts only after it passes "
        "(DESIGN_ROUND3.md).")
    if hold:
        held = not_run("held: the machine check says STOP (DESIGN_ROUND3.md: stop and check versions, weights and "
                       "data first); rerun with --analyse-despite-machine-check to compute it anyway")
        out["G3"], out["G4"], out["G1"], out["G2"] = dict(held), dict(held), dict(held), dict(held)
    else:
        out["G3"] = part_g3(res, ref, sel12)
        out["G4"] = part_g4(res, ref, sel12)
        out["G1"] = part_g1(res, ref, {"eligible_70": ref["eligible70"], "new_58": ref["new58"],
                                       "first_12": sel12})
        out["G2"] = part_g2(res, ref, sel12)
    recs = {}
    for var in ALL_VARIANTS:
        fr = FIRST_ROUND_MODEL.get(var)
        r = run_records(results_root, var, ref["first_round_revisions"][fr] if fr else None, ref["freeze_utc"])
        if r is not None:
            recs[var] = r
    out["run_records"] = {"status": "RUN" if recs else "NOT RUN", "variants": recs}
    if not recs:
        out["run_records"]["reason"] = "no per_config/*.run_meta.json file is present"
    out["files_read"] = reader.files
    return out


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
def f_pct(x, nd=1, sign=True):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}%" if sign else f"{x:.{nd}f}%"


def f_p(p):
    if p is None:
        return "n/a"
    return f"{p:.4f}" if p >= 1e-3 else f"{p:.2e}"


def f_g(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def f_iv(iv):
    if not iv or "pct_low" not in iv:
        return "not run"
    s = f"{f_pct(iv['pct_low'])} to {f_pct(iv['pct_high'])}"
    return s + (f" (df {iv['df']:.1f})" if "df" in iv else "")


def f_loo(loo, key):
    return f"{f_pct(loo[key]['effect_pct'])} (without {loo[key]['left_out']})" if key in loo else "not run"


def f_level(lc, mode):
    if not lc:
        return "only on the 70"
    r = lc[mode]
    return f"{r['two_sided_pct']:.2f}%; {f_pct(r['in_direction_of_observed_effect_pct'], 2, sign=False)}"


def f_S(S):
    s = f_pct(S["pct"], 2, sign=False)
    if "loo_smallest" in S:
        s += f" ({S['loo_smallest']['pct']:.2f}% to {S['loo_largest']['pct']:.2f}%)"
    return s


def f_t2(t2):
    if not t2.get("computed"):
        return ["not computed (" + t2.get("reason", "") + "; enters Holm with p = 1)", f_p(t2.get("p_holm")), "no"]
    return [f_p(t2["p_one_sided"]), f_p(t2["p_holm"]), "yes" if t2["rejects_at_2_5pct_after_holm"] else "no"]


def md_compare(name, x, published_where):
    """Lines for one reproduction comparison (G3 reading (a) / G2)."""
    if x["status"] != "RUN":
        return [f"{name}: not run ({x['reason']}).", ""]
    lines = [f"{name}: geometric mean of scaled MASE {f_g(x['gmean_scaled_ours'])} on "
             f"{len(x['configurations_used'])} configurations, against {f_g(x['gmean_scaled_published_same_configs'])} "
             f"from the published rows ({published_where}) on the same configurations; relative gap "
             f"{f_pct(x['relative_gap_pct'], 2)}; within 1%: {'yes' if x['within_1pct'] else 'no'}; "
             f"verdict {x['verdict']}."]
    if x["missing"]:
        lines.append(f"Missing configurations: {', '.join(x['missing'])}.")
    lines += ["", table(["Configuration", "Our MASE", "Published MASE", "Relative difference"],
                        [[r["config"], f_g(r["ours"]), f_g(r["published"]), f_pct(100 * r["relative_difference"], 2)]
                         for r in x["per_configuration"]]), ""]
    return lines


def md_family(fam, models, family_word, note):
    lines, rows1, rows2, rows3, dropped, disagree = [], [], [], [], [], []
    for m in models:
        b = fam["models"].get(m, {})
        if b.get("status") != "RUN":
            rows1.append([LABEL[m], "not run: " + b.get("reason", "")] + [""] * 12)
            if b.get("dropped"):
                dropped.append((m, b["dropped"]))
            continue
        src, bd = b["by_source"], b["by_base_dataset"]
        t1 = src["T1"]
        lc = b.get("level_check")
        rows1.append([LABEL[m], f"{b['n_configurations']} of {b['configurations_in_set']}", b["n_sources"],
                      f_pct(b["effect_pct"]), f_iv(src["interval_signflip_95"]),
                      f"{f_p(t1['p_two_sided'])} ({t1['sign_patterns_counted']} of {t1['n_sign_patterns']:,})",
                      f_p(t1["p_holm"]), "yes" if t1["rejects_at_5pct_after_holm"] else "no",
                      f_level(lc, "shared"), f_level(lc, "independent"), f_S(b["S"])] + f_t2(src["T2"]))
        rows2.append([LABEL[m], f_p(src["signed_rank"].get("p_two_sided")), f_iv(src["interval_cr2_95"]),
                      f_loo(b["leave_one_out"]["source"], "smallest"), f_loo(b["leave_one_out"]["source"], "largest"),
                      f_loo(b["leave_one_out"]["base_dataset"], "smallest"),
                      f_loo(b["leave_one_out"]["base_dataset"], "largest")])
        bt1 = bd["T1"]
        rows3.append([LABEL[m], b["n_base_datasets"], f_p(bt1["p_two_sided"]), f_p(bt1["p_holm"]),
                      "yes" if bt1["rejects_at_5pct_after_holm"] else "no"] + f_t2(bd["T2"]) +
                     [f_iv(bd["interval_signflip_95"]), f_iv(bd["interval_cr2_95"]),
                      f_p(bd["signed_rank"].get("p_two_sided"))])
        sr = src["signed_rank"].get("p_two_sided")
        if sr is not None and (t1["p_two_sided"] <= ALPHA_T1) != (sr <= ALPHA_T1):
            disagree.append(f"{LABEL[m]}: T1 (p = {f_p(t1['p_two_sided'])}) and the signed-rank test on source "
                            f"means (p = {f_p(sr)}) disagree at 5%. The text follows T1, because it tests the "
                            "board's own statistic, and says that the test of the typical source does not agree.")
        if b["dropped"]:
            dropped.append((m, b["dropped"]))
    lines += ["Tests of record, on sources:", "",
              table(["Model", "Configurations used", "Sources", "Effect", "95% interval (inverted sign-flip test)",
                     "T1 p, two-sided (sign patterns)", f"T1 p after Holm over {family_word}", "T1 rejects at 5%",
                     "Level check, one value per source: false claims, two-sided; in the effect's direction",
                     "Level check, one value per configuration: same", "S (leaving one base dataset out)",
                     "T2 p, one-sided", f"T2 p after Holm over {family_word}", "T2 rejects at 2.5%"], rows1), ""]
    if disagree:
        lines += disagree + [""]
    if rows2:
        lines += ["Reported beside the tests, on sources, deciding nothing:", "",
                  table(["Model", "Signed-rank sign-flip p, source means", "95% interval, CR2 with Satterthwaite df",
                         "Leave one source out: smallest", "largest", "Leave one base dataset out: smallest",
                         "largest"], rows2), "",
                  "The same tests on base datasets, not used for any claim:", "",
                  table(["Model", "Base datasets", "T1 p", "T1 p after Holm", "T1 rejects at 5%", "T2 p",
                         "T2 p after Holm", "T2 rejects at 2.5%", "95% interval, inverted sign-flip test",
                         "95% interval, CR2", "Signed-rank p, base-dataset means"], rows3), ""]
    if note:
        lines += [note, ""]
    if dropped:
        lines.append("Configurations dropped because a length is missing (each only for the models named):")
        same = {}
        for m, dr in dropped:
            same.setdefault(tuple((x["config"], tuple(x["missing_at"])) for x in dr), []).append(m)
        lines += [f"- {', '.join(LABEL[m] for m in ms)} ({len(key)} configurations): "
                  + "; ".join(f"{c} (missing at {' and '.join(at)})" for c, at in key) for key, ms in same.items()]
    else:
        lines.append("No configuration was dropped.")
    lines.append("")
    cc = fam.get("common_configurations")
    if cc:
        if cc["status"] != "RUN":
            lines += [f"Every model on the configurations all of them finished: not run ({cc['reason']}).", ""]
        elif cc["same_as_every_model_used"]:
            lines += [f"Every model finished the same {cc['n_configurations']} configurations, so the tables above "
                      "are already on the configurations all of them finished.", ""]
        else:
            rows = []
            for m in models:
                b = cc["models"][m]
                t1 = b["by_source"]["T1"]
                rows.append([LABEL[m], b["n_configurations"], b["n_sources"], f_pct(b["effect_pct"]),
                             f_iv(b["by_source"]["interval_signflip_95"]), f_p(t1["p_two_sided"]), f_p(t1["p_holm"]),
                             f_S(b["S"])] + f_t2(b["by_source"]["T2"]))
            lines += [f"Every model on the {cc['n_configurations']} configurations all of them finished (sources):", "",
                      table(["Model", "Configurations", "Sources", "Effect", "95% interval (inverted sign-flip test)",
                             "T1 p", "T1 p after Holm", "S", "T2 p", "T2 p after Holm", "T2 rejects at 2.5%"], rows), ""]
    return lines


def to_markdown(o):
    d = o["design"]
    L = ["# GPU round: results of the checks fixed in DESIGN_ROUND3.md", ""]
    L += [f"Freeze, from DESIGN_ROUND3.md: {d['freeze_sentence']}" if d["freeze_sentence"] else
          "Freeze: the freeze line was not found in DESIGN_ROUND3.md.", ""]
    L += [f"Computed by `analyze_gpu_round.py` from the results folder `{o['results_root']}`. Every number here is "
          "also in GPU_ROUND_RESULTS.json, next to the files it was computed from (path and md5 under `files_read`).",
          "",
          "The effect of a model is computed from d_c = ln(MASE at 512 / MASE at the documented length) for each "
          "configuration c: it is exp(mean of d_c) - 1, in percent. A positive effect means the model's error is "
          "higher when it is given only the last 512 steps than with its documented history length. Scaled MASE "
          "is MASE divided by Seasonal Naive's MASE from the 2026-09-07 snapshot. The groups of every test of "
          "record are sources: the three bizitobs datasets are one source, the two ETT datasets one, the two "
          "bitbrains datasets one, and every other base dataset is its own source (14 sources on the 70 "
          "configurations, 9 on the 12).", ""]
    rf = o["reference"]
    S = rf["S"]
    L += [f"Reference values computed from the board before any result: S, the spread of the displayed top ten, is "
          f"{S['eligible_70']['pct']:.2f}% on the 70 configurations, {S['new_58']['pct']:.2f}% on the 58 and "
          f"{S['first_12']['pct']:.2f}% on the 12 (the design wrote down 11.76%, 12.71% and 9.34%: "
          f"{'all match' if all(x['equals_design_to_2_decimals'] for x in S.values()) else 'DO NOT ALL MATCH'}). "
          f"Leaving one base dataset out of the 70 moves S between {S['eligible_70']['loo_smallest']['pct']:.2f}% "
          f"(without {S['eligible_70']['loo_smallest']['left_out']}) and {S['eligible_70']['loo_largest']['pct']:.2f}% "
          f"(without {S['eligible_70']['loo_largest']['left_out']}).", ""]

    # Machine check
    mc = o["machine_check"]
    L += ["## Check of the new machine", ""]
    if mc["status"] != "RUN":
        L += [f"Not run: {mc['reason']}.", ""]
    else:
        la = mc["largest_at"]
        L += [f"Outcome {mc['verdict']}: {mc['values_compared']} of {mc['values_expected']} MASE values compared "
              f"with the first round, {mc['values_missing']} missing; largest relative difference "
              f"{mc['largest_relative_difference']:.2e} ({LABEL[la['model']]}, {la['length']} length, {la['config']}); "
              f"{mc['count_at_or_above_1e-4']} at or above 1e-4, {mc['count_above_1e-3']} above 1e-3; "
              f"{len(mc['d_c_sign_changes'])} d_c with a different sign than in the first round. Below 1e-4 is "
              "PASS, from 1e-4 to 1e-3 is CONTINUE, above 1e-3 or any change of sign is STOP.", "",
              table(["Model", "Length", "Values compared", "Missing", "Largest relative difference",
                     "At or above 1e-4", "Above 1e-3"],
                    [[LABEL[r["model"]], r["length"], r["n_compared"], len(r["missing"]),
                      "file absent" if not r["file_present"] else
                      ("n/a" if r["largest_relative_difference"] is None else f"{r['largest_relative_difference']:.2e}"),
                      r.get("count_at_or_above_1e-4", ""), r.get("count_above_1e-3", "")] for r in mc["rows"]]), ""]
        bad = [(r["model"], r["length"], x) for r in mc["rows"] for x in r.get("per_configuration", [])
               if x["relative_difference"] >= MC_PASS_BELOW]
        if bad:
            L += ["Values that differ by 1e-4 or more:", "",
                  table(["Model", "Length", "Configuration", "New MASE", "First-round MASE", "Relative difference"],
                        [[LABEL[m], ln, x["config"], f_g(x["new"], 6), f_g(x["first_round"], 6),
                          f"{x['relative_difference']:.2e}"] for m, ln, x in bad]), ""]
        if mc["d_c_sign_changes"]:
            L += ["d_c with a different sign than in the first round:", "",
                  table(["Model", "Configuration", "d_c now", "d_c in the first round"],
                        [[LABEL[x["model"]], x["config"], f"{x['d_new']:+.5f}", f"{x['d_first_round']:+.5f}"]
                         for x in mc["d_c_sign_changes"]]), ""]
        for r in mc["rows"]:
            if r["missing"] and r["file_present"]:
                L.append(f"Missing for {LABEL[r['model']]} at the {r['length']} length: {', '.join(r['missing'])}.")
        if not mc["complete"]:
            L += ["", "The outcome above is on the values present."]
        L.append("")
        if mc["verdict"] == "CONTINUE":
            L += [f"Request (DESIGN_ROUND3.md): run one configuration twice on one card to show that repeat runs "
                  f"agree, for example {la['config']} with {LABEL[la['model']]} at the {la['length']} length, where "
                  "the difference is largest.", ""]
    L += [o["machine_check_gate"], ""]

    # G3
    g3 = o["G3"]
    L += ["## G3. Toto's reproduction gap", ""]
    if "reference" in g3:
        r3 = g3["reference"]
        L += [f"The board's value, the geometric mean of scaled MASE of the published Toto rows "
              f"(Toto_Open_Base_1.0) on the 12 configurations, is {r3['board_gmean_scaled_on_12']:.4f} "
              f"(the paper prints 0.7699: {'matches' if r3['board_value_equals_paper_0.7699_to_4_decimals'] else 'DOES NOT MATCH'}). "
              "The gap is read two ways at the documented length: (a) within 1% of the board's value; (b) within "
              "noise, meaning the board's value lies inside the range of the three seeds' values widened by half "
              "that range on each side.", ""]
    if g3["status"] != "RUN":
        L += [f"Not run: {g3['reason']}.", ""]
    else:
        L += ["### Reading (a): within 1%", ""]
        for v, x in g3["reading_a_within_1pct"].items():
            L += md_compare(LABEL[v], x, PUBLISHED_WHERE["toto"])
        b = g3["reading_b_within_noise"]
        L += ["### Reading (b): within noise", ""]
        if b["status"] != "RUN":
            L += [f"Not run: {b['reason']}.", ""]
        else:
            g = b["gmean_scaled_by_seed"]
            L += [f"Seeds 42, 43, 44 on {len(b['configurations_used'])} configurations: "
                  + ", ".join(f"{g[s]:.4f}" for s in ("42", "43", "44"))
                  + f"; range {b['range'][0]:.4f} to {b['range'][1]:.4f}, widened {b['widened_range'][0]:.4f} to "
                  f"{b['widened_range'][1]:.4f}; the board's value on the same configurations is "
                  f"{b['board_gmean_scaled_same_configs']:.4f}, {'inside' if b['inside'] else 'outside'}; verdict "
                  f"{b['verdict']}." + (f" Missing: {', '.join(b['missing'])}." if b["missing"] else ""), ""]
        a = g3["reading_a_within_1pct"][TOTO_A]
        if a["status"] == "RUN" and a["verdict"] == "PASS":
            L += ["Setting A meets the 1% criterion. The statement allowed by DESIGN_ROUND3.md is \"the notebook's "
                  "setting reproduces the board\"; that the sample count is the cause rests on the other changes "
                  "being unable to move the expected score.", ""]
        elif a["status"] == "RUN" and a["verdict"] == "FAIL" and b["status"] == "RUN" and b["verdict"] == "FAIL":
            L += ["Setting A misses both readings. DESIGN_ROUND3.md then runs one variant on the 12 configurations: "
                  "setting A with the faster matrix products off. Nothing else.", ""]
        L += ["Setting A is Toto's setting for G1 and G4 whatever this check shows (DESIGN_ROUND3.md).", ""]
    if "reference" in g3:
        L += md_compare("The first round's point setting (one point forecast, no sample count; the paper prints 0.8126)",
                        g3["reference"]["first_round_point_setting"], PUBLISHED_WHERE["toto"])

    # G4
    g4 = o["G4"]
    L += ["## G4. Repeat runs of Toto, setting A", ""]
    if g4["status"] != "RUN":
        L += [f"Not run: {g4['reason']}.", ""]
    else:
        rows = []
        for seed, x in g4["seeds"].items():
            if x["status"] != "RUN":
                rows.append([seed, "not run", "", "", "", ""])
            else:
                rows.append([seed, f"{x['configurations_documented']} / {x['configurations_512']} / "
                                   f"{x['configurations_both_lengths']}",
                             f_g(x["gmean_scaled_documented"]), f_g(x["gmean_scaled_512"]), f_pct(x["effect_pct"]),
                             "; ".join(dd["config"] for dd in x["dropped"]) or "none"])
        L += [table(["Seed", "Configurations (documented / 512 / both)", "Geometric mean of scaled MASE, documented",
                     "Geometric mean of scaled MASE, 512", "Effect", "Dropped"], rows), ""]
        rg = g4["range_over_seeds"]

        def rtxt(k, f):
            r = rg[k]
            return "n/a (fewer than two seeds)" if r["range"] is None else f"{f(r['min'])} to {f(r['max'])}"
        L += [f"Range over seeds {', '.join(rg['effect_pct']['seeds_with_value'])}: effect {rtxt('effect_pct', f_pct)}"
              + (f" ({rg['effect_pct']['range']:.2f} percentage points)" if rg['effect_pct']['range'] is not None else "")
              + f"; geometric mean at the documented length {rtxt('gmean_scaled_documented', f_g)}; at 512 "
              f"{rtxt('gmean_scaled_512', f_g)}. Seed 42 is the value of record. For m4_daily/D/short the 512 value "
              "is the documented value by rule (d_c = 0)."
              + ("" if g4["same_configurations_for_every_seed"] else
                 " The seeds do not all have the same configurations; see the dropped column."), ""]

    # G1
    g1 = o["G1"]
    L += ["## G1. The history-length test on every eligible configuration", ""]
    if g1["status"] != "RUN":
        L += [f"Not run: {g1['reason']}.", ""]
    else:
        lc = next((b.get("level_check") for b in g1["eligible_70"]["models"].values() if b.get("level_check")), None)
        L += ["The 70 configurations are the estimate and the test of record; the 58 and the 12 are reported beside "
              "them. T1 is the exact two-sided sign-flip test of a zero effect on source totals of d_c, Holm over the "
              "four models, nominal level 5%. T2 is the exact one-sided sign-flip test of the null \"mean of d_c is "
              "at most ln(1 + S)\" on the shifted source totals s_g - n_g x ln(1 + S), Holm over the four models, "
              "nominal level 2.5%; a model that did not pass T1 after Holm with a positive effect enters T2's Holm "
              "correction with p = 1. The interval of record is the inverted sign-flip interval on sources. S is the "
              "spread of the board's displayed top ten on the configurations the model used. A model with no result "
              "enters each Holm correction with p = 1.",
              "",
              "The level check repeats the review's simulation on each model's own d_c on the 70, centred to mean "
              "zero, drawn into the 14-source layout of the 70"
              + (f" ({lc['draws']:,} draws, seed {lc['seed']}" + ("" if lc["draws"] == LEVEL_DRAWS else
                 f"; the design fixes {LEVEL_DRAWS:,}") + ")" if lc else "")
              + ": the share of draws in which T1 claims an effect although there is none, in either direction and "
              "in the direction of the model's observed effect. The paper states that rate with the p-value.", ""]
        titles = {"eligible_70": "All 70 eligible configurations (estimate and test of record)",
                  "new_58": "The 58 configurations not among the first round's 12 (reported beside)",
                  "first_12": "The first round's 12 configurations (reported beside)"}
        for key in ("eligible_70", "new_58", "first_12"):
            fam = g1[key]
            L += [f"### {titles[key]}", ""]
            if fam["status"] != "RUN":
                L += [f"Not run: {fam['reason']}.", ""]
                continue
            L += md_family(fam, G1_MODELS, "the four models", None)
            if key == "new_58":
                cf = g1["confirmation_on_58"]
                for m in G1_MODELS:
                    x = cf["per_model"][m]
                    if x["T1_rejects_on_58"] is None:
                        L.append(f"- {LABEL[m]}: no result on the 58.")
                    else:
                        L.append(f"- {LABEL[m]}: T1 on sources {'also rejects' if x['T1_rejects_on_58'] else 'does not reject'} "
                                 f"on the 58 after Holm over the four models (effect {f_pct(x['effect_pct_58'])} on the 58, "
                                 f"{f_pct(x['effect_pct_70'])} on the 70).")
                L += ["", cf["overlap"]["sentence"] + " " + cf["rule"], ""]
        L += ["Limits stated with the result (DESIGN_ROUND3.md): S is used as a fixed yardstick although it is itself "
              "computed from data. Fourteen sources is few: the smallest two-sided p-value possible is 2 / 2^14, and a "
              "single large source can still carry a sum; the leave-one-out ranges are there for that reason. The "
              "sign-flip test is exact only when, with no true effect, a source's total is as likely to be positive "
              "as negative; the level check above shows how far that holds for each model.", ""]

    # G2
    g2 = o["G2"]
    L += ["## G2. Two newer models on the 12 first-round configurations", ""]
    if "reproduction" in g2:
        for m in G2_MODELS:
            x = g2["reproduction"][m]
            L.append(f"The board's value for {LABEL[m]} on the 12 configurations, from the published rows of "
                     f"{x['published_rows']}, is {x['board_gmean_scaled_on_12']:.4f}.")
        L.append("")
    if g2["status"] != "RUN":
        L += [f"Not run: {g2['reason']}.", ""]
    else:
        for m in G2_MODELS:
            x = g2["reproduction"][m]
            L += md_compare(f"Reproduction check, {LABEL[m]}", x, x["published_rows"])
            if x["status"] == "RUN" and x["verdict"] == "FAIL":
                L += [f"{LABEL[m]} misses the 1% criterion; its effect below is reported with that miss.", ""]
        fam = g2["tests_on_12"]
        if fam["status"] != "RUN":
            L += [f"Tests not run: {fam['reason']}.", ""]
        else:
            L += md_family(fam, G2_MODELS, "the two models",
                           "T2 uses S computed on the configurations the model used; on all 12 it is "
                           f"{S['first_12']['pct']:.2f}% (the paper's 9.3%). The level check uses the 9-source "
                           "layout of the 12. Twelve configurations from nine sources give wide intervals.")

    # Run records
    rr = o["run_records"]
    L += ["## Run records", ""]
    if rr["status"] != "RUN":
        L += [f"Not run: {rr['reason']}.", ""]
    else:
        rows = []
        for v, x in rr["variants"].items():
            def keys(d):
                return ", ".join(f"{k} ({n})" for k, n in d.items()) if isinstance(d, dict) else str(d)
            rev = keys(x["hf_revision"])
            if x["hf_revision_differs_from_first_round"]:
                rev += " DIFFERS FROM THE FIRST ROUND"
            pk = ("; ".join(f"{k} {v_['max']}" for k, v_ in x["peak_memory"].items())
                  if isinstance(x["peak_memory"], dict) else x["peak_memory"])
            wall = x["wall_time_hours_total"]
            rows.append([LABEL.get(v, v), f"{x['runs_ok']} ok, {x['runs_failed']} failed", keys(x["card"]),
                         keys(x["torch_version"]), keys(x["cuda_version"]), rev, keys(x["evaluate_batch_size"]), pk,
                         "not recorded" if isinstance(wall, str) else f"{wall:.2f} h"])
        L += [table(["Variant", "Runs", "Cards", "torch", "CUDA", "Hugging Face revision",
                     "Batch size passed to evaluate_model", "Peak memory (largest)", "Wall time, total"], rows), ""]
        trows = []
        for v, x in rr["variants"].items():
            if v.startswith("toto"):
                def keys(d):
                    return ", ".join(f"{k} ({n})" for k, n in d.items())
                trows.append([LABEL.get(v, v), keys(x["matmul_precision"]), keys(x["deterministic"]),
                              keys(x["xformers_installed"]) + (" FLAG: must not be installed"
                                                               if x["xformers_installed_flag"] else ""),
                              keys(x["samples_per_batch_final"]), keys(x["toto_batch_size"])])
        if trows:
            L += [table(["Toto variant", "Matmul precision", "Deterministic", "xFormers installed",
                         "Samples per batch after retries", "Batch size"], trows), ""]
        late = [(v, r) for v, x in rr["variants"].items() for r in x["runs_finished_after_freeze"]]
        if late:
            L += [f"Runs whose record says they finished after the freeze ({len(late)}):", ""]
            L += [f"- {LABEL.get(v, v)}: `{r['file']}` finished {r['finished_at']}" for v, r in late] + [""]
        bad = [(v, u) for v, x in rr["variants"].items() for u in x["unreadable"]]
        if bad:
            L += ["Run records that could not be read:", ""] + [f"- `{u}`" for v, u in bad] + [""]

    L += ["## How missing or odd values were handled", ""] + [f"- {r}" for r in o["rules_applied"]] + [""]
    L += ["- m4_daily/D/short rule, per Toto variant: "
          + "; ".join(f"{k}: {v}" for k, v in o["m4_daily_rule"].items()) + ".", ""]
    root = o["results_root"]
    probs = [(p, x["problems"]) for p, x in o["files_read"].items() if x.get("problems") and p.startswith(root)]
    if probs:
        L += ["Problems found in the result files:", ""]
        for p, ps in probs:
            L += [f"- `{p}`: " + "; ".join(ps)]
        L.append("")
    return "\n".join(L) + "\n"


def clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None, sets to sorted lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, set):
        return sorted(clean(v) for v in o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="results root: <root>/<variant>/ctx_<default|512>/all_results.csv")
    ap.add_argument("--out", required=True, help="folder for GPU_ROUND_RESULTS.json and GPU_ROUND_RESULTS.md")
    ap.add_argument("--analyse-despite-machine-check", action="store_true",
                    help="compute G1-G4 even when the machine check says STOP (they are labelled as such)")
    ap.add_argument("--level-draws", type=int, default=LEVEL_DRAWS,
                    help=f"draws of the level check (the design fixes {LEVEL_DRAWS}; fewer only for quick tests)")
    a = ap.parse_args(argv)
    out = clean(analyse(a.results, force=a.analyse_despite_machine_check, level_draws=a.level_draws))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "GPU_ROUND_RESULTS.json", "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    with open(od / "GPU_ROUND_RESULTS.md", "w") as f:
        f.write(to_markdown(out))
    status = {k: out[k].get("verdict", out[k]["status"]) for k in ("machine_check", "G3", "G4", "G1", "G2", "run_records")}
    print(json.dumps(status))
    return out


if __name__ == "__main__":
    main()
