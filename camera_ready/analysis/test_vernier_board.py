"""Regression test of vernier_board.py on the submitted paper's GIFT-Eval snapshot and git history."""
import os, csv, json, math, subprocess, io, time, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
import vernier_board as vb

SNAP = "fmts/snapshot_2026-09-07/results"
REPO = "fmts/poolcheck/gift"
PIN = "b61c48aa3197a1a111e405c0e6f600128a1efee3"
MASE = "eval_metrics/MASE[0.5]"
fails = []


def check(label, got, want, tol=None, rng_ok=None):
    ok = (got in rng_ok) if rng_ok is not None else (abs(got - want) <= tol if tol is not None else got == want)
    print(("OK   " if ok else "FAIL ") + f"{label}: got {got} expected {want}" + (f" (tol {tol})" if tol else ""))
    if not ok:
        fails.append(label)


# ---- load the snapshot as resolution_budget.py / pool_perturbations.py do
rows, cfg = defaultdict(dict), {}
for root, _, files in os.walk(SNAP):
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
names = sorted(rows)
V = np.array([[rows[m].get(c, np.nan) for c in configs] for m in names])
leak = np.array([cfg.get(m, {}).get("testdata_leakage", "No") == "Yes" for m in names])
disp = [i for i, m in enumerate(names) if m in cfg and not np.isnan(V[i]).any()]
bases = sorted({c.split("/")[0] for c in configs})
cl = np.array([bases.index(c.split("/")[0]) for c in configs])
print(f"{len(names)} entries, {len(disp)} displayable, {len(configs)} configs, {len(bases)} sources")

avg, ordn = vb.mean_rank_rule("average"), vb.mean_rank_rule("ordinal")


def build(rule):
    R = np.full(V.shape, np.nan)
    for j in range(V.shape[1]):
        ok = ~np.isnan(V[:, j])
        R[ok, j] = rankdata(V[ok, j], method=rule)
    Xd = R[disp]
    order = list(np.lexsort((Xd.mean(1), leak[disp])))
    return Xd, order, [names[i] for i in disp]


# ---- report_from_matrix, both tie conventions (control only for the paper's one)
Xa, oa, nd = build("average")
t0 = time.time()
rep = vb.report_from_matrix(Xa, cl, nd, oa)
rt = time.time() - t0
print(f"report_from_matrix runtime (with known-shift control): {rt:.0f}s")
c, rs = rep["counts"], rep["resolution"]
print("top10:", rep["top10"])
check("adjacent separated, paper rule", c["paper"]["adjacent"], 0)
check("all pairs separated, paper rule (11-13)", c["paper"]["all"], 12, rng_ok=range(11, 14))
check("all pairs after Holm (1-3)", c["paper"]["all_holm"], 2, rng_ok=range(1, 4))
check("paper resolution", rs["paper"], 6.16, tol=0.10)
check("median adjacent gap", rs["median_adjacent_gap"], 0.9072, tol=0.001)
check("sources needed (boot)", rs["sources_needed_boot"], 1269, tol=127)
check("leading tier size, paper rule", rep["leading_tier"]["paper"]["size"], 3)
check("leading tier members", rep["leading_tier"]["paper"]["members"],
      ["STRIDE w/ Synapse", "EXAONE-Forecast-Agent", "LS-MoE"])

Xo, oo, _ = build("ordinal")
rep_o = vb.report_from_matrix(Xo, cl, nd, oo, n_control=0)
print(f"[info] first-come (ordinal) ties on same matrix: resolution {rep_o['resolution']['paper']:.4f}, median gap "
      f"{rep_o['resolution']['median_adjacent_gap']:.4f}, paper adj/all/holm {rep_o['counts']['paper']['adjacent']}/"
      f"{rep_o['counts']['paper']['all']}/{rep_o['counts']['paper']['all_holm']}, tier {rep_o['leading_tier']['paper']['size']}")

# ---- pool tests
r = vb.pool_tests(V, avg, disp, tier=[(1 if leak[i] else 0) if i in set(disp) else None for i in range(len(names))])
check("removal reorders top ten", (r["removal"]["changed"], r["removal"]["of"]), (28, 121))
check("near-copy reorders top ten", (r["near_copy"]["changed"], r["near_copy"]["of"]), (42, 128))
print("   near-copy entries:", [names[i] for i in r["near_copy"]["entries"]][:12], "...")

# ---- history replay from git
git = lambda *a: subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True).stdout
cache = {}


def load_csv(bid):
    if bid not in cache:
        out = defaultdict(dict)
        for r in csv.DictReader(io.StringIO(git("cat-file", "-p", bid))):
            try:
                out[r["model"]][r["dataset"]] = float(r[MASE])
            except (KeyError, ValueError):
                pass
        cache[bid] = out
    return cache[bid]


def state(c, date):
    res, rb, cb, lk = defaultdict(dict), {}, {}, {}
    for line in git("ls-tree", "-r", c, "results").splitlines():
        _, _, bid, path = line.split(None, 3)
        if path.endswith("all_results.csv"):
            for m, d in load_csv(bid).items():
                res[m].update(d)
                rb[m] = bid
        elif path.endswith("config.json"):
            try:
                j = json.loads(git("cat-file", "-p", bid))
                cb[j["model"]] = bid
                lk[j["model"]] = j.get("testdata_leakage", "No") == "Yes"
            except (ValueError, KeyError):
                pass
    cf = sorted({d for m in res for d in res[m]})
    ents = {}
    for m in res:
        vec = [res[m].get(d, np.nan) for d in cf]
        shown = m in lk and not np.isnan(vec).any() and m.lower() not in ("seasonal_naive", "seasonal naive")
        ents[m] = (vec, (rb.get(m), cb.get(m)), (1 if lk.get(m) else 0) if shown else None)
    return dict(date=date, entries=ents, units=tuple(cf))


commits = [l.split() for l in git("log", "--reverse", "--format=%H %ad", "--date=short", PIN, "--", "results").splitlines()]
states = [state(c_, d_) for c_, d_ in commits]
print(f"{len(states)} commits replayed")


def gmean_rule(S, nm):
    sn = next((i for i, m in enumerate(nm) if m.lower() in ("seasonal_naive", "seasonal naive")), None)
    if sn is None or np.isnan(S[sn]).any():
        return None
    return np.log(S / S[sn]).mean(1)


h = vb.history_replay(states, avg)
check("replay MASE mean rank: transitions", h["transitions"], 141)
check("replay MASE mean rank: reversed pairs", h["reversed_pairs"], 18)
check("replay MASE mean rank: transitions with reversal", h["transitions_with_reversal"], 14)
hg = vb.history_replay(states, gmean_rule)
check("replay gmean: reversed pairs", hg["reversed_pairs"], 0)
print("   gmean transitions examined:", hg["transitions"])

# ---- self-consistency: bootstrap helper equals vernier_stats.pct_bootstrap given the same random stream
import vernier_stats as vs
d = Xa[oa[1]] - Xa[oa[0]]
a_ = vs.pct_bootstrap(d, cl, B=4000, rng=np.random.default_rng(1), scheme="iid")
b_ = vb._draws(d[None], cl, 4000, np.random.default_rng(1))[0][0]
check("iid draws reproduce vernier_stats.pct_bootstrap interval", tuple(np.percentile(b_, [2.5, 97.5]).round(9)), tuple(np.round(a_["ci95"], 9)))

print("\nknown-shift control, GIFT-Eval MASE_Rank:", json.dumps(rep["known_shift"]["detection"], indent=1))
print("counts:", json.dumps(c))
print("resolution:", json.dumps(rs))
print("tiers:", {k: v["size"] for k, v in rep["leading_tier"].items()})
json.dump(rep, open("gift_mase_rank_report.json", "w"))
assert not fails, fails
print("ALL OK")
