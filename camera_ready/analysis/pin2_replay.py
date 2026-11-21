"""Replay of GIFT-Eval updates after b61c48aa on the 2026-11-13 clone. Both tie conventions.
Run: python -I pin2_replay.py   (from camera_ready). Reads git objects only; never runs code from the clone."""
import sys, os, csv, io, json, math, subprocess
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vernier_board as vb
REPO = os.path.abspath("boards/pin_2026-11-13/gift"); OUT = "boards/pin_2026-11-13/"
PIN = "b61c48aa3197a1a111e405c0e6f600128a1efee3"
MASE = "eval_metrics/MASE[0.5]"
git = lambda *a: subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True).stdout
cache = {}
def load_csv(bid):
    if bid not in cache:
        out = defaultdict(dict)
        for r in csv.DictReader(io.StringIO(git("cat-file", "-p", bid))):
            try: out[r["model"]][r["dataset"]] = float(r[MASE])
            except (KeyError, ValueError): pass
        cache[bid] = out
    return cache[bid]
def raw(c):
    res, rb, cb, lk = defaultdict(dict), {}, {}, {}
    for line in git("ls-tree", "-r", c, "results").splitlines():
        _, _, bid, path = line.split(None, 3)
        if path.endswith("all_results.csv"):
            for m, d in load_csv(bid).items(): res[m].update(d); rb[m] = bid
        elif path.endswith("config.json"):
            try:
                j = json.loads(git("cat-file", "-p", bid)); cb[j["model"]] = bid; lk[j["model"]] = j.get("testdata_leakage", "No") == "Yes"
            except (ValueError, KeyError): pass
    return res, rb, cb, lk
# commits touching results, from the pin onward (pin itself is the starting state)
allc = [l.split() for l in git("log", "--reverse", "--format=%H %ad", "--date=short", "--", "results").splitlines()]
idx = [c for c, _ in allc].index(PIN)
commits = allc[idx:]
print(len(commits) - 1, "results-touching commits after pin")
raws = [(c, d) + raw(c) for c, d in commits]
# ---- list of updates
upd = []
for k in range(1, len(raws)):
    c, d, res, rb, cb, lk = raws[k]; _, _, pres, prb, pcb, plk = raws[k - 1]
    new = sorted(m for m in set(res) | set(cb) if m not in pres and m not in plk)
    chg_res = sorted(m for m in res if m in pres and rb.get(m) != prb.get(m))
    chg_cfg = sorted(m for m in cb if m in plk and cb.get(m) != pcb.get(m))
    gone = sorted(m for m in set(pres) | set(plk) if m not in res and m not in cb)
    subj = git("log", "-1", "--format=%s", c).strip()
    upd.append(dict(commit=c[:8], date=d, subject=subj, new=new, results_changed=chg_res, config_changed=chg_cfg, removed=gone,
                    n_entries_after=len(set(res) & set(lk))))
# ---- convention 1: vernier_board.history_replay, ranks averaged
def state(r):
    c, d, res, rb, cb, lk = r
    cf = sorted({x for m in res for x in res[m]})
    ents = {}
    for m in res:
        vec = [res[m].get(x, np.nan) for x in cf]
        shown = m in lk and not np.isnan(vec).any() and m.lower() not in ("seasonal_naive", "seasonal naive")
        ents[m] = (vec, (rb.get(m), cb.get(m)), (1 if lk.get(m) else 0) if shown else None)
    return dict(date=d, entries=ents, units=tuple(cf))
states = [state(r) for r in raws]
h = vb.history_replay(states, vb.mean_rank_rule("average"))
# ---- convention 2: poolcheck logic (copy of pool_dependence_replay_method_first.py scores/replay): ranks among all models with a score for the
# config, ties broken by alphabetical model order ("first"), score kept only for models with every config and a config.json
def scores(res, leak):
    configs = sorted({d for m in res for d in res[m]}); ok = [m for m in res if m in leak and all(d in res[m] for d in configs)]
    R = defaultdict(list); ties = 0
    for d in configs:
        ids = sorted(m for m in res if d in res[m] and not math.isnan(res[m][d]))
        vals = [res[m][d] for m in ids]
        ties += len(set(vals)) != len(vals)
        for m, r in zip(ids, rankdata(vals, method="ordinal")): R[m].append(r)
    return {m: float(np.mean(R[m])) for m in ok}, configs, ties
def poolcheck_replay(first=False):
    prev = None; steps = fs = flips = 0; ex = []; tie_cmp = 0; cfg_ties = 0
    for c, date, res, rb, cb, leak in raws:
        if first:
            s, cf, t = scores(res, leak); cfg_ties += t
        else:
            cf = sorted({d for m in res for d in res[m]}); ok = [m for m in res if m in leak and all(d in res[m] for d in cf)]
            R = defaultdict(list)
            for d in cf:
                ids = [m for m in res if d in res[m] and not math.isnan(res[m][d])]
                for m, r in zip(ids, rankdata([res[m][d] for m in ids], method="average")): R[m].append(r)
            s = {m: float(np.mean(R[m])) for m in ok}
        if len(s) < 10: prev = None; continue
        disp = sorted(s, key=lambda m: (leak[m], s[m]))
        if prev and set(cf) == set(prev[4]):
            pdate, ps, prb, pcb, pcf, pdisp = prev; steps += 1
            inc = [m for m in pdisp[:10] if m in s and rb.get(m) == prb.get(m) and cb.get(m) == pcb.get(m)]
            for i, a in enumerate(inc):
                for b in inc[i + 1:]:
                    if ps[a] == ps[b] or s[a] == s[b]: tie_cmp += 1
            fl = [(a, b) for i, a in enumerate(inc) for b in inc[i + 1:] if (ps[a] - ps[b]) * (s[a] - s[b]) < 0]
            if fl: fs += 1; flips += len(fl); ex.append(dict(from_date=pdate, to_date=date, pairs=fl, entries_added=sorted(m for m in res if rb.get(m) != prb.get(m))[:6]))
        prev = (date, s, rb, cb, cf, disp)
    return dict(transitions=steps, transitions_with_reversal=fs, reversed_pairs=flips, events=ex, exact_tie_comparisons=tie_cmp)
pc_avg = poolcheck_replay(False); pc_first = poolcheck_replay(True)
out = dict(updates=upd, avg_vernier_board=h, avg_poolcheck_code=pc_avg, exact_tiebreak_poolcheck_code=pc_first)
json.dump(out, open(OUT + "gift_replay_after_b61c48aa.json", "w"), indent=1, default=str)
print("avg (vernier_board.history_replay):", {k: h[k] for k in ("transitions", "transitions_with_reversal", "reversed_pairs")})
print("avg (poolcheck code):", {k: pc_avg[k] for k in ("transitions", "transitions_with_reversal", "reversed_pairs")})
print("exact (poolcheck first):", {k: pc_first[k] for k in ("transitions", "transitions_with_reversal", "reversed_pairs", "exact_tie_comparisons")})
for u in upd: print(u["date"], u["commit"], "new", u["new"], "res_chg", u["results_changed"], "cfg_chg", u["config_changed"], "gone", u["removed"], "n", u["n_entries_after"])
for name, e in (("avg", h["events"]), ("avg-poolcheck", pc_avg["events"]), ("exact", pc_first["events"])):
    print(name); [print("  ", x) for x in e]
