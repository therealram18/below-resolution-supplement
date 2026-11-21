"""N7/N8/N13 lookup. Copy of poolcheck/pool_dependence_replay_method_first_fast.py (exact tie-break = rank 'first' by
alphabetical model order, as scipy 'ordinal') that also prints commit hashes, full timestamps, places, and arrivals;
plus the second-round code (vernier_board.history_replay, tied ranks averaged). Read-only on the git clone.
Run: PYTHONDONTWRITEBYTECODE=1 python list_reversals_exact.py   (about 1 minute)"""
import subprocess, csv, io, json, math, sys, os
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vernier_board as vb
CR = os.path.dirname(os.path.abspath(__file__))
REPO = "fmts/poolcheck/gift"
PIN = "b61c48aa3197a1a111e405c0e6f600128a1efee3"
G = ["git", "-C", REPO]
git = lambda *a: subprocess.run(G + list(a), capture_output=True, text=True).stdout
MASE = "eval_metrics/MASE[0.5]"
commits = [l.split() for l in git("log", "--reverse", "--format=%H %aI", PIN, "--", "results").splitlines()]
tree = {}; blobs = set()
for c, _ in commits:
    e = []
    for line in git("ls-tree", "-r", c, "results").splitlines():
        _, _, bid, path = line.split(None, 3)
        if path.endswith("all_results.csv") or path.endswith("config.json"):
            e.append((bid, path)); blobs.add(bid)
    tree[c] = e
blobs = list(blobs)
out = subprocess.run(G + ["cat-file", "--batch"], input=("\n".join(blobs) + "\n").encode(), capture_output=True).stdout
content = {}; pos = 0
for bid in blobs:
    nl = out.index(b"\n", pos); size = int(out[pos:nl].split()[2])
    content[bid] = out[nl + 1:nl + 1 + size].decode(errors="replace"); pos = nl + 1 + size + 1
csvc = {}
def parse(bid):
    if bid not in csvc:
        d = defaultdict(dict)
        for r in csv.DictReader(io.StringIO(content[bid])):
            try: d[r["model"]][r["dataset"]] = float(r[MASE])
            except (KeyError, ValueError): pass
        csvc[bid] = d
    return csvc[bid]
def board(c):
    res = defaultdict(dict); rb = {}; cb = {}; lk = {}
    for bid, path in tree[c]:
        if path.endswith("all_results.csv"):
            for m, d in parse(bid).items(): res[m].update(d); rb[m] = bid
        else:
            try:
                j = json.loads(content[bid]); cb[j["model"]] = bid; lk[j["model"]] = j.get("testdata_leakage", "No") == "Yes"
            except (ValueError, KeyError): pass
    return res, rb, cb, lk
def mase_rank(res, lk, method, drop=()):
    """board's MASE_Rank per displayable entry. method 'ordinal' = board's rank(method='first') on alphabetical order."""
    res = {m: v for m, v in res.items() if m not in drop}
    cf = sorted({d for m in res for d in res[m]})
    ok = [m for m in res if m in lk and all(d in res[m] for d in cf)]
    R = defaultdict(list)
    for d in cf:
        ids = sorted(m for m in res if d in res[m])
        for m, r in zip(ids, rankdata([res[m][d] for m in ids], method=method)): R[m].append(r)
    return {m: float(np.mean(R[m])) for m in ok}, cf
B = [(c, d) + board(c) for c, d in commits]   # (hash, iso, res, rb, cb, lk)
def disp_order(s, lk): return sorted(s, key=lambda m: (lk[m], s[m]))

# ---------- exact tie-break (first-round code, ordinal)
def exact(method):
    ev = []; prev = None; steps = 0
    for c, d, res, rb, cb, lk in B:
        s, cf = mase_rank(res, lk, method)
        if len(s) < 10: prev = None; continue
        disp = disp_order(s, lk)
        if prev and set(cf) == set(prev["cf"]):
            steps += 1
            inc = [m for m in prev["disp"][:10] if m in s and rb.get(m) == prev["rb"].get(m) and cb.get(m) == prev["cb"].get(m)]
            fl = [(a, b) for i, a in enumerate(inc) for b in inc[i + 1:] if (prev["s"][a] - prev["s"][b]) * (s[a] - s[b]) < 0]
            new = sorted(m for m in res if rb.get(m) != prev["rb"].get(m))
            for a, b in fl:
                ev.append(dict(from_commit=prev["c"], from_time=prev["d"], to_commit=c, to_time=d,
                    arrived_or_changed=";".join(new), entry_1=a, entry_2=b,
                    place_before_1=prev["disp"].index(a) + 1, place_before_2=prev["disp"].index(b) + 1,
                    place_after_1=disp.index(a) + 1, place_after_2=disp.index(b) + 1,
                    meanrank_before_1=round(prev["s"][a], 4), meanrank_before_2=round(prev["s"][b], 4),
                    meanrank_after_1=round(s[a], 4), meanrank_after_2=round(s[b], 4),
                    tied_before=prev["s"][a] == prev["s"][b], tied_after=s[a] == s[b]))
        prev = dict(c=c, d=d, s=s, rb=rb, cb=cb, cf=cf, disp=disp)
    return ev, steps
ex_ev, ex_steps = exact("ordinal")
# ---------- second round code: vernier_board.history_replay with average ties
def vb_state(c, d, res, rb, cb, lk):
    cf = sorted({x for m in res for x in res[m]})
    ents = {}
    for m in res:
        vec = [res[m].get(x, np.nan) for x in cf]
        shown = m in lk and not np.isnan(vec).any() and m.lower() not in ("seasonal_naive", "seasonal naive")
        ents[m] = (vec, (rb.get(m), cb.get(m)), (1 if lk.get(m) else 0) if shown else None)
    return dict(date=c + "|" + d, entries=ents, units=tuple(cf))
sts = [vb_state(*b) for b in B]
avg = vb.mean_rank_rule("average")
h = vb.history_replay(sts, avg)
bymap = {b[0]: b for b in B}
av_ev = []
for e in h["events"]:
    fc, fd = e["from_date"].split("|"); tc, td = e["to_date"].split("|")
    b0, b1 = bymap[fc], bymap[tc]
    s0, _ = mase_rank(b0[2], b0[5], "average"); s1, _ = mase_rank(b1[2], b1[5], "average")
    d0, d1 = disp_order(s0, b0[5]), disp_order(s1, b1[5])
    for a, b in e["pairs"]:
        av_ev.append(dict(from_commit=fc, from_time=fd, to_commit=tc, to_time=td, arrived_or_changed=";".join(e["entries_added"]),
            entry_1=a, entry_2=b, place_before_1=d0.index(a) + 1, place_before_2=d0.index(b) + 1,
            place_after_1=d1.index(a) + 1, place_after_2=d1.index(b) + 1,
            meanrank_before_1=round(s0[a], 4), meanrank_before_2=round(s0[b], 4), meanrank_after_1=round(s1[a], 4), meanrank_after_2=round(s1[b], 4)))
def write(fn, rows):
    with open(os.path.join(CR, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write("reversals_exact_tiebreak.csv", ex_ev); write("reversals_ties_averaged.csv", av_ev)
ups = lambda ev: {(r["from_commit"], r["to_commit"]) for r in ev}
print("exact: transitions", ex_steps, "updates", len(ups(ex_ev)), "pairs", len(ex_ev))
print("averaged (vernier_board): transitions", h["transitions"], "updates", h["transitions_with_reversal"], "pairs", h["reversed_pairs"], "csv rows", len(av_ev))
key = lambda r: (r["to_commit"], frozenset((r["entry_1"], r["entry_2"])))
E = {key(r): r for r in ex_ev}; A = {key(r): r for r in av_ev}
print("in exact only:"); [print("  ", r["to_time"], r["entry_1"], r["entry_2"], r["place_before_1"], r["place_before_2"], r["meanrank_before_1"], r["meanrank_before_2"], r["meanrank_after_1"], r["meanrank_after_2"], "tied", r["tied_before"], r["tied_after"]) for k, r in E.items() if k not in A]
print("in averaged only:"); [print("  ", r["to_time"], r["entry_1"], r["entry_2"]) for k, r in A.items() if k not in E]
# ---------- N8 dates
SORT_COMMIT = "603993764aa48162852d7b1973021d0c1d52b361"
sort_t = git("show", "-s", "--format=%aI", SORT_COMMIT).strip()
from datetime import datetime
st = datetime.fromisoformat(sort_t)
print("sort change", sort_t)
for name, ev in (("exact", ex_ev), ("averaged", av_ev)):
    later = [r for r in ev if datetime.fromisoformat(r["from_time"]) > st]   # whole update after
    to_after = [r for r in ev if datetime.fromisoformat(r["to_time"]) > st]
    u_to = {(r["from_commit"], r["to_commit"]) for r in to_after}
    u_from = {(r["from_commit"], r["to_commit"]) for r in later}
    print(name, "updates ending after:", len(u_to), "pairs:", len(to_after), "| updates starting after:", len(u_from), "pairs:", len(later))
for r in ex_ev:
    if r["to_time"].startswith("2025-08-0"): print("2025-08 update:", r["from_commit"][:7], r["from_time"], r["to_commit"][:7], r["to_time"])
# ---------- N13
NEW = [c for c, d in commits[-3:]]
print("three newest commits", [(c[:7], d, git("show", "-s", "--format=%s", c).strip()) for c, d in commits[-3:]])
b0, b1 = B[-4], B[-1]
print("before commit", b0[0][:7], b0[1])
for meth in ("ordinal", "average"):
    s0, _ = mase_rank(b0[2], b0[5], meth); s1, _ = mase_rank(b1[2], b1[5], meth)
    newm = sorted(set(b1[2]) - set(b0[2])); print(meth, "new models:", newm)
    s1w, _ = mase_rank(b1[2], b1[5], meth, drop=set(newm))
    d0, d1, d1w = disp_order(s0, b0[5]), disp_order(s1, b1[5]), disp_order(s1w, b1[5])
    print(meth, "without-by-removal equals before-commit scores:", all(abs(s1w[m] - s0[m]) < 1e-9 for m in s0) and set(s1w) == set(s0), "orders equal:", d1w == d0)
    print(" top 12 before:", [(m, round(s0[m], 4)) for m in d0[:12]])
    print(" top 12 after :", [(m, round(s1[m], 4)) for m in d1[:12]])
    for m in ("limix_moe", "CastStar", "Falcon-Agent"):
        print(" ", m, "before place", d0.index(m) + 1, round(s0[m], 4), "| after place", d1.index(m) + 1, round(s1[m], 4), "| pin minus 3 place", d1w.index(m) + 1, round(s1w[m], 4))
