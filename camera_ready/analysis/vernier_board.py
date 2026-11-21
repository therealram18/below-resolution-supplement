"""Shared leaderboard-audit report (same code for every board). Statistics come from vernier_stats.

Conventions
- X[entry, unit], LOWER IS BETTER (callers negate where higher is better); an entry's displayed
  statistic is the plain mean of its row. `order` lists entry indices, best displayed first.
- Pair (a, b), a placed higher than b: d = X[b] - X[a]; positive d = the higher-placed entry is better.
- Bootstrap p = 2*min(share <= 0, share >= 0); paper-rule p = max(p_iid, p_cluster); "separated at 5%"
  under the paper rule = both percentile 95% intervals exclude zero. Other methods: p < 0.05.
- Paper resolution = 2.80 x cluster-bootstrap SE; corrected = vernier_stats.mde80(se_cr2, df).
- Tie rules behind the published GIFT-Eval numbers (checked in test_vernier_board.py):
  resolution 6.16 / median gap 0.9072 / 1269 sources / tier of 3 / 28-of-121 / 42-of-128 / replay 141,18,14
  come from the scripts resolution_budget.py, pool_perturbations.py, pool_dependence_replay.py, which rank
  with scipy 'average' ties over every entry that has a score for the unit. The audit code
  (load_gift_pairs.py) ranks with first-come ('ordinal') ties; that is a different X (see the test).
- history_replay / pool_tests accept an optional `tier` (None = not displayable, else a sort key such as
  0 normal / 1 leakage-flagged): the board's displayed order is (tier, aggregate).
"""
import inspect
import numpy as np
from scipy.stats import rankdata
import vernier_stats as vs


def _draws(D, cl, B, rng):
    """Bootstrap means of every row of D (P x N): iid over units and over whole clusters (as pct_bootstrap)."""
    D = np.asarray(D, float)
    P, N = D.shape
    _, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float)
    G = len(n)
    sums = np.stack([np.bincount(inv, weights=r) for r in D])
    iid, clu = np.empty((P, B)), np.empty((P, B))
    step = max(50, int(2e6 // max(N, G * P)))
    for s in range(0, B, step):
        b = min(step, B - s)
        idx = rng.integers(0, N, size=(b, N))
        for p in range(P):
            iid[p, s:s + b] = D[p][idx].mean(1)
        idc = rng.integers(0, G, size=(b, G))
        clu[:, s:s + b] = sums[:, idc].sum(2) / n[idc].sum(1)
    return iid, clu


def _boot_summary(x, null=0.0):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return dict(ci=(float(lo), float(hi)), se=float(x.std(ddof=1)),
                p=float(min(1.0, 2 * min((x <= null).mean(), (x >= null).mean()))))


def _pair_records(D, cl, B_boot, B_wild, rng):
    iid, clu = _draws(D, cl, B_boot, rng)
    out = []
    for p, d in enumerate(D):
        bi, bc = _boot_summary(iid[p]), _boot_summary(clu[p])
        c_icc, c_ind = vs.cr2_test(d, cl, df="icc"), vs.cr2_test(d, cl, df="indep")
        c1, mt = vs.cr1_test(d, cl), vs.means_t_test(d, cl)
        w = vs.wild_cluster_p(d, cl, B=B_wild, rng=rng)
        keep = lambda r: dict(se=float(r["se"]), df=float(r["df"]), p=float(r["p"]), ci95=[float(r["ci95"][0]), float(r["ci95"][1])])
        out.append(dict(
            theta=float(d.mean()), iid=bi, cluster=bc,
            p_paper=max(bi["p"], bc["p"]),
            sep_paper=bool((bi["ci"][0] > 0 or bi["ci"][1] < 0) and (bc["ci"][0] > 0 or bc["ci"][1] < 0)),
            cr2_icc=keep(c_icc), cr2_indep=keep(c_ind), cr1=keep(c1),
            wild_p=float(w["p"]), means_t_p=float(mt["p"]), icc_hat=float(vs.icc_hat(d, cl))))
    return out


def _sep(r, m):
    """Separated at 5% for method m."""
    if m == "paper":
        return r["sep_paper"]
    if m == "P-clu":
        return r["cluster"]["p"] < 0.05
    p = {"CR2-icc": r["cr2_icc"]["p"], "CR2-indep": r["cr2_indep"]["p"], "CR1": r["cr1"]["p"],
         "wild": r["wild_p"], "means-t": r["means_t_p"]}[m]
    return p < 0.05


METHODS = ["paper", "P-clu", "CR2-icc", "CR2-indep", "CR1", "wild", "means-t"]


def _pval(r, m):
    return {"paper": r["p_paper"], "P-clu": r["cluster"]["p"], "CR2-icc": r["cr2_icc"]["p"],
            "CR2-indep": r["cr2_indep"]["p"], "CR1": r["cr1"]["p"], "wild": r["wild_p"],
            "means-t": r["means_t_p"]}[m]


def _known_shift(d_pairs, cl, deltas, n_control, rng):
    """Detection rate per delta, per method, for each pair (rows of d_pairs). Sources resampled with replacement."""
    members = {g: np.flatnonzero(cl == g) for g in np.unique(cl)}
    labels = list(members)
    G = len(labels)
    names = ["paper", "CR2-icc", "wild"]
    rates = np.zeros((len(deltas), 3, len(d_pairs)))
    for q, d in enumerate(d_pairs):
        th = d.mean()
        for _ in range(n_control):
            pick = rng.integers(0, G, size=G)
            parts = [members[labels[g]] for g in pick]
            dr = np.concatenate([d[m] for m in parts])
            clr = np.repeat(np.arange(G), [len(m) for m in parts])
            iid, clu = _draws(dr[None], clr, 999, rng)
            ci_i, ci_c = np.percentile(iid[0], [2.5, 97.5]), np.percentile(clu[0], [2.5, 97.5])
            for k, dl in enumerate(deltas):
                null = th - dl  # testing theta_orig on (d_res + dl) is testing th - dl on d_res
                rates[k, 0, q] += (null < ci_i[0] or null > ci_i[1]) and (null < ci_c[0] or null > ci_c[1])
                rates[k, 1, q] += vs.cr2_test(dr, clr, null=null, df="icc")["p"] < 0.05
                rates[k, 2, q] += vs.wild_cluster_p(dr, clr, null=null, B=999, rng=rng)["p"] < 0.05
    rates /= n_control
    return {n: [float(rates[k, i].mean()) for k in range(len(deltas))] for i, n in enumerate(names)}, rates


def report_from_matrix(X, cl, names, order, seed=20261111, B_boot=20000, B_wild=99999, n_control=1000):
    X = np.asarray(X, float)
    cl = np.asarray(cl)
    assert not np.isnan(X).any()
    rng = np.random.default_rng(seed)
    G = len(np.unique(cl))
    top = list(order[:10])
    adj = [(i, i + 1) for i in range(9)]
    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    dvec = lambda a, b: X[b] - X[a]
    D = np.stack([dvec(top[i], top[j]) for i, j in pairs])
    recs = _pair_records(D, cl, B_boot, B_wild, rng)
    for (i, j), r in zip(pairs, recs):
        r.update(a=names[top[i]], b=names[top[j]], rank_a=i + 1, rank_b=j + 1, adjacent=(j == i + 1))
    pos = {p: k for k, p in enumerate(pairs)}
    adj_idx = [pos[p] for p in adj]

    counts = {}
    for m in METHODS:
        rej = np.array([_sep(r, m) for r in recs])
        hol = vs.holm([_pval(r, m) for r in recs])
        counts[m] = dict(adjacent=int(rej[adj_idx].sum()), all=int(rej.sum()),
                         adjacent_holm=int(hol[adj_idx].sum()), all_holm=int(hol.sum()))

    ar = [recs[k] for k in adj_idx]
    gaps = np.array([abs(r["theta"]) for r in ar])
    se_b = np.array([r["cluster"]["se"] for r in ar])
    se_c = np.array([r["cr2_icc"]["se"] for r in ar])
    res_paper = float(np.median(2.80 * se_b))
    res_icc = float(np.median([vs.mde80(r["cr2_icc"]["se"], r["cr2_icc"]["df"]) for r in ar]))
    res_ind = float(np.median([vs.mde80(r["cr2_indep"]["se"], r["cr2_indep"]["df"]) for r in ar]))
    med_gap = float(np.median(gaps))
    nz = gaps > 0
    resolution = dict(
        paper=res_paper, corrected_icc=res_icc, corrected_indep=res_ind, median_adjacent_gap=med_gap,
        ratio_paper=res_paper / med_gap, ratio_corrected_icc=res_icc / med_gap, ratio_corrected_indep=res_ind / med_gap,
        sources=G, n_nonzero_gap=int(nz.sum()),
        sources_needed_boot=float(np.median(G * (2.80 * se_b[nz] / gaps[nz]) ** 2)),
        sources_needed_cr2=float(np.median(G * (2.80 * se_c[nz] / gaps[nz]) ** 2)))

    k = min(39, len(order) - 1)
    lead = order[0]
    LD = np.stack([dvec(lead, order[t]) for t in range(1, k + 1)])
    lrec = _pair_records(LD, cl, B_boot, B_wild, rng)
    tier = {}
    for m, key in (("paper", "paper"), ("CR2-icc", "CR2-icc"), ("wild", "wild")):
        mem = [names[lead]] + [names[order[t + 1]] for t, r in enumerate(lrec) if not _sep(r, key)]
        tier[m] = dict(size=len(mem), members=mem)

    d_adj = [dvec(top[i], top[i + 1]) for i, _ in adj]
    deltas = [res_paper, res_icc, 0.0]
    kr, per_pair = _known_shift(d_adj, cl, deltas, n_control, rng) if n_control else (None, None)
    shift = None if not n_control else dict(deltas=dict(paper_resolution=res_paper, corrected_resolution=res_icc, zero=0.0), n_control=n_control,
                 detection={m: dict(at_paper_resolution=v[0], at_corrected_resolution=v[1], false_alarm_delta0=v[2])
                            for m, v in kr.items()},
                 per_pair={m: per_pair[:, i, :].tolist() for i, m in enumerate(["paper", "CR2-icc", "wild"])})

    return dict(seed=seed, B_boot=B_boot, B_wild=B_wild, n_entries=len(names), n_units=X.shape[1], n_sources=G,
                top10=[names[i] for i in top], pairs=recs, counts=counts, resolution=resolution,
                leading_tier=tier, known_shift=shift,
                n_adjacent_pairs=9, n_all_pairs=45)


# ---------------------------------------------------------------- pool dependence

def _call(rule_fn, S, names):
    try:
        two = len(inspect.signature(rule_fn).parameters) >= 2
    except (TypeError, ValueError):
        two = False
    return rule_fn(S, names) if two else rule_fn(S)


def _display_order(agg, tier, idx):
    t = np.zeros(len(agg)) if tier is None else np.asarray([0 if x is None else x for x in tier], float)
    return sorted(idx, key=lambda i: (t[i], agg[i]))  # stable: ties keep entry order


def pool_tests(S, rule_fn, displayed_idx, top_k=10, tier=None):
    S = np.asarray(S, float)
    n = len(S)
    disp = list(displayed_idx)
    agg = np.asarray(rule_fn(S), float)
    T = _display_order(agg, tier, disp)[:top_k]
    internal = lambda a: sorted(T, key=lambda i: a[i])  # stable sort of the same entries by new aggregate
    rem_cases = [x for x in range(n) if x not in T]
    rem_changed = []
    for x in rem_cases:
        keep = [i for i in range(n) if i != x]
        a = np.empty(n)
        a[keep] = rule_fn(S[keep])
        rem_changed.append(internal(a) != T)
    add_changed = []
    for m in disp:
        a = np.asarray(rule_fn(np.vstack([S, S[m] * (1 + 1e-6)])), float)[:n]
        add_changed.append(internal(a) != T)
    return dict(top=T, removal=dict(changed=int(sum(rem_changed)), of=len(rem_cases),
                                     entries=[x for x, c in zip(rem_cases, rem_changed) if c]),
                near_copy=dict(changed=int(sum(add_changed)), of=len(disp),
                               entries=[m for m, c in zip(disp, add_changed) if c]))


def history_replay(states, rule_fn, top_k=10):
    """entries: {name: (vector, hash[, tier])}; tier None = not displayable (default tier 0)."""
    def agg_state(st):
        nm = list(st["entries"])
        if not nm:
            return None
        S = np.array([st["entries"][m][0] for m in nm], float)
        a = _call(rule_fn, S, nm)
        if a is None:
            return None
        tier = [(st["entries"][m][2] if len(st["entries"][m]) > 2 else 0) for m in nm]
        agg = dict(zip(nm, np.asarray(a, float)))
        disp = _display_order(np.asarray(a, float), tier, [i for i, t in enumerate(tier) if t is not None])
        return agg, [nm[i] for i in disp], {m: st["entries"][m][1] for m in nm}, {m for m, t in zip(nm, tier) if t is not None}
    prev, steps, flip_steps, flips, events = None, 0, 0, 0, []
    for st in states:
        cur = agg_state(st)
        if cur is None or len(cur[1]) < 10:
            prev = None
            continue
        agg, disp, hs, shown = cur
        if prev and set(st["units"]) == set(prev[0]["units"]):
            pst, (pagg, pdisp, phs, _) = prev
            steps += 1
            inc = [m for m in pdisp[:top_k] if m in shown and hs[m] == phs[m]]
            fl = [(a, b) for i, a in enumerate(inc) for b in inc[i + 1:] if (pagg[a] - pagg[b]) * (agg[a] - agg[b]) < 0]
            if fl:
                flip_steps += 1
                flips += len(fl)
                events.append(dict(from_date=pst["date"], to_date=st["date"], pairs=fl,
                                   entries_added=sorted(m for m in hs if m not in phs or hs[m] != phs[m])))
        prev = (st, cur)
    return dict(transitions=steps, transitions_with_reversal=flip_steps, reversed_pairs=flips, events=events)


def mean_rank_rule(method="average"):
    assert method in ("average", "ordinal")
    def rule(S):
        S = np.asarray(S, float)
        R = np.full(S.shape, np.nan)
        for j in range(S.shape[1]):
            ok = ~np.isnan(S[:, j])
            R[ok, j] = rankdata(S[ok, j], method=method)
        return R.mean(1)  # entries missing a unit get NaN (they are not displayable)
    return rule
