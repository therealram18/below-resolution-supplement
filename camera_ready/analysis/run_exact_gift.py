"""Exact sign-flip p-values for the 45 pairs x 3 rules on GIFT-Eval (DESIGN_ROUND2.md S1). Usage: run_exact_gift.py [n_pairs_for_timing]"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = "fmts/camera_ready"
sys.path.insert(0, HERE)
import vernier_stats as v
import load_gift_pairs as lg
from exact_signflip import exact_sum_p, exact_student_p

RULES = ["MASE_Rank", "CRPS_Rank", "gmean"]
PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
ADJ = [k for k, (i, j) in enumerate(PAIRS) if j == i + 1]


def one(args):
    rule, k = args
    top, diffs, cl = lg.top10_pair_diffs(rule)
    d = diffs[PAIRS[k]]
    s = np.bincount(cl, weights=d); n = np.bincount(cl).astype(float)
    return rule, k, exact_sum_p(s), exact_student_p(s, n)


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    t0 = time.time()
    jobs = [(r, k) for r in RULES for k in range(lim)]
    with Pool() as pl:
        res = pl.map(one, jobs, chunksize=1)
    print("compute seconds", time.time() - t0, "for", len(jobs), "pair-rule jobs")
    if lim < 45:
        return
    st = json.load(open(f"{HERE}/retest_gift.json"))["pairs"]
    out = {}
    for rule in RULES:
        rr = sorted(x for x in res if x[0] == rule)
        psum = np.array([x[2] for x in rr]); pstu = np.array([x[3] for x in rr])
        wild = np.array([r["WCB-Rad"]["p"] for r in st[rule]])
        cr2 = np.array([r["CR2-icc"]["p"] for r in st[rule]])
        lab = lambda k: f"{PAIRS[k][0]+1}-{PAIRS[k][1]+1}"
        def cnt(p):
            h = v.holm(p); u = p < 0.05
            return dict(adj_unc=int(u[ADJ].sum()), all_unc=int(u.sum()),
                        adj_holm=int(v.holm(p[ADJ]).sum()), all_holm=int(h.sum()))
        cw = cnt(wild)
        def dec(p):
            u, h9, h45 = p < .05, np.zeros(45, bool), v.holm(p)
            h9[ADJ] = v.holm(p[ADJ])
            return u, h9, h45
        dw, de = {}, {}
        res_rule = dict(pairs={lab(k): dict(p_exact_student=float(pstu[k]), p_exact_sum=float(psum[k])) for k in range(45)},
                        counts=dict(CR2=cnt(cr2), wild_stored=cw, exact_student=cnt(pstu), exact_sum=cnt(psum)))
        for nm, p in (("exact_student", pstu), ("exact_sum", psum)):
            ku = dec(p); kw = dec(wild)
            res_rule[f"decision_differences_vs_wild_{nm}"] = dict(
                uncorrected_all45=int((ku[0] != kw[0]).sum()), uncorrected_neighbours=int((ku[0] != kw[0])[ADJ].sum()),
                holm_all45=int((ku[2] != kw[2]).sum()), holm_neighbours9=int((ku[1] != kw[1])[ADJ].sum()))
            dd = np.abs(p - wild); m = int(dd.argmax())
            res_rule[f"max_abs_diff_{nm}_vs_wild"] = dict(value=float(dd[m]), pair=lab(m), exact=float(p[m]), wild=float(wild[m]))
        out[rule] = res_rule
    out["meta"] = dict(note="gmean uses the pairs-order top ten (as the first round's 45-pair counts); stored wild = WCB-Rad, 99,999 draws", runtime_s=time.time() - t0)
    json.dump(out, open(f"{HERE}/exact_gift.json", "w"), indent=1)
    for r in RULES:
        print(r, json.dumps(out[r]["counts"]), out[r]["max_abs_diff_exact_student_vs_wild"], out[r]["max_abs_diff_exact_sum_vs_wild"])


if __name__ == "__main__":
    main()
