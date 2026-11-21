"""Main-session recheck of the signed-rank sign-flip test on the real geometric-mean pairs
(MASE_Rank top ten), written separately from gmean_level_check.py. Prints the counts and
compares one pair with scipy's exact Wilcoxon test."""
import json, os, sys
import numpy as np
from scipy.stats import rankdata, wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from exact_signflip import exact_sum_p
import vernier_stats as vs
z = np.load(os.path.join(HERE, "calib_d4_data.npz")); cl = z["gmean|cl"]
_, inv = np.unique(cl, return_inverse=True); n = np.bincount(inv).astype(float)
pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
def p4(d):
    x = np.bincount(inv, weights=d) / n
    return exact_sum_p(np.sign(x) * rankdata(np.abs(x)))
P = {k: p4(z["gmean|%d_%d" % k]) for k in pairs}
adj = [P[(i, i + 1)] for i in range(9)]; allp = [P[k] for k in pairs]
m = np.bincount(inv, weights=z["gmean|8_9"]) / n
out = dict(neighbours_of_9=int(sum(p < 0.05 for p in adj)), pairs_of_45=int(sum(p < 0.05 for p in allp)),
           holm_9=int(vs.holm(adj).sum()), holm_45=int(vs.holm(allp).sum()),
           pair_9_10_mine=P[(8, 9)], pair_9_10_scipy_exact=float(wilcoxon(m, mode="exact").pvalue))
print(out); json.dump(out, open(os.path.join(HERE, "verify_signedrank_counts.json"), "w"), indent=1)
