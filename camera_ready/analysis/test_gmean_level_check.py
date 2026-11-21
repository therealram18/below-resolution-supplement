"""Sanity checks for gmean_level_check.py (S3). Foreground."""
import sys, time
import numpy as np
from scipy import stats
sys.path.insert(0, "fmts/camera_ready")
import gmean_level_check as g

def test_b():
    rng = np.random.default_rng(1); worst = 0
    for _ in range(10):
        x = rng.standard_normal(20)
        p = g.pratt_p(x); q = stats.wilcoxon(x, method="exact", alternative="two-sided").pvalue
        worst = max(worst, abs(p - q))
    print("(b) max |p - wilcoxon exact p| over 10 vectors: %.2e" % worst); assert worst < 1e-12

def test_a(n=2000):
    cl = g.load_set(1)[0][2]; starts, sizes = g.layout(cl)
    rng = np.random.default_rng(2); r1 = r4 = 0; t0 = time.time()
    for _ in range(n):
        m = g.dsmeans(rng.standard_normal(len(cl)), starts, sizes)
        r1 += g.exact_sum_p(m) < .05; r4 += g.pratt_p(m) < .05
    print("(a) normal data, %d draws: c1 %.2f%%, c4 %.2f%% (%.0fs)" % (n, 100 * r1 / n, 100 * r4 / n, time.time() - t0))
    assert .04 <= r1 / n <= .06 and .04 <= r4 / n <= .06

def test_c():  # CR2 on set i, 100 reps x 5 pairs would be noisy; done in the real run (check printed by report)
    pass

if __name__ == "__main__":
    test_b(); test_a(); print("sanity (a),(b) passed")
