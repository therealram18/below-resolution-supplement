import sys, json
import numpy as np
HERE = "fmts/camera_ready"
sys.path.insert(0, HERE)
import vernier_stats as v
from exact_signflip import exact_sum_p, exact_student_p, TOL


def brute(s, n):
    G = len(s); N = n.sum()
    W = 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) - 1
    A = W @ s
    ps = np.mean(np.abs(A) >= abs(s.sum()) * (1 - TOL))
    t = v._cr1_t(W * s, n, N)
    t0 = v._cr1_t(s[None, :], n, N)[0]
    return float(ps), float(np.mean(np.abs(t) >= abs(t0) * (1 - TOL)))


def test_random_layouts():
    rng = np.random.default_rng(20261111)
    for _ in range(20):
        G = 12
        n = rng.integers(1, 9, G).astype(float)
        cl = np.repeat(np.arange(G), n.astype(int))
        d = rng.standard_t(3, len(cl)) + rng.normal(0, 1, G)[cl] * 0.5 + 0.2
        s = np.bincount(cl, weights=d)
        bs, bt = brute(s, n)
        assert abs(exact_sum_p(s) - bs) < 1e-12
        assert abs(exact_student_p(s, n) - bt) < 1e-12
        assert abs(exact_student_p(s, n) - v.wild_cluster_p(d, cl, exact_max_G=16)["p"]) < 1e-12


def test_odd_G_and_big_chunks():
    rng = np.random.default_rng(1)
    n = rng.integers(1, 5, 11).astype(float); s = rng.normal(size=11)
    bs, bt = brute(s, n)
    assert abs(exact_sum_p(s) - bs) < 1e-12 and abs(exact_student_p(s, n, chunk_elems=50) - bt) < 1e-12


def test_equal_sizes_same_p():
    rng = np.random.default_rng(2)
    for _ in range(5):
        s = rng.normal(0.3, 1, 12)
        assert abs(exact_sum_p(s) - exact_student_p(s, np.full(12, 4.0))) < 1e-12


def test_real_pair_vs_stored_wcb():
    import load_gift_pairs as lg
    top, diffs, cl = lg.top10_pair_diffs("MASE_Rank")
    st = json.load(open(f"{HERE}/retest_gift.json"))["pairs"]["MASE_Rank"]
    PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    k = PAIRS.index((8, 9))
    d = diffs[(8, 9)]
    s = np.bincount(cl, weights=d); n = np.bincount(cl).astype(float)
    p = exact_student_p(s, n); q = st[k]["WCB-Rad"]["p"]
    se = np.sqrt(p * (1 - p) / 99999)
    print("pair 9-10 MASE exact", p, "stored", q, "4 MC se", 4 * se)
    assert abs(p - q) <= 4 * se


if __name__ == "__main__":
    for f in (test_random_layouts, test_odd_G_and_big_chunks, test_equal_sizes_same_p, test_real_pair_vs_stored_wcb):
        f(); print("PASS", f.__name__)
