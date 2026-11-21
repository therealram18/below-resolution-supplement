"""Exact sign-flip p-values by full enumeration (DESIGN_ROUND2.md S1). s = cluster sums of (d - null), n = cluster sizes."""
import numpy as np

TOL = 1e-9


def _signed_sums(x):
    """All 2^len(x) values of sum w_g x_g (w = +-1), as rows of the sign matrix applied to x."""
    k = len(x)
    out = np.zeros(1)
    for v in x:
        out = np.concatenate([out + v, out - v])
    return out


def exact_sum_p(s):
    s = np.asarray(s, float)
    h = len(s) // 2
    a, b = _signed_sums(s[:h]), np.sort(_signed_sums(s[h:]))
    thr = abs(s.sum()) * (1 - TOL)
    # |a+b| >= thr  <=>  b >= thr-a  or  b <= -thr-a  (disjoint when thr > 0; thr == 0 counts everything)
    hi = len(b) - np.searchsorted(b, thr - a, side="left")
    lo = np.searchsorted(b, -thr - a, side="right")
    cnt = hi + lo
    if thr == 0:
        cnt = np.full(len(a), len(b))
    return float(cnt.sum() / (len(a) * len(b)))


def _half_AC(s, n):
    A = np.zeros(1); C = np.zeros(1)
    for sv, nv in zip(s, n):
        A, C = np.concatenate([A + sv, A - sv]), np.concatenate([C + nv * sv, C - nv * sv])
    return A, C


def _t2(A, C, N, G, Q, R):
    """t*^2 for arrays A, C; 0 where se* is 0 (as vernier_stats._cr1_t)."""
    den = G / (G - 1) * (Q - 2 * A * C / N + A * A * R / N ** 2)
    ok = den > 1e-13 * max(Q, 1e-300)
    return np.where(ok, A * A / np.where(ok, den, 1.0), 0.0)


def exact_student_p(s, n, chunk_elems=4_000_000):
    """Exact two-sided p of the CR1 studentized statistic over all 2^G sign patterns.
    The overall sign flip leaves |t*| unchanged, so the first cluster is fixed at +1 and the count doubled."""
    s = np.asarray(s, float); n = np.asarray(n, float)
    G, N = len(s), n.sum()
    Q, R = (s ** 2).sum(), (n ** 2).sum()
    t_obs2 = float(_t2(np.array([s.sum()]), np.array([(n * s).sum()]), N, G, Q, R)[0])
    thr = t_obs2 * (1 - TOL) ** 2
    h = (G - 1) // 2 + 1                       # first half includes cluster 0 (fixed +)
    A1, C1 = _half_AC(s[1:h], n[1:h])
    A1, C1 = A1 + s[0], C1 + n[0] * s[0]
    A2, C2 = _half_AC(s[h:], n[h:])
    m = max(1, chunk_elems // len(A2))
    cnt = 0
    for i in range(0, len(A1), m):
        t2 = _t2(A1[i:i + m, None] + A2[None, :], C1[i:i + m, None] + C2[None, :], N, G, Q, R)
        cnt += int((t2 >= thr).sum())
    return float(cnt / (len(A1) * len(A2)))
