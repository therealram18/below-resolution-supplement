"""Small-sample cluster statistics shared by the camera-ready analyses (see DESIGN.md).

d = 1-D float array of per-unit differences, cl = 1-D int array of cluster labels.
Plain numpy/scipy; no classes.
"""
import numpy as np
from scipy import stats


def _groups(d, cl):
    d = np.asarray(d, float)
    cl = np.asarray(cl)
    _, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float)
    return d, inv, n


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def cluster_summary(d, cl):
    d, inv, n = _groups(d, cl)
    theta = d.mean()
    S = np.bincount(inv, weights=d - theta)
    return dict(N=len(d), G=len(n), theta=theta, sizes=n, S=S)


def icc_hat(d, cl):
    d, inv, n = _groups(d, cl)
    N, G = len(d), len(n)
    if N == G or G < 2:
        return 0.0
    theta = d.mean()
    means = np.bincount(inv, weights=d) / n
    msb = (n * (means - theta) ** 2).sum() / (G - 1)
    msw = ((d - means[inv]) ** 2).sum() / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    den = msb + (n0 - 1) * msw
    if den <= 0:
        return 0.0
    return float(min(max((msb - msw) / den, 0.0), 1.0))


def se_cr1(d, cl):
    s = cluster_summary(d, cl)
    return np.sqrt(s["G"] / (s["G"] - 1) * (s["S"] ** 2).sum() / s["N"] ** 2)


def se_cr2(d, cl):
    s = cluster_summary(d, cl)
    return np.sqrt((s["S"] ** 2 / (1 - s["sizes"] / s["N"])).sum() / s["N"] ** 2)


def satterthwaite_df(cl, rho=0.0):
    # Cluster-level form of the same quantity as _satterthwaite_df_explicit below
    # (which builds N x N matrices and is unusable beyond a few hundred units).
    # (I-H)a_g is constant within clusters: U[g, j] = c_g*[g==j] - n_g*c_g/N, and
    # Omega acts on such vectors as a per-cluster factor 1 + rho*(n_j - 1).
    cl = np.asarray(cl)
    N = len(cl)
    n = np.bincount(np.unique(cl, return_inverse=True)[1]).astype(float)
    c = (1 / N) * (1 - n / N) ** -0.5
    U = np.diag(c) - np.outer(n * c / N, np.ones(len(n)))
    M = (U * (n * (1 + rho * (n - 1)))) @ U.T
    return float(np.trace(M) ** 2 / np.sum(M * M))


def _satterthwaite_df_explicit(cl, rho=0.0):
    cl = np.asarray(cl)
    N = len(cl)
    _, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float)
    G = len(n)
    H = np.ones((N, N)) / N
    A = np.zeros((N, G))
    A[np.arange(N), inv] = (1 / N) * (1 - n[inv] / N) ** -0.5
    Om = np.where(inv[:, None] == inv[None, :], rho, 0.0)
    np.fill_diagonal(Om, 1.0)
    IH = np.eye(N) - H
    M = A.T @ IH @ Om @ IH @ A
    return float(np.trace(M) ** 2 / np.trace(M @ M))


def _t_result(theta, se, df, null):
    t = (theta - null) / se
    tc = stats.t.ppf(0.975, df)
    return dict(theta=theta, se=se, df=df, t=t, p=2 * stats.t.sf(abs(t), df),
                ci95=(theta - tc * se, theta + tc * se))


def cr2_test(d, cl, null=0.0, df="icc"):
    rho = icc_hat(d, cl) if df == "icc" else 0.0
    nu = satterthwaite_df(cl, rho)
    return _t_result(np.mean(d), se_cr2(d, cl), nu, null)


def cr1_test(d, cl, null=0.0):
    G = len(np.unique(cl))
    return _t_result(np.mean(d), se_cr1(d, cl), G - 1, null)


def _cr1_t(sums, n, N):
    """t statistics for rows of cluster-level sums of u (shape (draws, G))."""
    G = len(n)
    mean = sums.sum(1) / N
    S = sums - n[None, :] * mean[:, None]
    se = np.sqrt(G / (G - 1) * (S ** 2).sum(1)) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, mean / se, 0.0)


def wild_cluster_p(d, cl, null=0.0, B=9999, rng=None, weights="rademacher", exact_max_G=16):
    d, inv, n = _groups(d, cl)
    u = d - null
    N, G = len(u), len(n)
    s = np.bincount(inv, weights=u)
    t_obs = float(_cr1_t(s[None, :], n, N)[0])
    if weights == "rademacher" and G <= exact_max_G:
        bits = (np.arange(2 ** G)[:, None] >> np.arange(G)[None, :]) & 1
        ts = _cr1_t((2.0 * bits - 1) * s, n, N)
        return dict(t=t_obs, p=float(np.mean(np.abs(ts) >= abs(t_obs) - 1e-12)), B_or_exact="exact")
    rng = _rng(rng)
    vals = (np.array([-1.0, 1.0]) if weights == "rademacher" else
            np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)]))
    count, done = 0, 0
    while done < B:
        b = min(20000, B - done)
        w = rng.choice(vals, size=(b, G))
        count += int((np.abs(_cr1_t(w * s, n, N)) >= abs(t_obs)).sum())
        done += b
    return dict(t=t_obs, p=(1 + count) / (B + 1), B_or_exact=B)


def signflip_exact_p(d, null=0.0):
    u = np.asarray(d, float) - null
    n = len(u)
    assert n <= 20
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
    m = np.abs(((2.0 * bits - 1) * u).sum(1) / n)
    return float(np.mean(m >= abs(u.mean()) - 1e-12))


def means_t_test(d, cl, null=0.0):
    d, inv, n = _groups(d, cl)
    m = np.bincount(inv, weights=d) / n
    G = len(m)
    r = _t_result(m.mean(), m.std(ddof=1) / np.sqrt(G), G - 1, null)
    r["theta_means"] = r.pop("theta")
    return r


def pct_bootstrap(d, cl, B=4000, rng=None, scheme="cluster"):
    d, inv, n = _groups(d, cl)
    rng = _rng(rng)
    sums = np.bincount(inv, weights=d)
    out = np.empty(B)
    done = 0
    while done < B:
        b = min(20000, B - done)
        if scheme == "iid":
            out[done:done + b] = d[rng.integers(0, len(d), size=(b, len(d)))].mean(1)
        else:
            idx = rng.integers(0, len(n), size=(b, len(n)))
            out[done:done + b] = sums[idx].sum(1) / n[idx].sum(1)
        done += b
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(ci95=(lo, hi), se=out.std(ddof=1))


def paired_t_test(d, null=0.0):
    d = np.asarray(d, float)
    return _t_result(d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1, null)


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    rej = np.zeros(m, bool)
    for k, i in enumerate(np.argsort(p)):
        if p[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break
    return rej


def mde80(se, df=None):
    if df is None:
        return (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se
    return (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se
