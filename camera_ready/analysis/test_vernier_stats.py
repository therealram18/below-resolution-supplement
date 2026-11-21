import itertools
import time
import numpy as np
import scipy.linalg
from scipy import stats
import vernier_stats as v

LAYOUT = [1] * 11 + [2] + [3] * 4 + [4] * 5 + [6] + [7] * 2 + [8] * 4
CL97 = np.repeat(np.arange(28), LAYOUT)
assert len(CL97) == 97

# satterthwaite df
for rho, want in [(0, 17.07), (0.1, 14.39), (0.3, 12.30), (0.5, 11.44), (0.7, 10.98)]:
    got = v.satterthwaite_df(CL97, rho)
    assert abs(got - want) < 0.01, (rho, got, want)
print("OK satterthwaite_df matches 17.07/14.39/12.30/11.44/10.98")
assert abs(v.satterthwaite_df(np.repeat(np.arange(28), 4)) - 27) < 1e-8
assert abs(v.satterthwaite_df(np.arange(12)) - 11) < 1e-8
print("OK df = 27 (28x4) and 11 (12 singletons)")

# se_cr2 brute force
rng = np.random.default_rng(0)
d = rng.normal(size=97) + rng.normal(size=28)[CL97]
N = 97
e = (np.eye(N) - np.ones((N, N)) / N) @ d
tot = 0.0
for g in range(28):
    ix = np.where(CL97 == g)[0]
    Hgg = np.ones((len(ix), len(ix))) / N
    Ag = scipy.linalg.fractional_matrix_power(np.eye(len(ix)) - Hgg, -0.5).real
    tot += (np.ones(len(ix)) / N @ Ag @ e[ix]) ** 2
assert abs(np.sqrt(tot) - v.se_cr2(d, CL97)) < 1e-10
print("OK se_cr2 equals brute force to 1e-10")

# wild cluster p: exact vs sampled (G=10)
cl10 = np.repeat(np.arange(10), [1, 2, 3, 4, 2, 1, 5, 3, 2, 4])
d10 = rng.normal(0.4, 1, size=len(cl10)) + rng.normal(0, 0.5, size=10)[cl10]
ex = v.wild_cluster_p(d10, cl10)
sm = v.wild_cluster_p(d10, cl10, B=200000, rng=np.random.default_rng(1), exact_max_G=0)
assert ex["B_or_exact"] == "exact" and abs(ex["p"] - sm["p"]) < 0.005, (ex, sm)
print("OK wild exact vs sampled", round(ex["p"], 4), round(sm["p"], 4))
# singleton: direct studentized sign flip
du = rng.normal(0.5, 1, size=10)
def stud(u):
    m = u.mean(); return m / np.sqrt(10 / 9 * ((u - m) ** 2).sum() / 100)
t0 = abs(stud(du))
cnt = sum(abs(stud(np.array(s) * du)) >= t0 - 1e-12 for s in itertools.product([-1, 1], repeat=10))
assert abs(v.wild_cluster_p(du, np.arange(10))["p"] - cnt / 1024) < 1e-12
print("OK wild (singletons) equals direct studentized sign-flip")

# sign-flip
assert v.signflip_exact_p(np.array([1.0, 2, 3])) == 0.25
print("OK signflip d=[1,2,3] p=0.25")

# means t, paired t
dm = rng.normal(0.3, 1, 97)
cm = np.array([dm[CL97 == g].mean() for g in range(28)])
r = v.means_t_test(dm, CL97); ref = stats.ttest_1samp(cm, 0)
assert abs(r["t"] - ref.statistic) < 1e-10 and abs(r["p"] - ref.pvalue) < 1e-12 and abs(r["theta_means"] - cm.mean()) < 1e-12
dp = rng.normal(0.3, 1, 12); r = v.paired_t_test(dp); ref = stats.ttest_1samp(dp, 0)
assert abs(r["t"] - ref.statistic) < 1e-10 and abs(r["p"] - ref.pvalue) < 1e-12
print("OK means_t_test and paired_t_test match scipy")

# holm: p=[.01,.04,.03,.005], m=4: sorted .005<=.0125 y, .01<=.0167 y, .03<=.025 n -> stop
assert list(v.holm([0.01, 0.04, 0.03, 0.005])) == [True, False, False, True]
assert not v.holm([0.02, 0.02]).any() or True
assert list(v.holm([0.02, 0.02])) == [True, True]  # .02<=.025, then .02<=.05
print("OK holm hand example")

# pct bootstrap cluster(singletons) vs iid
d12 = rng.normal(0.5, 1, 60)
a = v.pct_bootstrap(d12, np.arange(60), B=100000, rng=np.random.default_rng(2), scheme="cluster")
b = v.pct_bootstrap(d12, np.arange(60), B=100000, rng=np.random.default_rng(3), scheme="iid")
assert all(abs(x - y) < 0.02 for x, y in zip(a["ci95"], b["ci95"])), (a, b)
assert abs(a["se"] / (d12.std() / np.sqrt(60)) - 1) < 0.03
print("OK pct_bootstrap cluster(singletons) ~ iid; se within 3%")

# other small checks
cs = v.cluster_summary(d, CL97); assert cs["N"] == 97 and cs["G"] == 28 and abs(cs["S"].sum()) < 1e-9
assert v.icc_hat(d, np.arange(97)) == 0.0 and 0 <= v.icc_hat(d, CL97) <= 1
assert abs(v.mde80(1.0) - 2.8016) < 1e-3 and v.mde80(1.0, 10) > v.mde80(1.0)
print("OK summary, icc, mde80 sanity")

# real-data regression
import load_gift_pairs as g
names, diffs, cl = g.top10_pair_diffs("MASE_Rank")
dd = diffs[(8, 9)]
assert abs(dd.mean() - 1.33) < 0.01, dd.mean()
c = v.pct_bootstrap(dd, cl, B=50000, rng=np.random.default_rng(20260907), scheme="cluster")["ci95"]
i = v.pct_bootstrap(dd, cl, B=50000, rng=np.random.default_rng(20260908), scheme="iid")["ci95"]
print("regression theta=%.4f clustered=[%.3f,%.3f] iid=[%.3f,%.3f]" % (dd.mean(), *c, *i))
assert abs(c[0] - 0.13) < 0.06 and abs(c[1] - 2.41) < 0.06
assert abs(i[0] + 0.05) < 0.06 and abs(i[1] - 2.70) < 0.06
for rule in ("CRPS_Rank", "gmean"):
    n2, d2, cl2 = g.top10_pair_diffs(rule); assert len(d2) == 45 and len(cl2) == 97
print("OK real-data regression: pair 9-10 MASE_Rank reproduces published numbers")

t = time.time(); w = v.wild_cluster_p(dd, cl, B=99999, rng=np.random.default_rng(5))
print("OK wild B=99999 N=97 G=28: p=%.4f in %.2fs" % (w["p"], time.time() - t))


# cluster-level degrees of freedom equal the explicit N x N computation
import numpy as _np, vernier_stats as _vs
_rng = _np.random.default_rng(3)
for _sizes in ([1]*11 + [2] + [3]*4 + [4]*5 + [6] + [7]*2 + [8]*4, list(_rng.integers(1, 9, 39)), [1]*12, [5]*9):
    _cl = _np.repeat(_np.arange(len(_sizes)), _sizes)
    _rng.shuffle(_cl)
    for _rho in (0.0, 0.1, 0.3, 0.5, 0.7):
        assert abs(_vs.satterthwaite_df(_cl, _rho) - _vs._satterthwaite_df_explicit(_cl, _rho)) < 1e-8
print("OK cluster-level satterthwaite_df equals the explicit N x N computation")
