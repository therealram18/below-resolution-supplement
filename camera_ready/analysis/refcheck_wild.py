"""Run with _refenv/py/bin/python. Python wildboottest vs stored WCB-Rad p (and exact_gift.json if present)."""
import json, os, numpy as np, pandas as pd, statsmodels.api as sm, warnings
from wildboottest.wildboottest import wildboottest
CR = os.path.dirname(os.path.abspath(__file__)); B = 99999
df = pd.read_csv(f'{CR}/_refenv/pairs_mase_rank.csv'); ours = pd.read_csv(f'{CR}/_refenv/ours.csv').set_index('pair')
ex = json.load(open(f'{CR}/exact_gift.json'))['MASE_Rank']['pairs'] if os.path.exists(f'{CR}/exact_gift.json') else None
out = []
for pair, g in df.groupby('pair', sort=False):
    m = sm.OLS(g.d.values, np.ones((len(g), 1)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = wildboottest(m, cluster=g.dataset.values, B=B, weights_type='rademacher', impose_null=True,
                         bootstrap_type='11', seed=20261111, show=False)
    p, t = float(r['p-value'].iloc[0]), float(np.ravel(r['statistic'].iloc[0])[0])
    ps, tv = ours.loc[pair, 'wild_p'], ours.loc[pair, 'wild_t']
    row = dict(pair=pair, p_ref=p, t_ref=t, p_stored=ps, t_ours=tv, t_absdiff=abs(t - tv),
               p_diff=abs(p - ps), p_tol=4 * np.sqrt(2 * ps * (1 - ps) / B))
    row['pass_stored'] = bool(row['p_diff'] < row['p_tol'])
    if ex:
        pe = ex[pair]['p_exact_student']; row.update(p_exact_student=pe, diff_exact=abs(p - pe),
            tol_exact=4 * np.sqrt(pe * (1 - pe) / B)); row['pass_exact'] = bool(row['diff_exact'] < row['tol_exact'])
    out.append(row)
json.dump(out, open(f'{CR}/_refenv/wild_py.json', 'w'), indent=1)
o = pd.DataFrame(out)
print('pass vs stored', o.pass_stored.sum(), 'maxdiff', o.p_diff.max(), 'max t diff', o.t_absdiff.max())
if ex: print('pass vs exact', o.pass_exact.sum(), 'maxdiff', o.diff_exact.max())
