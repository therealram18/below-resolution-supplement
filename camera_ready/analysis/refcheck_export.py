import json, os, sys, numpy as np, pandas as pd
CR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, CR)
import vernier_stats as vs
from load_gift_pairs import top10_pair_diffs
names, diffs, cl = top10_pair_diffs('MASE_Rank')
stored = json.load(open(f'{CR}/retest_gift.json'))['pairs']['MASE_Rank']
rows, ours = [], []
for (i, j), d in diffs.items():
    s = stored[len(ours)]
    assert s['pair'] == f'{i+1}-{j+1}', (s['pair'], i, j)
    pair = f'{i+1}-{j+1}'
    rows += [(pair, k, int(cl[k]), float(d[k])) for k in range(len(d))]
    se = vs.se_cr2(d, cl); rho = vs.icc_hat(d, cl)
    d0 = vs.satterthwaite_df(cl, 0.0); d1 = vs.satterthwaite_df(cl, rho)
    r0 = vs.cr2_test(d, cl, df='indep'); r1 = vs.cr2_test(d, cl)
    chk = [(se, s['CR2-icc']['se']), (se, s['CR2-indep']['se']), (d0, s['CR2-indep']['df']), (d1, s['CR2-icc']['df']),
           (r0['p'], s['CR2-indep']['p']), (r1['p'], s['CR2-icc']['p']), (rho, s['icc_hat'])]
    assert all(abs(a - b) < 1e-10 for a, b in chk), pair
    ours.append(dict(pair=pair, theta=float(d.mean()), se_cr2=se, df_indep=d0, p_indep=r0['p'], icc_hat=rho,
                     df_icc=d1, p_icc=r1['p'], wild_p=s['WCB-Rad']['p'], wild_t=s['WCB-Rad']['t']))
os.makedirs(f'{CR}/_refenv', exist_ok=True)
pd.DataFrame(rows, columns=['pair', 'config_index', 'dataset', 'd']).to_csv(f'{CR}/_refenv/pairs_mase_rank.csv', index=False)
pd.DataFrame(ours).to_csv(f'{CR}/_refenv/ours.csv', index=False)
print(len(rows), len(ours), 'stored values match to 1e-10')
