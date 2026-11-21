"""Run with the chronos_bolt_base python after refcheck.R and refcheck_wild.py. Writes refcheck_results.json."""
import json, os, sys, numpy as np, pandas as pd
CR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, CR)
import vernier_stats as vs
from load_gift_pairs import top10_pair_diffs
_, diffs, cl = top10_pair_diffs('MASE_Rank')
o = pd.read_csv(f'{CR}/_refenv/ours.csv', dtype={'pair': str}).set_index('pair')
r = pd.read_csv(f'{CR}/_refenv/ref_R.csv', dtype={'pair': str}).set_index('pair').loc[o.index]
w = pd.read_csv(f'{CR}/_refenv/wild_py.json'.replace('.json', '.json')) if False else pd.DataFrame(json.load(open(f'{CR}/_refenv/wild_py.json'))).set_index('pair').loc[o.index]
rel = lambda a, b: np.abs(a - b) / np.abs(b)
R = {}
# a. clubSandwich
se_rel, df_rel = rel(o.se_cr2, r.club_se), rel(o.df_indep, r.club_df)
R['a_clubSandwich'] = dict(max_rel_se=se_rel.max(), max_rel_df=df_rel.max(), pass_=bool((se_rel < 1e-8).all() and (df_rel < 1e-6).all()),
                           max_rel_p=rel(o.p_indep, r.club_p).max())
# b(i). dfadjust: se equals ours; correlation estimate converted to ours
sig, rik = r.dfa_sig.values, r.dfa_rho_ik.values
corr = np.where(sig > 0, rik / (sig + rik), 1.0)          # IK covariance sig*I + rho*J  ->  correlation rho/(sig+rho); sig floored at 0 -> 1
corr0 = np.where(r.dfa0_sig.values > 0, np.maximum(rik, 0) / (r.dfa0_sig.values + np.maximum(rik, 0) + 0 * rik), 1.0)
ours_given = []; ours_given0 = []
for k, p in enumerate(o.index):
    i, j = map(int, p.split('-')); d = diffs[(i - 1, j - 1)]
    ours_given.append(vs.satterthwaite_df(cl, corr[k]))
    s0 = r.dfa0_sig.values[k]; rk0 = r.dfa0_rho_ik.values[k]
    ours_given0.append(vs.satterthwaite_df(cl, 1.0 if s0 <= 0 else rk0 / (s0 + rk0)))
ours_given = np.array(ours_given); ours_given0 = np.array(ours_given0)
bi = rel(ours_given, r.dfa_df.values); bi0 = rel(ours_given0, r.dfa0_df.values)
R['b_dfadjust'] = dict(se_vs_ours_max_rel=rel(r.dfa_se, o.se_cr2).max(),
    i_max_rel_df=bi.max(), i_pass=bool((bi < 1e-6).all()), i_rho0_max_rel_df=bi0.max(), i_rho0_pass=bool((bi0 < 1e-6).all()),
    n_sig_floored=int((sig <= 0).sum()), n_rho_negative=int((rik < 0).sum()),
    rho_ik_range=[float(rik.min()), float(rik.max())], corr_range=[float(corr.min()), float(corr.max())],
    ours_icc_range=[float(o.icc_hat.min()), float(o.icc_hat.max())])
def counts(p):
    p = np.asarray(p); nb = np.array([int(a.split('-')[1]) - int(a.split('-')[0]) == 1 for a in o.index])
    return dict(neigh_of_9=int((p[nb] < .05).sum()), all_of_45=int((p < .05).sum()),
                holm_9=int(vs.holm(p[nb]).sum()), holm_45=int(vs.holm(p).sum()))
R['b_counts'] = dict(ours_icc=counts(o.p_icc), ours_indep=counts(o.p_indep), dfadjust_shipped=counts(r.dfa_p),
                     dfadjust_rho_floored=counts(r.dfa0_p))
R['b_diffs_vs_ours_icc'] = dict(max_abs_df=float((r.dfa_df - o.df_icc).abs().max()), max_abs_p=float((r.dfa_p - o.p_icc).abs().max()),
                                max_abs_df_rho0=float((r.dfa0_df - o.df_icc).abs().max()), max_abs_p_rho0=float((r.dfa0_p - o.p_icc).abs().max()))
R['b_dfadjust_noIK_df_vs_clubSandwich_max_rel'] = float(rel(r.dfa_noik_df, r.club_df).max())
# c. wildboottest
R['c_wildboottest'] = dict(n_pass_vs_stored=int(w.pass_stored.sum()), max_p_diff_vs_stored=float(w.p_diff.max()),
    worst_ratio_to_tol=float((w.p_diff / w.p_tol).max()), max_abs_t_diff=float(w.t_absdiff.max()))
if 'pass_exact' in w:
    R['c_wildboottest'].update(n_pass_vs_exact_student=int(w.pass_exact.sum()), max_p_diff_vs_exact=float(w.diff_exact.max()),
                               worst_ratio_to_tol_exact=float((w.diff_exact / w.tol_exact).max()))
pairs = pd.concat([o, r.drop(columns='theta'), w.add_prefix('wild_')], axis=1)
R['pairs'] = pairs.reset_index().to_dict('records')
json.dump(R, open(f'{CR}/refcheck_results.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
print(json.dumps({k: v for k, v in R.items() if k != 'pairs'}, indent=1, default=str))
