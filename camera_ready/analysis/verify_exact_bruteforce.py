# Independent brute-force check of the exact sign-flip p-values: lists all 2^28 patterns bit by bit.
import sys, json, time, numpy as np
sys.path.insert(0, 'fmts/camera_ready')
from load_gift_pairs import top10_pair_diffs
def brute(d, cl):
    _, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float); s = np.bincount(inv, weights=d)
    G, N = len(n), n.sum(); Q, R = (s**2).sum(), (n**2).sum()
    def tstat(A, C):
        m = A / N
        se2 = G/(G-1) * (Q - 2*m*C + m*m*R) / N**2
        return np.where(se2 > 0, m/np.sqrt(np.maximum(se2, 1e-300)), 0.0)
    A0, C0 = s.sum(), (n*s).sum(); t0 = abs(float(tstat(np.array([A0]), np.array([C0]))[0]))
    cs = ct = 0; ch = 1 << 18
    ar = np.arange(G)
    for start in range(0, 1 << G, ch):
        k = np.arange(start, start + ch, dtype=np.int64)
        w = 2.0 * ((k[:, None] >> ar[None, :]) & 1) - 1
        A = w @ s; C = w @ (n*s)
        cs += int((np.abs(A) >= abs(A0) * (1 - 1e-9)).sum())
        ct += int((np.abs(tstat(A, C)) >= t0 * (1 - 1e-9)).sum())
    return cs / 2**G, ct / 2**G, t0
out = {}
for rule, pairs in (('MASE_Rank', [(8, 9), (0, 1)]), ('gmean', [(0, 9)]), ('CRPS_Rank', [(2, 7)])):
    names, D, cl = top10_pair_diffs(rule)
    for ij in pairs:
        t = time.time(); ps, pt, t0 = brute(np.asarray(D[ij], float), np.asarray(cl))
        out['%s|%d-%d' % (rule, ij[0]+1, ij[1]+1)] = dict(p_sum=ps, p_student=pt, t_obs=t0)
        print(rule, ij, ps, pt, t0, round(time.time()-t, 1), 's', flush=True)
json.dump(out, open('fmts/camera_ready/verify_exact_bruteforce.json', 'w'), indent=1)
