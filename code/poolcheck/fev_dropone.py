"""Drop-one pool check on fev-bench: does removing any single entry from the
pool reverse the order of any pair among the displayed (win-rate) top-10?
Win rate per the board: for each task, fraction of other models with worse
relative error (ties 0.5), averaged over tasks. Usage: python fev_dropone.py pivot_SQL.csv"""
import csv, sys
import numpy as np
rows = list(csv.reader(open(sys.argv[1])))
models = rows[0][1:]
E = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
mi = {m: i for i, m in enumerate(models)}
Rel = np.clip(E / E[:, [mi["Seasonal Naive"]]], 0.01, 100)
def win_rates(idx):
    A = Rel[:, idx]
    W = (A[:, :, None] < A[:, None, :]) + 0.5 * (A[:, :, None] == A[:, None, :])
    W = W.astype(float)
    n = len(idx)
    W[:, np.arange(n), np.arange(n)] = np.nan
    return np.nanmean(np.nanmean(W, axis=2), axis=0) * 100
full = list(range(len(models)))
wr = win_rates(full)
T = [full[i] for i in np.argsort(-wr)[:10]]
print("top-10 by win rate:", [models[i] for i in T])
reversals = 0
for drop in [i for i in full if i not in T]:
    keep = [i for i in full if i != drop]
    w = dict(zip(keep, win_rates(keep)))
    for a_pos, a in enumerate(T):
        for b in T[a_pos+1:]:
            if w[a] < w[b]:
                reversals += 1
                print("REVERSED:", models[a], "vs", models[b], "when dropping", models[drop])
print(f"dropping any one of {len(full)-10} non-top-10 entries: {reversals} reversed top-10 pairs")
