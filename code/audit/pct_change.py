"""Percentage change in geometric mean of scaled MASE, 512 vs documented
default, per model, with paired bootstrap CIs (B=4000, seed 20260914 --
same seed as result2_grid.py's paired differences). Run from audit/."""
import csv, json, math, random
sn = {r['dataset']: float(r['eval_metrics/MASE[0.5]']) for r in csv.DictReader(open('../snapshot_2026-09-07/results/seasonal_naive/all_results.csv'))}
cfgs = sorted(s['config'] for s in json.load(open('../selection/selected_configs.json'))['selected'])
def scaled(model, ctx):
    d = {r['dataset']: float(r['eval_metrics/MASE[0.5]']) for r in csv.DictReader(open(f'../grid_results_h100/results/{model}/ctx_{ctx}/all_results.csv'))}
    return {c: d[c]/sn[c] for c in cfgs}
def gm(vals): return math.exp(sum(math.log(v) for v in vals)/len(vals))
out = {}
for m in ['timesfm25', 'toto', 'moirai2_small', 'chronos_bolt_base']:
    a, b = scaled(m, 'default'), scaled(m, '512')
    rng = random.Random(20260914)
    reps = []
    for _ in range(4000):
        s = [rng.choice(cfgs) for _ in cfgs]
        reps.append(gm([b[c] for c in s]) / gm([a[c] for c in s]) - 1)
    reps.sort()
    out[m] = {"pct_change": round(100*(gm(b.values())/gm(a.values())-1), 1),
              "ci95": [round(100*reps[99], 1), round(100*reps[3899], 1)]}
json.dump(out, open('pct_change.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
