"""Pool perturbations on a GIFT-Eval snapshot: does removing one entry, or adding a near-copy of one entry,
reorder the displayed MASE_Rank top-10? Optional: reconstruct an earlier board by removing named entries.
Usage: python pool_perturbations.py SNAPSHOT_RESULTS_DIR [ENTRY_TO_REMOVE ...]"""
import os, sys, csv, json
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata
SNAP=sys.argv[1]; REMOVE=sys.argv[2:]
MASE="eval_metrics/MASE[0.5]"
rows=defaultdict(dict); cfg={}
for root,dirs,files in os.walk(SNAP):
    if "all_results.csv" in files:
        for r in csv.DictReader(open(os.path.join(root,"all_results.csv"))):
            try: rows[r["model"]][r["dataset"]]=float(r[MASE])
            except (KeyError,ValueError): pass
    if "config.json" in files:
        try: j=json.load(open(os.path.join(root,"config.json"))); cfg[j["model"]]=j
        except (ValueError,KeyError): pass
configs=sorted({k for m in rows for k in rows[m]}); names=sorted(rows)
V=np.array([[rows[m].get(c,np.nan) for c in configs] for m in names])
leak={m:cfg.get(m,{}).get("testdata_leakage","No")=="Yes" for m in names}
display=[m for m in names if m in cfg and not np.isnan(V[names.index(m)]).any()]
def mean_rank(pool, extra=None):
    X=V[pool] if extra is None else np.vstack([V[pool],extra])
    R=np.full(X.shape,np.nan)
    for j in range(X.shape[1]):
        ok=~np.isnan(X[:,j]); R[ok,j]=rankdata(X[ok,j],method="average")
    return {names[p]:R[k].mean() for k,p in enumerate(pool)}
def top10(pos): return sorted([m for m in display if m in pos],key=lambda m:(leak[m],pos[m]))[:10]
full=list(range(len(names))); base=mean_rank(full); T=top10(base)
print("displayed top-10:",T)
if REMOVE:
    pool=[i for i in full if names[i] not in REMOVE]; pos=mean_rank(pool); keep=[m for m in T if m not in REMOVE]
    print("order of the remaining top-10 entries without",REMOVE,":",sorted(keep,key=lambda m:pos[m]))
    print("order with them:",sorted(keep,key=lambda m:base[m]))
reorder_rm=0
for x in [i for i in full if names[i] not in T]:
    pos=mean_rank([i for i in full if i!=x])
    reorder_rm+= sorted(T,key=lambda m:pos[m])!=T
reorder_add=0
for m in display:
    pos=mean_rank(full,extra=V[names.index(m)]*(1+1e-6))
    reorder_add+= sorted(T,key=lambda q:pos[q])!=T
print(f"removing one non-top-10 entry reorders the top-10 in {reorder_rm} of {len(full)-10} cases")
print(f"adding a near-copy of one displayed entry reorders the top-10 in {reorder_add} of {len(display)} cases")
