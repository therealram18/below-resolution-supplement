"""Resolution budget and statistical tiers for a GIFT-Eval snapshot (CPU only).
Usage: python resolution_budget.py SNAPSHOT_RESULTS_DIR [SELECTED_CONFIGS_JSON]
Prints, for MASE_Rank, CRPS_Rank and the geometric mean of seasonal-naive-scaled MASE:
top-tier membership, minimum detectable adjacent gap at 80% power, datasets needed to resolve
the median adjacent gap, and (optionally) the top-10 spread on a config subset."""
import os, sys, csv, json
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata
SNAP=sys.argv[1]; SEL=sys.argv[2] if len(sys.argv)>2 else None
MASE="eval_metrics/MASE[0.5]"; CRPS="eval_metrics/mean_weighted_sum_quantile_loss"
rows=defaultdict(dict); cfg={}
for root,dirs,files in os.walk(SNAP):
    if "all_results.csv" in files:
        for r in csv.DictReader(open(os.path.join(root,"all_results.csv"))):
            try: rows[r["model"]][r["dataset"]]=(float(r[MASE]),float(r[CRPS]))
            except (KeyError,ValueError): pass
    if "config.json" in files:
        try: j=json.load(open(os.path.join(root,"config.json"))); cfg[j["model"]]=j
        except (ValueError,KeyError): pass
configs=sorted({k for m in rows for k in rows[m]}); C=len(configs); names=sorted(rows)
def rank_matrix(k):
    R=np.full((len(names),C),np.nan)
    for j,c in enumerate(configs):
        ids=[i for i,m in enumerate(names) if c in rows[m]]
        R[ids,j]=rankdata([rows[names[i]][c][k] for i in ids],method="average")
    return R
elig=[i for i,m in enumerate(names) if m in cfg and len(rows[m])==C]
E=[names[i] for i in elig]; e={m:k for k,m in enumerate(E)}
Rm=rank_matrix(0)[elig]; Rc=rank_matrix(1)[elig]
mase=np.array([[rows[m][c][0] for c in configs] for m in E])
L=np.log(mase/mase[e["Seasonal_Naive"]])
leak=np.array([cfg[m].get("testdata_leakage","No")=="Yes" for m in E])
bases=sorted({c.split("/")[0] for c in configs}); bidx=np.array([bases.index(c.split("/")[0]) for c in configs])
B=4000; rng=np.random.default_rng(20260907); iid=rng.integers(0,C,size=(B,C)); W=np.zeros((B,C))
for b in range(B): W[b]=np.bincount(rng.integers(0,len(bases),size=len(bases)),minlength=len(bases))[bidx]
boot=lambda d:(d[iid].mean(1),(W*d).sum(1)/W.sum(1))
lo=lambda x: np.percentile(x,2.5)
def separated(X,a,b):
    it,cl=boot(X[b]-X[a]); return lo(it)>0 and lo(cl)>0
T=list(np.lexsort((Rm.mean(1),leak)))[:10]
print("displayed top-10:",[E[i] for i in T])
for X,label in ((Rm,"MASE_Rank"),(Rc,"CRPS_Rank"),(L,"log gmean scaled MASE")):
    od=list(np.lexsort((X.mean(1),leak))); Tr=od[:10]   # adjacency follows this rule's own displayed order
    tier=[od[0]]+[i for i in od[1:40] if not separated(X,od[0],i)]
    ses=[];gaps=[];need=[]
    for a,b in zip(Tr,Tr[1:]):
        d=X[b]-X[a]; ses.append(boot(d)[1].std()); gaps.append(abs(d.mean()))
        need.append(len(bases)*(2.8*ses[-1]/gaps[-1])**2 if gaps[-1]>0 else np.inf)
    mde=2.8*np.median(ses)
    n45=sum(1 for x,a in enumerate(Tr) for b in Tr[x+1:] if abs((X[b]-X[a]).mean())>mde)
    print(f"\n[{label}] top tier ({len(tier)}): {[E[i] for i in tier]}")
    print(f"[{label}] MDE80 with {len(bases)} base datasets: {mde:.4f}; median adjacent gap {np.median(gaps):.4f}; "
          f"base datasets needed for the median adjacent gap: {np.median(need):.0f} ({np.median(need)/len(bases):.0f}x); top-10 pairs above MDE: {n45}/45")
if SEL:
    sel=json.load(open(SEL)); lst=sel if isinstance(sel,list) else sel.get("selected",sel.get("configs",[]))
    sc=[x if isinstance(x,str) else (x.get("ds_config") or x.get("config") or x.get("name")) for x in lst]
    si=[configs.index(c) for c in sc if c in configs]
    r12=Rm[:,si].mean(1); g12=np.exp(L[:,si].mean(1))
    print(f"\n[subset of {len(si)} configs] top-10 MASE_Rank span {r12[T].max()-r12[T].min():.2f} places; top-10 gmean relative spread {100*(g12[T].max()/g12[T].min()-1):.1f}%")
