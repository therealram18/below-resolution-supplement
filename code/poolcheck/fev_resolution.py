"""Replicate the fev-bench leaderboard from its published pivot table and measure adjacent-gap resolution.
Usage: python fev_resolution.py pivot_SQL.csv leaderboard_SQL.csv"""
import csv, sys
import numpy as np
P,LB=sys.argv[1],sys.argv[2]
rows=list(csv.reader(open(P))); models=rows[0][1:]; tasks=[r[0] for r in rows[1:]]
E=np.array([[float(x) for x in r[1:]] for r in rows[1:]]); mi={m:i for i,m in enumerate(models)}
Rel=np.clip(E/E[:,[mi["Seasonal Naive"]]],0.01,100)
A=Rel; Wn=(A[:,:,None]<A[:,None,:])+0.5*(A[:,:,None]==A[:,None,:]); Wn=Wn.astype(float)
n=len(models); Wn[:,np.arange(n),np.arange(n)]=np.nan; WT=np.nanmean(Wn,axis=2)*100
wr=WT.mean(0); sk=(1-np.exp(np.log(Rel).mean(0)))*100
off={r["model_name"]:(float(r["win_rate"]),float(r["skill_score"])) for r in csv.DictReader(open(LB))}
print("max |win rate - published|:",round(max(abs(wr[mi[m]]-v[0]) for m,v in off.items() if m in mi),6),
      "| max |skill - published|:",round(max(abs(sk[mi[m]]-v[1]) for m,v in off.items() if m in mi),6))
T=list(np.argsort(-wr))[:10]; print("top-10 by win rate:",[models[i] for i in T])
fam=[t.rsplit("_",1)[0] for t in tasks]; fams=sorted(set(fam)); fidx=np.array([fams.index(f) for f in fam])
B=4000; rng=np.random.default_rng(20260907); N=len(tasks); iid=rng.integers(0,N,size=(B,N)); W=np.zeros((B,N))
for b in range(B): W[b]=np.bincount(rng.integers(0,len(fams),size=len(fams)),minlength=len(fams))[fidx]
boot=lambda d:(d[iid].mean(1),(W*d).sum(1)/W.sum(1))
for X,label in ((WT,"win rate (displayed sort)"),(-np.log(Rel),"skill (log gmean)")):
    surv=0; ses=[]; gaps=[]
    for a,b in zip(T,T[1:]):
        d=X[:,a]-X[:,b]; it,cl=boot(d); surv+=np.percentile(it,2.5)>0 and np.percentile(cl,2.5)>0
        ses.append(cl.std()); gaps.append(abs(d.mean()))
    need=[len(fams)*(2.8*s/g)**2 for s,g in zip(ses,gaps) if g>0]
    print(f"[{label}] adjacent gaps surviving both schemes: {surv}/9; MDE80 {2.8*np.median(ses):.3f}; median adjacent gap {np.median(gaps):.3f}; "
          f"dataset families needed {np.median(need):.0f} vs {len(fams)} ({np.median(need)/len(fams):.1f}x)")
