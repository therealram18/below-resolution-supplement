"""Replay a GIFT-Eval HF-space clone commit by commit and count order reversals between top-10 entries
whose own results AND config did not change (pure effect of other submissions). CPU only.
Usage: git clone https://huggingface.co/spaces/Salesforce/GIFT-Eval gift && python pool_dependence_replay.py gift
By default replays only up to the snapshot commit the paper's numbers are pinned to
(b61c48aa3197a1a111e405c0e6f600128a1efee3, 2026-09-07) so the counts stay reproducible even as
the board keeps adding commits. Pass --unpinned to replay all the way to HEAD instead."""
import subprocess, csv, io, json, math, sys
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata
REPO=sys.argv[1]; G=["git","-C",REPO]
SNAPSHOT_COMMIT="b61c48aa3197a1a111e405c0e6f600128a1efee3"
REV="HEAD" if "--unpinned" in sys.argv[2:] else SNAPSHOT_COMMIT
git=lambda *a: subprocess.run(G+list(a),capture_output=True,text=True).stdout
MASE="eval_metrics/MASE[0.5]"; CRPS="eval_metrics/mean_weighted_sum_quantile_loss"
cache={}
def load_csv(bid):
    if bid not in cache:
        out=defaultdict(dict)
        for r in csv.DictReader(io.StringIO(git("cat-file","-p",bid))):
            try: out[r["model"]][r["dataset"]]=(float(r[MASE]),float(r.get(CRPS) or "nan"))
            except (KeyError,ValueError): pass
        cache[bid]=out
    return cache[bid]
def board(c):
    res=defaultdict(dict); rblob={}; cblob={}; leak={}
    for line in git("ls-tree","-r",c,"results").splitlines():
        _,_,bid,path=line.split(None,3)
        if path.endswith("all_results.csv"):
            for m,d in load_csv(bid).items(): res[m].update(d); rblob[m]=bid
        elif path.endswith("config.json"):
            try: j=json.loads(git("cat-file","-p",bid)); cblob[j["model"]]=bid; leak[j["model"]]=j.get("testdata_leakage","No")=="Yes"
            except (ValueError,KeyError): pass
    return res,rblob,cblob,leak
def scores(res,leak,rule):
    configs=sorted({d for m in res for d in res[m]}); ok=[m for m in res if m in leak and all(d in res[m] for d in configs)]
    if rule=="gmean":
        sn=next((m for m in res if m.lower() in ("seasonal_naive","seasonal naive")),None)
        if sn is None: return None,configs
        return {m:float(np.mean([math.log(res[m][d][0]/res[sn][d][0]) for d in configs])) for m in ok if m!=sn},configs
    k=0 if rule=="MASE_Rank" else 1; R=defaultdict(list)
    for d in configs:
        ids=[m for m in res if d in res[m] and not math.isnan(res[m][d][k])]
        for m,r in zip(ids,rankdata([res[m][d][k] for m in ids],method="average")): R[m].append(r)
    return {m:float(np.mean(R[m])) for m in ok},configs
commits=[l.split() for l in git("log","--reverse","--format=%H %ad","--date=short",REV,"--","results").splitlines()]
print(f"# replaying {REPO} up to {REV} ({len(commits)} commits on results)")
boards=[(date,)+board(c) for c,date in commits]
for rule in ("MASE_Rank","CRPS_Rank","gmean"):
    prev=None; steps=flip_steps=flips=0; ex=[]
    for date,res,rb,cb,leak in boards:
        s,cf=scores(res,leak,rule)
        if s is None or len(s)<10: prev=None; continue
        disp=sorted(s,key=lambda m:(leak[m],s[m]))
        if prev and set(cf)==set(prev[4]):
            pdate,ps,prb,pcb,pcf,pdisp=prev; steps+=1
            inc=[m for m in pdisp[:10] if m in s and rb.get(m)==prb.get(m) and cb.get(m)==pcb.get(m)]
            fl=[(a,b) for i,a in enumerate(inc) for b in inc[i+1:] if (ps[a]-ps[b])*(s[a]-s[b])<0]
            if fl:
                flip_steps+=1; flips+=len(fl)
                new=sorted(m for m in res if rb.get(m)!=prb.get(m))
                ex.append((pdate,date,fl,new[:4]))
        prev=(date,s,rb,cb,cf,disp)
    print(f"\n[{rule}] transitions {steps}; transitions reversing unchanged top-10 entries {flip_steps}; reversed pairs {flips}")
    for x in ex: print("   ",x)
