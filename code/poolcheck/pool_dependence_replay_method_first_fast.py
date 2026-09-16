"""Same logic as pool_dependence_replay_method_first.py (claim-6 sensitivity check: rank with
scipy rankdata(method="ordinal"), scipy's equivalent of pandas rank(method="first"), ties broken
by a FIXED alphabetical model order), but fetches all git blobs in one `git cat-file --batch`
subprocess instead of one `git cat-file -p` subprocess per blob. Pure performance rewrite of the
diagnostic script -- not one of the critique's four Appendix A scripts -- to avoid ~19,000 process
spawns. Usage: python pool_dependence_replay_method_first_fast.py gift
By default replays only up to the snapshot commit the paper's numbers are pinned to
(b61c48aa3197a1a111e405c0e6f600128a1efee3, 2026-09-07) so the counts stay reproducible even as
the board keeps adding commits. Pass --unpinned to replay all the way to HEAD instead.
"""
import subprocess, csv, io, json, math, sys
import numpy as np
from collections import defaultdict
from scipy.stats import rankdata
REPO=sys.argv[1]; G=["git","-C",REPO]
SNAPSHOT_COMMIT="b61c48aa3197a1a111e405c0e6f600128a1efee3"
REV="HEAD" if "--unpinned" in sys.argv[2:] else SNAPSHOT_COMMIT
git=lambda *a: subprocess.run(G+list(a),capture_output=True,text=True).stdout
MASE="eval_metrics/MASE[0.5]"; CRPS="eval_metrics/mean_weighted_sum_quantile_loss"

commits=[l.split() for l in git("log","--reverse","--format=%H %ad","--date=short",REV,"--","results").splitlines()]
print(f"# replaying {REPO} up to {REV} ({len(commits)} commits on results)")

# Pass 1: collect, for every commit, the tree listing (blob sha -> path), and the set of all blob shas needed.
tree_by_commit={}
all_blobs=set()
for c,date in commits:
    entries=[]
    for line in git("ls-tree","-r",c,"results").splitlines():
        _,_,bid,path=line.split(None,3)
        if path.endswith("all_results.csv") or path.endswith("config.json"):
            entries.append((bid,path))
            all_blobs.add(bid)
    tree_by_commit[c]=entries

# Pass 2: one batched git cat-file --batch call for every blob we will ever need.
# Binary mode throughout: git's <size> in the batch header is a BYTE count, so slicing a
# decoded (text) string by that many characters drifts out of sync as soon as any blob
# contains multi-byte UTF-8. Decode each blob's bytes individually after slicing instead.
blob_input=("\n".join(all_blobs)+"\n").encode()
proc=subprocess.run(G+["cat-file","--batch"],input=blob_input,capture_output=True)
blob_content={}
out=proc.stdout
pos=0
for bid in all_blobs:
    nl=out.index(b"\n",pos)
    header=out[pos:nl]  # b"<sha> <type> <size>" or b"<sha> missing"
    parts=header.split()
    if len(parts)<3:
        pos=nl+1; blob_content[bid]=""; continue
    size=int(parts[2])
    start=nl+1
    blob_content[bid]=out[start:start+size].decode(errors="replace")
    pos=start+size+1  # skip trailing newline after content

def board(c):
    res=defaultdict(dict); rblob={}; cblob={}; leak={}
    for bid,path in tree_by_commit[c]:
        if path.endswith("all_results.csv"):
            for r in csv.DictReader(io.StringIO(blob_content[bid])):
                try:
                    m=r["model"]
                    res[m][r["dataset"]]=(float(r[MASE]),float(r.get(CRPS) or "nan"))
                    rblob[m]=bid
                except (KeyError,ValueError): pass
        elif path.endswith("config.json"):
            try:
                j=json.loads(blob_content[bid])
                cblob[j["model"]]=bid; leak[j["model"]]=j.get("testdata_leakage","No")=="Yes"
            except (ValueError,KeyError): pass
    return res,rblob,cblob,leak

per_config_ties = 0

def scores(res,leak,rule):
    global per_config_ties
    configs=sorted({d for m in res for d in res[m]}); ok=[m for m in res if m in leak and all(d in res[m] for d in configs)]
    if rule=="gmean":
        sn=next((m for m in res if m.lower() in ("seasonal_naive","seasonal naive")),None)
        if sn is None: return None,configs
        return {m:float(np.mean([math.log(res[m][d][0]/res[sn][d][0]) for d in configs])) for m in ok if m!=sn},configs
    k=0 if rule=="MASE_Rank" else 1; R=defaultdict(list)
    for d in configs:
        ids=sorted(m for m in res if d in res[m] and not math.isnan(res[m][d][k]))
        vals=[res[m][d][k] for m in ids]
        if len(set(vals))!=len(vals): per_config_ties += 1
        for m,r in zip(ids,rankdata(vals,method="ordinal")): R[m].append(r)
    return {m:float(np.mean(R[m])) for m in ok},configs

boards=[(date,)+board(c) for c,date in commits]
for rule in ("MASE_Rank","CRPS_Rank","gmean"):
    prev=None; steps=flip_steps=flips=0; ex=[]; score_ties=0
    for date,res,rb,cb,leak in boards:
        s,cf=scores(res,leak,rule)
        if s is None or len(s)<10: prev=None; continue
        disp=sorted(s,key=lambda m:(leak[m],s[m]))
        if prev and set(cf)==set(prev[4]):
            pdate,ps,prb,pcb,pcf,pdisp=prev; steps+=1
            inc=[m for m in pdisp[:10] if m in s and rb.get(m)==prb.get(m) and cb.get(m)==pcb.get(m)]
            for i,a in enumerate(inc):
                for b in inc[i+1:]:
                    if ps[a]==ps[b] or s[a]==s[b]: score_ties+=1
            fl=[(a,b) for i,a in enumerate(inc) for b in inc[i+1:] if (ps[a]-ps[b])*(s[a]-s[b])<0]
            if fl:
                flip_steps+=1; flips+=len(fl)
                new=sorted(m for m in res if rb.get(m)!=prb.get(m))
                ex.append((pdate,date,fl,new[:4]))
        prev=(date,s,rb,cb,cf,disp)
    print(f"\n[{rule}] (method=ordinal, fixed alpha order) transitions {steps}; transitions reversing unchanged top-10 entries {flip_steps}; reversed pairs {flips}; exact-tie comparisons among untouched top-10 pairs {score_ties}")
    for x in ex: print("   ",x)
print(f"\nper-config exact ties in raw MASE/CRPS values (any rule, any config): {per_config_ties}")
