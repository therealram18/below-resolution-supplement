import sys, os, csv, io, subprocess
sys.argv=['x']; 
REPO=os.path.abspath("boards/pin_2026-11-13/gift")
def git(*a): return subprocess.run(["git","-C",REPO]+list(a),capture_output=True,text=True).stdout
def load(c):
    out={}
    for line in git("ls-tree","-r",c,"results").splitlines():
        _,_,bid,p=line.split(None,3)
        if p.endswith("all_results.csv"):
            for r in csv.DictReader(io.StringIO(git("cat-file","-p",bid))):
                out.setdefault(r["model"],{})[r["dataset"]]=r["eval_metrics/MASE[0.5]"]
    return out
a=load("5c42c06f^"); b=load("5c42c06f")
ch=[m for m in b if m in a and a[m]!=b[m]]
print("models with MASE differing at 5c42c06f:",len(ch),ch)
for m in ch[:30]:
    d=[(k,a[m].get(k),b[m].get(k)) for k in b[m] if a[m].get(k)!=b[m].get(k)]
    print(m,len(d),d[:1])
print("only in before:",[m for m in a if m not in b]," only in after:",[m for m in b if m not in a])
