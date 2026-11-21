"""Main-session recheck of one post-snapshot reversal on GIFT-Eval: mean rank on MASE (ties
averaged) of limix_moe and EXAONE-Forecast-Agent before and after HART-36M arrived (commit
58e77d5c, 2026-10-29), read straight from the board repository's history."""
import io, json, os, subprocess
import pandas as pd
from scipy.stats import rankdata
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boards/pin_2026-11-13/gift")
def git(*a): return subprocess.run(["git", "-C", G, *a], capture_output=True, text=True, check=True).stdout
def state(c):
    files = git("ls-tree", "-r", "--name-only", c, "results/").split("\n")
    res = {f.split("/")[1] for f in files if f.endswith("/all_results.csv")}
    cfg = {f.split("/")[1] for f in files if f.endswith("/config.json")}
    out = {}
    for m in sorted(res & cfg):
        df = pd.read_csv(io.StringIO(git("show", f"{c}:results/{m}/all_results.csv")))
        col = [x for x in df.columns if "MASE" in x][0]
        s = df.drop_duplicates("dataset").set_index("dataset")[col]
        try: name = json.loads(git("show", f"{c}:results/{m}/config.json")).get("model", m)
        except Exception: name = m
        if len(s) == 97 and s.notna().all(): out[name] = s
    return pd.DataFrame(out)
c = "58e77d5c"; prev = git("log", "-2", "--format=%h", c, "--", "results/").split()[1]
res = {}
for lab, cc in (("before", prev), ("after", c)):
    M = state(cc); r = pd.DataFrame(rankdata(M.values, axis=1, method="average"), index=M.index, columns=M.columns).mean()
    res[lab] = dict(commit=cc, entries=int(M.shape[1]), limix_moe=round(float(r["limix_moe"]), 4), EXAONE=round(float(r["EXAONE-Forecast-Agent"]), 4))
print(res); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_pin2_reversal.json"), "w"), indent=1)
