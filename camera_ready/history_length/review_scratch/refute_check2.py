"""Reading 2: d_c per model, arms with equal history, run_meta consistency, retries."""
import csv, json, math, glob, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "fmts/camera_ready/gpu_round/review_scratch")
exec(open("fmts/camera_ready/gpu_round/review_scratch/refute_check.py").read().split("# ---------- Reading 1 ----------")[0])
MODELS = ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto__A__seed42"]
D = {}
for v in MODELS:
    a, b = per_config(v, "default"), per_config(v, "512")
    d = {}
    for c in elig:
        if c not in a or (c not in b and not (v.startswith("toto") and c == "m4_daily/D/short")):
            print("missing", v, c); continue
        x = float(a[c][0]["eval_metrics/MASE[0.5]"])
        y = x if c not in b else float(b[c][0]["eval_metrics/MASE[0.5]"])
        if v.startswith("toto") and c == "m4_daily/D/short": y = x
        d[c] = math.log(y / x)
    D[v] = d
    cu = Counter((a[c][1].get("context_length_used"), b[c][1].get("context_length_used") if c in b else None) for c in elig)
    eb = Counter((a[c][1].get("evaluate_batch_size"), b[c][1].get("evaluate_batch_size") if c in b else None) for c in elig)
    revs = Counter(m.get("model_revision") for c,(r,m) in list(a.items())+list(b.items()))
    sha = Counter(m.get("run_grid_sha256","")[:12] for c,(r,m) in list(a.items())+list(b.items()))
    cbs = Counter(m.get("chronos_batch_size_env") for c,(r,m) in list(a.items())+list(b.items()))
    tf = Counter(m.get("float32_matmul_precision") for c,(r,m) in list(a.items())+list(b.items()))
    arr = np.array([d[c] for c in elig])
    print(f"\n{v}: effect {100*(math.exp(arr.mean())-1):+.2f}%  n={len(d)}  |d|<1e-9: {sum(abs(arr)<1e-9)}  |d|<1e-3: {sum(abs(arr)<1e-3)}  |d|<0.005: {sum(abs(arr)<0.005)}")
    print("  context (doc,512) top:", cu.most_common(6))
    print("  eval batch (doc,512):", dict(eb), " revisions:", dict(revs), " run_grid sha:", dict(sha), " CHRONOS_BATCH_SIZE env:", dict(cbs), " matmul:", dict(tf))
# available history per config: shortest series and median series, latest window
print("\nconfig: min_len, median_len, pred_len, windows, toto_maximal_context, median history at latest window, |d_c| per model")
rows = []
for c in elig:
    i = info[c]; m = med[c]["median_length"]
    hist_med_last = m - i["prediction_length"]
    hist_med_first = m - i["windows"] * i["prediction_length"]
    rows.append((c, i["min_series_length"], m, i["prediction_length"], i["windows"], i["toto_maximal_context"], hist_med_first, hist_med_last, [round(D[v][c], 4) for v in MODELS]))
rows.sort(key=lambda r: r[7])
for r in rows[:18]: print(" ", r)
print("\nToto maximal context <= 512:", [(r[0], r[5]) for r in rows if r[5] <= 512])
print("Toto maximal context <= 1024:", sorted([(r[5], r[0]) for r in rows if r[5] <= 1024]))
print("median series history at latest window <= 512:", [r[0] for r in rows if r[7] <= 512])
print("median series history at earliest window <= 512:", [r[0] for r in rows if r[6] <= 512])
json.dump({v: D[v] for v in MODELS}, open("fmts/camera_ready/gpu_round/review_scratch/refute_dc70.json", "w"), indent=1)
