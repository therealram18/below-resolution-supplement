"""Reading 5: G2 reproduction exactness, columns, contexts, and the pin mix."""
import csv, json, math, glob
from collections import Counter
import numpy as np
exec(open("fmts/camera_ready/gpu_round/review_scratch/refute_check.py").read().split("# ---------- Reading 1 ----------")[0])
snr = read_rows(SNAP + "/seasonal_naive/all_results.csv"); sn = {k: float(v["eval_metrics/MASE[0.5]"]) for k, v in snr.items()}
snp = read_rows(PIN + "/Seasonal_Naive/all_results.csv")
for v, pubp in [("timesfm3", SNAP + "/TimesFM-3/all_results.csv"), ("timestral209m", PIN + "/Timestral-209M/all_results.csv")]:
    pub = read_rows(pubp); a = per_config(v, "default"); b = per_config(v, "512")
    cols = [c for c in next(iter(pub.values())).keys() if c.startswith("eval_metrics/") and c in next(iter(a.values()))[0]]
    print(f"\n{v}: published model column {Counter(r['model'] for r in pub.values())}")
    for col in cols:
        rel = [float(a[c][0][col]) / float(pub[c][col]) - 1 for c in sel12]
        eq = sum(float(a[c][0][col]) == float(pub[c][col]) for c in sel12)
        print(f"  {col}: exact {eq}/12 max|rel| {max(map(abs, rel)):.2e}")
    print("  example MASE ours vs published:", [(c, a[c][0]['eval_metrics/MASE[0.5]'], pub[c]['eval_metrics/MASE[0.5]']) for c in sel12[:3]])
    m = a[sel12[0]][1]
    print("  meta keys g2:", {k: m[k] for k in m if k.startswith("g2") or k in ("context_length_used", "model_revision", "evaluate_batch_size", "torch_version", "run_grid_sha256", "g2_models_sha256")})
    print("  context (doc,512):", Counter((a[c][1]["context_length_used"], b[c][1]["context_length_used"]) for c in sel12))
    print("  longest history given (doc,512):", [(c, a[c][1].get("g2_longest_history_given"), b[c][1].get("g2_longest_history_given")) for c in sel12])
    gm = lambda vals: math.exp(np.mean([math.log(vals[c] / sn[c]) for c in sel12]))
    print(f"  gmean ours {gm({c: float(a[c][0]['eval_metrics/MASE[0.5]']) for c in sel12}):.6f} pub {gm({c: float(pub[c]['eval_metrics/MASE[0.5]']) for c in sel12}):.6f}")
# is Timestral present in Sept snapshot? is TimesFM-3 row identical in pin vs snapshot? top-10 at pin vs snapshot
import os
print("\nTimestral in snapshot:", os.path.exists(SNAP + "/Timestral-209M"))
t3s, t3p = read_rows(SNAP + "/TimesFM-3/all_results.csv"), read_rows(PIN + "/TimesFM-3/all_results.csv")
print("TimesFM-3 snapshot vs pin max|rel| on 12:", max(abs(float(t3s[c]['eval_metrics/MASE[0.5]'])/float(t3p[c]['eval_metrics/MASE[0.5]'])-1) for c in sel12))
TOP = ("STRIDE w/ Synapse", "EXAONE-Forecast-Agent", "LS-MoE", "LS-Agent", "Falcon-Agent", "CastStar", "limix_moe", "TIMEHEDGE", "RegLab-AH-1", "RacineCast-1-NC")
def folder_of(root):
    f = {}
    for d in os.listdir(root):
        p = f"{root}/{d}/all_results.csv"
        if os.path.exists(p):
            f[next(csv.DictReader(open(p)))["model"]] = p
    return f
fs, fp = folder_of(SNAP), folder_of(PIN)
def spread(fmap, cs):
    g = []
    for n in TOP:
        if n not in fmap: g.append(None); continue
        r = read_rows(fmap[n]); g.append(math.exp(np.mean([math.log(float(r[c]['eval_metrics/MASE[0.5]']) / sn[c]) for c in cs])))
    return g
gs, gp = spread(fs, sel12), spread(fp, sel12)
print("top10 on 12, snapshot:", [round(x, 4) for x in gs], "S", round(100 * (max(gs) / min(gs) - 1), 2))
print("top10 on 12, pin     :", [None if x is None else round(x, 4) for x in gp])
