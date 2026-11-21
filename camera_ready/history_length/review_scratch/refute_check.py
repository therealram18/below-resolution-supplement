"""Independent checks of readings 1, 2, 5 (read-only). Reads per_config CSVs and run_meta files directly."""
import csv, json, math, glob, os, sys
from collections import Counter, defaultdict
import numpy as np
R = "fmts"
GR = R + "/camera_ready/gpu_round"
RES = GR + "/pulled/harness/results"
SNAP = R + "/snapshot_2026-09-07/results"
PIN = R + "/camera_ready/boards/pin_2026-11-13/gift/results"
med = json.load(open(R + "/selection/config_median_lengths.json"))
elig = sorted(c for c, v in med.items() if v["median_length"] >= 1024)
sel12 = sorted(x["config"] for x in json.load(open(R + "/selection/selected_configs.json"))["selected"])
info = json.load(open(GR + "/harness/jobs/config_info.json"))
SRC = {"bizitobs_application": "bizitobs", "bizitobs_l2c": "bizitobs", "bizitobs_service": "bizitobs",
       "ett1": "ett", "ett2": "ett", "bitbrains_fast_storage": "bitbrains", "bitbrains_rnd": "bitbrains"}
src = lambda c: SRC.get(c.split("/")[0], c.split("/")[0])

def read_rows(path):
    with open(path, newline="") as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}

def per_config(variant, ctx):
    out = {}
    for p in glob.glob(f"{RES}/{variant}/ctx_{ctx}/per_config/*.csv"):
        rows = list(csv.DictReader(open(p)))
        assert len(rows) == 1, p
        r = rows[0]
        meta = json.load(open(p[:-4] + ".run_meta.json"))
        out[r["dataset"]] = (r, meta)
    return out

print("eligible", len(elig), "bases", len({c.split('/')[0] for c in elig}), "sources", sorted(Counter(map(src, elig)).values(), reverse=True))
print("12 sources", len(set(map(src, sel12))), "12 subset of 70", all(c in elig for c in sel12))
print("median-length rule check: min median among 70", min(med[c]["median_length"] for c in elig),
      "max median among excluded", max(v["median_length"] for c, v in med.items() if c not in elig), "total configs", len(med))

# ---------- Reading 1 ----------
sn = {k: float(v["eval_metrics/MASE[0.5]"]) for k, v in read_rows(SNAP + "/seasonal_naive/all_results.csv").items()}
board = read_rows(SNAP + "/Toto_Open_Base_1.0/all_results.csv")
ours = per_config("toto__A__seed42", "default")
allr = read_rows(f"{RES}/toto__A__seed42/ctx_default/all_results.csv")
print("\n[R1] board Toto model column:", Counter(r["model"] for r in board.values()))
cols = [c for c in next(iter(board.values())).keys() if c.startswith("eval_metrics/") and c in next(iter(ours.values()))[0]]
exact = Counter(); rel = defaultdict(list)
for c in elig:
    r, m = ours[c]
    assert abs(float(r["eval_metrics/MASE[0.5]"]) - float(allr[c]["eval_metrics/MASE[0.5]"])) == 0
    for col in cols:
        a, b = float(r[col]), float(board[c][col])
        exact[col] += (a == b)
        rel[col].append(a / b - 1 if b else float("nan"))
for col in cols:
    v = np.array(rel[col]); print(f"  {col}: exact-equal {exact[col]}/70, median |rel| {np.nanmedian(np.abs(v)):.2e}, max |rel| {np.nanmax(np.abs(v)):.4f}")
def gm(vals, cs): return math.exp(np.mean([math.log(vals[c] / sn[c]) for c in cs]))
o = {c: float(ours[c][0]["eval_metrics/MASE[0.5]"]) for c in elig}
b = {c: float(board[c]["eval_metrics/MASE[0.5]"]) for c in elig}
for name, cs in [("12", sel12), ("70", elig), ("58", [c for c in elig if c not in sel12])]:
    print(f"  gmean ours {gm(o,cs):.5f} board {gm(b,cs):.5f} gap {100*(gm(o,cs)/gm(b,cs)-1):+.3f}%")
d = sorted(((o[c]/b[c]-1)*100, c) for c in elig)
print("  largest |diff| 70:", sorted(d, key=lambda t: -abs(t[0]))[:5])
# other Toto entries on the board, for wrong-entry check
for e in ["Toto-2.0-22m", "Toto-2.0-4m", "Toto-2.0-313m"]:
    p = read_rows(f"{SNAP}/{e}/all_results.csv"); bb = {c: float(p[c]["eval_metrics/MASE[0.5]"]) for c in sel12}
    print(f"  vs {e}: gmean on 12 {gm(bb, sel12):.4f}")
# run meta of Toto
ctxm = Counter(); bad = []
for ctx in ("default", "512"):
    pc = per_config("toto__A__seed42", ctx)
    for c, (r, m) in pc.items():
        ok = (m["status"] == "ok" and m["num_samples"] == 256 and m["use_kv_cache"] and m["seed"] == 42 and m["deterministic_algorithms"]
              and m["float32_matmul_precision"] == "high" and not m["xformers_installed"] and m["evaluate_batch_size"] == 256
              and m["toto_setting"] == "A" and m["model_revision"] == "0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5")
        if not ok: bad.append((ctx, c))
        ctxm[(ctx, m["run_grid_sha256"][:12])] += 1
print("  toto seed42 meta problems:", bad, " run_grid sha by arm:", dict(ctxm))
