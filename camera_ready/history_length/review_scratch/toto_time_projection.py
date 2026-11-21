"""Review scratch (round 3): rough projection of Toto's run time under setting A
(256 samples, key-value cache, notebook batch-size formula) on the 70
configurations, to judge the 90-minute rule (DESIGN_ROUND3.md line 111).

Batch size: toto.ipynb cell 5, calculate_optimal_batch_size, with safety factor
0.01 and an 80 GB card. ASSUMPTION: the model has 12 layers of width 768 (the
exact values are in Datadog/Toto-Open-Base-1.0 config.json, not on this disk);
for any plausible size the key-value term dominates and the batch at 4,096 steps
is 1 or 2 series.

Per-batch time is calibrated on the notebook's own printed timings (cell 7 output,
GPU not stated): bizitobs_l2c/H/short, 6 batches at 2.22 per second, one batch =
7 variates x 256 samples x 37 patches = 66k patch tokens, one decoding step:
0.45 s, so about 6.8 microseconds per prefill patch token; bizitobs_l2c/H/long,
1 batch of 7 x 256 x 20 patches plus 11 more decoding steps at 1.10 per second:
0.91 s, so about 0.061 s per extra decoding step for 1,792 sequences; scaled
linearly with sequences. These are rough (one GPU, two data points); the point
is the order of magnitude and which configurations are heaviest.
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/toto_time_projection.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "toto_context_70.json").read_text())

WIDTH, DEPTH = 768, 12
PARAM_MB = 151e6 * 4 / 2 ** 20
GPU_MB = 81_000
AVAIL = (GPU_MB - PARAM_MB - 1024) * 0.01
SAMPLES = 256
PATCH = 64
US_PER_TOKEN = 0.45 / (7 * 256 * math.ceil(2328 / 64)) * 1e6
S_PER_STEP_PER_SEQ = (0.91 - 7 * 256 * math.ceil(1224 / 64) * US_PER_TOKEN / 1e6) / 11 / (7 * 256)


def batch_size(dim, ctx, pred):
    base = WIDTH * DEPTH * 4 / 2 ** 20
    io = dim * (ctx + pred) * 4 / 2 ** 20
    kv = DEPTH * WIDTH * 2 * ctx * 4 / 2 ** 20
    per = (base + io + kv) * dim
    return min(16, max(1, int(AVAIL / per)))


out = []
for r in rows:
    inst_series = r["n_series"] * r["windows"]            # forecast instances (each with all variates)
    steps = math.ceil(r["pred"] / PATCH)
    rec = {"config": r["config"]}
    for arm, ctx in (("doc", r["nb_doc"]), ("512", r["nb_512"])):
        b = batch_size(r["target_dim"], ctx, r["pred"])
        nb = math.ceil(inst_series / b)
        seqs = b * r["target_dim"] * SAMPLES
        t_batch = seqs * math.ceil(ctx / PATCH) * US_PER_TOKEN / 1e6 + (steps - 1) * seqs * S_PER_STEP_PER_SEQ
        rec[f"batch_{arm}"] = b
        rec[f"batches_{arm}"] = nb
        rec[f"minutes_{arm}"] = round(nb * t_batch / 60, 1)
    out.append(rec)

out.sort(key=lambda x: -x["minutes_doc"])
tot_doc = sum(x["minutes_doc"] for x in out); tot_512 = sum(x["minutes_512"] for x in out)
print(f"calibration: {US_PER_TOKEN:.2f} us per prefill patch token; {S_PER_STEP_PER_SEQ*1e6:.1f} us per sequence per extra step")
print(f"{'config':34s} {'b_doc':>5s} {'batches':>8s} {'min_doc':>8s} | {'b_512':>5s} {'min_512':>8s}")
for x in out[:12]:
    print(f"{x['config']:34s} {x['batch_doc']:5d} {x['batches_doc']:8d} {x['minutes_doc']:8.1f} | {x['batch_512']:5d} {x['minutes_512']:8.1f}")
print(f"total over 70: documented {tot_doc/60:.1f} h, 512 {tot_512/60:.1f} h (on the notebook's GPU, TF32 as in the notebook)")
(HERE / "toto_time_projection.json").write_text(json.dumps(
    {"assumptions": {"width": WIDTH, "depth": DEPTH, "gpu_mb": GPU_MB}, "rows": out,
     "total_hours_doc": tot_doc / 60, "total_hours_512": tot_512 / 60}, indent=1))
