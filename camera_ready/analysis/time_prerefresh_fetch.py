"""Step 2: download the metrics.npz objects that the head files do not already contain, from the same HF dataset repo,
into boards/time_output_pre_refresh/objects/<sha256>. Each file is checked against its LFS sha256."""
import json, os, hashlib, urllib.request, urllib.parse, concurrent.futures as cf
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boards"); OUT = os.path.join(B, "time_output_pre_refresh", "objects")
os.makedirs(OUT, exist_ok=True)
need = json.load(open(os.path.join(B, "time_prerefresh_manifest.json")))["need"]
def get(item):
    oid, v = item; dst = os.path.join(OUT, oid)
    if os.path.exists(dst): return oid, "have", 0
    url = f"https://huggingface.co/datasets/Real-TSF/TIME-Output/resolve/{v['commit']}/" + urllib.parse.quote(v["path"])
    for _ in range(3):
        try:
            b = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(b).hexdigest() != oid: return oid, "BAD_HASH", len(b)
            open(dst + ".part", "wb").write(b); os.replace(dst + ".part", dst); return oid, "ok", len(b)
        except Exception as e: err = str(e)
    return oid, "FAIL " + err, 0
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(get, need.items()))
from collections import Counter
print(Counter(r[1][:8] for r in res), "bytes", sum(r[2] for r in res))
for r in res:
    if r[1] not in ("ok", "have"): print(r)
