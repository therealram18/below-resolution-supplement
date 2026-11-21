import os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
B='boards/pin_2026-11-13/'; C='3731cc5b91682f681320ecd7080192da1e06cb70'
files=[l.strip() for l in open(B+'simons_files.txt')]
def get(f):
    out=B+'time_results/'+f
    os.makedirs(os.path.dirname(out),exist_ok=True)
    if os.path.exists(out): return
    urllib.request.urlretrieve(f'https://huggingface.co/datasets/Real-TSF/TIME-Output/resolve/{C}/{f}',out)
with ThreadPoolExecutor(8) as ex: list(ex.map(get,files))
print(len(files),'fetched')
