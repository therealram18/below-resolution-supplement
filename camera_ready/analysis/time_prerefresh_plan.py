"""Step 1: from the existing history clone (LFS pointers only), list for every commit that touches results/ the
(model, unit) -> LFS oid and size of metrics.npz; find which oids are not among the head files already on disk;
write boards/time_prerefresh_manifest.json. Read-only git."""
import subprocess, json, os, re, collections
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boards")
HIST = os.path.join(H, "time_output_history")
def git(*a, inp=None):
    return subprocess.run(["git", "-C", HIST] + list(a), capture_output=True, check=True, input=inp).stdout
commits = [l.split(" ", 2) for l in git("log", "--reverse", "--format=%H %ad %s", "--date=short", "--", "results").decode().splitlines()]
trees, blobs = [], set()
for c, d, s in commits:
    t = {}
    for line in git("ls-tree", "-r", c, "--", "results").decode().splitlines():
        meta, path = line.split("\t", 1); p = path.split("/")
        if len(p) == 6 and p[5] in ("metrics.npz", "predictions.npz", "config.json"):
            t.setdefault(p[1], {}).setdefault("/".join(p[2:5]), {})[p[5]] = meta.split()[2]
    trees.append(t)
    for m in t.values():
        for u in m.values(): blobs.add(u["metrics.npz"])
blobs = sorted(blobs)
buf = git("cat-file", "--batch", inp=("\n".join(blobs) + "\n").encode())
pos, info = 0, {}
while pos < len(buf):
    nl = buf.index(b"\n", pos); b, typ, size = buf[pos:nl].decode().split(); size = int(size)
    body = buf[nl + 1: nl + 1 + size]; pos = nl + 1 + size + 1
    txt = body.decode("utf8", "replace")
    assert txt.startswith("version https://git-lfs"), b
    oid = re.search(r"oid sha256:(\w+)", txt).group(1); sz = int(re.search(r"size (\d+)", txt).group(1))
    info[b] = (oid, sz)
head = commits[-1][0]
head_oids = {info[u["metrics.npz"]][0] for m in trees[-1].values() for u in m.values()}
need = {}   # oid -> (size, commit, path) first place seen
for (c, d, s), t in zip(commits, trees):
    for m, us in t.items():
        for u, f in us.items():
            oid, sz = info[f["metrics.npz"]]
            if oid not in head_oids and oid not in need: need[oid] = dict(size=sz, commit=c, path=f"results/{m}/{u}/metrics.npz")
state = [dict(commit=c, date=d, subject=s, files={m: {u: info[f["metrics.npz"]][0] for u, f in us.items()} for m, us in t.items()},
              n_pred={m: sum("predictions.npz" in f for f in us.values()) for m, us in t.items()}) for (c, d, s), t in zip(commits, trees)]
json.dump(dict(head=head, states=state, need=need), open(os.path.join(H, "time_prerefresh_manifest.json"), "w"))
print(len(commits), "commits;", len(blobs), "distinct metrics blobs;", len(need), "oids not at head; MB", sum(v["size"] for v in need.values()) / 1e6)
tot_all = {}
for t in trees:
    for m, us in t.items():
        for u, f in us.items(): tot_all[info[f["metrics.npz"]][0]] = info[f["metrics.npz"]][1]
print("all distinct oids", len(tot_all), "MB", sum(tot_all.values()) / 1e6)
