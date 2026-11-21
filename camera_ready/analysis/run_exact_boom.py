"""Exact sign-flip p-values for the displayed BOOM top ten (DESIGN_ROUND2.md S1). Loads exactly as audit_boom.stage_report does."""
import sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = "fmts/camera_ready"
sys.path.insert(0, HERE)
import audit_boom as ab
import vernier_stats as v
from exact_signflip import exact_sum_p, exact_student_p

PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
ADJ = [k for k, (i, j) in enumerate(PAIRS) if j == i + 1]
SEED = 20261111
_z = np.load(f"{ab.PART}/matrices.npz")
X = _z["X"]; names = list(_z["names"])
cs, _ = ab.clusters(list(_z["units"]))
top = list(np.argsort(X.mean(1), kind="stable")[:10])
CL = {"domain": cs["c"]["cl"], "domain_type": cs["b"]["cl"]}


def one(args):
    k = args
    i, j = PAIRS[k]
    d = X[top[j]] - X[top[i]]
    r = {}
    for nm, cl in CL.items():
        s = np.bincount(cl, weights=d); n = np.bincount(cl).astype(float)
        r[nm] = dict(sum=exact_sum_p(s))
        if nm == "domain":
            r[nm]["student"] = exact_student_p(s, n)
        else:  # 2^36 combinations of a two-number statistic: random draws, as specified
            r[nm]["student"] = v.wild_cluster_p(d, cl, B=999_999, rng=np.random.default_rng(SEED + k), exact_max_G=16)["p"]
    return k, r


def main():
    t0 = time.time()
    G = {nm: len(np.unique(c)) for nm, c in CL.items()}
    print("group counts", G)
    if G != {"domain": 23, "domain_type": 36}:
        print("STOP: group counts are not 23 and 36"); return
    with Pool() as pl:
        res = dict(pl.map(one, range(45), chunksize=1))
    st = {nm: json.load(open(f"{ab.PART}/report_{c}.json"))["crps"] for nm, c in (("domain", "c"), ("domain_type", "b"))}
    assert [names[i] for i in top] == st["domain"]["top10"]
    out = dict(top10=[names[i] for i in top], groups=G)
    lab = lambda k: f"{PAIRS[k][0]+1}-{PAIRS[k][1]+1}"
    for nm in CL:
        wild = np.array([p["wild_p"] for p in st[nm]["pairs"]]); cr2 = np.array([p["cr2_icc"]["p"] for p in st[nm]["pairs"]])
        assert all(lab(k) == f"{p['rank_a']}-{p['rank_b']}" for k, p in enumerate(st[nm]["pairs"]))
        cnt = lambda p: dict(adj_unc=int((p < .05)[ADJ].sum()), all_unc=int((p < .05).sum()), adj_holm=int(v.holm(p[ADJ]).sum()), all_holm=int(v.holm(p).sum()))
        R = dict(pairs={lab(k): dict(p_exact_sum=res[k][nm]["sum"], p_student=res[k][nm]["student"]) for k in range(45)},
                 student_method="exact enumeration" if nm == "domain" else "wild_cluster_p B=999999 seed 20261111+pair index")
        ps = np.array([res[k][nm]["sum"] for k in range(45)]); pt = np.array([res[k][nm]["student"] for k in range(45)])
        R["counts"] = dict(CR2=cnt(cr2), wild_stored=cnt(wild), student=cnt(pt), exact_sum=cnt(ps))
        for q, p in (("student", pt), ("exact_sum", ps)):
            dd = np.abs(p - wild); m = int(dd.argmax())
            R[f"max_abs_diff_{q}_vs_stored_wild"] = dict(value=float(dd[m]), pair=lab(m))
            R[f"decision_differences_vs_wild_{q}"] = dict(unc_all45=int(((p < .05) != (wild < .05)).sum()), holm_all45=int((v.holm(p) != v.holm(wild)).sum()),
                                                          holm_neighbours9=int((v.holm(p[ADJ]) != v.holm(wild[ADJ])).sum()), unc_neighbours=int(((p < .05) != (wild < .05))[ADJ].sum()))
        out[nm] = R
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(f"{ab.OUT}/exact_boom.json", "w"), indent=1)
    for nm in CL:
        print(nm, json.dumps(out[nm]["counts"]), out[nm]["max_abs_diff_student_vs_stored_wild"], out[nm]["max_abs_diff_exact_sum_vs_stored_wild"])
    print("seconds", out["runtime_s"])


if __name__ == "__main__":
    main()
