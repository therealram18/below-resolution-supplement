"""Per-pair tables under CR2 (DESIGN_ROUND2.md S4). Reads retest_gift.json and exact_gift.json only; CR2 values are not recomputed."""
import json
import numpy as np, pandas as pd
from scipy import stats
HERE = "fmts/camera_ready"
R = json.load(open(f"{HERE}/retest_gift.json")); E = json.load(open(f"{HERE}/exact_gift.json"))
PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
ADJ = [k for k, (i, j) in enumerate(PAIRS) if j == i + 1]


def holm_adj(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); a = np.empty(m)
    a[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return a


md = ["# Per-pair tables, displayed top ten, CR2 (degrees of freedom from the estimated within-dataset correlation)\n",
      "Source: retest_gift.json and exact_gift.json. Gap = mean difference theta (lower-placed minus higher-placed; log ratio for gmean; positive favours the higher-placed entry). "
      "Holm columns are step-down adjusted p-values (capped at 1, monotone). Holm-9 and resolution are for neighbouring pairs only. "
      "Resolution = (t_.975,df + t_.80,df) x CR2 SE.\n"]
for rule in ("MASE_Rank", "CRPS_Rank", "gmean"):
    st = R["pairs"][rule]
    rows = []
    for k, r in enumerate(st):
        lab = r["pair"]; ex = E[rule]["pairs"][lab]; c = r["CR2-icc"]
        rows.append(dict(places=lab, name_higher=r["names"][0], name_lower=r["names"][1], gap_theta=r["theta"], cr2_se=c["se"], df=c["df"],
                         ci95_lo=c["ci95"][0], ci95_hi=c["ci95"][1], p_cr2=c["p"], p_wild_99999=r["WCB-Rad"]["p"],
                         p_exact_studentized=ex["p_exact_student"], p_exact_sum=ex["p_exact_sum"]))
    df = pd.DataFrame(rows)
    df["holm45_cr2"] = holm_adj(df.p_cr2); df["holm45_exact_studentized"] = holm_adj(df.p_exact_studentized)
    for c9, src in (("holm9_cr2", "p_cr2"), ("holm9_exact_studentized", "p_exact_studentized")):
        df[c9] = np.nan; df.loc[ADJ, c9] = holm_adj(df.loc[ADJ, src].values)
    df["resolution"] = np.nan
    df.loc[ADJ, "resolution"] = [(stats.t.ppf(.975, d) + stats.t.ppf(.80, d)) * s for d, s in zip(df.df[ADJ], df.cr2_se[ADJ])]
    df.to_csv(f"{HERE}/pair_table_{rule}.csv", index=False)
    f = lambda x: "" if pd.isna(x) else (f"{x:.4g}" if abs(x) < 1e-3 else f"{x:.4f}")
    md.append(f"\n## {rule}\n")
    md.append("| places | higher | lower | gap | CR2 SE | df | 95% interval | p CR2 | p wild | p exact stud. | p exact sum | Holm-45 CR2 | Holm-45 exact stud. | Holm-9 CR2 | Holm-9 exact stud. | resolution |")
    md.append("|" + "---|" * 16)
    for _, r in df.iterrows():
        md.append(f"| {r.places} | {r.name_higher} | {r.name_lower} | {r.gap_theta:.4f} | {r.cr2_se:.4f} | {r.df:.1f} | [{r.ci95_lo:.4f}, {r.ci95_hi:.4f}] | " +
                  " | ".join(f(r[c]) for c in ("p_cr2", "p_wild_99999", "p_exact_studentized", "p_exact_sum", "holm45_cr2", "holm45_exact_studentized", "holm9_cr2", "holm9_exact_studentized", "resolution")) + " |")
open(f"{HERE}/PAIR_TABLES.md", "w").write("\n".join(md) + "\n")
