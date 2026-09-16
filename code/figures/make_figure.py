"""Build fig_resolution.pdf: adjacent top-10 gaps vs. the smallest gap the board
can resolve at 80% power, for GIFT-Eval (MASE_Rank) and fev-bench (win rate).

Left panel reads table1_200k.json directly (clustered bootstrap CIs already
computed at B=200000, seed 20260907 -- see audit/table1_200k.py).

Right panel extends poolcheck/fev_resolution.py's logic (same pinned CSVs) to
dump per-pair point gaps + family-clustered 95% CIs at B=4000, seed 20260907,
and saves them to fev_pairs.json alongside this script, so the number in the
figure and the number on disk are the same file.

Usage: bin/python make_figure.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
POOLCHECK = os.path.normpath(os.path.join(HERE, "..", "poolcheck"))
AUDIT = os.path.normpath(os.path.join(HERE, "..", "audit"))

GIFT_MDE80 = 6.16   # audit/table1_200k.py / poolcheck/resolution_budget.py, MASE_Rank, 28 base datasets
FEV_MDE80 = 9.52    # poolcheck/fev_resolution.py, win rate (displayed sort), 39 task families

# ---------------------------------------------------------------------------
# Left panel data: read straight from the audited table.
# ---------------------------------------------------------------------------

def load_gift_pairs():
    d = json.load(open(os.path.join(AUDIT, "table1_200k.json")))
    rows = d["computation1_table1_at_200k"]["table1_rows"]
    assert len(rows) == 9
    pairs, gaps, los, his = [], [], [], []
    for r in rows:
        pairs.append(r["pair"])
        gaps.append(r["delta_mase_rank"])
        lo, hi = r["clustered_ci"]
        los.append(lo)
        his.append(hi)
    return pairs, np.array(gaps), np.array(los), np.array(his)


# ---------------------------------------------------------------------------
# Right panel: replicate fev_resolution.py's win-rate computation, but dump
# per-pair point estimate + family-clustered CI instead of just survival counts.
# ---------------------------------------------------------------------------

def compute_fev_pairs():
    pivot = os.path.join(POOLCHECK, "pivot_SQL.csv")
    rows = list(csv.reader(open(pivot)))
    models = rows[0][1:]
    tasks = [r[0] for r in rows[1:]]
    E = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    mi = {m: i for i, m in enumerate(models)}
    Rel = np.clip(E / E[:, [mi["Seasonal Naive"]]], 0.01, 100)
    A = Rel
    Wn = (A[:, :, None] < A[:, None, :]) + 0.5 * (A[:, :, None] == A[:, None, :])
    Wn = Wn.astype(float)
    n = len(models)
    Wn[:, np.arange(n), np.arange(n)] = np.nan
    WT = np.nanmean(Wn, axis=2) * 100
    wr = WT.mean(0)

    T = list(np.argsort(-wr))[:10]
    top10 = [models[i] for i in T]

    fam = [t.rsplit("_", 1)[0] for t in tasks]
    fams = sorted(set(fam))
    fidx = np.array([fams.index(f) for f in fam])

    # Same rng call order as fev_resolution.py: iid draw first, then the
    # per-replicate family draw loop -- reproduce both so the random stream
    # lines up and the clustered CIs match a from-scratch run of that script.
    B = 4000
    rng = np.random.default_rng(20260907)
    N = len(tasks)
    iid = rng.integers(0, N, size=(B, N))  # noqa: F841 (kept to match rng stream)
    W = np.zeros((B, N))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(fams), size=len(fams)), minlength=len(fams))[fidx]

    def boot_clustered(d):
        return (W * d).sum(1) / W.sum(1)

    out_pairs = []
    for a, b in zip(T, T[1:]):
        d = WT[:, a] - WT[:, b]
        cl = boot_clustered(d)
        lo, hi = np.percentile(cl, [2.5, 97.5])
        out_pairs.append({
            "pair": f"{T.index(a) + 1}-{T.index(b) + 1}",
            "modelA": models[a],
            "modelB": models[b],
            "delta_win_rate": float(d.mean()),
            "clustered_ci": [float(lo), float(hi)],
        })

    ses = [np.std(boot_clustered(WT[:, a] - WT[:, b])) for a, b in zip(T, T[1:])]
    mde80 = 2.8 * float(np.median(ses))

    return {
        "meta": {
            "source": "poolcheck/pivot_SQL.csv, poolcheck/leaderboard_SQL.csv",
            "logic": "extends poolcheck/fev_resolution.py (win rate, displayed sort)",
            "B": B,
            "seed": 20260907,
            "scheme": "family-clustered bootstrap (block bootstrap over task families), 95% percentile CI",
            "metric": "win_rate (0-100 points)",
            "top10_win_rate_order": top10,
            "mde80_computed": mde80,
            "mde80_reported": FEV_MDE80,
        },
        "pairs": out_pairs,
    }


def load_fev_pairs():
    result = compute_fev_pairs()
    out_path = os.path.join(HERE, "fev_pairs.json")
    json.dump(result, open(out_path, "w"), indent=2)
    pairs = result["pairs"]
    labels = [p["pair"] for p in pairs]
    gaps = np.array([p["delta_win_rate"] for p in pairs])
    los = np.array([p["clustered_ci"][0] for p in pairs])
    his = np.array([p["clustered_ci"][1] for p in pairs])
    return labels, gaps, los, his, result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def draw_panel(ax, labels, gaps, los, his, mde_line, title, ylabel):
    x = np.arange(1, len(labels) + 1)
    yerr = np.vstack([gaps - los, his - gaps])

    ax.axhline(0, color="black", linewidth=1.0, zorder=2)
    ax.axhline(mde_line, color="black", linewidth=1.1, linestyle=(0, (4, 2)), zorder=2)

    ax.errorbar(
        x, gaps, yerr=yerr, fmt="o", color="black",
        ecolor="black", elinewidth=1.3, capsize=3, capthick=1.3,
        markersize=5, markerfacecolor="black", markeredgecolor="black",
        zorder=3,
    )

    # Direct label for the dashed threshold line (no legend). White backing so
    # it stays readable wherever a whisker happens to cross underneath it.
    ax.text(
        0.02, mde_line, "smallest detectable gap (80% power)",
        transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=7.0, color="black", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.2),
    )

    # Direct label for any interval that clears zero (the notable case) --
    # sparing, selective: only the surviving pair gets a label.
    for xi, lo in zip(x, los):
        if lo > 0:
            ax.annotate(
                "excludes 0", xy=(xi, lo), xytext=(0, -7),
                textcoords="offset points", ha="center", va="top", fontsize=7.0,
                zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("adjacent top-10 pair", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.06)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,  # embed as real text, not paths
        "ps.fonttype": 42,
    })

    gift_labels, gift_gaps, gift_los, gift_his = load_gift_pairs()
    fev_labels, fev_gaps, fev_los, fev_his, fev_result = load_fev_pairs()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))

    draw_panel(
        axes[0], gift_labels, gift_gaps, gift_los, gift_his, GIFT_MDE80,
        "GIFT-Eval, mean rank", "gap (rank places)",
    )
    draw_panel(
        axes[1], fev_labels, fev_gaps, fev_los, fev_his, FEV_MDE80,
        "fev-bench, win rate", "gap (win-rate points)",
    )

    fig.tight_layout(w_pad=2.2)
    out_pdf = os.path.join(HERE, "fig_resolution.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)

    return gift_labels, gift_gaps, gift_los, gift_his, fev_labels, fev_gaps, fev_los, fev_his, fev_result, out_pdf


def _self_check():
    """ponytail: one runnable check -- shapes, sign conventions, output files exist."""
    gl, gg, glo, ghi, fl, fg, flo, fhi, fev_result, out_pdf = main()

    assert len(gl) == 9 and len(fl) == 9, "expected 9 adjacent pairs per panel"
    assert np.all(glo <= gg) and np.all(gg <= ghi), "GIFT CI must bracket the point gap"
    assert np.all(flo <= fg) and np.all(fg <= fhi), "fev CI must bracket the point gap"
    assert os.path.getsize(out_pdf) > 1000, "PDF looks empty"
    fev_pairs_path = os.path.join(HERE, "fev_pairs.json")
    assert os.path.exists(fev_pairs_path)
    reloaded = json.load(open(fev_pairs_path))
    assert len(reloaded["pairs"]) == 9

    clearing = [p for p in reloaded["pairs"] if p["clustered_ci"][0] > 0]
    print("fev-bench per-pair win-rate gaps (family-clustered 95% CI, B=4000, seed=20260907):")
    for p in reloaded["pairs"]:
        print(f"  {p['pair']:5s} {p['modelA']:16s} vs {p['modelB']:16s}  "
              f"gap={p['delta_win_rate']:6.3f}  CI=[{p['clustered_ci'][0]:7.3f}, {p['clustered_ci'][1]:7.3f}]")
    print(f"MDE80 computed here: {reloaded['meta']['mde80_computed']:.3f} "
          f"(reported/used for dashed line: {reloaded['meta']['mde80_reported']})")
    print(f"Pairs whose clustered CI excludes 0: {[p['pair'] for p in clearing]}")
    print("OK: self-check passed.")


if __name__ == "__main__":
    _self_check()
