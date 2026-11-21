"""Camera-ready Figure 1 (fig_resolution.pdf).

For two leaderboards, the gaps between neighbouring entries of the displayed top
ten, each with its CR2 95% interval (t interval from the cluster-robust CR2
standard error with Satterthwaite degrees of freedom), against the board's
corrected resolution (the smallest difference the board can detect).

Only stored values are plotted. Nothing is recomputed from raw scores; the only
arithmetic is a unit change for fev-bench (stored as fractions, plotted as
win-rate points, times 100) and consistency checks on the stored numbers.

Style, palette and size follow figure/make_figure.py (the accepted figure).
Changes: CR2 intervals and corrected resolutions instead of percentile
bootstrap; labels without background boxes, placed where they touch no marker,
whisker or line; the line is labelled "resolution".

Outputs (paths relative to camera_ready/paper):
  fig_resolution.pdf               the figure (vector, TrueType fonts embedded)
  figs/fig_resolution_preview.png  200 dpi preview
  figs/fig_resolution_facts.md     every plotted number with its source

Run from any directory:
  uv run --no-project --with matplotlib --with numpy --with seaborn --with scipy \
      python fmts/camera_ready/paper/figs/make_fig_resolution.py
"""
import csv
import json
import os
import re
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)          # camera_ready/paper
CR = os.path.dirname(PAPER)            # camera_ready

GIFT_CSV = os.path.join(CR, "pair_table_MASE_Rank.csv")
GIFT_JSON = os.path.join(CR, "retest_gift.json")
FEV_JSON = os.path.join(CR, "boards", "fev_family_report.json")

OUT_PDF = os.path.join(PAPER, "fig_resolution.pdf")
OUT_PNG = os.path.join(HERE, "fig_resolution_preview.png")
OUT_FACTS = os.path.join(HERE, "fig_resolution_facts.md")
RUN_COMMAND = (
    "uv run --no-project --with matplotlib --with numpy --with seaborn --with scipy "
    "python " + os.path.abspath(__file__)
)

# Palette and marks from figure/make_figure.py (accepted figure).
COLOR_GIFT = "#2a78d6"    # blue: GIFT-Eval panel
COLOR_FEV = "#199e70"     # teal: fev-bench panel
COLOR_ACCENT = "#eb6834"  # orange: resolution line
GRID_COLOR = "#e1e0d9"
ELINEWIDTH = 1.3
CAPSIZE = 3.0
MARKER_S = 30
MARKER_LW = 1.1
ZERO_LW = 1.0
RES_LW = 1.4
RES_ALPHA = 0.9
LABEL_FONTSIZE = 6.5
CLEARANCE_PT = 1.0        # minimum free space around every label

NEIGHBOURS = [f"{i}-{i + 1}" for i in range(1, 10)]

# The printed GIFT-Eval table (gap, interval low, interval high), 2 decimals.
PRINTED_GIFT = {
    "1-2": (4.25, -2.40, 10.89),
    "2-3": (0.44, -6.68, 7.57),
    "3-4": (1.94, -0.15, 4.02),
    "4-5": (0.91, -3.82, 5.64),
    "5-6": (0.07, -4.75, 4.90),
    "6-7": (0.01, -2.98, 3.00),
    "7-8": (1.42, -4.17, 7.02),
    "8-9": (0.65, -6.63, 7.93),
    "9-10": (1.33, 0.09, 2.57),
}


# ---------------------------------------------------------------------------
# Stored values
# ---------------------------------------------------------------------------

def check_t_intervals(name, gaps, los, his, ses, dfs, resolution):
    """Stored interval = gap +- t(0.975, df) * SE, and the stored resolution is the
    median over the nine pairs of (t(0.975, df) + t(0.80, df)) * SE. Checks only;
    nothing here is plotted."""
    q = student_t.ppf(0.975, dfs)
    assert np.allclose(los, gaps - q * ses, atol=1e-6, rtol=0), f"{name}: low ends differ"
    assert np.allclose(his, gaps + q * ses, atol=1e-6, rtol=0), f"{name}: high ends differ"
    per_pair = (q + student_t.ppf(0.80, dfs)) * ses
    assert abs(float(np.median(per_pair)) - resolution) < 1e-6, f"{name}: resolution differs"


def load_gift():
    rows = {r["places"]: r for r in csv.DictReader(open(GIFT_CSV))}
    js = json.load(open(GIFT_JSON))
    js_pairs = {p["pair"]: (k, p) for k, p in enumerate(js["pairs"]["MASE_Rank"])}
    summary = js["resolution"]["MASE_Rank"]["summary"]

    out = {"labels": NEIGHBOURS, "names": [], "gaps": [], "los": [], "his": [],
           "ses": [], "dfs": [], "json_index": []}
    for pl in NEIGHBOURS:
        r = rows[pl]
        gap, lo, hi = float(r["gap_theta"]), float(r["ci95_lo"]), float(r["ci95_hi"])
        se, df = float(r["cr2_se"]), float(r["df"])
        # Same numbers in the second stored file.
        k, p = js_pairs[pl]
        c = p["CR2-icc"]
        assert abs(p["theta"] - gap) < 1e-9 and abs(c["ci95"][0] - lo) < 1e-9 \
            and abs(c["ci95"][1] - hi) < 1e-9 and abs(c["se"] - se) < 1e-9 \
            and abs(c["df"] - df) < 1e-9, f"GIFT {pl}: csv and json differ"
        assert p["names"] == [r["name_higher"], r["name_lower"]]
        # The printed table.
        pg, plo, phi = PRINTED_GIFT[pl]
        for v, w, what in ((gap, pg, "gap"), (lo, plo, "low"), (hi, phi, "high")):
            assert abs(v - w) <= 0.005 + 1e-9, f"GIFT {pl} {what}: stored {v:.4f}, printed {w}"
        out["names"].append((r["name_higher"], r["name_lower"]))
        out["gaps"].append(gap); out["los"].append(lo); out["his"].append(hi)
        out["ses"].append(se); out["dfs"].append(df); out["json_index"].append(k)
    for key in ("gaps", "los", "his", "ses", "dfs"):
        out[key] = np.array(out[key])

    out["resolution"] = float(summary["corr_res_icc"])
    out["median_gap"] = float(summary["median_gap"])
    out["holm9_cr2"] = {pl: float(rows[pl]["holm9_cr2"]) for pl in NEIGHBOURS}
    assert min(out["holm9_cr2"].values()) > 0.05   # the caption says none survives Holm
    assert f"{out['resolution']:.2f}" == "6.80" and abs(out["resolution"] - 6.8010) < 5e-5
    assert abs(statistics.median(out["gaps"]) - out["median_gap"]) < 1e-12
    check_t_intervals("GIFT-Eval", out["gaps"], out["los"], out["his"],
                      out["ses"], out["dfs"], out["resolution"])
    return out


def load_fev():
    d = json.load(open(FEV_JSON))
    rep = d["reports"]["A_wr"]          # fev-bench (100 tasks), win rate
    assert rep["n_entries"] == 40 and d["base"]["repro_A"]["entries_shown"] == 40
    assert d["base"]["repro_A"]["top10_equal"] and rep["top10"] == d["base"]["repro_A"]["top10_board"]
    adj = sorted(((k, p) for k, p in enumerate(rep["pairs"]) if p["adjacent"]),
                 key=lambda kp: kp[1]["rank_a"])
    assert len(adj) == 9
    top10 = rep["top10"]
    out = {"labels": [], "names": [], "gaps": [], "los": [], "his": [],
           "ses": [], "dfs": [], "json_index": []}
    for i, (k, p) in enumerate(adj, start=1):
        assert p["rank_a"] == i and p["rank_b"] == i + 1
        assert p["a"] == top10[i - 1] and p["b"] == top10[i]
        c = p["cr2_icc"]
        out["labels"].append(f"{i}-{i + 1}")
        out["names"].append((p["a"], p["b"]))
        out["gaps"].append(100 * p["theta"])
        out["los"].append(100 * c["ci95"][0]); out["his"].append(100 * c["ci95"][1])
        out["ses"].append(100 * c["se"]); out["dfs"].append(c["df"])
        out["json_index"].append(k)
    assert out["labels"] == NEIGHBOURS
    for key in ("gaps", "los", "his", "ses", "dfs"):
        out[key] = np.array(out[key])

    out["resolution"] = 100 * rep["resolution"]["corrected_icc"]
    out["median_gap"] = 100 * rep["resolution"]["median_adjacent_gap"]
    out["n_units"], out["n_sources"] = rep["n_units"], rep["n_sources"]
    assert f"{out['resolution']:.2f}" == "7.51"
    assert abs(out["resolution"] - d["summaries"]["A_wr"]["resolution_corrected"]) < 5e-4
    assert f"{out['median_gap']:.2f}" == "2.08"
    assert abs(statistics.median(out["gaps"]) - out["median_gap"]) < 1e-9
    excl = (out["los"] > 0) | (out["his"] < 0)
    assert excl.sum() == 0 == d["summaries"]["A_wr"]["adjacent_of_9"]["CR2-icc"], \
        "fev-bench: expected none of the nine CR2 intervals to exclude zero"
    check_t_intervals("fev-bench", out["gaps"], out["los"], out["his"],
                      out["ses"], out["dfs"], out["resolution"])
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def excludes_zero(p):
    return (p["los"] > 0) | (p["his"] < 0)


def draw_panel(ax, p, title, ylabel, color):
    x = np.arange(1, 10)
    gaps, los, his = p["gaps"], p["los"], p["his"]
    yerr = np.vstack([gaps - los, his - gaps])

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.axhline(0, color="black", linewidth=ZERO_LW, zorder=2)
    # Solid, on top of whiskers and markers, as in the accepted figure.
    ax.axhline(p["resolution"], color=COLOR_ACCENT, linewidth=RES_LW, alpha=RES_ALPHA, zorder=5)

    ax.errorbar(x, gaps, yerr=yerr, fmt="none", ecolor=color, elinewidth=ELINEWIDTH,
                capsize=CAPSIZE, capthick=ELINEWIDTH, zorder=3)
    ex = excludes_zero(p)
    ax.scatter(x[~ex], gaps[~ex], s=MARKER_S, facecolor="white", edgecolor=color,
               linewidth=MARKER_LW, zorder=4)
    ax.scatter(x[ex], gaps[ex], s=MARKER_S, facecolor=color, edgecolor=color,
               linewidth=MARKER_LW, zorder=4)

    # Labels have no background box; their positions are chosen later so that
    # they touch nothing.
    res_label = ax.annotate(
        f"resolution {p['resolution']:.2f}", xy=(1, p["resolution"]), xytext=(0, 2.0),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=LABEL_FONTSIZE, color=COLOR_ACCENT, zorder=6,
    )
    zero_labels = []
    for xi in x[ex]:
        zero_labels.append((xi, ax.annotate(
            "excludes 0", xy=(xi, 0), xytext=(0, -4), textcoords="offset points",
            ha="center", va="top", fontsize=LABEL_FONTSIZE, zorder=6,
        )))

    ax.set_xticks(x)
    ax.set_xticklabels(p["labels"], fontsize=7)
    ax.set_xlabel("neighbouring places in the top ten", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.06)
    return res_label, zero_labels


# ---------------------------------------------------------------------------
# Collision checks (display pixels)
# ---------------------------------------------------------------------------

def _pt(fig, v):
    return v * fig.dpi / 72.0


def mark_boxes(ax, p):
    """Boxes around every marker, whisker, cap and horizontal line in one panel."""
    fig = ax.figure
    T = ax.transData.transform
    hw, cap, r = _pt(fig, ELINEWIDTH / 2), _pt(fig, CAPSIZE), _pt(fig, np.sqrt(MARKER_S) / 2 + MARKER_LW / 2)
    boxes = []
    for i, (g, lo, hi) in enumerate(zip(p["gaps"], p["los"], p["his"]), start=1):
        (px, plo), (_, phi), (_, pg) = T([(i, lo), (i, hi), (i, g)])
        boxes.append((f"whisker {i}", (px - hw, plo - hw, px + hw, phi + hw)))
        boxes.append((f"low cap {i}", (px - cap, plo - hw, px + cap, plo + hw)))
        boxes.append((f"high cap {i}", (px - cap, phi - hw, px + cap, phi + hw)))
        boxes.append((f"marker {i}", (px - r, pg - r, px + r, pg + r)))
    ax_x0, _, ax_x1, _ = ax.get_window_extent().extents
    for name, y, lw in (("zero line", 0.0, ZERO_LW), ("resolution line", p["resolution"], RES_LW)):
        py = T((0, y))[1]
        boxes.append((name, (ax_x0, py - _pt(fig, lw / 2), ax_x1, py + _pt(fig, lw / 2))))
    return boxes


def _hit(a, b, pad):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def text_hits(fig, txt, boxes, renderer):
    bb = txt.get_window_extent(renderer).extents
    pad = _pt(fig, CLEARANCE_PT)
    return [name for name, b in boxes if _hit(bb, b, pad)]


def inside_axes(ax, txt, renderer):
    bb = txt.get_window_extent(renderer).extents
    a = ax.get_window_extent().extents
    return bb[0] >= a[0] and bb[2] <= a[2] and bb[1] >= a[1] and bb[3] <= a[3]


def place_labels(fig, panels):
    """Resolution label: leftmost spot just above the line that touches nothing.
    'excludes 0' label: under the zero line below its interval; upright if it
    fits between the neighbouring whiskers, otherwise turned on its side."""
    renderer = fig.canvas.get_renderer()
    for ax, p, res_label, zero_labels in panels:
        boxes = mark_boxes(ax, p)
        lo_x, hi_x = ax.get_xlim()
        for xc in np.arange(lo_x + 0.05, hi_x, 0.02):
            res_label.xy = (xc, p["resolution"])
            if not text_hits(fig, res_label, boxes, renderer) and inside_axes(ax, res_label, renderer):
                break
        else:
            raise AssertionError("no free place for the resolution label")
        for xi, lab in zero_labels:
            ok = False
            for rotation in (0, 90):
                lab.set_rotation(rotation)
                lab.xy = (xi, min(0.0, p["los"][xi - 1]))
                if not text_hits(fig, lab, boxes, renderer) and inside_axes(ax, lab, renderer):
                    ok = True
                    break
            assert ok, "no free place for the 'excludes 0' label"


def all_texts(fig):
    out = []
    for ax in fig.axes:
        out += [t for t in ax.texts]
        out += [ax.title, ax.xaxis.label, ax.yaxis.label]
        out += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def verify_layout(fig, panels):
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    texts = all_texts(fig)
    # No text touches a marker, whisker, cap, the zero line or the resolution line.
    for ax, p, _, _ in panels:
        boxes = mark_boxes(ax, p)
        for t in texts:
            hits = text_hits(fig, t, boxes, renderer)
            assert not hits, f"text {t.get_text()!r} touches {hits}"
    # No two texts overlap.
    bbs = [(t.get_text(), t.get_window_extent(renderer).extents) for t in texts]
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            assert not _hit(bbs[i][1], bbs[j][1], 0.0), f"{bbs[i][0]!r} overlaps {bbs[j][0]!r}"
    # No background box on any text.
    for t in texts:
        assert t.get_bbox_patch() is None, f"text {t.get_text()!r} has a background box"
    # All nine pairs labelled in each panel.
    for ax, p, _, _ in panels:
        assert [t.get_text() for t in ax.get_xticklabels()] == NEIGHBOURS
    # Wording.
    for t in texts:
        assert not re.search(r"\bconfigs?\b", t.get_text(), re.I)


def check_line_in_png(fig, panels, dpi):
    """Every pixel column across each panel shows the resolution line's colour."""
    img = mpimg.imread(OUT_PNG)[..., :3]
    H = img.shape[0]
    s = dpi / fig.dpi
    acc = np.array(matplotlib.colors.to_rgb(COLOR_ACCENT))
    target = RES_ALPHA * acc + (1 - RES_ALPHA) * np.ones(3)
    report = []
    for ax, p, _, _ in panels:
        x0, _, x1, _ = ax.get_window_extent().extents
        py = ax.transData.transform((0, p["resolution"]))[1]
        row = int(round(H - py * s))
        cols = np.arange(int(np.ceil(x0 * s)) + 1, int(np.floor(x1 * s)) - 1)
        band = img[row - 1:row + 2, cols, :]                       # 3 rows around the line
        dist = np.abs(band - target).max(axis=2).min(axis=0)      # best row per column
        bad = int((dist > 0.15).sum())
        assert bad == 0, f"resolution line hidden in {bad} of {len(cols)} pixel columns"
        report.append((len(cols), float(dist.max())))
    return report


def check_pdf_fonts():
    data = open(OUT_PDF, "rb").read()
    assert b"/FontFile2" in data, "no embedded TrueType font program found"
    assert b"/Type3" not in data, "Type 3 font found"
    return len(re.findall(rb"/FontFile2", data))


# ---------------------------------------------------------------------------
# Facts file
# ---------------------------------------------------------------------------

def facts_table(p, unit_scale_note):
    lines = ["| Places | Entries (higher / lower) | Gap | 95% interval | CR2 standard error | Degrees of freedom | Interval excludes 0 | Gap reaches resolution |",
             "|---|---|---|---|---|---|---|---|"]
    ex = excludes_zero(p)
    for i in range(9):
        lines.append(
            f"| {p['labels'][i]} | {p['names'][i][0]} / {p['names'][i][1]} | {p['gaps'][i]:.2f} | "
            f"[{p['los'][i]:.2f}, {p['his'][i]:.2f}] | {p['ses'][i]:.3f} | {p['dfs'][i]:.2f} | "
            f"{'yes' if ex[i] else 'no'} | {'yes' if p['gaps'][i] >= p['resolution'] else 'no'} |")
    return "\n".join(lines)


def write_facts(gift, fev, png_check, n_fonts):
    def counts(p):
        ex = excludes_zero(p)
        return int(ex.sum()), [p["labels"][i] for i in np.where(ex)[0]], int((p["gaps"] >= p["resolution"]).sum())
    g_ex, g_ex_pairs, g_reach = counts(gift)
    f_ex, f_ex_pairs, f_reach = counts(fev)
    text = f"""# Figure 1 (`fig_resolution.pdf`): what is plotted and where each number comes from

Written by `figs/make_fig_resolution.py` in the same run that drew the figure, so the numbers below are the plotted numbers.

Command (one line, clean shell; the throwaway environment needs matplotlib, numpy, seaborn and scipy):

```
{RUN_COMMAND}
```

## What each panel plots

Both panels: for the nine neighbouring pairs of the displayed top ten (places 1-2 to 9-10), the marker is the gap between the two entries (positive means the higher-placed entry is ahead) and the whisker is its 95% interval from the CR2 method: a t interval, gap plus or minus t(0.975, df) times the cluster-robust CR2 standard error, with df from Satterthwaite's approximation. A filled marker means the interval excludes zero; a hollow marker means it does not. The orange line is the board's corrected resolution: for each pair, (t(0.975, df) + t(0.80, df)) times its CR2 standard error, the smallest true gap detected eight times in ten at the 5% level; the board's value is the median over the nine pairs.

Panel (a), GIFT-Eval, mean rank on MASE, snapshot of 2026-09-07. Gaps in rank places. Resolution {gift['resolution']:.2f} rank places; median neighbour gap {gift['median_gap']:.2f}.

Panel (b), fev-bench ({fev['n_units']} tasks, {fev['n_sources']} task families, from `reports.A_wr.n_units` and `n_sources`), win rate, the board as pinned on 2026-11-11 (autogluon/fev commit 9cf943c, 40 entries; unchanged at the 2026-11-13 re-pin). Gaps in win-rate points (0 to 100). Resolution {fev['resolution']:.2f} win-rate points; median neighbour gap {fev['median_gap']:.2f}.

## Counts

| Panel | Intervals that exclude zero (of 9) | Gaps that reach the resolution line (of 9) | Largest gap | Resolution |
|---|---|---|---|---|
| (a) GIFT-Eval | {g_ex} ({', '.join(g_ex_pairs) or 'none'}) | {g_reach} | {gift['gaps'].max():.2f} | {gift['resolution']:.2f} |
| (b) fev-bench | {f_ex} ({', '.join(f_ex_pairs) or 'none'}) | {f_reach} | {fev['gaps'].max():.2f} | {fev['resolution']:.2f} |

The zero counts are uncorrected (each pair tested at 5% on its own). After the Holm correction over the nine pairs no GIFT-Eval pair separates: the smallest corrected p-value is {min(gift['holm9_cr2'].values()):.2f} (column `holm9_cr2` of `pair_table_MASE_Rank.csv`; places 9-10 has {gift['holm9_cr2']['9-10']:.2f}).

## Plotted numbers, panel (a) GIFT-Eval

Source: `camera_ready/pair_table_MASE_Rank.csv`, the row whose `places` is the pair; gap = `gap_theta`, interval = `ci95_lo`, `ci95_hi`; standard error = `cr2_se`; degrees of freedom = `df`. The script checks that the same row in `camera_ready/retest_gift.json` (`pairs.MASE_Rank[k].theta` and `pairs.MASE_Rank[k]["CR2-icc"]`, with k = {gift['json_index']} for the nine pairs in order) holds identical values, and asserts every value against the printed table to 2 decimals (passed).
Resolution line: `retest_gift.json` key `resolution.MASE_Rank.summary.corr_res_icc` = {gift['resolution']:.4f}. Median gap: `resolution.MASE_Rank.summary.median_gap` = {gift['median_gap']:.4f}.

{facts_table(gift, '')}

## Plotted numbers, panel (b) fev-bench

Source: `camera_ready/boards/fev_family_report.json`, report `reports.A_wr` (fev-bench, 100 tasks, win rate; `n_entries` = 40). The nine pairs are the entries of `reports.A_wr.pairs` with `adjacent` true (indices {fev['json_index']}, in place order); gap = `theta`, interval = `cr2_icc.ci95`, standard error = `cr2_icc.se`, degrees of freedom = `cr2_icc.df`. The file stores win rates as fractions, so gap, interval and standard error are multiplied by 100 to give points; nothing else is changed.
Resolution line: `reports.A_wr.resolution.corrected_icc` x 100 = {fev['resolution']:.4f} (the summary `summaries.A_wr.resolution_corrected` = 7.513 agrees). Median gap: `reports.A_wr.resolution.median_adjacent_gap` x 100 = {fev['median_gap']:.4f}.

{facts_table(fev, '')}

## Checks run by the script (all passed)

- The nine GIFT-Eval values equal the printed table to 2 decimals; the CSV and `retest_gift.json` agree exactly.
- For both panels, each stored interval equals gap plus or minus t(0.975, df) times the stored standard error, and the stored resolution equals the median over the nine pairs of (t(0.975, df) + t(0.80, df)) times the standard error (differences below 1e-6). These are checks on stored numbers; the plotted values are the stored intervals.
- fev-bench: the stored count of neighbouring pairs separated under CR2 (`summaries.A_wr.adjacent_of_9.CR2-icc`) is 0, matching the nine plotted intervals.
- No text touches a marker, a whisker, a cap, the zero line or the resolution line (at least {CLEARANCE_PT:.0f} pt of free space); no two texts overlap; no text has a background box; all nine pairs are labelled on each x axis; no text in the figure uses a shortened form of "configuration".
- In the 200 dpi preview every pixel column across each panel shows the resolution line's colour (panel (a): {png_check[0][0]} columns, panel (b): {png_check[1][0]} columns).
- The PDF embeds its font as TrueType ({n_fonts} embedded font program; no Type 3 fonts).

## Sentences a caption can use

Gaps between neighbouring entries of each board's displayed top ten, with 95% intervals from the CR2 cluster-robust standard error (t intervals with Satterthwaite degrees of freedom); the orange line is the board's resolution, the smallest gap it can detect eight times in ten ({gift['resolution']:.2f} rank places on GIFT-Eval, {fev['resolution']:.2f} win-rate points on fev-bench).
No gap reaches the line, and only one of the eighteen intervals excludes zero (GIFT-Eval places 9-10, filled marker, uncorrected; it does not survive the Holm correction over nine pairs).
"""
    assert not re.search(r"\bconfigs?\b", text, re.I)
    open(OUT_FACTS, "w").write(text)
    return (g_ex, g_reach, f_ex, f_reach)


# ---------------------------------------------------------------------------

def main():
    gift = load_gift()
    fev = load_fev()

    sns.set_theme(context="paper", style="white")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,   # embed TrueType, keep text as text
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
    panels = []
    for ax, p, title, ylabel, color in (
        (axes[0], gift, "(a) GIFT-Eval", "gap in mean rank on MASE (places)", COLOR_GIFT),
        (axes[1], fev, "(b) fev-bench", "gap in win rate (points)", COLOR_FEV),
    ):
        res_label, zero_labels = draw_panel(ax, p, title, ylabel, color)
        panels.append((ax, p, res_label, zero_labels))

    # Layout, then place labels, then re-layout until stable.
    for _ in range(3):
        fig.tight_layout(w_pad=2.2)
        fig.canvas.draw()
        place_labels(fig, panels)
    fig.tight_layout(w_pad=2.2)
    fig.canvas.draw()
    verify_layout(fig, panels)

    fig.savefig(OUT_PDF)
    png_dpi = 200
    fig.savefig(OUT_PNG, dpi=png_dpi)
    png_check = check_line_in_png(fig, panels, png_dpi)
    n_fonts = check_pdf_fonts()
    plt.close(fig)

    counts = write_facts(gift, fev, png_check, n_fonts)

    for name, p in (("GIFT-Eval (rank places)", gift), ("fev-bench (win-rate points)", fev)):
        print(f"{name}: resolution {p['resolution']:.2f}, median gap {p['median_gap']:.2f}")
        for i in range(9):
            print(f"  {p['labels'][i]:5s} {p['gaps'][i]:5.2f} [{p['los'][i]:6.2f}, {p['his'][i]:6.2f}]")
    print("intervals excluding zero / gaps reaching resolution: "
          f"GIFT-Eval {counts[0]} / {counts[1]}, fev-bench {counts[2]} / {counts[3]}")
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}\nwrote {OUT_FACTS}")
    print("OK: all checks passed")


if __name__ == "__main__":
    main()
