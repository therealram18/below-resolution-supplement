"""Build fig_received.pdf: documented (nominal) vs. received (median-window
effective) context length, for the 48 default-setting runs in
effective_context/effective_context.csv.

Usage: bin/python make_received.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(HERE, "..", "effective_context", "effective_context.csv"))


def load_default_rows():
    with open(CSV_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "default"]
    assert len(rows) == 48, f"expected 48 default-setting rows, found {len(rows)}"
    return rows


def jitter_x(nominal):
    """Deterministic, plot-only jitter: points sharing the same documented
    (nominal) value are fanned out a few percent in log space by group index,
    so identical points don't stack exactly on top of each other. No
    randomness (reproducible without a seed) and nothing stored -- the
    `nominal` array used for the below-diagonal count and every annotation
    is untouched."""
    groups = defaultdict(list)
    for i, v in enumerate(nominal):
        groups[v].append(i)
    x_plot = np.array(nominal, dtype=float)
    STEP = 0.014  # log10 units, ~3% per step
    for v, idxs in groups.items():
        n = len(idxs)
        for k, i in enumerate(idxs):
            x_plot[i] = v * (10 ** ((k - (n - 1) / 2) * STEP))
    return x_plot


def main():
    rows = load_default_rows()
    nominal = np.array([float(r["nominal"]) for r in rows])
    eff_median = np.array([float(r["eff_median"]) for r in rows])
    below = eff_median < nominal
    assert below.sum() == 13, f"expected 13 below-diagonal points, found {below.sum()}"

    x_plot = jitter_x(nominal)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.85, 3.85))

    lo, hi = 60, 18500  # lo covers the smallest received value (toto, m4_daily: 79)
    ax.plot([lo, hi], [lo, hi], color="0.45", linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)
    ax.text(0.60, 0.685, "setting binds", transform=ax.transAxes, fontsize=7.3,
            rotation=45, rotation_mode="anchor", ha="left", va="bottom", color="0.3")

    ax.scatter(x_plot[~below], eff_median[~below], s=30, facecolor="none",
               edgecolor="black", linewidth=0.8, zorder=3, label="received = documented")
    ax.scatter(x_plot[below], eff_median[below], s=30, facecolor="black",
               edgecolor="black", linewidth=0.8, zorder=3, label="received < documented")

    # us_births callout: two default-setting configs claim different context
    # lengths (2048, 15360) but both received the same 983 -- the argument
    # that "documented" and "received" are different numbers.
    us_idx = [
        i for i, r in enumerate(rows)
        if r["config"].startswith("us_births")
        and float(r["eff_median"]) == 983.0
        and float(r["nominal"]) in (2048.0, 15360.0)
    ]
    assert len(us_idx) == 2, f"expected exactly 2 us_births 983-median points at nominal 2048/15360, found {len(us_idx)}"
    i_lo, i_hi = sorted(us_idx, key=lambda i: x_plot[i])
    y = 983.0
    # Bracket hangs *below* the two points, not above -- the space above them
    # is the 1,944-median cluster (spans the same x-range); the space below,
    # down to the lone (4096, 79) point, is clear.
    tick_bot = y / 1.75
    for i in (i_lo, i_hi):
        ax.plot([x_plot[i], x_plot[i]], [tick_bot, y], color="black", linewidth=0.7, zorder=4)
    x_mid = np.sqrt(x_plot[i_lo] * x_plot[i_hi])
    ax.plot([x_plot[i_lo], x_plot[i_hi]], [tick_bot, tick_bot], color="black", linewidth=0.7, zorder=4)
    ax.text(x_mid, tick_bot * 0.85,
            "us_births: documented 2048 & 15360,\nboth received 983",
            fontsize=6.6, ha="center", va="top", linespacing=1.25)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("documented context length (nominal)", fontsize=8)
    ax.set_ylabel("received context length (median window)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_aspect("equal", adjustable="box")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, handletextpad=0.4,
              borderaxespad=0.2)

    fig.tight_layout()
    out_pdf = os.path.join(HERE, "fig_received.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)

    return rows, nominal, eff_median, below, out_pdf


def _self_check():
    """ponytail: one runnable check -- row count, below-diagonal count, the
    us_births pair, output PDF exists and is non-trivial in size."""
    rows, nominal, eff_median, below, out_pdf = main()

    assert len(rows) == 48
    assert below.sum() == 13
    assert os.path.getsize(out_pdf) > 1000, "PDF looks empty"

    print(f"default-setting rows: {len(rows)}")
    print(f"below-diagonal (received < documented): {int(below.sum())} of {len(rows)}")
    print("below-diagonal points (model, config, nominal -> eff_median):")
    for r, n, e, b in zip(rows, nominal, eff_median, below):
        if b:
            print(f"  {r['model']:16s} {r['config']:38s} {int(n):6d} -> {e:g}")
    print(f"Wrote {out_pdf} ({os.path.getsize(out_pdf)} bytes)")
    print("OK: self-check passed.")


if __name__ == "__main__":
    _self_check()
