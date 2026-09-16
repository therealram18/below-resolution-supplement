"""Build fig_timeline.pdf: bystander replay timeline for the displayed rule
(MASE_Rank) vs. the geometric-mean rule (gmean), over the same 141 leaderboard
updates.

Parses poolcheck/replay_output.txt directly -- the [MASE_Rank] and [gmean]
blocks -- so the figure and the numbers in its caption come from the same
parse, not a second hand-typed copy.

Usage: bin/python make_timeline.py
"""
import ast
import os
import re
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
POOLCHECK = os.path.normpath(os.path.join(HERE, "..", "poolcheck"))
REPLAY_TXT = os.path.join(POOLCHECK, "replay_output.txt")


def parse_block(text, name):
    """Parse one `[name] transitions T; transitions reversing unchanged
    top-10 entries N; reversed pairs P` header plus its N tuple lines below
    it. Each line is a valid Python tuple literal
    (date_from, date_to, [reversed pairs], [what arrived]), so
    ast.literal_eval reads it directly instead of a hand-rolled regex."""
    m = re.search(
        r"\[" + re.escape(name) + r"\] transitions (\d+); "
        r"transitions reversing unchanged top-10 entries (\d+); "
        r"reversed pairs (\d+)",
        text,
    )
    if not m:
        raise ValueError(f"block [{name}] not found in {REPLAY_TXT}")
    transitions, n_reversing, n_pairs = (int(g) for g in m.groups())
    rest = text[m.end():]
    nxt = re.search(r"\n\[", rest)
    body = rest[: nxt.start()] if nxt else rest
    tuples = [ast.literal_eval(ln.strip()) for ln in body.splitlines() if ln.strip()]
    assert len(tuples) == n_reversing, (
        f"[{name}]: header says {n_reversing} reversing transitions, "
        f"found {len(tuples)} tuple lines"
    )
    pairs_total = sum(len(t[2]) for t in tuples)
    assert pairs_total == n_pairs, (
        f"[{name}]: header says {n_pairs} reversed pairs, tuples sum to {pairs_total}"
    )
    return dict(transitions=transitions, n_reversing=n_reversing, n_pairs=n_pairs, tuples=tuples)


def load():
    text = open(REPLAY_TXT).read()
    mase = parse_block(text, "MASE_Rank")
    gmean = parse_block(text, "gmean")
    assert mase["transitions"] == gmean["transitions"] == 141, "expected 141 updates under both rules"
    return mase, gmean


def _ymd(s):
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def main():
    mase, gmean = load()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    x_start, x_end = date(2024, 11, 1), date(2026, 9, 30)
    y_top, y_bot = 1.0, 0.0

    # Continuous baselines stand in for "every one of the 141 updates" -- the
    # update dates for non-reversing transitions aren't all in the txt, so the
    # lane label carries the "141 updates" count instead of per-update ticks.
    ax.hlines([y_top, y_bot], x_start, x_end, color="0.65", linewidth=1.1, zorder=1)

    # Top lane: a marker at each of the 14 reversing updates (date_to), area
    # scaled by how many top-10 pairs that single update reversed.
    dates_to = [_ymd(t[1]) for t in mase["tuples"]]
    n_pairs_each = [len(t[2]) for t in mase["tuples"]]
    AREA_PER_PAIR = 22.0
    ax.scatter(
        dates_to, [y_top] * len(dates_to),
        s=[AREA_PER_PAIR * n for n in n_pairs_each],
        facecolor="black", edgecolor="white", linewidth=0.6, zorder=3,
    )

    # Bottom lane: zero markers by construction (gmean reversed nothing).
    # Nothing to plot -- the empty lane next to the busy one is the figure.

    # Lane labels as plain in-axes text (not y-tick labels) so they don't eat
    # into the horizontal margin the date axis needs.
    ax.text(x_start, y_top + 0.30, "Mean rank (the displayed rule) -- 141 updates, 14 reversing",
            fontsize=7.3, ha="left", va="bottom")
    ax.text(x_start, y_bot - 0.22, "Geometric mean of scaled error -- no reversals in the same 141 updates",
            fontsize=7.3, ha="left", va="top")
    ax.set_yticks([])

    # Three callouts: the RacineCast pair reversing 2026-07-31 and reversing
    # back 2026-08-07, and the 2026-09-07 update where one arrival reversed
    # three pairs at once (a snapshot week). Anchors sit within ~5 weeks of
    # each other at the very right edge of a 22-month axis, so labels are
    # fanned out with leader lines rather than placed at their anchor's own x.
    by_date_to = {t[1]: t for t in mase["tuples"]}
    racine_out = by_date_to["2026-07-31"]
    racine_back = by_date_to["2026-08-07"]
    snapshot = by_date_to["2026-09-07"]
    assert racine_out[2] == [("RacineCast-1-NC", "RacineCast-1")]
    assert racine_back[2] == [("RacineCast-1", "RacineCast-1-NC")]
    assert len(snapshot[2]) == 3

    callouts = [
        (_ymd(racine_out[1]), "RacineCast pair\nreverses, 07-31", (-88, 30)),
        (_ymd(racine_back[1]), "reverses back\n08-07", (-14, 48)),
        (_ymd(snapshot[1]), "1 arrival reverses\n3 pairs, 09-07\n(snapshot week)", (46, 26)),
    ]
    for xy_date, label, offset in callouts:
        ax.annotate(
            label, xy=(xy_date, y_top), xycoords="data",
            xytext=offset, textcoords="offset points",
            fontsize=6.4, ha="center", va="bottom", linespacing=1.2,
            arrowprops=dict(arrowstyle="-|>", color="0.35", linewidth=0.6,
                             mutation_scale=6, shrinkA=1, shrinkB=1),
            zorder=5,
        )

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-0.55, 2.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("0.5")

    fig.tight_layout()
    out_pdf = os.path.join(HERE, "fig_timeline.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)

    return mase, gmean, out_pdf


def _self_check():
    """ponytail: one runnable check -- parsed counts match the header, three
    named callouts exist among the 14 tuples, output PDF exists and is
    non-trivial in size."""
    mase, gmean, out_pdf = main()

    assert mase["n_reversing"] == 14, mase["n_reversing"]
    assert mase["n_pairs"] == 18, mase["n_pairs"]
    assert mase["transitions"] == 141
    assert gmean["n_reversing"] == 0 and gmean["n_pairs"] == 0
    assert os.path.getsize(out_pdf) > 1000, "PDF looks empty"

    sizes = [len(t[2]) for t in mase["tuples"]]
    ones = sizes.count(1)
    threes = sizes.count(3)
    assert ones * 1 + threes * 3 == 18
    assert ones + threes == 14

    print(f"[MASE_Rank] transitions={mase['transitions']} reversing={mase['n_reversing']} pairs={mase['n_pairs']}")
    print(f"  pairs-per-update breakdown: {ones} updates reversed 1 pair, {threes} updates reversed 3 pairs "
          f"({ones} + {threes} = {ones + threes} tuples; {ones}*1 + {threes}*3 = {ones + 3 * threes} pairs)")
    for t in mase["tuples"]:
        print(f"    {t[0]} -> {t[1]}  reversed={t[2]}  arrivals={t[3]}")
    print(f"[gmean] transitions={gmean['transitions']} reversing={gmean['n_reversing']} pairs={gmean['n_pairs']}")
    print(f"Wrote {out_pdf} ({os.path.getsize(out_pdf)} bytes)")
    print("OK: self-check passed.")


if __name__ == "__main__":
    _self_check()
