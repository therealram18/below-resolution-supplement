"""A leaderboard as a table of scores, entries x units, and the tests that say which of its places differ.

A unit is one dataset configuration and its source is the dataset it was cut from; units of one source are
correlated, so every test clusters by source. Lower scores, ranks and statistics are better.
"""
import csv
import io
import json
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import rankdata

from .stats import ALPHA, Test, cr2_test, holm, resolution

MASE = "eval_metrics/MASE[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
LEAKAGE_ORDER = {"No": 0, "Yes": 1}  # then N/A, as GIFT-Eval sorts its table
TOP = 10  # the top places whose pairs are tested
TIER_POOL = 40  # the first places tried for the leading tier


class Board(NamedTuple):
    entries: list[str]  # alphabetical
    units: list[str]
    sources: list[str]  # the source of each unit
    scores: np.ndarray  # entries x units, NaN where an entry has no result
    leakage: list[Optional[str]]  # declared test-data leakage; None where no config.json names the entry


class Pair(NamedTuple):
    places: tuple
    entries: tuple
    test: Test  # of the lower-placed value minus the higher-placed one
    holm: bool  # separated after Holm over all pairs of the top entries


class Report(NamedTuple):
    pairs: list[Pair]
    neighbours: list[Pair]  # the pairs of neighbouring places, in place order
    resolution: float
    median_gap: float
    neighbours_separated: int
    neighbours_after_holm: int
    pairs_separated: int
    pairs_after_holm: int
    tier: list[str]


def read_results(text, metric=MASE):
    """Return {entry: {unit: score}} from the text of an all_results.csv file."""
    scores = {}
    for row in csv.DictReader(io.StringIO(text)):
        scores.setdefault(row["model"], {})[row["dataset"]] = float(row[metric])
    return scores


def read_leakage(text):
    """Return the entry a config.json names and its declared leakage."""
    config = json.loads(text)
    return config["model"], config.get("testdata_leakage", "N/A")


def make_board(results, leakage) -> Board:
    """Return the Board of {entry: {unit: score}}, with `leakage` given for the entries a config.json names."""
    entries = sorted(results)
    units = sorted({unit for scores in results.values() for unit in scores})
    scores = np.array([[results[entry].get(unit, np.nan) for unit in units] for entry in entries])
    sources = [unit.split("/")[0] for unit in units]
    return Board(entries, units, sources, scores, [leakage.get(entry) for entry in entries])


def load_gift_eval(results_dir, metric=MASE) -> Board:
    """Return the Board of a GIFT-Eval results folder, one subfolder per entry."""
    results = {}
    for path in sorted(Path(results_dir).glob("*/all_results.csv")):
        results.update(read_results(path.read_text(), metric))
    leakage = dict(read_leakage(path.read_text()) for path in sorted(Path(results_dir).glob("*/config.json")))
    return make_board(results, leakage)


def load_csv(path) -> Board:
    """Return the Board of a table with the columns entry, unit, source, score."""
    results, source_of = {}, {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            results.setdefault(row["entry"], {})[row["unit"]] = float(row["score"])
            source_of[row["unit"]] = row["source"]
    board = make_board(results, dict.fromkeys(results, ""))
    return board._replace(sources=[source_of[unit] for unit in board.units])


def ranks(scores, ties="first") -> np.ndarray:
    """Return each entry's rank on each unit among the entries scored there (1 = lowest score), NaN if unscored."""
    # 'first' breaks ties by entry order, as GIFT-Eval's rank(method="first") on its alphabetical table.
    method = {"first": "ordinal", "average": "average"}[ties]
    return rankdata(scores, method=method, axis=0, nan_policy="omit")


def log_scaled_error(scores, baseline: int) -> np.ndarray:
    """Return log(score / the baseline entry's score) on each unit; its mean is the log of the geometric mean."""
    return np.log(scores / scores[baseline])


def gift_eval_order(board: Board, values) -> list[int]:
    """Return the entries GIFT-Eval displays with a value on every unit, by declared leakage and then mean value."""
    shown = [i for i, row in enumerate(values) if board.leakage[i] is not None and not np.isnan(row).any()]
    return sorted(shown, key=lambda i: (LEAKAGE_ORDER.get(board.leakage[i], 2), values[i].mean()))


def report(values, sources, entries, order) -> Report:
    """Return the CR2 tests among the top entries of `order`, the board's resolution and its leading tier."""
    places = list(combinations(range(TOP), 2))
    tests = [cr2_test(values[order[b]] - values[order[a]], sources) for a, b in places]
    pairs = [Pair((a + 1, b + 1), (entries[order[a]], entries[order[b]]), t, rejected)
             for (a, b), t, rejected in zip(places, tests, holm([t.p for t in tests]))]
    neighbours = [pair for pair in pairs if pair.places[1] == pair.places[0] + 1]
    leader = order[0]
    # `not p < ALPHA` keeps an entry with exactly the leader's values: its p is NaN and it is not separated.
    tier = [entries[leader]] + [entries[k] for k in order[1:TIER_POOL]
                                if not cr2_test(values[k] - values[leader], sources).p < ALPHA]
    return Report(
        pairs=pairs,
        neighbours=neighbours,
        resolution=float(np.median([resolution(pair.test.se, pair.test.df) for pair in neighbours])),
        median_gap=float(np.median([abs(pair.test.gap) for pair in neighbours])),
        neighbours_separated=sum(pair.test.p < ALPHA for pair in neighbours),
        neighbours_after_holm=int(holm([pair.test.p for pair in neighbours]).sum()),
        pairs_separated=sum(pair.test.p < ALPHA for pair in pairs),
        pairs_after_holm=sum(pair.holm for pair in pairs),
        tier=tier,
    )
