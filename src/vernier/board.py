"""A leaderboard as a table of scores, entries x units, and the rules that turn its scores into an order.

A unit is one evaluation task, such as a dataset configuration, and its source is the data it was cut from.
Units of one source are correlated, so every test clusters by source. Lower scores and values are better.
"""
import csv
import gzip
from typing import NamedTuple

import numpy as np
from scipy.stats import rankdata

from .sources import BoardError

RULES = ("mean-rank", "geometric-mean", "win-rate")
MASE = "eval_metrics/MASE[0.5]"  # the score columns of GIFT-Eval's and BOOM's result files
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
TOP = 10  # the top places whose pairs are tested
TIER_POOL = 40  # the first places tried for the leading tier


class Board(NamedTuple):
    entries: list  # in the board's own order, which breaks tied ranks under ties="first"
    units: list
    sources: list  # the source of each unit
    scores: np.ndarray  # entries x units, NaN where an entry has no result
    band: list  # the board lists the entries of band 0 first, then band 1 and so on; None: not listed
    files: dict  # entry -> the identity of its own files, which changes when their content does


def make_board(results, source_of, band=None, files=None) -> Board:
    """Return the Board of {entry: {unit: score}}, its entries in the given order and its units sorted."""
    entries = list(results)
    units = sorted({unit for scores in results.values() for unit in scores})
    scores = np.array([[results[entry].get(unit, np.nan) for unit in units] for entry in entries], float)
    band = band or {}
    return Board(entries, units, [source_of(unit) for unit in units], scores.reshape(len(entries), len(units)),
                 [band.get(entry, 0) for entry in entries], files or {})


def load_csv(path, metric="score", sources="source") -> Board:
    """Return the Board of a table with one row per entry and unit.

    The table has the columns entry and unit, the score column `metric` and the source column `sources`. An
    empty score is a missing result. A column named band, if present, gives the board's listing band of each
    entry, empty for an entry the board does not list. Entries keep the order of their first row.
    """
    try:
        with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except (OSError, UnicodeDecodeError) as error:
        raise BoardError(f"{path}: {getattr(error, 'strerror', None) or 'not a UTF-8 text file'}") from None
    absent = [column for column in ("entry", "unit", sources, metric) if not rows or column not in rows[0]]
    if absent:
        raise BoardError(f"{path} needs the columns entry, unit, {sources} and {metric}, "
                         f"one row per entry and unit; it has no {' or '.join(absent)}")
    results, source_of, band = {}, {}, {}
    for number, row in enumerate(rows, 2):
        if None in row or None in row.values():  # csv puts surplus values under None and fills absent ones with it
            raise BoardError(f"{path}, line {number}: not as many values as the header has columns")
        scores = results.setdefault(row["entry"], {})
        if row["unit"] in scores:
            raise BoardError(f"{path}, line {number}: a second row for {row['entry']} on {row['unit']}")
        score, listed = row[metric].strip(), row.get("band", "0").strip()
        try:
            scores[row["unit"]] = float(score) if score else np.nan
        except ValueError:
            raise BoardError(f"{path}, line {number}: {metric} is {score!r}, not a number") from None
        if np.isinf(scores[row["unit"]]) or not listed.lstrip("-").isdigit() and listed:
            raise BoardError(f"{path}, line {number}: {metric} {score!r} must be finite (empty if missing) "
                             f"and band {listed!r} a whole number")
        band[row["entry"]] = int(listed) if listed else None
        source_of[row["unit"]] = row[sources]
    # An entry's own results are its scores: the bystander test asks whether they changed between two tables.
    return make_board(results, source_of.get, band, {entry: tuple(sorted(scores.items()))
                                                     for entry, scores in results.items()})


def ranks(scores, ties="first") -> np.ndarray:
    """Return each entry's rank on each unit among the entries scored there (1 = lowest score), NaN if unscored."""
    # 'first' breaks ties by entry order, as a board's rank(method="first") on its own table.
    method = {"first": "ordinal", "average": "average"}[ties]
    return rankdata(scores, method=method, axis=0, nan_policy="omit")


def log_scaled_error(scores, baseline: int) -> np.ndarray:
    """Return log(score / the baseline entry's score) on each unit; its mean is the log of the geometric mean."""
    return np.log(scores / scores[baseline])


def win_rate(scores) -> np.ndarray:
    """Return the share of the other entries scored on a unit that each entry beats there; a tie counts half."""
    scored = (~np.isnan(scores)).sum(0)
    with np.errstate(invalid="ignore"):  # a unit with one scored entry has no win rate
        return (scored - rankdata(scores, method="average", axis=0, nan_policy="omit")) / (scored - 1)


def values_under(rule, board: Board, ties="first", baseline=None) -> np.ndarray:
    """Return entries x units values whose mean orders the entries under `rule`, lower being better.

    mean-rank gives rank places, geometric-mean the log of error over the baseline entry's error, and win-rate
    minus the win rate in percentage points.
    """
    if rule == "mean-rank":
        return ranks(board.scores, ties)
    if rule == "win-rate":
        return -100 * win_rate(board.scores)
    if rule != "geometric-mean":
        raise BoardError(f"the rule is {rule!r}; the rules are {', '.join(RULES)}")
    if baseline not in board.entries:
        raise BoardError(f"the geometric mean scales every score by the entry named {baseline!r}, and the board "
                         "has no such entry; name the baseline entry with --baseline")
    scale = board.scores[board.entries.index(baseline)]
    if np.isnan(scale).any() or not (board.scores[~np.isnan(board.scores)] > 0).all():
        raise BoardError(f"the geometric mean needs a score of {baseline} on every unit, and every score above zero")
    return log_scaled_error(board.scores, board.entries.index(baseline))


def listed_order(board: Board, values) -> list:
    """Return the entries the board lists that have a value on every unit, by band and then by mean value."""
    shown = [i for i, row in enumerate(values) if board.band[i] is not None and not np.isnan(row).any()]
    return sorted(shown, key=lambda i: (board.band[i], values[i].mean()))
