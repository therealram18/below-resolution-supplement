"""The report: which of a board's top places its data can tell apart, and how small a difference it can detect."""
import math
from itertools import combinations
from typing import NamedTuple

import numpy as np

from .board import TIER_POOL, TOP
from .sources import BoardError
from .stats import (
    ALPHA,
    EXACT_SUM_SOURCES,
    EXACT_T_SOURCES,
    WILD_DRAWS,
    Test,
    cr2_test,
    holm,
    percentile_separated,
    resolution,
    sign_flip_p,
    sign_flip_t_p,
    signed_rank_p,
    source_totals,
    wild_p,
)

TESTS = {  # name -> what the report calls it and the largest number of sources it can list every pattern for
    "cr2": ("CR2 t-test", None),
    "wild": ("wild cluster bootstrap", None),
    "exact": ("exact sign-flip, t statistic", EXACT_T_SOURCES),
    "exact-sum": ("exact sign-flip, plain sum", EXACT_SUM_SOURCES),
    "signed-rank": ("signed-rank on source means", EXACT_SUM_SOURCES),
    "percentile": ("percentile bootstrap", None),
}


class Pair(NamedTuple):
    places: tuple
    entries: tuple
    test: Test  # the CR2 test of the lower-placed value minus the higher-placed one
    holm: bool  # separated by it after Holm over all pairs of the top entries


class Verdict(NamedTuple):
    """What one test says about the top entries."""
    p: list | None  # of each pair, in the order of Report.pairs; None for the percentile rule
    neighbours_separated: int
    neighbours_after_holm: int | None
    pairs_separated: int
    pairs_after_holm: int | None
    tier: list  # the leader and the entries not separated from it among the first TIER_POOL places


class Report(NamedTuple):
    pairs: list
    neighbours: list  # the pairs of neighbouring places, in place order
    resolution: float
    median_gap: float
    neighbours_separated: int  # these five repeat verdicts["cr2"]
    neighbours_after_holm: int
    pairs_separated: int
    pairs_after_holm: int
    tier: list
    verdicts: dict  # test name -> Verdict

    @property
    def ratio(self) -> float:
        """The resolution in median gaps between neighbours: above 1, the board cannot resolve its neighbours."""
        return self.resolution / self.median_gap if self.median_gap else math.inf


def _p_values(test, differences, sources):
    """Return the p-value of a zero mean for each row of `differences` under `test`."""
    if test == "wild":
        return wild_p(differences, sources)
    one = {"cr2": lambda row: cr2_test(row, sources).p,
           "exact": lambda row: sign_flip_t_p(row, sources),
           "exact-sum": lambda row: sign_flip_p(source_totals(row, sources)[0]),
           "signed-rank": lambda row: signed_rank_p(row, sources)}[test]
    return np.array([one(row) for row in differences])


def _verdict(test, pairs, leader, neighbour, sources, entries, order) -> Verdict:
    """Return what `test` says of the pair differences and of the differences from the leader."""
    if test == "percentile":
        apart, behind = percentile_separated(pairs, sources), percentile_separated(leader, sources)
        p = after_holm = neighbours_after_holm = None
    else:
        p, behind = _p_values(test, pairs, sources), _p_values(test, leader, sources) < ALPHA
        apart, after_holm, neighbours_after_holm = p < ALPHA, int(holm(p).sum()), int(holm(p[neighbour]).sum())
    # `not separated` keeps an entry with exactly the leader's values: its CR2 p is NaN and it is not separated.
    tier = [entries[order[0]]] + [entries[k] for k, separated in zip(order[1:TIER_POOL], behind) if not separated]
    return Verdict(None if p is None else p.tolist(), int(apart[neighbour].sum()), neighbours_after_holm,
                   int(apart.sum()), after_holm, tier)


def report(values, sources, entries, order, tests=("cr2",)) -> Report:
    """Return the tests among the top entries of `order`, the board's resolution and its leading tier.

    `values` is entries x units with lower being better, and `order` lists the entries from the first place
    down. Each test in `tests` gives a Verdict. The CR2 test always runs: it gives the resolution.
    """
    n_sources = len(set(sources))
    if len(order) < 2 or n_sources < 2:
        raise BoardError(f"the report needs two entries with a value on every unit and two sources; "
                         f"this board has {len(order)} such entries and {n_sources} sources")
    for test in tests:
        if test not in TESTS:
            raise BoardError(f"{test!r} is not a test; the tests are {', '.join(TESTS)}")
        if (TESTS[test][1] or n_sources) < n_sources:
            raise BoardError(f"the test {test} lists every sign pattern of the sources, and {n_sources} sources "
                             f"have too many; the test wild draws {WILD_DRAWS:,} of them")
    values, top = np.asarray(values, float), order[:TOP]
    places = list(combinations(range(len(top)), 2))
    neighbour = np.array([b == a + 1 for a, b in places])
    pair_differences = np.array([values[top[b]] - values[top[a]] for a, b in places])
    leader_differences = np.array([values[k] - values[order[0]] for k in order[1:TIER_POOL]])
    cr2 = [cr2_test(row, sources) for row in pair_differences]
    pairs = [Pair((a + 1, b + 1), (entries[top[a]], entries[top[b]]), test, bool(rejected))
             for (a, b), test, rejected in zip(places, cr2, holm([test.p for test in cr2]))]
    neighbours = [pair for pair in pairs if pair.places[1] == pair.places[0] + 1]
    verdicts = {test: _verdict(test, pair_differences, leader_differences, neighbour, sources, entries, order)
                for test in dict.fromkeys(("cr2", *tests))}
    return Report(
        pairs=pairs,
        neighbours=neighbours,
        resolution=float(np.median([resolution(pair.test.se, pair.test.df) for pair in neighbours])),
        median_gap=float(np.median([abs(pair.test.gap) for pair in neighbours])),
        **{field: getattr(verdicts["cr2"], field) for field in Verdict._fields[1:]},
        verdicts=verdicts,
    )


def as_text(result: Report, entries, values, order, shown=lambda mean: mean) -> str:
    """Return the report as a plain-text table; `shown` turns a mean value into the number the board displays."""
    with_p = [test for test, verdict in result.verdicts.items() if verdict.p is not None]
    lines = [f"{'place':>5}  {'entry':<34}{'mean':>9}  {'gap to next':>11}  {'95% interval':>19}"
             + "".join(f"  {'p, ' + test:>14}" for test in with_p)]
    for place, row in enumerate(order[:TOP], 1):
        line = f"{place:>5}  {entries[row]:<34}{shown(values[row].mean()):9.4f}"
        if place <= len(result.neighbours):
            pair = result.neighbours[place - 1]
            line += f"  {pair.test.gap:11.4f}  {pair.test.low:8.4f} to {pair.test.high:<7.4f}"
            line += "".join(f"  {result.verdicts[test].p[result.pairs.index(pair)]:14.4f}" for test in with_p)
        lines.append(line)
    resolution, gap, ratio = result.resolution, result.median_gap, result.ratio
    lines += [f"Resolution, the gap a 5% test detects 8 times in 10 (median over neighbours): {resolution:.4f}",
              f"Median gap between neighbours: {gap:.4f}, so the resolution is {ratio:.3g} times the gap",
              f"Separated at 5%, and after Holm:  {'neighbouring pairs':<20}{'all pairs':<20}leading tier"]
    for test, verdict in result.verdicts.items():
        near, every = (f"{separated} of {size}" + ("" if after_holm is None else f", {after_holm}")
                       for separated, size, after_holm in (
                           (verdict.neighbours_separated, len(result.neighbours), verdict.neighbours_after_holm),
                           (verdict.pairs_separated, len(result.pairs), verdict.pairs_after_holm)))
        lines.append(f"  {TESTS[test][0]:<32}{near:<20}{every:<20}{len(verdict.tier)}")
    lines.append(f"Leading tier: the leader and the entries not separated from it among the first {TIER_POOL} places")
    for test, verdict in result.verdicts.items():
        same = test != "cr2" and verdict.tier == result.tier
        lines.append(f"  {TESTS[test][0]}: {'the same entries' if same else ', '.join(verdict.tier)}")
    return "\n".join(lines)


def as_dict(result: Report, entries, values, order, shown=lambda mean: mean) -> dict:
    """Return the report as plain data for JSON; a value that is not a finite number becomes None."""
    def number(value):
        return value if math.isfinite(value) else None

    return {
        "places": [{"place": place, "entry": entries[row], "mean": float(shown(values[row].mean()))}
                   for place, row in enumerate(order[:TOP], 1)],
        "resolution": result.resolution,
        "median_gap": result.median_gap,
        "ratio": number(result.ratio),
        "pairs": [{"places": pair.places, "entries": pair.entries,
                   **{name: number(value) for name, value in pair.test._asdict().items()},
                   "p": {test: number(verdict.p[index]) for test, verdict in result.verdicts.items() if verdict.p}}
                  for index, pair in enumerate(result.pairs)],
        "tests": {test: {"name": TESTS[test][0], **verdict._asdict(), "p": verdict.p and [number(p) for p in verdict.p]}
                  for test, verdict in result.verdicts.items()},
    }
