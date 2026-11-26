"""Tests for the mean of per-unit differences when the units come in correlated groups (sources).

cr2_test is a t test whose standard error is clustered by source, and resolution turns that standard error into
the smallest difference the test detects. The sign-flip tests give each source's differences one sign: wild_p
draws the signs, sign_flip_t_p and sign_flip_p list every pattern, and signed_rank_p lists them on the ranks of
the sources' means. percentile_separated is the bootstrap rule of the submitted paper, kept for comparison.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy import stats

ALPHA = 0.05  # the level of every test: a difference is separated when p < ALPHA
# A sign pattern whose total equals the observed one up to rounding counts as at least as far from zero.
TOLERANCE = 1e-9
SEED = 20261111  # of every random draw, so that a report is the same on every run
WILD_DRAWS = 99_999
BOOTSTRAP_DRAWS = 20_000
EXACT_T_SOURCES = 28  # sign_flip_t_p lists 2^(G-1) patterns; beyond this it takes minutes per pair
EXACT_SUM_SOURCES = 50  # sign_flip_p sorts 2^(G/2) sums


class Test(NamedTuple):
    gap: float
    se: float
    df: float
    p: float
    low: float
    high: float


def source_totals(values, sources):
    """Return the sum of `values` over each source and each source's number of units (sources sorted)."""
    _, index = np.unique(sources, return_inverse=True)
    return np.bincount(index, weights=values), np.bincount(index).astype(float)


def _correlation(differences, sources):
    """Return the ANOVA estimate of the correlation between units of one source, floored at zero."""
    totals, sizes = source_totals(differences, sources)
    n_units, n_sources = len(differences), len(sizes)
    if n_units == n_sources:  # every source has one unit, so there is no within-source variation
        return 0.0
    explained = (totals ** 2 / sizes).sum()
    between = (explained - n_units * np.mean(differences) ** 2) / (n_sources - 1)
    within = ((differences ** 2).sum() - explained) / (n_units - n_sources)
    typical_size = (n_units - (sizes ** 2).sum() / n_units) / (n_sources - 1)
    total = between + (typical_size - 1) * within
    return max(0.0, (between - within) / total) if total > 0 else 0.0  # differences that are all equal: no variation


def _satterthwaite_df(sizes, correlation):
    """Return the degrees of freedom of the CR2 variance when units of one source are equally correlated.

    They are trace(C)^2 / sum(C^2), where C is the covariance of the sources' weighted totals after the overall
    mean is removed. Source g's row of weights is `weight[g]` on its own units and `-shift[g]` on every unit, so
    C = diag(weight^2 spread) - outer(pull, shift) - outer(shift, pull) + spread.sum() outer(shift, shift)
    with pull = weight * spread. Its trace and its sum of squares are written out from that form, which costs
    one pass over the sources where the matrix costs their cube.
    """
    n_units = sizes.sum()
    weight = (1 - sizes / n_units) ** -0.5 / n_units
    spread = sizes * (1 + correlation * (sizes - 1))  # the variance of a source's total, in units of one unit's
    shift, pull, total = sizes * weight / n_units, weight * spread, spread.sum()
    diagonal = weight ** 2 * spread
    rest = total * shift ** 2 - 2 * pull * shift  # the diagonal of the three outer products
    squares = ((diagonal + rest) ** 2).sum() - (rest ** 2).sum() + 2 * (pull @ pull) * (shift @ shift) \
        + (total * (shift @ shift)) ** 2 + 2 * (pull @ shift) ** 2 - 4 * total * (pull @ shift) * (shift @ shift)
    return (diagonal + rest).sum() ** 2 / squares


def cr2_test(differences, sources) -> Test:
    """Return the two-sided t test of a zero mean difference, with the CR2 standard error clustered by source."""
    differences = np.asarray(differences, float)
    n_units, gap = len(differences), differences.mean()
    totals, sizes = source_totals(differences - gap, sources)
    se = np.sqrt((totals ** 2 / (1 - sizes / n_units)).sum()) / n_units
    df = _satterthwaite_df(sizes, _correlation(differences, sources))
    half_width = stats.t.ppf(1 - ALPHA / 2, df) * se
    with np.errstate(divide="ignore", invalid="ignore"):  # differences that are all equal have no standard error:
        p = 2 * stats.t.sf(abs(gap) / se, df)  # p is 0, or NaN when they are all zero
    return Test(gap, se, df, p, gap - half_width, gap + half_width)


def resolution(se: float, df: float) -> float:
    """Return the smallest true difference a 5% test detects eight times in ten."""
    return (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(0.80, df)) * se


def holm_adjusted(p_values) -> np.ndarray:
    """Return Holm-adjusted p-values: the k-th smallest p times (m - k + 1), made non-decreasing, capped at 1."""
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values, kind="stable")
    steps = np.maximum.accumulate(p_values[order] * (len(p_values) - np.arange(len(p_values))))
    adjusted = np.empty(len(p_values))
    adjusted[order] = np.minimum(steps, 1.0)
    return adjusted


def holm(p_values) -> np.ndarray:
    """Return which hypotheses the Holm procedure rejects at level ALPHA."""
    return holm_adjusted(p_values) <= ALPHA


def _signed_sums(values):
    """Return the 2^G sums of +-values over every sign pattern."""
    sums = np.zeros(1)
    for value in values:
        sums = np.concatenate([sums + value, sums - value])
    return sums


def sign_flip_p(totals) -> float:
    """Return the share of the 2^G sign patterns whose total is at least as far from zero as the observed total."""
    totals = np.asarray(totals, float)
    # The patterns of the first half of the sources meet the sorted patterns of the second half, so 2^G sums
    # are counted from two lists of 2^(G/2).
    first, second = _signed_sums(totals[:len(totals) // 2]), np.sort(_signed_sums(totals[len(totals) // 2:]))
    observed = abs(totals.sum()) * (1 - TOLERANCE)
    above = len(second) - np.searchsorted(second, observed - first, side="left")
    below = np.searchsorted(second, -observed - first, side="right")
    # The two sides share patterns only when the observed total is zero, and then every pattern counts.
    return float(np.minimum(above + below, len(second)).sum() / (len(first) * len(second)))


def sign_flip_interval(differences, sources):
    """Return the lowest and highest shift at which sign_flip_p of the totals of differences - shift is >= ALPHA."""
    totals, sizes = source_totals(np.asarray(differences, float), sources)
    signed_totals, signed_sizes = _signed_sums(totals), _signed_sums(sizes)
    total, n_units = totals.sum(), sizes.sum()
    # A pattern counts at a shift exactly between these two ends, an interval that contains the estimate;
    # the all-plus and all-minus patterns count at every shift.
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (total - signed_totals) / (n_units - signed_sizes)
        second = (total + signed_totals) / (n_units + signed_sizes)
    always = np.abs(signed_sizes) == n_units
    low = np.where(always, -np.inf, np.minimum(first, second))
    high = np.where(always, np.inf, np.maximum(first, second))
    needed = math.ceil(ALPHA * 2 ** len(totals))
    return np.sort(low)[needed - 1], np.sort(high)[-needed]


def signed_rank_p(differences, sources) -> float:
    """Return the exact Wilcoxon signed-rank p-value of the sources' mean differences, each source counted once."""
    totals, sizes = source_totals(np.asarray(differences, float), sources)
    means = totals / sizes
    return sign_flip_p(np.sign(means) * stats.rankdata(np.abs(means)))


def _t_squared(total, weighted, sizes, squares):
    """Return the squared t statistic of sign patterns from total = sum(w s), weighted = sum(w n s), sum(s^2).

    s is a source's total, n its number of units and w its sign. The standard error is clustered by source
    (CR1). A pattern with no variation between sources gets 0.
    """
    n_units, n_sources = sizes.sum(), len(sizes)
    spread = squares - 2 * total * weighted / n_units + total ** 2 * (sizes ** 2).sum() / n_units ** 2
    spread = spread * n_sources / (n_sources - 1)
    defined = spread > 1e-13 * squares
    return np.where(defined, total ** 2 / np.where(defined, spread, 1.0), 0.0)


def _pattern_sums(totals, sizes):
    """Return sum(w s) and sum(w n s) over every sign pattern w of the given sources."""
    total, weighted = np.zeros(1), np.zeros(1)
    for s, n in zip(totals, sizes):
        total, weighted = np.concatenate([total + s, total - s]), np.concatenate([weighted + n * s, weighted - n * s])
    return total, weighted


def sign_flip_t_p(differences, sources, chunk=4_000_000) -> float:
    """Return the share of all sign patterns whose t statistic is at least as far from zero as the observed one.

    This is the exact form of wild_p. Changing every sign leaves the statistic unchanged, so the first source
    keeps its sign and 2^(G-1) patterns are listed, in two halves that are combined chunk by chunk.
    """
    totals, sizes = source_totals(np.asarray(differences, float), sources)
    squares, half = (totals ** 2).sum(), (len(totals) + 1) // 2
    observed = _t_squared(totals.sum(), (sizes * totals).sum(), sizes, squares) * (1 - TOLERANCE) ** 2
    total_a, weighted_a = _pattern_sums(totals[1:half], sizes[1:half])
    total_a, weighted_a = total_a + totals[0], weighted_a + sizes[0] * totals[0]
    total_b, weighted_b = _pattern_sums(totals[half:], sizes[half:])
    rows, count = max(1, chunk // len(total_b)), 0
    for start in range(0, len(total_a), rows):
        block = slice(start, start + rows)
        count += int((_t_squared(total_a[block, None] + total_b, weighted_a[block, None] + weighted_b,
                                 sizes, squares) >= observed).sum())
    return count / (len(total_a) * len(total_b))


def wild_p(differences, sources, draws=WILD_DRAWS) -> np.ndarray:
    """Return the wild cluster bootstrap p-value of a zero mean for each row of `differences` (pairs x units).

    Each draw gives every source one random sign, the null imposed, and p = (1 + draws at least as far from
    zero) / (1 + draws). When the sources have no more sign patterns than `draws`, every pattern is listed.
    """
    differences = np.atleast_2d(np.asarray(differences, float))
    _, sizes = source_totals(differences[0], sources)
    if 2 ** (len(sizes) - 1) <= draws:
        return np.array([sign_flip_t_p(row, sources) for row in differences])
    totals = np.array([source_totals(row, sources)[0] for row in differences])
    squares = (totals ** 2).sum(1)
    observed = _t_squared(totals.sum(1), (totals * sizes).sum(1), sizes, squares) * (1 - TOLERANCE) ** 2
    rng, count = np.random.default_rng(SEED), np.zeros(len(totals))
    for start in range(0, draws, 2_000):
        signs = rng.choice([-1.0, 1.0], size=(min(2_000, draws - start), len(sizes)))
        count += (_t_squared(signs @ totals.T, signs @ (totals * sizes).T, sizes, squares) >= observed).sum(0)
    return (1 + count) / (1 + draws)


def _excludes_zero(means) -> np.ndarray:
    """Return whether the central 95% of each column of bootstrap means lies on one side of zero."""
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=0)
    return (low > 0) | (high < 0)


def percentile_separated(differences, sources, draws=BOOTSTRAP_DRAWS) -> np.ndarray:
    """Return, for each row of `differences`, whether two 95% percentile bootstrap intervals both exclude zero.

    One bootstrap resamples units and the other whole sources. This is the rule of the submitted paper. It
    rejects too often when units of one source move together, so it is reported for comparison only.
    """
    differences = np.atleast_2d(np.asarray(differences, float))
    _, index = np.unique(sources, return_inverse=True)
    sizes = np.bincount(index).astype(float)
    totals = np.array([np.bincount(index, weights=row) for row in differences])
    n_pairs, n_units, n_sources = *differences.shape, len(sizes)
    rng, step = np.random.default_rng(SEED), max(50, int(2e6 // max(n_units, n_sources * n_pairs)))
    by_unit, by_source = np.empty((draws, n_pairs)), np.empty((draws, n_pairs))
    for start in range(0, draws, step):
        block = slice(start, min(start + step, draws))
        units = rng.integers(0, n_units, size=(block.stop - start, n_units))
        by_unit[block] = np.stack([row[units].mean(1) for row in differences], axis=1)
        picked = rng.integers(0, n_sources, size=(block.stop - start, n_sources))
        by_source[block] = (totals[:, picked].sum(2) / sizes[picked].sum(1)).T
    return _excludes_zero(by_unit) & _excludes_zero(by_source)
