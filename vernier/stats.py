"""Tests for the mean of per-unit differences when the units come in correlated groups (sources).

cr2_test is a t test whose standard error is clustered by source; resolution turns that standard error into the
smallest difference the test detects; sign_flip_p and sign_flip_interval are exact tests on the sources' totals.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy import stats

ALPHA = 0.05  # the level of every test: a difference is separated when p < ALPHA
# A sign pattern whose total equals the observed one up to rounding counts as at least as far from zero.
TOLERANCE = 1e-9


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
    return max(0.0, (between - within) / (between + (typical_size - 1) * within))


def _satterthwaite_df(sizes, correlation):
    """Return the degrees of freedom of the CR2 variance when units of one source are equally correlated."""
    n_units = sizes.sum()
    weights = (1 - sizes / n_units) ** -0.5 / n_units
    # Row g is source g's CR2 weight after removing the overall mean; it is constant within each source,
    # so the N x N computation reduces to one over sources.
    loadings = np.diag(weights) - (sizes * weights / n_units)[:, None]
    covariance = (loadings * sizes * (1 + correlation * (sizes - 1))) @ loadings.T
    return np.trace(covariance) ** 2 / (covariance ** 2).sum()


def cr2_test(differences, sources) -> Test:
    """Return the two-sided t test of a zero mean difference, with the CR2 standard error clustered by source."""
    differences = np.asarray(differences, float)
    n_units, gap = len(differences), differences.mean()
    totals, sizes = source_totals(differences - gap, sources)
    se = np.sqrt((totals ** 2 / (1 - sizes / n_units)).sum()) / n_units
    df = _satterthwaite_df(sizes, _correlation(differences, sources))
    half_width = stats.t.ppf(1 - ALPHA / 2, df) * se
    return Test(gap, se, df, 2 * stats.t.sf(abs(gap) / se, df), gap - half_width, gap + half_width)


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
    observed = abs(np.sum(totals)) * (1 - TOLERANCE)
    return float(np.mean(np.abs(_signed_sums(totals)) >= observed))


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
