"""The tests of a zero mean difference, each checked against a slower form that is plainly right."""
from itertools import product

import numpy as np
import pytest
from scipy import stats

from vernier.stats import (
    _satterthwaite_df,
    cr2_test,
    percentile_separated,
    sign_flip_p,
    sign_flip_t_p,
    signed_rank_p,
    source_totals,
    wild_p,
)

RNG = np.random.default_rng(7)
SOURCES = np.repeat(np.arange(9), [1, 3, 2, 1, 4, 1, 2, 5, 1])  # 20 units from 9 sources
DIFFERENCES = RNG.normal(0.4, 1, 20) + RNG.normal(0, 0.7, 9)[SOURCES]


def every_pattern(n_sources):
    return np.array(list(product([1.0, -1.0], repeat=n_sources)))


def t_statistic(differences, sources):
    """Return the mean over its standard error clustered by source (CR1), written out unit by unit."""
    residual = differences - differences.mean()
    totals = np.array([residual[sources == source].sum() for source in np.unique(sources)])
    n_sources = len(totals)
    return differences.mean() / (np.sqrt(n_sources / (n_sources - 1) * (totals ** 2).sum()) / len(differences))


def test_sign_flip_p_counts_every_pattern():
    totals, _ = source_totals(DIFFERENCES, SOURCES)
    sums = every_pattern(9) @ totals
    assert sign_flip_p(totals) == np.mean(np.abs(sums) >= abs(totals.sum()) * (1 - 1e-9))
    assert sign_flip_p(np.zeros(5)) == 1.0
    assert sign_flip_p([2.0]) == 1.0


def test_sign_flip_t_p_counts_every_pattern():
    observed = abs(t_statistic(DIFFERENCES, SOURCES))
    flipped = [abs(t_statistic(DIFFERENCES * signs[SOURCES], SOURCES)) for signs in every_pattern(9)]
    assert sign_flip_t_p(DIFFERENCES, SOURCES) == pytest.approx(np.mean(np.array(flipped) >= observed * (1 - 1e-9)))


def test_wild_p_lists_every_pattern_of_few_sources_and_draws_them_of_many():
    assert wild_p(DIFFERENCES, SOURCES)[0] == sign_flip_t_p(DIFFERENCES, SOURCES)
    sources = np.repeat(np.arange(24), 3)  # 2^23 patterns: more than the draws, and few enough to list
    differences = RNG.normal(0.3, 1, (3, 72))
    exact = [sign_flip_t_p(row, sources) for row in differences]
    assert wild_p(differences, sources) == pytest.approx(exact, abs=0.006)  # four standard errors of 99,999 draws
    assert list(wild_p(differences, sources)) == list(wild_p(differences, sources))  # the same on every run


def test_tests_of_identical_entries_do_not_separate_them():
    zero = np.zeros(20)
    assert wild_p(zero, SOURCES)[0] == 1.0
    assert sign_flip_t_p(zero, SOURCES) == 1.0
    assert signed_rank_p(zero, SOURCES) == 1.0
    assert not percentile_separated(zero, SOURCES)[0]


def test_signed_rank_p_is_the_exact_wilcoxon_test_on_source_means():
    means = np.array([DIFFERENCES[SOURCES == source].mean() for source in range(9)])
    assert signed_rank_p(DIFFERENCES, SOURCES) == pytest.approx(stats.wilcoxon(means, method="exact").pvalue)


def test_degrees_of_freedom_equal_the_matrix_form():
    for correlation in (0.0, 0.3):
        sizes = np.bincount(SOURCES).astype(float)
        n_units = len(SOURCES)
        # Row g of `weights` holds source g's CR2 weight on each unit; the overall mean is then removed.
        weights = np.zeros((9, n_units))
        weights[SOURCES, np.arange(n_units)] = ((1 - sizes / n_units) ** -0.5 / n_units)[SOURCES]
        centred = weights @ (np.eye(n_units) - np.ones((n_units, n_units)) / n_units)
        within = np.where(SOURCES[:, None] == SOURCES[None, :], correlation, 0.0)
        np.fill_diagonal(within, 1.0)
        covariance = centred @ within @ centred.T
        expected = np.trace(covariance) ** 2 / (covariance ** 2).sum()
        assert _satterthwaite_df(sizes, correlation) == pytest.approx(expected, rel=1e-10)


def test_cr2_with_one_unit_per_source_is_the_paired_t_test():
    differences = RNG.normal(0.5, 1, 12)
    test, paired = cr2_test(differences, np.arange(12)), stats.ttest_1samp(differences, 0)
    assert (test.p, test.df) == pytest.approx((paired.pvalue, 11))


def test_percentile_rule_needs_both_intervals_to_exclude_zero():
    sources = np.repeat(np.arange(6), 10)
    steady = RNG.normal(1.0, 0.5, 60)  # far from zero on every unit: separated
    one_source = np.where(sources == 0, 12.0, 0.0) + RNG.normal(0, 0.1, 60)  # all of it from one source
    assert list(percentile_separated(np.array([steady, one_source]), sources)) == [True, False]
