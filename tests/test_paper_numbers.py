"""The paper's key numbers, recomputed from the files in this repository.

Each expected number is written as the paper prints it, and a computed value passes when it rounds to it.
The replay reads the leaderboard's git history, which is not in this repository: its test runs only when the
environment variable GIFT_EVAL_CLONE names a local clone of the GIFT-Eval leaderboard.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from vernier import (effects, eligible_units, gift_eval_order, load_csv, load_gift_eval, log_scaled_error, ranks,
                     report, reversals, snapshots_from_git)
from vernier.board import CRPS, MASE
from vernier.stats import holm, holm_adjusted

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "snapshot" / "snapshot_2026-09-07" / "results"
RUNS = ROOT / "runs"
SUBMITTED = ROOT / "record" / "submitted" / "selection" / "selected_configs.json"
HEADLINE = ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto__A__seed42"]
PIN = "b61c48aa3197a1a111e405c0e6f600128a1efee3"  # the commit the snapshot was taken from


def printed(*texts):
    """Return comparisons that accept any value which rounds to each printed number."""
    return [pytest.approx(float(text), abs=0.5 * 10 ** -len(text.partition(".")[2])) for text in texts]


def mean_rank_report(metric):
    board = load_gift_eval(SNAPSHOT, metric)
    values = ranks(board.scores, ties="first")
    return report(values, board.sources, board.entries, gift_eval_order(board, values))


def separated_neighbours(result):
    return [pair for pair in result.neighbours if pair.test.p < 0.05]


def test_mean_rank_on_mase():
    result = mean_rank_report(MASE)
    assert [result.resolution, result.median_gap] == printed("6.801", "0.9072")
    [pair] = separated_neighbours(result)
    assert pair.places == (9, 10)
    assert list(pair.test) == printed("1.3299", "0.5857", "17.07", "0.0364", "0.09", "2.57")
    assert (result.neighbours_separated, result.neighbours_after_holm) == (1, 0)
    assert (result.pairs_separated, result.pairs_after_holm) == (12, 0)
    assert result.tier == ["STRIDE w/ Synapse", "EXAONE-Forecast-Agent", "LS-MoE"]


def test_mean_rank_on_crps():
    result = mean_rank_report(CRPS)
    assert [result.resolution] == printed("7.38")
    assert (result.neighbours_separated, result.neighbours_after_holm) == (0, 0)
    assert (result.pairs_separated, result.pairs_after_holm) == (9, 0)
    assert len(result.tier) == 3


def test_geometric_mean_of_scaled_error():
    board = load_gift_eval(SNAPSHOT)
    errors = log_scaled_error(board.scores, board.entries.index("Seasonal_Naive"))
    own_order = report(errors, board.sources, board.entries, gift_eval_order(board, errors))
    assert [own_order.resolution, own_order.median_gap] == printed("0.0497", "0.0045")
    assert len(own_order.tier) == 17
    mean_rank_order = gift_eval_order(board, ranks(board.scores, ties="first"))
    mean_rank_top = report(errors, board.sources, board.entries, mean_rank_order)
    assert (mean_rank_top.neighbours_separated, mean_rank_top.neighbours_after_holm) == (0, 0)
    assert (mean_rank_top.pairs_separated, mean_rank_top.pairs_after_holm) == (10, 0)


def submitted_units():
    return [unit["config"] for unit in json.loads(SUBMITTED.read_text())["selected"]]


def test_history_length_on_70_eligible_configurations():
    found = effects(RUNS / "history_length", HEADLINE, eligible_units(RUNS / "eligible_configurations.json"))
    assert [(e.n_units, e.n_sources) for e in found] == [(70, 14)] * 4
    assert [100 * e.effect for e in found] == printed("2.7", "6.2", "15.3", "15.3")
    assert [e.p for e in found] == printed("0.1146", "0.0051", "0.000244", "0.000244")
    assert [e.p_holm for e in found] == printed("0.1146", "0.0103", "0.000977", "0.000977")
    assert [100 * e.low for e in found] == printed("-0.9", "1.6", "4.1", "4.0")
    assert [100 * e.high for e in found] == printed("6.2", "10.4", "32.6", "27.1")


def test_history_length_on_12_submitted_configurations():
    found = effects(RUNS / "history_length", HEADLINE + ["timesfm3", "timestral209m"], submitted_units())
    assert [e.n_units for e in found] == [12] * 6
    assert [100 * e.effect for e in found] == printed("-2.0", "6.8", "22.2", "22.0", "31.8", "28.9")
    assert [e.p for e in found[4:]] == printed("0.0156", "0.0039")


def test_holm_on_fixed_p_values():
    assert list(holm([0.01, 0.04, 0.03])) == [True, False, False]
    assert list(holm_adjusted([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_report_on_a_csv_board(tmp_path):
    scores = np.random.default_rng(1).random((12, 6))
    rows = [f"e{entry:02},u{unit},s{unit // 2},{scores[entry, unit]}" for entry in range(12) for unit in range(6)]
    (tmp_path / "board.csv").write_text("\n".join(["entry,unit,source,score", *rows]))
    board = load_csv(tmp_path / "board.csv")
    values = ranks(board.scores)
    order = gift_eval_order(board, values)
    result = report(values, board.sources, board.entries, order)
    assert len(result.neighbours) == 9
    assert sum(np.isfinite(pair.test.p) for pair in result.pairs) == 45
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report(values, board.units, board.entries, order)  # every unit its own source


def test_an_entry_with_the_leaders_values_is_in_the_tier():
    values = np.random.default_rng(1).random((12, 6))
    order = list(np.argsort(values.mean(1)))
    values[order[1]] = values[order[0]]
    entries = [f"e{entry:02}" for entry in range(12)]
    with np.errstate(invalid="ignore"):  # the entry's differences from the leader are all zero, so 0 / 0
        result = report(values, [f"s{unit // 2}" for unit in range(6)], entries, order)
    assert entries[order[1]] in result.tier


@pytest.mark.skipif("GIFT_EVAL_CLONE" not in os.environ, reason="needs a clone of the GIFT-Eval leaderboard")
def test_replay_of_the_leaderboard_history():
    snapshots = snapshots_from_git(os.environ["GIFT_EVAL_CLONE"], until=PIN)
    first = reversals(snapshots, ties="first")
    assert first.updates == 141
    assert len({reversal.commit for reversal in first.reversals}) == 14
    assert len(first.reversals) == 20
    assert len(reversals(snapshots, ties="average").reversals) == 18
