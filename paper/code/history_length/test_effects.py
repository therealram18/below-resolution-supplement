"""The history-length numbers of the paper, recomputed from the stored runs in paper/results/runs."""
import json
from pathlib import Path

import pytest
from effects import effects, eligible_units

RESULTS = Path(__file__).resolve().parents[2] / "results"
RUNS = RESULTS / "runs"
SUBMITTED = RESULTS / "record" / "submitted" / "selection" / "selected_configs.json"
HEADLINE = ["chronos_bolt_base", "moirai2_small", "timesfm25", "toto__A__seed42"]


def printed(*texts):
    """Return comparisons that accept any value which rounds to each printed number."""
    return [pytest.approx(float(text), abs=0.5 * 10 ** -len(text.partition(".")[2])) for text in texts]


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
