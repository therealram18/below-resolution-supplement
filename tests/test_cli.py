"""The command line on tables written by hand: what it prints, and what it says when it cannot go on."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RNG = np.random.default_rng(3)
ROOT = Path(__file__).resolve().parent.parent


def table(n_entries=12, header="entry,unit,source,score", n_units=6):
    scores = RNG.random((n_entries, n_units)) + 0.5
    rows = [f"e{entry:02},u{unit},s{unit // 2},{scores[entry, unit]}"
            for entry in range(n_entries) for unit in range(n_units)]
    return "\n".join([header, *rows])


def vernier(*args, **environment):
    command = [sys.executable, "-m", "vernier", *map(str, args)]
    return subprocess.run(command, capture_output=True, text=True, cwd=ROOT, check=False, env=os.environ | environment)


@pytest.fixture
def scores(tmp_path):
    (tmp_path / "scores.csv").write_text(table())
    return tmp_path / "scores.csv"


def test_report_on_a_table(scores):
    done = vernier("report", scores)
    assert done.returncode == 0 and done.stderr == ""
    assert "mean-rank on score, ties first: 12 entries listed with all 6 units from 3 sources" in done.stdout
    assert "CR2 t-test" in done.stdout and "wild cluster bootstrap" in done.stdout


def test_json_report_is_json(scores):
    found = json.loads(vernier("report", scores, "--json", "--tests", "all").stdout)
    assert [place["place"] for place in found["places"]] == list(range(1, 11))
    assert len(found["pairs"]) == 45 and found["fallback"] is False
    assert found["ratio"] == pytest.approx(found["resolution"] / found["median_gap"])
    assert set(found["tests"]) == {"cr2", "wild", "exact", "exact-sum", "signed-rank", "percentile"}
    assert found["tests"]["percentile"]["p"] is None and len(found["tests"]["wild"]["p"]) == 45


def test_a_board_with_fewer_than_ten_entries_reports_all_of_them(tmp_path):
    (tmp_path / "six.csv").write_text(table(n_entries=6))
    done = vernier("report", tmp_path / "six.csv")
    assert done.returncode == 0 and "6 entries listed" in done.stdout
    counts = next(line for line in done.stdout.splitlines() if line.strip().startswith("CR2 t-test"))
    assert " of 5, " in counts and " of 15, " in counts  # five neighbouring pairs and fifteen pairs


def test_entries_with_the_same_scores_give_a_clean_report(tmp_path):
    rows = [f"e{entry:02},u{unit},s{unit // 2},{1 + (entry > 3) * (1 + entry + unit % 3)}"
            for entry in range(12) for unit in range(6)]
    (tmp_path / "same.csv").write_text("\n".join(["entry,unit,source,score", *rows]))
    for ties, p in (("first", 0.0), ("average", None)):  # split by their order, or tied: the t test then has no p
        done = vernier("report", tmp_path / "same.csv", "--json", "--ties", ties)
        found = json.loads(done.stdout, parse_constant=pytest.fail)  # NaN is not JSON
        assert done.stderr == "" and found["pairs"][0]["p"]["cr2"] == p


def test_higher_is_better_reverses_the_order(scores):
    lower = json.loads(vernier("report", scores, "--json").stdout)["places"]
    higher = json.loads(vernier("report", scores, "--json", "--higher-is-better").stdout)["places"]
    all_twelve = [f"e{entry:02}" for entry in range(12)]
    assert lower[0]["entry"] != higher[0]["entry"]
    assert higher[0]["entry"] in set(all_twelve) - {place["entry"] for place in lower[:2]}


def test_geometric_mean_and_win_rate_on_a_table(tmp_path, scores):
    done = vernier("report", scores, "--rule", "geometric-mean", "--baseline", "e11")
    assert done.returncode == 0 and "signed-rank on source means" in done.stdout and "holds its level" in done.stdout
    assert vernier("report", scores, "--rule", "win-rate").returncode == 0


def test_order_by_lists_by_one_rule_and_tests_under_another(scores):
    by_rank = json.loads(vernier("report", scores, "--json").stdout)
    geometric = json.loads(vernier("report", scores, "--json", "--rule", "geometric-mean", "--baseline", "e11").stdout)
    mixed = json.loads(vernier("report", scores, "--json", "--rule", "geometric-mean", "--baseline", "e11",
                               "--order-by", "mean-rank").stdout)
    assert (mixed["rule"], mixed["ordered_by"], geometric["ordered_by"]) == ("geometric-mean", "mean-rank", "geometric-mean")
    assert [place["entry"] for place in mixed["places"]] == [place["entry"] for place in by_rank["places"]]
    means = {place["entry"]: place["mean"] for place in geometric["places"]}
    assert all(place["mean"] == means[place["entry"]] for place in mixed["places"] if place["entry"] in means)


def test_other_columns_name_the_score_and_the_source(tmp_path):
    (tmp_path / "wide.csv").write_text(table(header="entry,unit,family,accuracy"))
    done = vernier("report", tmp_path / "wide.csv", "--metric", "accuracy", "--sources", "family", "--higher-is-better")
    assert done.returncode == 0 and "mean-rank on accuracy" in done.stdout


@pytest.mark.parametrize("arguments, message", [
    (["report"], "name a leaderboard with --board or give a table"),
    (["report", "absent.csv"], "absent.csv: No such file or directory"),
    (["report", "{wrong}"], "needs the columns entry, unit, source and score"),
    (["report", "{scores}", "--rule", "geometric-mean"], "name the baseline entry with --baseline"),
    (["report", "{scores}", "--tests", "bootstrap"], "'bootstrap' is not a test"),
    (["report", "{one}"], "the report needs two entries"),
    (["report", "{text}"], "line 3: score is 'fast', not a number"),
    (["report", "{twice}"], "line 3: a second row for a on u"),
    (["report", "{comma}"], "line 2: not as many values as the header has columns"),
    (["report", "{scores}", "--board", "boom", "--pinned"], "name it with --board and give no source"),
    (["report", "--pinned"], "name a leaderboard"),
    (["report", "--board", "boom", "--pinned", "--metric", "SQL"], "BOOM has the metrics CRPS, MASE"),
    (["report", "--board", "boom", "--pinned", "--tests", "exact"], "2554 sources have too many"),
    (["report", "--board", "boom", "--pinned", "--higher-is-better"], "--higher-is-better is for a table of scores"),
    (["report", "{folder}", "--board", "fev-bench"], "fev-bench needs complete results of Seasonal Naive"),
])
def test_a_request_that_cannot_run_says_why(tmp_path, arguments, message):
    files = {"scores": table(), "wrong": table(header="model,dataset,group,value"), "one": table(n_entries=1),
             "text": "entry,unit,source,score\na,u,s,1\na,v,s,fast",
             "twice": "entry,unit,source,score\na,u,s,1\na,u,s,2",
             "comma": "entry,unit,source,score\nModel, large,u,s,1\nb,u,s,2"}
    for name, content in files.items():
        (tmp_path / f"{name}.csv").write_text(content)
    (tmp_path / "folder").mkdir()
    paths = {name: tmp_path / f"{name}.csv" for name in files} | {"folder": tmp_path / "folder"}
    done = vernier(*(argument.format(**paths) for argument in arguments))
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("vernier: ") and message in done.stderr and "Traceback" not in done.stderr


def test_every_board_in_one_command():
    done = vernier("report", "--board", "all", "--pinned", "--json")
    rows = json.loads(done.stdout)
    assert done.returncode == any("error" in row for row in rows.values())  # 1 until TIME's snapshot is held
    assert list(rows) == ["gift-eval", "fev-bench", "fev-bench-mini", "chronos-bench-ii", "time", "boom"]
    assert rows["gift-eval"]["resolution"] == pytest.approx(6.801, abs=5e-4) and rows["boom"]["sources"] == 2554
    assert all(row["schema"] == 1 for row in rows.values())
    assert "takes no source" in vernier("report", "scores.csv", "--board", "all").stderr
    lines = vernier("report", "--board", "all", "--pinned").stdout.splitlines()
    assert lines[1].split()[:8] == ["gift-eval", "mean-rank", "128", "97", "28", "6.8010", "0.9072", "7.5"]


def test_a_board_that_cannot_be_reached_falls_back_to_its_snapshot_and_says_so(tmp_path):
    offline = {"GIT_ALLOW_PROTOCOL": "file", "VERNIER_CACHE": str(tmp_path)}  # git refuses to clone over https
    done = vernier("report", "--board", "fev-bench", **offline)
    assert done.returncode == 0 and done.stdout.startswith("fev-bench, NOT the board today: the paper's snapshot")
    assert "cannot reach https://github.com/autogluon/fev" in done.stderr
    asked = vernier("report", "--board", "fev-bench", "--at", "9cf943c3", **offline)  # a named commit has no stand-in
    assert asked.returncode == 1 and "cannot reach" in asked.stderr and asked.stdout == ""


def test_json_from_replay_and_boards_and_the_version(tmp_path):
    (tmp_path / "2026-01.csv").write_text(table())
    (tmp_path / "2026-02.csv").write_text(table())
    replay = json.loads(vernier("replay", tmp_path, "--json").stdout)
    assert (replay["updates"], replay["rule"], replay["schema"]) == (1, "mean-rank", 1)
    boards = json.loads(vernier("boards", "--json").stdout)
    assert boards["time"]["url"].startswith("https://huggingface.co/") and boards["boom"]["snapshot"] is True
    assert vernier("--version").stdout.strip() == "vernier 1.0.0"


def test_boards_lists_every_board():
    lines = vernier("boards").stdout.splitlines()
    names = ["gift-eval", "fev-bench", "fev-bench-mini", "chronos-bench-ii", "time", "boom"]
    assert [line.split()[0] for line in lines[1:]] == names


def test_pinned_report_names_its_snapshot():
    done = vernier("report", "--board", "fev-bench", "--pinned")
    first = "fev-bench, the paper's snapshot of 2026-11-11 (commit 9cf943c3 of https://github.com/autogluon/fev)"
    assert done.stdout.startswith(first)
    assert "win-rate on SQL: 40 entries listed with all 100 units from 39 sources" in done.stdout
