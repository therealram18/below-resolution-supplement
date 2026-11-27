"""Every board's snapshot against the paper's tables, and every loader against a small board written by hand.

Each expected number is written as the paper prints it, and a computed value passes when it rounds to it.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import commit, printed, write

from vernier import BOARDS, BoardError, Commit, Folder, listed_order, load_csv, load_pin, report
from vernier.board import CRPS, MASE, ranks, values_under
from vernier.boards import pin_file, write_table
from vernier.loaders import boom, chronos_bench_ii, fev_bench, gift_eval, time_board
from vernier.sources import Unreachable, hub_file

ROOT = Path(__file__).resolve().parent.parent
# Table 1 (entries, units, sources) and Table 8 (resolution, median gap, ratio; neighbouring pairs separated
# under CR2, the wild bootstrap and the percentile bootstrap). Geometric means are in log units: the paper
# prints fev-bench's and Chronos Benchmark II's times 100.
PAPER = [
    ("gift-eval", "mean-rank", "source", (128, 97, 28), ("6.80", "0.91", "7.5"), (1, 1, 0)),
    ("gift-eval", "geometric-mean", "source", (128, 97, 28), ("0.0497", "0.0045", "11"), None),
    ("fev-bench", "win-rate", "source", (40, 100, 39), ("7.51", "2.08", "3.6"), (0, 0, 0)),
    ("fev-bench", "geometric-mean", "source", (40, 100, 39), ("0.0320", "0.0083", "3.8"), (0, 0, 0)),
    ("fev-bench-mini", "win-rate", "source", (40, 20, 15), ("15.3", "2.18", "7.0"), (1, 1, 1)),
    ("chronos-bench-ii", "mean-rank", "source", (21, 27, 19), ("4.24", "0.22", "19"), (0, 0, 1)),
    ("chronos-bench-ii", "geometric-mean", "source", (21, 27, 19), ("0.0929", "0.0059", "16"), (0, 0, 0)),
    ("time", "geometric-mean", "source", (43, 98, 39), ("0.0238", "0.0021", "11"), (0, 1, 1)),
    ("time", "mean-rank", "source", (43, 98, 39), ("4.01", "0.378", "11"), (1, 1, 1)),
    ("boom", "mean-rank", "source", (23, 6750, 2554), ("0.196", "0.312", "0.63"), (9, 9, 9)),
    ("boom", "mean-rank", "domain-type", (23, 6750, 36), ("0.420", "0.312", "1.35"), (6, 6, 7)),
    ("boom", "mean-rank", "domain", (23, 6750, 23), ("0.635", "0.312", "2.04"), (5, 6, 7)),
]


def pinned_report(name, rule, sources="source", metric=None, order_by=None, tests=("cr2",)):
    named = BOARDS[name]
    board = load_pin(name, metric or named.metrics[0], sources)
    values = values_under(rule, board, named.ties, named.baseline)
    order = listed_order(board, values_under(order_by or rule, board, named.ties, named.baseline))
    return board, order, report(values, board.sources, board.entries, order, tests)


@pytest.mark.parametrize("name, rule, sources, size, numbers, separated", PAPER)
def test_snapshot_gives_the_papers_numbers(name, rule, sources, size, numbers, separated):
    board, order, result = pinned_report(name, rule, sources, tests=("cr2", "wild", "percentile"))
    assert (len(order), len(board.units), len(set(board.sources))) == size
    assert [result.resolution, result.median_gap, result.resolution / result.median_gap] == printed(*numbers)
    if separated:
        assert tuple(result.verdicts[test].neighbours_separated for test in ("cr2", "wild", "percentile")) == separated


def test_gift_eval_geometric_mean_on_the_displayed_top_ten():
    # Table 8 counts this row's pairs among the ten entries the board displays, which mean rank orders.
    _, _, result = pinned_report("gift-eval", "geometric-mean", order_by="mean-rank",
                                 tests=("cr2", "wild", "exact-sum", "signed-rank", "percentile"))
    verdicts = result.verdicts
    assert [verdicts[test].neighbours_separated for test in ("cr2", "wild", "percentile")] == [0, 0, 2]
    counts = {test: (v.neighbours_separated, v.neighbours_after_holm, v.pairs_separated, v.pairs_after_holm)
              for test, v in verdicts.items() if test != "percentile"}
    assert counts == {"cr2": (0, 0, 10, 0), "wild": (0, 0, 19, 4), "exact-sum": (0, 0, 18, 4),
                      "signed-rank": (1, 0, 12, 1)}


def test_gift_eval_counts_and_tiers_under_each_test():
    _, _, on_mase = pinned_report("gift-eval", "mean-rank", tests=("cr2", "wild", "exact-sum"))
    assert [(v.neighbours_separated, v.neighbours_after_holm, v.pairs_separated, v.pairs_after_holm)
            for v in on_mase.verdicts.values()] == [(1, 0, 12, 0), (1, 0, 13, 1), (1, 0, 12, 1)]
    assert on_mase.verdicts["wild"].tier == on_mase.tier == ["STRIDE w/ Synapse", "EXAONE-Forecast-Agent", "LS-MoE"]
    _, _, on_crps = pinned_report("gift-eval", "mean-rank", metric="CRPS", tests=("cr2", "wild", "exact-sum"))
    assert [(v.neighbours_separated, v.neighbours_after_holm, v.pairs_separated, v.pairs_after_holm)
            for v in on_crps.verdicts.values()] == [(0, 0, 9, 0), (0, 0, 12, 0), (0, 0, 12, 0)]
    _, _, geometric = pinned_report("gift-eval", "geometric-mean", tests=("cr2", "wild", "signed-rank"))
    assert [len(v.tier) for v in geometric.verdicts.values()] == [17, 17, 16]
    left, joined = set(geometric.tier) - set(geometric.verdicts["signed-rank"].tier), \
        set(geometric.verdicts["signed-rank"].tier) - set(geometric.tier)
    assert left == {"ForecastMate", "FastCastrage", "Taichu-TimeSeries-Agent", "STRIDE (+Chronos-2)"}
    assert joined == {"RacineCast-1", "RacineCast-1-NC", "Toto-2.0-FnF"}


@pytest.mark.skipif(not os.environ.get("VERNIER_SLOW"), reason="lists 2^27 sign patterns per pair: `just test-slow`")
@pytest.mark.parametrize("metric, rule, order_by, counts", [
    ("MASE", "mean-rank", None, (1, 0, 13, 1)),
    ("CRPS", "mean-rank", None, (0, 0, 11, 0)),
    ("MASE", "geometric-mean", "mean-rank", (0, 0, 19, 4)),
])
def test_gift_eval_counts_under_the_exact_t_test(metric, rule, order_by, counts):
    # The paper's table of counts, columns "Exact, same statistic": of 9, after Holm, of 45, after Holm.
    _, _, result = pinned_report("gift-eval", rule, metric=metric, order_by=order_by, tests=("exact",))
    exact = result.verdicts["exact"]
    assert counts == (exact.neighbours_separated, exact.neighbours_after_holm,
                      exact.pairs_separated, exact.pairs_after_holm)


CHECKED = {  # sha256 of each snapshot table; `just test-live` compares its content with the board's files
    "gift-eval": "911f8db873d9561dc37650eadedf56d0ee9e845f4bff19c642d05ab54d9dd027",
    "fev-bench": "0a0500979680449385ee5fa162cd40b29617841c415a599657a1ea52447b13ec",
    "fev-bench-mini": "019df88ba4d2ce0e0756a660d308c00bae0092b5295dd7a5ef2de3f7768cdaf9",
    "chronos-bench-ii": "5324d46e01cf07bb27b22b3b4da6fccf18aa974a136c8685837637e150584a95",
    "time": "079e8a4853e88d6cda3cf0458a6ee5131a0abaa38e68b4adf5230ef71c8991d4",
    "boom": "609df6e5c307ae0dabb2db04762d37303e86f72928f9c9ece11b0544afac8160",
}


@pytest.mark.parametrize("name", CHECKED)
def test_a_pinned_table_has_its_checksum(name):
    assert hashlib.sha256(pin_file(name).read_bytes()).hexdigest() == CHECKED[name]


def test_boom_snapshot_gives_the_boards_published_ranks():
    board, order, _ = pinned_report("boom", "mean-rank")
    ranks = values_under("mean-rank", board, "first").mean(1)
    # The first eight rows of boom/leaderboards/boom_leaderboard.csv at the pinned commit, column Rank-6750-scaled.
    assert [board.entries[i] for i in order[:8]] == ["toto_2.0_2.5b", "toto_2.0_1b", "toto_2.0_313m", "toto_2.0_22m",
                                                    "toto", "toto_2.0_4m", "chronos_2", "timesfm_2_5_200m"]
    published = ("3.876", "3.964", "4.258", "5.525", "6.939", "7.173", "7.390", "7.702")
    assert [ranks[i] for i in order[:8]] == printed(*published)


def test_a_missing_result_has_no_rank():
    scores = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 3.0]])  # SciPy before 1.13 gave it a whole number
    assert np.isnan(ranks(scores)[0, 1]) and ranks(scores)[:, 0].tolist() == [1, 2, 3]


def test_win_rate_and_mean_rank_give_the_same_order_on_chronos_benchmark_ii():
    board, by_rank, _ = pinned_report("chronos-bench-ii", "mean-rank")
    assert listed_order(board, values_under("win-rate", board)) == by_rank


# The loaders, each on a board small enough to work out by hand.

def fev_file(model, errors, leaked=()):
    rows = [f"{task},{model},{error},{error},{task in leaked}" for task, error in errors.items()]
    return "\n".join(["task_name,model_name,test_error,MASE,trained_on_this_dataset", *rows])


FEV = {
    "tasks.yaml": "tasks:\n- {dataset_config: solar_1D}\n- {dataset_config: solar_1W}\n"
                  "- {task_name: ett_1H, dataset_config: ett}\n",
    "models.yaml": "naive: {hidden: false}\nsmall: {hidden: true}\n",
    "results/naive.csv": fev_file("Seasonal Naive", {"solar_1D": 2.0, "solar_1W": 4.0, "ett_1H": 1.0}),
    "results/bolt.csv": fev_file("Chronos-Bolt", {"solar_1D": 1.0, "solar_1W": 1.0, "ett_1H": 0.5}),
    "results/new.csv": fev_file("New", {"solar_1D": 0.5, "solar_1W": 0.001, "ett_1H": 0.8}, leaked={"solar_1D"}),
    "results/partial.csv": fev_file("Partial", {"solar_1D": 3.0, "ett_1H": 500.0, "other_task": 1.0}),
    "results/small.csv": fev_file("Small", {"solar_1D": 0.1, "solar_1W": 0.1, "ett_1H": 0.1}),
}


def test_fev_bench_table_follows_the_boards_arithmetic(tmp_path):
    write(tmp_path, FEV)
    board = fev_bench(Folder(tmp_path))
    assert board.entries == ["Chronos-Bolt", "New", "Partial", "Seasonal Naive"]  # Small is hidden
    assert (board.units, board.sources) == (["ett_1H", "solar_1D", "solar_1W"], ["ett", "solar", "solar"])
    assert board.scores.tolist() == [[0.5, 0.5, 0.25],
                                     [0.8, 0.5, 0.01],  # trained on solar_1D: Chronos-Bolt's value; 0.00025 clipped
                                     [100.0, 1.5, 1.0],  # 500 clipped; no result on solar_1W: the baseline's value
                                     [1.0, 1.0, 1.0]]
    # On solar_1D the two entries at 0.5 beat two others and tie with each other: 2.5 of 3. The baseline beats one.
    assert values_under("win-rate", board)[:, 1].tolist() == pytest.approx([-250 / 3, -250 / 3, 0, -100 / 3])


def test_chronos_benchmark_ii_keeps_complete_entries_and_groups_collections(tmp_path):
    tasks = ["monash_m3_monthly", "monash_m3_yearly", "m4_quarterly", "ETTh", "ETTm", "nn5"]
    files = {"tasks.yaml": "tasks:\n" + "".join(f"- {{dataset_config: {task}}}\n" for task in tasks)}
    for model, missing in (("a", ()), ("b", ("nn5",))):
        rows = [f"{model},{task},{1 + len(task)},{1 + len(task)}" for task in tasks if task not in missing]
        files[f"results/{model}.csv"] = "\n".join(["model_name,dataset_config,test_error,MASE", *rows])
    write(tmp_path, files)
    board = chronos_bench_ii(Folder(tmp_path))
    assert board.entries == ["a"]
    assert dict(zip(board.units, board.sources)) == {"ETTh": "ETT", "ETTm": "ETT", "m4_quarterly": "m4",
                                                     "monash_m3_monthly": "m3", "monash_m3_yearly": "m3", "nn5": "nn5"}


def metrics_file(mase, crps):
    content = io.BytesIO()
    np.savez_compressed(content, MASE=np.array(mase, float), CRPS=np.array(crps, float))  # as TIME's saver writes it
    return content.getvalue()


TIME = {
    "seasonal_naive/Traffic/15T/short/metrics.npz": metrics_file([[[2.0, 4.0]], [[np.nan, 6.0]]], [[[1.0]]]),
    "seasonal_naive/Traffic/15T/long/metrics.npz": metrics_file([[[8.0]]], [[[2.0]]]),
    "model/Traffic/15T/short/metrics.npz": metrics_file([[[1.0, 1.0]], [[1.0, 3.0]]], [[[0.5]]]),
    "model/Traffic/15T/long/metrics.npz": metrics_file([[[2.0]]], [[[4.0]]]),
    "model/Solar/H/short/metrics.npz": metrics_file([[[1.0]]], [[[1.0]]]),  # the baseline has no result here
}


def test_time_reduces_each_array_to_its_mean_and_skips_units_without_a_baseline(tmp_path):
    write(tmp_path, TIME)
    board = time_board(Folder(tmp_path))
    assert (board.entries, board.units, board.sources) == (["model", "seasonal_naive"],
                                                           ["Traffic/15T/long", "Traffic/15T/short"], ["Traffic"] * 2)
    assert board.scores.tolist() == [[2.0, 1.5], [8.0, 4.0]]  # the mean skips the missing value
    assert time_board(Folder(tmp_path), "CRPS").scores.tolist() == [[4.0, 0.5], [2.0, 1.0]]
    scaled = values_under("geometric-mean", board, baseline="seasonal_naive")
    assert scaled[0].tolist() == pytest.approx(np.log([0.25, 0.375]))


def test_time_reads_large_files_through_their_pointers(tmp_path):
    # In TIME's repository each metrics.npz is a pointer to a file kept outside git.
    pointers = {f"results/{path}": f"version https://git-lfs.github.com/spec/v1\noid sha256:{i}\n"
                for i, path in enumerate(TIME)}
    revision = commit(tmp_path, pointers, "2026-01-01T00:00:00")
    fetched = []

    def large(at, path):
        fetched.append((at, path))
        return TIME[path.removeprefix("results/")]

    write(tmp_path / "plain", TIME)
    board, plain = time_board(Commit(tmp_path, revision, "results", large)), time_board(Folder(tmp_path / "plain"))
    assert (board.entries, board.units, board.scores.tolist()) == (plain.entries, plain.units, plain.scores.tolist())
    assert sorted(fetched) == sorted((revision, path) for path in pointers)
    time_board(Commit(tmp_path, revision, "results", large))
    assert len(fetched) == len(pointers)  # each file is fetched once


def boom_file(scores):
    rows = [f"{unit},{mase},{crps}" for unit, (mase, crps) in scores.items()]
    return "\n".join(["dataset,eval_metrics/MASE[0.5],eval_metrics/mean_weighted_sum_quantile_loss", *rows])


BOOM = {
    "utils/leaderboard.py": 'LOW_VARIANCE_DATASETS = [\n    "ds-9-T",\n]\n',
    "boom_properties.json": json.dumps({"ds-1-T": {"domain": ["Database"], "type": "gauge"},
                                        "ds-2-T": {"domain": ["Networking", "Database"], "type": "count"}}),
    "results/seasonalnaive/all_results.csv": boom_file({
        "ds-1-T/Short/short": (1.0, 2.0), "ds-1-T/Short/long": (1.0, 4.0), "ds-2-T/Short/short": (1.0, 8.0),
        "ds-3-T/Short/short": (0.0, 1.0), "ds-3-T/Short/long": (1.0, 1.0), "ds-9-T/Short/short": (1.0, 1.0)}),
    "results/toto/all_results.csv": boom_file({
        "ds-1-T/Short/short": (1.0, 1.0), "ds-1-T/Short/long": (1.0, "inf"), "ds-2-T/Short/short": (1.0, 3.0),
        "ds-3-T/Short/short": (1.0, 1.0), "ds-3-T/Short/long": (1.0, 1.0), "ds-9-T/Short/short": (1.0, 1.0)}),
    "results/later/all_results.csv": boom_file({
        "ds-1-T/Short/short": (1.0, 2.0), "ds-1-T/Short/long": (1.0, 2.0), "ds-2-T/Short/short": (1.0, ""),
        "ds-3-T/Short/short": (1.0, 1.0), "ds-3-T/Short/long": (1.0, 1.0), "ds-9-T/Short/short": (1.0, 1.0)}),
}


def test_a_large_file_that_cannot_be_fetched_makes_the_board_unreachable(monkeypatch, tmp_path):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)  # the library then asks no server
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(Unreachable, match=r"cannot fetch results/a.npz of Real-TSF/TIME-Output from Hugging Face"):
        hub_file("Real-TSF/TIME-Output", "dataset")("0" * 40, "results/a.npz")


def test_boom_table_follows_the_boards_arithmetic(tmp_path):
    write(tmp_path, BOOM)
    board = boom(Folder(tmp_path))
    assert board.entries == ["toto", "later", "seasonalnaive"]  # the known listing, later entries, the baseline
    # ds-9-T is low-variance and ds-3-T has a zero baseline MASE on one horizon: the board ranks both apart.
    assert board.units == ["ds-1-T/Short/long", "ds-1-T/Short/short", "ds-2-T/Short/short"]
    assert board.scores.tolist() == [[2.0 / 4, 1.0 / 2, 3.0 / 8],  # the infinite score becomes toto's mean, 2
                                     [2.0 / 4, 2.0 / 2, 2.0 / 8],  # the missing score becomes the entry's mean, 2
                                     [1.0, 1.0, 1.0]]
    assert values_under("mean-rank", board, "first")[:, 0].tolist() == [1, 2, 3]  # the tie goes to the earlier entry
    assert board.sources == ["ds-1-T", "ds-1-T", "ds-2-T"]
    assert boom(Folder(tmp_path), sources="domain").sources == ["Database", "Database", "Database|Networking#count"]
    by_type = boom(Folder(tmp_path), sources="domain-type").sources
    assert by_type == ["Database#gauge", "Database#gauge", "Database|Networking#count"]


def test_gift_eval_files_are_read_at_a_commit_whatever_their_names(tmp_path):
    results = f"dataset,model,{MASE},{CRPS}\nm4/D/short,Modèle,1.5,0.5"
    revision = commit(tmp_path, {"results/Modèle é/all_results.csv": results,
                                 "results/Modèle é/config.json": '{"model": "Modèle", "testdata_leakage": "No"}',
                                 "results/unnamed/config.json": "{}"}, "2026-01-01T00:00:00")
    board = gift_eval(Commit(tmp_path, revision, "results"))
    assert (board.entries, board.units, board.scores.tolist()) == (["Modèle"], ["m4/D/short"], [[1.5]])


def test_a_board_file_that_cannot_be_read_is_named(tmp_path):
    write(tmp_path, {"broken/all_results.csv": "dataset,eval_metrics/MASE[0.5]\nm4/D/short,1.5"})
    with pytest.raises(BoardError, match=r"broken/all_results.csv cannot be read \(KeyError: 'model'\)"):
        gift_eval(Folder(tmp_path))


def test_a_written_table_reads_back_as_the_same_board(tmp_path):
    write(tmp_path, BOOM)
    write_table(tmp_path / "table.csv.gz", "boom", Folder(tmp_path))
    for metric in ("CRPS", "MASE"):
        for sources, column in (("source", "source"), ("domain", "domain")):
            board, table = boom(Folder(tmp_path), metric, sources), load_csv(tmp_path / "table.csv.gz", metric, column)
            assert table[:3] == board[:3] and table.band == board.band  # entries, units and sources
            assert np.array_equal(table.scores, board.scores)
