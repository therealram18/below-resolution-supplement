"""The leaderboards Vernier reads directly: where each keeps its files, its rules, and the paper's snapshot of it."""
import csv
import gzip
import io
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .board import Board, load_csv
from .loaders import BOOM_BASELINE, FEV_BASELINE, boom, chronos_bench_ii, fev_bench, gift_eval, time_board
from .sources import BoardError, hub_file

PINS = Path(__file__).parent / "pins"  # the paper's snapshot of each board, as a table load_csv reads


class Named(NamedTuple):
    title: str
    url: str  # the git repository that holds the board's result files
    path: str  # the folder of the board inside it
    results: str  # the folder of result files inside it; a commit that changes it is an update of the board
    pin: str  # the commit of the paper's snapshot
    pinned: str  # its date
    load: Callable  # (tree, metric, sources) -> Board
    metrics: tuple  # the first is the board's own
    rules: tuple  # the first is the rule the board sorts by
    ties: str  # how the board breaks tied ranks
    baseline: str | None = None  # the entry whose scores scale every other entry's
    sources: tuple = ("source",)  # the definitions of a source; the first is the one the paper fixed in advance
    large: Callable | None = None  # downloads a file the repository keeps outside git


# The repository of fev-bench and of Chronos Benchmark II.
FEV = {"url": "https://github.com/autogluon/fev", "pin": "9cf943c388e5cfaf352175ea3c69e408b197d997",
       "pinned": "2026-11-11", "ties": "average"}
FEV_BENCH = {**FEV, "path": "benchmarks/fev_bench", "results": "benchmarks/fev_bench/results",
             "metrics": ("SQL", "MASE"), "rules": ("win-rate", "geometric-mean"), "baseline": FEV_BASELINE}
BOARDS = {
    "gift-eval": Named(
        "GIFT-Eval", url="https://huggingface.co/spaces/Salesforce/GIFT-Eval", path="results", results="results",
        pin="b61c48aa3197a1a111e405c0e6f600128a1efee3", pinned="2026-09-07", load=gift_eval,
        metrics=("MASE", "CRPS"), rules=("mean-rank", "geometric-mean"), ties="first", baseline="Seasonal_Naive"),
    "fev-bench": Named("fev-bench", load=fev_bench, **FEV_BENCH),
    "fev-bench-mini": Named("fev-bench-mini", load=partial(fev_bench, tasks_file="tasks_mini.yaml"), **FEV_BENCH),
    "chronos-bench-ii": Named(
        "Chronos Benchmark II", path="benchmarks/chronos_zeroshot", results="benchmarks/chronos_zeroshot/results",
        load=chronos_bench_ii, metrics=("MASE", "WQL"), rules=("mean-rank", "geometric-mean", "win-rate"),
        baseline="seasonal_naive", **FEV),
    "time": Named(
        "TIME", url="https://huggingface.co/datasets/Real-TSF/TIME-Output", path="results", results="results",
        pin="3bf103cb8c2b418c4a8b17bda21b425034dc869c", pinned="2026-10-31", load=time_board,
        metrics=("MASE", "CRPS"), rules=("geometric-mean", "mean-rank"), ties="first", baseline="seasonal_naive",
        large=hub_file("Real-TSF/TIME-Output", "dataset")),
    "boom": Named(
        # The commit is of 2026-10-26; it was the board's latest when the snapshot was taken on 2026-11-11.
        "BOOM", url="https://github.com/DataDog/toto", path="boom", results="boom/results",
        pin="1d3e1c97ccf615016d38fe910602c905ae31fbb7", pinned="2026-11-11", load=boom,
        metrics=("CRPS", "MASE"), rules=("mean-rank",), ties="first", baseline=BOOM_BASELINE,
        sources=("source", "domain-type", "domain")),
}


def pin_file(name) -> Path:
    return PINS / f"{name}_{BOARDS[name].pinned}.csv.gz"


def load_pin(name, metric, sources="source") -> Board:
    """Return the paper's snapshot of a board from the table kept in this package."""
    if not pin_file(name).exists():
        raise BoardError(f"this copy of Vernier holds no snapshot of {BOARDS[name].title}")
    return load_csv(pin_file(name), metric, sources)


def write_table(path, name, tree):
    """Write a named board's table as a file load_csv reads: a column per metric and per definition of a source."""
    named = BOARDS[name]
    boards = {metric: named.load(tree, metric, "source") for metric in named.metrics}
    others = {sources: named.load(tree, named.metrics[0], sources).sources for sources in named.sources[1:]}
    first = boards[named.metrics[0]]
    if any((board.entries, board.units) != (first.entries, first.units) for board in boards.values()):
        raise BoardError(f"{named.title} has different entries or units under {' and '.join(boards)}")
    with open(path, "wb") as file:
        # No name and no time in the gzip header: the same table always gives the same bytes.
        packed = gzip.GzipFile("", "wb", fileobj=file, mtime=0) if str(path).endswith(".gz") else file
        with io.TextIOWrapper(packed, newline="", encoding="utf-8") as text:
            table = csv.writer(text, lineterminator="\n")
            table.writerow(["entry", "unit", "source", *boards, "band", *others])
            for i, entry in enumerate(first.entries):
                for j, unit in enumerate(first.units):
                    scores = ["" if np.isnan(board.scores[i, j]) else repr(float(board.scores[i, j]))
                              for board in boards.values()]
                    table.writerow([entry, unit, first.sources[j], *scores,
                                    "" if first.band[i] is None else first.band[i],
                                    *(column[j] for column in others.values())])
