"""The boards' own repositories, read over the network: each snapshot is the board's files at its pinned commit,
and each replay gives the paper's counts.

These tests run only when the environment variable VERNIER_LIVE names the hosts to reach:
VERNIER_LIVE=github reads fev-bench, Chronos Benchmark II and BOOM, and VERNIER_LIVE=github,huggingface also
GIFT-Eval and TIME. `just test-live` runs them.
"""
import gzip
import os

import pytest

from vernier import BOARDS, Commit, clone_of, reversals, snapshots_from_git
from vernier.boards import pin_file, write_table

HOSTS = os.environ.get("VERNIER_LIVE", "").split(",")
GITHUB = ["fev-bench", "fev-bench-mini", "chronos-bench-ii", "boom"]
HUGGING_FACE = ["gift-eval", "time"]
# Table 8: the board, the rule, then updates examined, updates with a reversal and reversed pairs.
REPLAYS = [
    ("fev-bench", "win-rate", "average", (30, 2, 2)),
    ("fev-bench", "geometric-mean", "average", (30, 0, 0)),
    ("chronos-bench-ii", "mean-rank", "average", (11, 3, 8)),
    ("chronos-bench-ii", "geometric-mean", "average", (11, 0, 0)),
    ("gift-eval", "mean-rank", "first", (141, 14, 20)),
    ("gift-eval", "mean-rank", "average", (141, None, 18)),
    ("gift-eval", "geometric-mean", "first", (None, 0, 0)),  # fewer updates: early states have no baseline entry
    ("time", "geometric-mean", "first", (44, 0, 0)),
    ("time", "mean-rank", "first", (44, 12, 13)),
]


def reachable(name):
    host = "huggingface" if name in HUGGING_FACE else "github"
    return pytest.mark.skipif(host not in HOSTS, reason=f"set VERNIER_LIVE={host} to read {name} over the network")


@pytest.mark.parametrize("name", [pytest.param(name, marks=reachable(name)) for name in GITHUB + HUGGING_FACE])
def test_snapshot_is_the_boards_files_at_the_pinned_commit(name, tmp_path):
    named = BOARDS[name]
    if not pin_file(name).exists():
        pytest.skip(f"no snapshot of {name} in src/vernier/pins: write it with `just pin {name}`")
    write_table(tmp_path / "table.csv.gz", name, Commit(clone_of(named.url), named.pin, named.path, named.large))
    with gzip.open(tmp_path / "table.csv.gz") as written, gzip.open(pin_file(name)) as held:
        assert written.read() == held.read()


@pytest.mark.parametrize("name, rule, ties, expected",
                         [pytest.param(*replay, marks=reachable(replay[0])) for replay in REPLAYS])
def test_replay_gives_the_papers_counts(name, rule, ties, expected):
    named = BOARDS[name]
    replay = reversals(snapshots_from_git(clone_of(named.url), named.pin, name), ties, rule, named.baseline)
    found = (replay.updates, len({reversal.commit for reversal in replay.reversals}), len(replay.reversals))
    assert [(a, b) for a, b in zip(found, expected) if b is not None and a != b] == []
