"""The bystander test: replay a board's history and list the pairs of top-ten entries that swapped order
although neither entry's own files changed, so that other entries arriving or changing swapped them.
"""
from itertools import combinations
from typing import NamedTuple

from .board import TOP, listed_order, values_under
from .boards import BOARDS
from .sources import BoardError, Commit, updates

TIE = 1e-9  # two means closer than this are tied: a win rate of equal mean ranks differs by rounding only


class Reversal(NamedTuple):
    date: str
    commit: str  # the update that reversed the pair
    entry_a: str  # placed higher before the update
    entry_b: str


class Replay(NamedTuple):
    updates: int  # consecutive snapshots with the same units
    reversals: list


def snapshots_from_git(clone_dir, until=None, board="gift-eval", metric=None) -> list:
    """Return (commit, date, Board) for each commit that changed the board's files and holds a board, oldest first."""
    named = BOARDS[board]
    snapshots = []
    for commit, date in updates(clone_dir, named.results, until):
        try:
            state = named.load(Commit(clone_dir, commit, named.path, named.large), metric or named.metrics[0], "source")
        except BoardError:  # the board's own code cannot build its table from these files, for example with no baseline
            state = None
        if state is None or state.entries:  # GIFT-Eval's first commits hold no results and are no update
            snapshots.append((commit, date, state))
    return snapshots


def _side(difference) -> int:
    """Return the sign of a difference of means, with 0 for a tie up to rounding."""
    return 0 if abs(difference) < TIE else (1 if difference > 0 else -1)


def _listing(board, rule, ties, baseline):
    """Return each entry's mean value and the entries the board lists, or None if it lists fewer than ten."""
    try:
        values = values_under(rule, board, ties, baseline)
    except BoardError:  # the rule needs a baseline entry that this state of the board lacks
        return None
    listed = [board.entries[i] for i in listed_order(board, values)]
    return (dict(zip(board.entries, values.mean(1))), listed) if len(listed) >= TOP else None


def _untouched(leaders, earlier, board, listed) -> list:
    """Return (name before, name now) of the earlier leaders that are still listed with their own files unchanged.

    An entry that was renamed with its files unchanged is the same entry.
    """
    renamed = {files: entry for entry, files in board.files.items()}
    now = {entry: entry if board.files.get(entry) == earlier.files[entry] else renamed.get(earlier.files[entry])
           for entry in leaders}
    return [(entry, name) for entry, name in now.items() if name in listed]


def reversals(snapshots, ties="first", rule="mean-rank", baseline=None) -> Replay:
    """Return the updates examined and the reversed pairs of earlier top entries whose own files are unchanged.

    An update is examined when the board lists ten entries before and after it and its units are the same.
    """
    found, examined, before = [], 0, None
    for commit, date, board in snapshots:
        now = board and _listing(board, rule, ties, baseline)
        if now and before and before[0].units == board.units:
            earlier, mean_before, leaders = before
            mean, listed = now
            examined += 1
            found += [Reversal(date, commit, a, b)
                      for (a, a_now), (b, b_now) in combinations(_untouched(leaders, earlier, board, listed), 2)
                      if _side(mean_before[a] - mean_before[b]) * _side(mean[a_now] - mean[b_now]) < 0]
        before = now and (board, now[0], now[1][:TOP])
    return Replay(examined, found)
