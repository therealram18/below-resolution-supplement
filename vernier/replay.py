"""Replays a leaderboard's git history under mean rank and lists the pairs of top-ten entries that swapped order
although neither entry's own result and config files changed: other entries arriving or changing swapped them.
"""
import json
import subprocess
from itertools import combinations
from typing import NamedTuple

from .board import TOP, gift_eval_order, make_board, ranks, read_leakage, read_results


class Reversal(NamedTuple):
    date: str
    commit: str  # the update that reversed the pair
    entry_a: str  # placed higher before the update
    entry_b: str


class Replay(NamedTuple):
    updates: int  # consecutive snapshots with the same units
    reversals: list[Reversal]


def _read_blobs(git, blobs):
    """Return {blob: text} for all `blobs`, read by one git process."""
    output = git("cat-file", "--batch", input="".join(blob + "\n" for blob in blobs).encode())
    texts, start = {}, 0
    for blob in blobs:  # git writes each as "<blob> <type> <size>\n<content>\n"
        header_end = output.index(b"\n", start)
        size = int(output[start:header_end].split()[2])
        texts[blob] = output[header_end + 1:header_end + 1 + size].decode()
        start = header_end + size + 2
    return texts


def _leakage(text):
    """Return {entry: declared leakage} from the text of a config.json file."""
    try:
        entry, leakage = read_leakage(text)
    except json.JSONDecodeError:  # one config.json in GIFT-Eval's history is not valid JSON
        return {}
    return {entry: leakage}


def snapshots_from_git(clone_dir, until=None):
    """Return (commit, date, Board, own files) for each commit that changed results/ and holds results, oldest first.

    Own files maps each entry to the git blob ids of its all_results.csv and its config.json.
    """
    def git(*args, input=None):
        return subprocess.run(["git", "-C", str(clone_dir), *args], input=input, capture_output=True, check=True).stdout

    log = git("log", "--reverse", "--format=%H %aI", until or "HEAD", "--", "results").decode()
    commits = [line.split() for line in log.splitlines()]  # [commit, date]
    files = {}
    for commit, _ in commits:
        listing = git("ls-tree", "-r", commit, "results").decode().splitlines()
        files[commit] = [line.split(None, 3)[2:] for line in listing  # [blob, path]
                         if line.endswith(("all_results.csv", "config.json"))]
    paths = {blob: path for listing in files.values() for blob, path in listing}
    texts = _read_blobs(git, sorted(paths))
    # Each file is parsed once; most stay unchanged over many commits.
    parsed = {blob: read_results(texts[blob]) if path.endswith(".csv") else _leakage(texts[blob])
              for blob, path in paths.items()}
    snapshots = []
    for commit, date in commits:
        results, leakage, result_file, config_file = {}, {}, {}, {}
        for blob, path in files[commit]:
            if path.endswith(".csv"):
                results.update(parsed[blob])
                result_file.update(dict.fromkeys(parsed[blob], blob))
            else:
                leakage.update(parsed[blob])
                config_file.update(dict.fromkeys(parsed[blob], blob))
        if results:  # the board's first commits hold none
            own_files = {entry: (result_file[entry], config_file.get(entry)) for entry in results}
            snapshots.append((commit, date, make_board(results, leakage), own_files))
    return snapshots


def reversals(snapshots, ties="first") -> Replay:
    """Return the updates examined and the reversed pairs of earlier top entries whose own files are unchanged."""
    found, updates, units_before = [], 0, None
    for commit, date, board, own_files in snapshots:
        values = ranks(board.scores, ties)
        shown = [board.entries[i] for i in gift_eval_order(board, values)]
        if len(shown) < TOP:
            units_before = None
            continue
        mean = dict(zip(board.entries, values.mean(1)))
        if units_before is not None and units_before == board.units:
            updates += 1
            kept = [entry for entry in leaders_before
                    if entry in shown and own_files[entry] == own_files_before[entry]]
            found += [Reversal(date, commit, a, b) for a, b in combinations(kept, 2)
                      if (mean_before[a] - mean_before[b]) * (mean[a] - mean[b]) < 0]
        units_before, own_files_before, mean_before, leaders_before = board.units, own_files, mean, shown[:TOP]
    return Replay(updates, found)
