"""The bystander test on a small board with a history written by hand."""
import json

from helpers import commit, write

from vernier import reversals, snapshots_from_git
from vernier.__main__ import main
from vernier.board import make_board

HEADER = "dataset,model,eval_metrics/MASE[0.5],eval_metrics/mean_weighted_sum_quantile_loss"
UNITS = ["a/D/short", "a/D/long", "b/H/short"]


def entry(name, scores, units=UNITS):
    rows = [f"{unit},{name},{score},{score}" for unit, score in zip(units, scores)]
    return {f"results/{name}/all_results.csv": "\n".join([HEADER, *rows]),
            f"results/{name}/config.json": json.dumps({"model": name, "testdata_leakage": "No"})}


def board(*entries):
    return {path: content for files in entries for path, content in files.items()}


# Leader and Second are first on every unit but the second one, where three newcomers arrive between them.
FIELD = [entry(f"field{i}", [10 + i] * 3) for i in range(9)] + [entry("Seasonal_Naive", [30, 30, 30])]
LEADER, SECOND = entry("Leader", [1, 5, 1]), entry("Second", [3, 4, 3])
NEWCOMERS = [entry(f"newcomer{i}", [20, 4.2 + i / 10, 20]) for i in range(3)]


def history(root):
    """Return the commits of a board with one bystander reversal, one own change and one change of units."""
    start = commit(root, board(*FIELD, LEADER, SECOND), "2026-01-01T00:00:00")
    arrival = commit(root, board(*NEWCOMERS), "2026-02-01T00:00:00")  # Leader falls to rank 5 on the second unit
    own_change = commit(root, entry("Leader", [1, 3.9, 1]), "2026-03-01T00:00:00")  # Leader's own file changes
    wider = [entry(name, [*scores, 1], [*UNITS, "c/D/short"])
             for name, scores in (("Leader", [1, 3.9, 1]), ("Second", [3, 4, 3]))]
    new_units = commit(root, board(*wider), "2026-04-01T00:00:00")  # two entries gain a unit: the units changed
    return start, arrival, own_change, new_units


def test_a_pair_reversed_by_newcomers_is_found_and_an_own_change_is_not(tmp_path):
    _, arrival, _, _ = history(tmp_path)
    snapshots = snapshots_from_git(tmp_path)
    replay = reversals(snapshots)
    assert replay.updates == 2  # the arrival and the own change; the update that changed the units is skipped
    assert [(r.commit, r.entry_a, r.entry_b) for r in replay.reversals] == [(arrival, "Leader", "Second")]
    assert reversals(snapshots, rule="geometric-mean", baseline="Seasonal_Naive").reversals == []
    assert reversals(snapshots_from_git(tmp_path, until=snapshots[0][0])).updates == 0


def test_a_renamed_entry_with_unchanged_files_is_the_same_entry():
    def state(entries):
        results = {name: dict(zip(UNITS, scores)) for name, scores, _ in entries}
        return make_board(results, lambda unit: unit[0], files={name: files for name, _, files in entries})

    field = [(f"field{i}", [10 + i] * 3, f"field{i} v1") for i in range(9)]
    newcomers = [(f"newcomer{i}", [20, 4.2 + i / 10, 20], f"newcomer{i} v1") for i in range(3)]
    before = state(field + [("Leader", [1, 5, 1], "leader v1"), ("Second", [3, 4, 3], "second v1")])
    renamed = state(field + newcomers + [("Renamed", [1, 5, 1], "leader v1"), ("Second", [3, 4, 3], "second v1")])
    replay = reversals([("c1", "2026-01-01", before), ("c2", "2026-02-01", renamed)])
    assert [(r.entry_a, r.entry_b) for r in replay.reversals] == [("Leader", "Second")]
    rerun = state(field + newcomers + [("Renamed", [1, 5, 1], "leader v2"), ("Second", [3, 4, 3], "second v1")])
    assert reversals([("c1", "2026-01-01", before), ("c2", "2026-02-01", rerun)]).reversals == []  # its files changed


def test_replay_of_a_folder_of_dated_tables(tmp_path, capsys, monkeypatch):
    def table(entries):
        rows = [f"{name},{unit},{unit[0]},{score}" for name, scores in entries for unit, score in zip(UNITS, scores)]
        return "\n".join(["entry,unit,source,score", *rows])

    field = [(f"field{i}", [10 + i] * 3) for i in range(9)] + [("Leader", [1, 5, 1]), ("Second", [3, 4, 3])]
    write(tmp_path, {"2026-01.csv": table(field),
                     "2026-02.csv": table(field + [(f"newcomer{i}", [20, 4.2 + i / 10, 20]) for i in range(3)])})
    monkeypatch.setattr("sys.argv", ["vernier", "replay", str(tmp_path)])
    main()
    printed = capsys.readouterr().out.splitlines()
    assert "1 updates examined, 1 reversed 1 pairs" in printed[0]
    assert printed[2].split() == ["2026-02.csv", "Leader", "Second"]


def run(monkeypatch, capsys, *arguments):
    monkeypatch.setattr("sys.argv", ["vernier", *map(str, arguments)])
    main()
    return capsys.readouterr().out


def test_the_command_line_reads_a_clone_at_a_commit_replays_it_until_one_and_exports_it(tmp_path, capsys, monkeypatch):
    clone = tmp_path / "board"
    clone.mkdir()
    start, arrival, _, _ = history(clone)
    whole = json.loads(run(monkeypatch, capsys, "replay", clone, "--json"))
    assert (whole["updates"], len(whole["reversals"])) == (2, 1)
    assert json.loads(run(monkeypatch, capsys, "replay", clone, "--until", start, "--json"))["updates"] == 0
    then = json.loads(run(monkeypatch, capsys, "report", clone, "--at", start, "--json"))
    now = json.loads(run(monkeypatch, capsys, "report", clone, "--at", arrival, "--json"))
    assert (then["entries_listed"], now["entries_listed"]) == (12, 15) and start[:8] in then["board"]
    run(monkeypatch, capsys, "export", clone, "--board", "gift-eval", "--at", arrival, "-o", tmp_path / "table.csv")
    table = json.loads(run(monkeypatch, capsys, "report", tmp_path / "table.csv", "--board", "gift-eval", "--json"))
    assert all(table[field] == now[field] for field in ("places", "resolution", "median_gap", "tests"))
    folder = json.loads(run(monkeypatch, capsys, "report", clone / "results", "--json"))  # no board named: GIFT-Eval's
    assert folder["board"].startswith("GIFT-Eval, the folder") and folder["entries_listed"] == 2
