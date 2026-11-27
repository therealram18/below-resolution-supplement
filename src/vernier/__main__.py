"""The command line: vernier report | replay | boards | export.

Exit status: 0 when the command ran, 1 when it could not (the message on stderr says why) or when --board all
could not read one of the boards (its row says why), 2 for a wrong option.
"""
import argparse
import json
import math
import signal
import sys
from pathlib import Path

from . import __doc__ as summary
from . import __version__
from .board import RULES, listed_order, load_csv, values_under
from .boards import BOARDS, load_pin, pin_file, write_table
from .replay import reversals, snapshots_from_git
from .report import TESTS, as_dict, as_text, report
from .sources import BoardError, Commit, Folder, Unreachable, clone_of, git

SCHEMA = 1  # of the --json output: it goes up when a field is renamed or removed
UNITS = {"mean-rank": "rank places",
         "geometric-mean": "the log of error over the error of {baseline}",
         "win-rate": "the win rate in percentage points"}
# The paper's Appendix A.2: on a geometric mean the CR2 test and the wild bootstrap reject a true zero about 16% of
# the time, and the signed-rank test holds its level.
CAUTION = ("On a geometric mean the CR2 test and the wild bootstrap separate too many pairs. "
           "Read the signed-rank line: that test holds its level on this rule.")


def is_table(source) -> bool:
    return str(source).endswith((".csv", ".csv.gz"))


def is_clone(folder) -> bool:
    return (Path(folder) / ".git").exists() or (Path(folder) / "HEAD").exists()  # a clone, or a bare one


def settings(args, tabular):
    """Return the named board a command reads, or None for tables, and its metric, sources, rule, ties and baseline."""
    if not (args.board or args.source):
        raise BoardError("name a leaderboard with --board or give a table of scores; `vernier boards` lists the boards")
    # A folder or a git address with no --board is read as GIFT-Eval's files.
    named = BOARDS[args.board] if args.board else None if tabular else BOARDS["gift-eval"]
    if args.higher_is_better and not tabular:
        raise BoardError("--higher-is-better is for a table of scores, not for a board's own files")
    metric, sources = args.metric or (named.metrics[0] if named else "score"), args.sources or "source"
    if named and not tabular and (metric not in named.metrics or sources not in named.sources):
        raise BoardError(f"{named.title} has the metrics {', '.join(named.metrics)} and the sources "
                         f"{', '.join(named.sources)}; --metric is {metric} and --sources is {sources}")
    return (named, metric, sources, args.rule or (named.rules[0] if named else "mean-rank"),
            args.ties or (named.ties if named else "first"),
            args.baseline or (named.baseline if named else "Seasonal_Naive"))


def lower_is_better(args, board):
    return board._replace(scores=-board.scores) if args.higher_is_better else board


def open_tree(named, source, at=None):
    """Return the tree of a named board's files and a description of it: a folder, or a commit of a clone."""
    if source and "://" not in source:
        folder = Path(source)
        if not folder.is_dir():
            raise BoardError(f"{source} is not a folder, a table (.csv) or the address of a git repository")
        if not is_clone(folder):
            return Folder(folder / named.path if (folder / named.path).is_dir() else folder), f"the folder {source}"
        clone, origin = folder, str(folder)
    else:
        clone, origin = clone_of(source or named.url), source or named.url
    commit, date = git(clone, "log", "-1", "--format=%H %cs", "--end-of-options", at or "HEAD").decode().split()
    return Commit(clone, commit, named.path, named.large), f"commit {commit[:8]} of {date}, {origin}"


def read_board(args, named, metric, sources):
    """Return the Board a report asks for, where it came from, and whether the snapshot stands in for the board."""
    snapshot = f"the paper's snapshot of {named.pinned} (commit {named.pin[:8]} of {named.url})" if named else ""
    if args.pinned:
        if args.source or not args.board:
            raise BoardError("--pinned reads the paper's snapshot of a board: name it with --board and give no source")
        return load_pin(args.board, metric, sources), f"{named.title}, {snapshot}", False
    if args.source and is_table(args.source):
        title = f"{named.title + ', ' if named else ''}the table {args.source}"
        return load_csv(args.source, metric, sources), title, False
    try:
        tree, where = open_tree(named, args.source, args.at)
        return named.load(tree, metric, sources), f"{named.title}, {where}", False
    except Unreachable as error:
        if args.source or args.at:
            raise
        # The board as it is now was asked for and cannot be had: the snapshot is shown, and every output says so.
        print(f"vernier: {error}; this report is of {snapshot}", file=sys.stderr)
        return load_pin(args.board, metric, sources), f"{named.title}, NOT the board today: {snapshot}", True


def one_report(args):
    """Return one board's report as facts for JSON and as text."""
    named, metric, sources, rule, ties, baseline = settings(args, tabular=is_table(args.source))
    board, where, fallback = read_board(args, named, metric, sources)
    board = lower_is_better(args, board)
    values = values_under(rule, board, ties, baseline)
    order = listed_order(board, values_under(args.order_by, board, ties, baseline) if args.order_by else values)
    n_sources = len(set(board.sources))
    fitting = [test for test, (_, most_sources) in TESTS.items() if n_sources <= (most_sources or n_sources)]
    usual = ["cr2", "wild"] + ["signed-rank"] * (rule == "geometric-mean" and "signed-rank" in fitting)
    tests = usual if not args.tests else fitting if args.tests == "all" else args.tests.split(",")
    result = report(values, board.sources, board.entries, order, tests)
    shown = (lambda mean: -mean) if rule == "win-rate" else (lambda mean: mean)
    facts = {"schema": SCHEMA, "board": where, "fallback": fallback, "rule": rule, "metric": metric, "ties": ties,
             "ordered_by": args.order_by or rule, "entries_listed": len(order), "units": len(board.units),
             "sources": n_sources, **as_dict(result, board.entries, values, order, shown)}
    listed = (f"{rule} on {metric}{', ties ' + ties if rule == 'mean-rank' else ''}"
              f"{', entries ordered by ' + args.order_by if args.order_by else ''}: {len(order)} entries listed with "
              f"all {len(board.units)} units from {n_sources} sources")
    lines = [where, listed, f"Values are {UNITS[rule].format(baseline=baseline)}; a lower value is better",
             as_text(result, board.entries, values, order, shown)] + [CAUTION] * (rule == "geometric-mean")
    return facts, "\n".join(lines)


def run_report(args):
    if args.board != "all":
        facts, text = one_report(args)
        print(json.dumps(facts, indent=1) if args.json else text)
        return
    if args.source or args.at:
        raise BoardError("--board all reads every board from its own repository; it takes no source and no --at")
    # Every board under the rule it sorts by. A board that cannot be read keeps its row, with the reason.
    rows = {}
    for name in BOARDS:
        try:
            rows[name] = one_report(argparse.Namespace(**{**vars(args), "board": name}))[0]
        except BoardError as error:
            rows[name] = {"schema": SCHEMA, "error": str(error)}
    failed = any("error" in row for row in rows.values())
    if args.json:
        print(json.dumps(rows, indent=1))
        sys.exit(failed)
    print(f"{'board':<18}{'rule':<16}{'entries':>8}{'units':>7}{'sources':>9}{'resolution':>12}{'median gap':>12}"
          f"{'ratio':>7}  neighbours separated at 5%, and after Holm")
    for name, row in rows.items():
        if "error" in row:
            print(f"{name:<18}{row['error']}")
            continue
        apart = "; ".join(f"{test} {verdict['neighbours_separated']} of {len(row['places']) - 1}"
                          + ("" if verdict["p"] is None else f", {verdict['neighbours_after_holm']}")
                          for test, verdict in row["tests"].items())
        ratio = math.inf if row["ratio"] is None else row["ratio"]
        print(f"{name:<18}{row['rule']:<16}{row['entries_listed']:>8}{row['units']:>7}{row['sources']:>9}"
              f"{row['resolution']:>12.4f}{row['median_gap']:>12.4f}{ratio:>7.3g}  {apart}"
              + "; NOT the board today, the paper's snapshot" * row["fallback"])
    sys.exit(failed)


def run_replay(args):
    local = Path(args.source) if args.source and "://" not in args.source else None
    if local and not local.is_dir():
        raise BoardError(f"{local} is not a git clone of a board, a folder of dated tables or a git address")
    named, metric, sources, rule, ties, baseline = settings(args, tabular=bool(local) and not is_clone(local))
    if named is None:  # a folder of dated tables, replayed in the order of their names
        tables = sorted(path for path in local.iterdir() if is_table(path))
        if not tables:
            raise BoardError(f"{local} is not a git clone of a board and holds no tables (.csv)")
        snapshots = [(path.name, "", lower_is_better(args, load_csv(path, metric, sources))) for path in tables]
        title = f"The {len(tables)} tables of {local}"
    else:
        clone = local or clone_of(args.source or named.url)
        snapshots, title = snapshots_from_git(clone, args.until, args.board or "gift-eval", metric), named.title
    replay = reversals(snapshots, ties, rule, baseline)
    with_reversal = len({reversal.commit for reversal in replay.reversals})
    if args.json:
        print(json.dumps({"schema": SCHEMA, "board": title, "rule": rule, "metric": metric, "ties": ties,
                          "updates": replay.updates, "updates_with_reversal": with_reversal,
                          "reversals": [reversal._asdict() for reversal in replay.reversals]}, indent=1))
        return
    print(f"{title}, {rule} on {metric}{', ties ' + ties if rule == 'mean-rank' else ''}: {replay.updates} updates "
          f"examined, {with_reversal} reversed {len(replay.reversals)} pairs of top-ten entries whose own results "
          "did not change")
    print(f"{'date':<10}  {'update':<8}  {'placed higher before':<28}  placed lower before")
    for reversal in replay.reversals:
        update = reversal.commit[:8] if named else reversal.commit  # a commit, or the name of a table
        print(f"{reversal.date[:10]:<10}  {update:<8}  {reversal.entry_a:<28}  {reversal.entry_b}")


def run_boards(args):
    if args.json:
        print(json.dumps({name: {"schema": SCHEMA} | {field: value for field, value in named._asdict().items()
                                                      if field not in ("load", "large")}  # all but the two functions
                          for name, named in BOARDS.items()}, indent=1))
        return
    print(f"{'--board':<18}{'sorts by':<16}{'other rules':<26}{'metrics':<12}{'sources':<28}{'snapshot':<12}files")
    for name, named in BOARDS.items():
        print(f"{name:<18}{named.rules[0]:<16}{', '.join(named.rules[1:]) or '-':<26}{', '.join(named.metrics):<12}"
              f"{', '.join(named.sources):<28}{named.pinned:<12}{named.url}")


def run_export(args):
    named = BOARDS[args.board]
    output = args.output or pin_file(args.board)  # --pinned alone renews the snapshot held in this package
    tree, where = open_tree(named, args.source, named.pin if args.pinned else args.at)
    write_table(output, args.board, tree)
    print(f"Wrote {output}: {named.title}, {where}")


def board_options(command, source, boards):
    command.add_argument("source", nargs="?", help=source)
    command.add_argument("--board", choices=boards, help="the leaderboard; `vernier boards` lists them")
    command.add_argument("--rule", choices=RULES, help="how per-unit scores become an order; default: the board's own")
    command.add_argument("--metric", help="the score: a metric of the board, or a column of the table (default: score)")
    command.add_argument("--ties", choices=["first", "average"], help="tied ranks under mean-rank: by entry order, or averaged")
    command.add_argument("--baseline", help="the entry whose scores scale the others under the geometric mean")
    command.add_argument("--sources", help="what counts as a source: a definition of the board, or a column of the "
                                           "table (default: source)")
    command.add_argument("--higher-is-better", action="store_true", help="the table's scores are better when higher")
    command.add_argument("--json", action="store_true", help="print the result as JSON")


def main():
    parser = argparse.ArgumentParser(prog="vernier", description=summary,
                                     epilog="Example: vernier report --board gift-eval")
    parser.add_argument("--version", action="version", version=f"vernier {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("report", help="which places of a board its data can tell apart")
    board_options(command, "a table (.csv), a folder of the board's files, or the address of a git repository that "
                           "holds them; default: the board's own repository, as it is now", [*BOARDS, "all"])
    moment = command.add_mutually_exclusive_group()
    moment.add_argument("--pinned", action="store_true", help="read the paper's snapshot of the board; no network")
    moment.add_argument("--at", metavar="COMMIT", help="read the board's files as of this commit")
    command.add_argument("--order-by", choices=RULES, help="list the entries by this rule and test them under --rule")
    command.add_argument("--tests", help=f"comma-separated, from {', '.join(TESTS)}; or all. Default: cr2,wild, "
                                         "and signed-rank on a geometric mean")
    command.set_defaults(run=run_report)
    command = commands.add_parser("replay", help="top-ten pairs reversed by other entries in the board's history")
    board_options(command, "a git clone or address of the board, or a folder of dated tables; default: the board's own",
                  list(BOARDS))
    command.add_argument("--until", metavar="COMMIT", help="replay the history up to this commit")
    command.set_defaults(run=run_replay)
    command = commands.add_parser("boards", help="the leaderboards --board names")
    command.add_argument("--json", action="store_true", help="print the boards as JSON")
    command.set_defaults(run=run_boards)
    command = commands.add_parser("export", help="write a board's per-unit scores as a table")
    command.add_argument("source", nargs="?", help="a folder of the board's files or a git address; default: its own")
    command.add_argument("--board", choices=list(BOARDS), required=True)
    moment = command.add_mutually_exclusive_group()
    moment.add_argument("--at", metavar="COMMIT", help="read the board's files as of this commit")
    moment.add_argument("--pinned", action="store_true", help="read them as of the commit of the paper's snapshot")
    command.add_argument("-o", "--output", help="the table to write (.csv or .csv.gz); required without --pinned")
    command.set_defaults(run=run_export)
    args = parser.parse_args()
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `vernier report ... | head` ends quietly
    if args.command == "export" and not (args.output or args.pinned):
        parser.error("export needs -o FILE, or --pinned to renew the snapshot held in this package")
    try:
        args.run(args)
    except (BoardError, OSError) as error:  # OSError: a file the command names cannot be read or written
        sys.exit(f"vernier: {error}")


if __name__ == "__main__":
    main()
