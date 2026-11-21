"""The command line: python -m vernier report | replay | history-length, each printing a short plain-text table."""
import argparse
from pathlib import Path

from .board import (CRPS, MASE, TIER_POOL, TOP, gift_eval_order, load_csv, load_gift_eval, log_scaled_error, ranks,
                    report)
from .history_length import effects, eligible_units
from .replay import reversals, snapshots_from_git

BASELINE = "Seasonal_Naive"  # the entry whose scores scale every other entry's for the geometric mean


def run_report(args):
    settings = [args.rule]
    if args.board.endswith(".csv"):
        board = load_csv(args.board)
    else:
        board = load_gift_eval(args.board, {"MASE": MASE, "CRPS": CRPS}[args.metric])
        settings.append(f"metric {args.metric}")
    if args.rule == "mean-rank":
        values = ranks(board.scores, args.ties)
        settings.append(f"ties {args.ties}")
    else:
        values = log_scaled_error(board.scores, board.entries.index(BASELINE))
    order = gift_eval_order(board, values)
    result = report(values, board.sources, board.entries, order)
    print(f"{', '.join(settings)}: {len(order)} entries shown with all {len(board.units)} units "
          f"from {len(set(board.sources))} sources")
    if args.rule == "geometric-mean":
        print(f"Values are the log of scaled error, log(error / {BASELINE}'s error), on each unit")
    print(f"{'place':>5}  {'entry':<34}{'mean':>9}  {'gap to next':>11}  {'95% interval':>19}  {'p':>6}")
    for place, row in enumerate(order[:TOP], 1):
        line = f"{place:>5}  {board.entries[row]:<34}{values[row].mean():9.4f}"
        if place < TOP:
            test = result.neighbours[place - 1].test
            line += f"  {test.gap:11.4f}  {test.low:8.4f} to {test.high:<7.4f}  {test.p:6.4f}"
        print(line)
    print(f"Resolution, the gap a 5% test detects 8 times in 10 (median over neighbours): {result.resolution:.4f}")
    print(f"Median gap between neighbours: {result.median_gap:.4f}")
    print(f"Neighbouring pairs separated at 5%: {result.neighbours_separated} of {len(result.neighbours)}, "
          f"{result.neighbours_after_holm} after Holm")
    print(f"All pairs separated at 5%: {result.pairs_separated} of {len(result.pairs)}, "
          f"{result.pairs_after_holm} after Holm")
    print(f"Leading tier, not separated from the leader at 5% among the first {TIER_POOL} places: "
          f"{len(result.tier)} ({', '.join(result.tier)})")


def run_replay(args):
    replay = reversals(snapshots_from_git(args.clone), args.ties)
    updates_with_reversal = len({reversal.commit for reversal in replay.reversals})
    print(f"mean rank, ties {args.ties}: {replay.updates} updates examined, {updates_with_reversal} reversed "
          f"{len(replay.reversals)} pairs of top-ten entries whose own results did not change")
    print(f"{'date':<10}  {'commit':<7}  {'placed higher before':<28}  placed lower before")
    for reversal in replay.reversals:
        print(f"{reversal.date[:10]}  {reversal.commit[:7]}  {reversal.entry_a:<28}  {reversal.entry_b}")


def run_history_length(args):
    units = eligible_units(args.eligible)
    variants = [folder.name for folder in sorted(Path(args.runs).iterdir()) if folder.is_dir()]
    print(f"MASE at 512 steps of history over MASE at the documented length, {len(units)} eligible configurations")
    print(f"{'variant':<20}{'units':>6}{'sources':>8}{'effect':>8}  {'95% interval':<17}{'p':>10}{'Holm p':>11}")
    for e in effects(args.runs, variants, units):
        print(f"{e.variant:<20}{e.n_units:>6}{e.n_sources:>8}{e.effect:>+8.1%}  {e.low:>+6.1%} to {e.high:<+7.1%}"
              f"{e.p:>10.4g}{e.p_holm:>11.4g}")


def main():
    parser = argparse.ArgumentParser(prog="python -m vernier")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("report", help="which places of a board its data can tell apart")
    command.add_argument("board", help="a GIFT-Eval results folder, or a CSV with columns entry, unit, source, score")
    command.add_argument("--rule", choices=["mean-rank", "geometric-mean"], default="mean-rank")
    command.add_argument("--metric", choices=["MASE", "CRPS"], default="MASE")
    command.add_argument("--ties", choices=["first", "average"], default="first")
    command.set_defaults(run=run_report)
    command = commands.add_parser("replay", help="top-ten pairs reversed by other entries in the board's history")
    command.add_argument("clone", help="a git clone of the board's repository")
    command.add_argument("--ties", choices=["first", "average"], default="first")
    command.set_defaults(run=run_replay)
    command = commands.add_parser("history-length", help="effect of 512 steps of history on each model")
    command.add_argument("runs", help="a folder of variant folders, each with ctx_default/ and ctx_512/")
    command.add_argument("--eligible", required=True, help="a JSON file of each configuration's median_length")
    command.set_defaults(run=run_history_length)
    args = parser.parse_args()
    args.run(args)


main()
