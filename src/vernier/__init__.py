"""Vernier reports what a leaderboard's own table of scores can establish about the order of its entries."""
from .board import Board, listed_order, load_csv, log_scaled_error, ranks, values_under, win_rate
from .boards import BOARDS, load_pin
from .replay import reversals, snapshots_from_git
from .report import report
from .sources import BoardError, Commit, Folder, clone_of
from .stats import cr2_test, holm, resolution, sign_flip_interval, sign_flip_p

__version__ = "1.0.0"
gift_eval_order = listed_order  # the name this function had when GIFT-Eval was the only board

__all__ = [
    "BOARDS", "Board", "BoardError", "Commit", "Folder", "clone_of", "cr2_test",
    "gift_eval_order", "holm", "listed_order", "load_csv", "load_pin", "log_scaled_error", "ranks",
    "report", "resolution", "reversals", "sign_flip_interval", "sign_flip_p", "snapshots_from_git", "values_under",
    "win_rate",
]
