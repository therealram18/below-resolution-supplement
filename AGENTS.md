# For agents

To use the tool, load [skills/vernier](skills/vernier/SKILL.md). This file is for changing the repository.

```bash
uv sync                      # the tool and pytest, in .venv
just test                    # the tests that need no network
just test-live               # the tests that read the boards' repositories on GitHub
uvx ruff check src tests paper/code
```

| File | Holds |
|---|---|
| `src/vernier/board.py` | `Board` and the rules: mean rank, geometric mean, win rate |
| `src/vernier/stats.py` | The tests of a zero mean difference, clustered by source |
| `src/vernier/report.py` | The report, as text and as data |
| `src/vernier/replay.py` | The bystander test |
| `src/vernier/sources.py` | A folder or a commit of a cached clone as a tree of files; `BoardError` |
| `src/vernier/loaders.py` | One function per leaderboard: the board's files to a `Board` |
| `src/vernier/boards.py` | `BOARDS`, the table of leaderboards, and their snapshots in `src/vernier/pins/` |
| `src/vernier/__main__.py` | The command line |

- A message for the user is a `BoardError` that says what is wrong and what to change. The command line prints it and exits with 1. Catch an exception only where data enters: a board's file, the network.
- A loader repeats the arithmetic of the board's own code, so that Vernier lists the order the board displays. Name the board's file or function that it follows.
- To add a leaderboard: write one loader in `loaders.py`, add one entry to `BOARDS`, test the loader on a small board written by hand, and run `just pin NAME` to write its snapshot.
- A test asserts a number as the paper prints it. If the code and the paper disagree, find the cause; do not change the number. `tests/test_paper_numbers.py` and `paper/results/` stay as they are.
- Keep lines to 120 characters and modules small. Use a library before new code.
