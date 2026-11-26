# For agents

Vernier reports how small a difference a forecasting leaderboard can detect. This file tells an agent how to use the tool and how to change this repository.

## Use the tool

Run it with no install (needs `uv`, and `git` to read a board's repository):

```bash
uvx vernier-leaderboard report --board gift-eval --json
```

In a clone, `uv run vernier ...` runs the same command.

| Question | Command |
|---|---|
| Which boards are there? | `vernier boards --json` |
| What can one board resolve today? | `vernier report --board NAME --json` |
| What can each board resolve? | `vernier report --board all --json` |
| What did the paper measure? | add `--pinned` (no network) |
| What can my own table of scores resolve? | `vernier report scores.csv --json` (columns `entry,unit,source,score`) |
| Did other entries reverse a pair of top-ten entries? | `vernier replay --board NAME --json` |

Read the result as follows.

- `ratio` is the resolution over the median gap between neighbouring top-ten places. Above 1, the board cannot tell its neighbours apart.
- `tests.cr2.tier` lists the leader and the entries that the data does not separate from it.
- `tests.NAME.neighbours_separated` counts the neighbouring pairs that a test separates at 5%, and `neighbours_after_holm` counts them after the Holm correction.
- On `--rule geometric-mean`, read `tests["signed-rank"]`. The CR2 test and the wild bootstrap separate too many pairs on that rule.
- `fallback: true` means that the board could not be reached and the paper's snapshot was read. Do not report such a result as the board today.
- A `null` p-value belongs to two entries with the same value on every unit.

Exit status: 0 the command ran, 1 it could not, 2 a wrong option. On 1, stderr holds one line that starts with `vernier:` and says what to change. With `--board all`, status 1 means that a board has `error` in place of its report.

All fields and options: [docs/reference.md](docs/reference.md).

## Change this repository

```bash
uv sync          # the tool and pytest, in .venv
just test        # the tests that need no network, under two minutes
just test-live   # the tests that read the boards' repositories on GitHub
uvx ruff check src tests paper/code
```

| File | Holds |
|---|---|
| `vernier/board.py` | `Board` and the rules: mean rank, geometric mean, win rate |
| `vernier/stats.py` | The tests of a zero mean difference, clustered by source |
| `vernier/report.py` | The report, as text and as data |
| `vernier/replay.py` | The bystander test |
| `vernier/sources.py` | A folder or a commit of a cached clone as a tree of files; `BoardError` |
| `vernier/loaders.py` | One function per leaderboard: the board's files to a `Board` |
| `vernier/boards.py` | `BOARDS`, the table of leaderboards, and the pinned snapshots |
| `vernier/__main__.py` | The command line |

Rules:

- A message for the user is a `BoardError`. It says what is wrong and what to change. The command line prints it and exits with 1. Catch an exception only where data enters: a board's file, the network.
- A loader repeats the arithmetic of the board's own code, so that Vernier lists the order the board displays. Name the board's file or function that it follows.
- To add a leaderboard: write one loader in `loaders.py`, add one entry to `BOARDS`, and test the loader on a small board written by hand. Then run `just pin NAME` to write its snapshot.
- Do not edit `tests/test_paper_numbers.py` or `paper/results/`: they are the record of the paper. `vernier export --pinned` writes the tables in `src/vernier/pins/`; do not write them by hand.
- A test asserts a number as the paper prints it. If the code and the paper disagree, find the cause; do not change the number.
- Keep lines to 120 characters and modules small. Use a library before new code.
