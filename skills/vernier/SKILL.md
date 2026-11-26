---
name: vernier
description: Report how small a difference a forecasting leaderboard can detect, and which of its top places the data can tell apart. Use when asked whether a gap between two models on GIFT-Eval, fev-bench, Chronos Benchmark II, TIME, BOOM or any table of per-task scores is real, whether a leaderboard rank is meaningful, or which entries tie with the leader.
---

# Vernier

Vernier reads the scores a leaderboard publishes and tests the order the board displays. It needs `uv` and `git`.

## Run

```bash
uvx vernier-leaderboard report --board gift-eval --json
```

- `--board` is `gift-eval`, `fev-bench`, `fev-bench-mini`, `chronos-bench-ii`, `time`, `boom`, or `all` for one result per board.
- With no other option the command reads the board as it is today. `--pinned` reads the snapshot of the paper (arXiv:2611.31619) with no network.
- For another leaderboard, write a CSV file with the columns `entry,unit,source,score`, one row per entry and task, and run `vernier report scores.csv --json`. A source groups the tasks that were cut from one dataset. Add `--higher-is-better` if a higher score is better.
- `vernier replay --board NAME --json` lists the top-ten pairs that swapped order in an update although neither entry's own results changed.

## Read the result

| Field | Meaning |
|---|---|
| `resolution` | The smallest gap between two entries that a 5% test detects eight times in ten |
| `median_gap` | The median gap between neighbouring top-ten places |
| `ratio` | `resolution / median_gap`. Above 1, the board cannot tell its neighbours apart |
| `tests.cr2.tier` | The leader and the entries that the data does not separate from it |
| `tests.cr2.neighbours_separated` | How many of the nine neighbouring pairs the test separates at 5% |
| `pairs` | Each pair of the top ten with its gap, 95% interval (`low`, `high`) and p-values |
| `fallback` | `true`: the board could not be reached, and the result is of the paper's snapshot |

On `--rule geometric-mean`, read `tests["signed-rank"]` in place of `tests.cr2`.

## Say

- State the resolution next to the gap in question: "the two entries differ by 0.9 rank places, and the board's resolution is 6.8".
- Say "the data does not separate A and B". Do not say that A and B are equal.
- Name the board's state that was read: the `board` field holds the commit and its date.
- If `fallback` is `true`, say that the result is of the snapshot and not of the board today.

Exit status 1 means that the command could not run. stderr then holds one line that says why.
