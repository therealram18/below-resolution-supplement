---
name: vernier
description: Tests whether the gaps on a time-series forecasting leaderboard are real. Reports a board's resolution (the smallest gap its data can detect), which top places its data can tell apart, and the leading tier of models that tie with the leader. Use this whenever someone compares forecasting or time-series foundation models on GIFT-Eval, fev-bench, Chronos Benchmark II, TIME, BOOM or their own per-task scores, asks whether one model really beats another, whether a leaderboard rank or a new state of the art is meaningful, or which models are statistically tied, even if they do not mention Vernier.
---

# Vernier

A leaderboard ranks models by an average over tasks, but tasks cut from one dataset move together, so small gaps are often noise. Vernier repeats the board's own arithmetic on its published scores and tests each gap with the tasks clustered by dataset. It needs `uv`, and `git` for a named board.

## 1. Run

For a public board, as it is today:

```bash
uvx vernier-leaderboard report --board gift-eval --json
```

`--board` is `gift-eval`, `fev-bench`, `fev-bench-mini`, `chronos-bench-ii`, `time`, `boom`, or `all` for one row per board. Add `--pinned` for the paper's snapshot, which needs no network.

For any other scores, write a long table with one row per model and task, then pass the file:

```python
import pandas as pd
df = pd.read_csv("results.csv")                      # e.g. columns model, dataset, MASE
df["source"] = df["dataset"].str.split("/").str[0]   # the dataset a task was cut from
df.rename(columns={"model": "entry", "dataset": "unit", "MASE": "score"})[
    ["entry", "unit", "source", "score"]].to_csv("scores.csv", index=False)
```

```bash
uvx vernier-leaderboard report scores.csv --json
```

Lower scores are better; add `--higher-is-better` for accuracy or skill scores. The `source` column matters most: tasks that share data must share a source, or every gap looks more certain than it is.

The report tests the top ten. To test two particular models, keep only their rows and the baseline's, and add `--rule geometric-mean --baseline NAME`: a geometric mean does not depend on which other models are on the board.

## 2. Read

| Field | Meaning |
|---|---|
| `resolution` | The smallest gap a 5% test detects eight times in ten, in the board's own units |
| `median_gap` | The median gap between neighbouring top-ten places |
| `ratio` | `resolution / median_gap`; above 1, the board cannot tell its neighbours apart |
| `tests.cr2.tier` | The leader and the models the data does not separate from it |
| `pairs` | Each top-ten pair with its `gap`, 95% interval (`low`, `high`) and p-values |
| `board` | The state that was read: a commit and its date, or the snapshot |
| `fallback` | `true` when the board was unreachable and the paper's snapshot was read |

On `--rule geometric-mean`, read `tests["signed-rank"]` instead of `tests.cr2`; the other tests reject too often on that rule.

## 3. Say

- Put the gap next to the resolution: "A leads B by 0.9 rank places; this board resolves 6.8."
- Say "the data does not separate A and B", not "A and B are equal": a test that fails to separate two models does not show they are the same.
- Name the leading tier when someone asks who is best.
- Name the board state from `board`. If `fallback` is `true`, say the result is of the snapshot, not of the board today.

Exit status 1 means the command could not run; stderr says why in one line. `vernier --help` lists every option, and `vernier replay --board NAME` lists top-ten pairs that swapped order only because other models arrived.
