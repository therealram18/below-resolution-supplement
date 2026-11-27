---
name: vernier
description: Tests whether the gaps on a time-series forecasting leaderboard are real. Reports a board's resolution (the smallest gap its data can detect), which top places its data can tell apart, and the leading tier of models that tie with the leader. Use this whenever someone compares forecasting or time-series foundation models on GIFT-Eval, fev-bench, Chronos Benchmark II, TIME, BOOM or their own per-task scores, asks whether one model really beats another, whether a leaderboard rank or a new state of the art is meaningful, or which models are statistically tied, even if they do not mention Vernier.
---

# Vernier

A leaderboard ranks models by an average over tasks, but tasks cut from one dataset move together, so small gaps are often noise. Vernier repeats the board's own arithmetic on its published scores and tests each gap with the tasks clustered by dataset. It needs `uv`, and `git` for a board as it is today.

## 1. Run

Who leads a public board, and which places its data can tell apart:

```bash
uvx vernier-leaderboard report --board gift-eval --json
```

`--board` is `gift-eval`, `fev-bench`, `fev-bench-mini`, `chronos-bench-ii`, `time`, `boom`, or `all` for one row per board. Add `--pinned` for the paper's snapshot, which needs no network.

The report covers the top ten. For any two models, wherever they rank, export the board's scores, keep the two, and compare them on a geometric mean, which does not depend on the other models:

```bash
uvx vernier-leaderboard export --board boom -o boom.csv
python3 -c "import pandas as pd; d = pd.read_csv('boom.csv'); d[d.entry.isin(['A', 'B'])].to_csv('pair.csv', index=False)"
uvx vernier-leaderboard report pair.csv --board boom --rule geometric-mean --baseline A --json
```

For your own scores, write a long table with one row per model and task:

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

## 2. Read

| Field | Meaning |
|---|---|
| `board` | The state that was read: a commit and its date, or the paper's snapshot |
| `resolution` | The smallest gap a 5% test detects eight times in ten, in the board's own units |
| `median_gap` | The median gap between neighbouring top-ten places |
| `ratio` | `resolution / median_gap`; above 1, the board cannot tell its neighbours apart |
| `tests.cr2.tier` | The leader and the models not separated from it at 5% |
| `pairs` | Each pair with its `gap`, 95% interval (`low`, `high`) and a p-value per test |
| `fallback` | `true` when the board was unreachable and the paper's snapshot was read |

On `--rule geometric-mean`, read the `signed-rank` test instead of `cr2`; the others reject too often on that rule. A board with more than 50 sources, such as BOOM, has no signed-rank test; there a `cr2` separation on a geometric mean is weak evidence, and a `cr2` non-separation stands.

## 3. Say

- Give the gap, its interval and the board's resolution: "A leads B by 0.9 rank places (95% interval −2.1 to 3.9); this board resolves 6.8." The interval decides the pair; the resolution says how fine the board is.
- Say "the data does not separate A and B", not "A and B are equal": a test that fails to separate two models does not show they are the same.
- Name the leading tier when someone asks who is best. If the model asked about is not the leader, say where it ranks.
- The board's rule answers "who ranks higher on this board"; the geometric mean answers "which of the two forecasts better". When they disagree, give both.
- Name the board state from `board`. If `fallback` is `true`, say the result is of the snapshot, not of the board today.

Exit status 1 means the command could not run; stderr says why in one line. `uvx vernier-leaderboard --help` lists every option, and `uvx vernier-leaderboard replay --board NAME` lists top-ten pairs that swapped order only because other models arrived.
