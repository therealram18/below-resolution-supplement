# Reference

`vernier --help` and `vernier COMMAND --help` print the options. `vernier boards` lists the leaderboards.

## `vernier report`

Where the report reads the board from:

| Option | The report reads |
|---|---|
| none | the board's own repository as it is now. A copy is kept in `~/.cache/vernier` (or `$VERNIER_CACHE`) and updated on each run |
| `--pinned` | the paper's snapshot of the board, held in this package; no network |
| `--at COMMIT` | the board's repository as of a commit |
| a folder or a git address after `report` | your own copy of the board's files, for example `vernier report --board gift-eval https://github.com/SalesforceAIResearch/gift-eval` |
| a `.csv` or `.csv.gz` file after `report` | a table of scores of any leaderboard |

If the board's repository cannot be reached, the report reads the paper's snapshot. Its first line then says `NOT the board today`, stderr says why, and the JSON has `"fallback": true`.

TIME keeps its score files outside git. The first report of `time` downloads them into the Hugging Face cache, and later reports read them from there.

`--board all` prints one row per board. Its exit status is 1 if a board could not be read, and that board's row says why.

What the report computes:

| Option | Meaning |
|---|---|
| `--rule mean-rank`, `geometric-mean`, `win-rate` | How per-unit scores become an order. The default is the rule the board sorts by. Values are rank places, the log of error over the baseline's error, and win rate in percentage points |
| `--metric` | The score. `vernier boards` lists each board's metrics |
| `--ties first`, `average` | Tied ranks go to the entry the board lists first, or are averaged. The default is the board's own way |
| `--sources` | What counts as a source. BOOM has `source` (each series), `domain-type` (36 groups) and `domain` (23 groups) |
| `--order-by RULE` | List the entries by one rule and test them under `--rule` |
| `--baseline ENTRY` | The entry whose scores scale the others under the geometric mean |
| `--tests` | Comma-separated tests, or `all`. The default is `cr2,wild`, and `signed-rank` on a geometric mean |
| `--higher-is-better` | The scores of a table are better when higher |
| `--json` | Print the report as JSON |

The tests of a zero mean difference between two entries:

| `--tests` | Test | Notes |
|---|---|---|
| `cr2` | t-test with a standard error clustered by source (CR2) and Satterthwaite degrees of freedom | Always runs. It gives the resolution |
| `wild` | Wild cluster bootstrap: each draw gives every source one random sign; 99,999 draws; the same result on every run | Lists every sign pattern when there are no more of them than draws |
| `exact` | The exact form of `wild`: every sign pattern | Up to 28 sources; about two seconds per pair at 28 |
| `exact-sum` | Every sign pattern, with the plain sum of the differences as the statistic | Up to 50 sources |
| `signed-rank` | Exact Wilcoxon signed-rank test on the sources' mean differences | Up to 50 sources. The test to read on a geometric mean |
| `percentile` | The percentile-bootstrap rule of the submitted paper | For comparison only: it separates too many pairs |

The first number of a count is uncorrected, and the second is after the Holm correction.

On a geometric mean the CR2 test and the wild bootstrap separate too many pairs (the paper's Appendix A.2: they reject a true zero about 16% of the time where 5% is intended). The report then also runs the signed-rank test, which holds its level on that rule, and says so.

The resolution is the median of (t<sub>0.975</sub> + t<sub>0.80</sub>) × SE over the pairs of neighbouring top-ten places. Each pair has its own CR2 standard error and Satterthwaite degrees of freedom.

## A table of any leaderboard

```csv
entry,unit,source,score
Auto_Arima,bitbrains_fast_storage/5T/long,bitbrains_fast_storage,1.14
```

- One row per entry and unit. A source groups the units that share data, such as the frequencies and horizons cut from one dataset.
- A lower score is better. Add `--higher-is-better` for accuracy, skill scores and the like.
- `--metric COLUMN` and `--sources COLUMN` name other columns, so one file can hold several scores and several definitions of a source.
- An empty score is a missing result. Only entries with a score on every unit are listed. The top ten of them are tested, or all of them if the board lists fewer.
- Tied ranks go to the entry that comes first in the file.
- `--rule geometric-mean` needs `--baseline ENTRY`.
- A column named `band`, if present, gives the board's listing band of each entry: the board lists band 0 first. An empty band is an entry that the board does not list.

`vernier export --board NAME -o table.csv` writes a named board's per-unit scores in this format.

## `vernier replay`

An update is a commit that changed the board's result files. An update is examined when the board lists ten entries before and after it and its units are the same. Among the entries in the top ten before the update, Vernier takes those whose own files are unchanged and lists the pairs that swapped order.

`replay` takes `--board`, `--rule`, `--metric`, `--ties`, `--baseline` and `--json` as `report` does, and `--until COMMIT` to stop at a commit. Under the geometric mean, which scores an entry against a fixed baseline, the paper found no such pair on any board: add `--rule geometric-mean`.

For a board that is not named, put one table per date in a folder. `vernier replay FOLDER` replays the tables in the order of their names. An entry's own results are its scores.

## JSON

`vernier report --json` prints one object. `--board all --json` prints one such object per board, keyed by the board's name. A board that could not be read has `error` in place of the report.

| Field | Meaning |
|---|---|
| `schema` | 1. It goes up when a field is renamed or removed |
| `board` | The board and where it was read from: a commit and its date, a snapshot, a folder or a table |
| `fallback` | `true` when the board could not be reached and the paper's snapshot was read |
| `rule`, `metric`, `ties`, `ordered_by` | What was computed |
| `entries_listed`, `units`, `sources` | The size of the board |
| `places` | The top ten: `place`, `entry`, `mean` (the board's statistic) |
| `resolution`, `median_gap`, `ratio` | The resolution, the median gap between neighbouring places, and the first over the second (`null` if the gap is zero) |
| `pairs` | Each pair of the top ten: `places`, `entries`, `gap`, `se`, `df`, `low`, `high` and `p` by test |
| `tests` | By test: `name`, `p` of each pair, `neighbours_separated`, `neighbours_after_holm`, `pairs_separated`, `pairs_after_holm`, `tier` |

A number that does not exist is `null`: the p-value of two entries with the same value on every unit, and each p-value of the `percentile` rule.

`vernier replay --json` prints `schema`, `board`, `rule`, `metric`, `ties`, `updates`, `updates_with_reversal` and `reversals` (`date`, `commit`, `entry_a` placed higher before, `entry_b`). `vernier boards --json` prints each board's address, pinned commit, metrics, rules and whether its snapshot is held.

Exit status: 0 when the command ran, 1 when it could not, 2 for a wrong option. On 1, stderr holds one line that starts with `vernier:`.

## Python

```text
from vernier import listed_order, load_pin, report, values_under

board = load_pin("fev-bench", metric="SQL")            # or load_csv("scores.csv")
values = values_under("win-rate", board)               # entries x units, lower is better
result = report(values, board.sources, board.entries, listed_order(board, values), tests=("cr2", "wild"))
print(result.resolution, result.median_gap, result.ratio, result.tier, result.verdicts["wild"].pairs_separated)
```

| Module | Holds |
|---|---|
| `vernier.board` | `Board`, the table of scores, and the rules that turn scores into an order |
| `vernier.stats` | The tests: `cr2_test`, `wild_p`, `sign_flip_t_p`, `sign_flip_p`, `signed_rank_p`, `percentile_separated`, `resolution`, `holm` |
| `vernier.report` | `report`, and the report as text and as plain data |
| `vernier.replay` | The bystander test |
| `vernier.sources` | A folder, or one commit of a cached clone, as a tree of files |
| `vernier.loaders` | One function per leaderboard that turns the board's files into a `Board` |
| `vernier.boards` | `BOARDS`: where each leaderboard keeps its files, its rules and its pinned snapshot |
