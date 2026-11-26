# The paper's numbers

Each pinned snapshot is a table in `src/vernier/pins/`, and `--pinned` reads it. The tests in `tests/` assert each number below as the paper prints it.

| Board | Snapshot | Commit | Entries, units, sources |
|---|---|---|---|
| GIFT-Eval | 2026-09-07 | `b61c48aa` | 128, 97, 28 datasets |
| fev-bench | 2026-11-11 | `9cf943c3` | 40, 100, 39 families |
| fev-bench-mini | 2026-11-11 | `9cf943c3` | 40, 20, 15 families |
| Chronos Benchmark II | 2026-11-11 | `9cf943c3` | 21, 27, 19 sources |
| TIME | 2026-10-31 | `3bf103cb` | 43, 98, 39 datasets |
| BOOM | 2026-11-11 | `1d3e1c97` | 23, 6,750, 2,554 series |

## Resolution and separated pairs

The command of each row starts with `vernier report --pinned --board`. The last column counts the neighbouring pairs that CR2, the wild bootstrap and the percentile bootstrap separate (add `--tests cr2,wild,percentile`).

| Row of the paper's table of all boards | Command | Resolution | Median gap | Separated |
|---|---|---|---|---|
| GIFT-Eval, mean rank on MASE | `gift-eval` | 6.80 | 0.91 | 1 / 1 / 0 |
| GIFT-Eval, geometric mean | `gift-eval --rule geometric-mean` | 0.0497 | 0.0045 | see below |
| fev-bench, win rate | `fev-bench` | 7.51 | 2.08 | 0 / 0 / 0 |
| fev-bench, skill score | `fev-bench --rule geometric-mean` | 0.0320 | 0.0083 | 0 / 0 / 0 |
| fev-bench-mini, win rate | `fev-bench-mini` | 15.3 | 2.18 | 1 / 1 / 1 |
| Chronos Benchmark II, mean rank | `chronos-bench-ii` | 4.24 | 0.22 | 0 / 0 / 1 |
| Chronos Benchmark II, geometric mean | `chronos-bench-ii --rule geometric-mean` | 0.0929 | 0.0059 | 0 / 0 / 0 |
| TIME, geometric mean | `time` | 0.0238 | 0.0021 | 0 / 1 / 1 |
| TIME, MASE rank column | `time --rule mean-rank` | 4.01 | 0.378 | 1 / 1 / 1 |
| BOOM, each series a source | `boom` | 0.196 | 0.312 | 9 / 9 / 9 |
| BOOM, 36 groups by domain and type | `boom --sources domain-type` | 0.420 | 0.312 | 6 / 6 / 7 |
| BOOM, 23 groups by domain | `boom --sources domain` | 0.635 | 0.312 | 5 / 6 / 7 |

- A geometric mean is in log units here. The paper prints fev-bench's skill score and Chronos Benchmark II's geometric mean times 100 (3.20 and 0.83; 9.29 and 0.59).
- The GIFT-Eval geometric-mean row of the paper counts pairs among the ten entries the board displays, which mean rank orders: `gift-eval --rule geometric-mean --order-by mean-rank` gives its 0 / 0 / 2. Without `--order-by` the report tests the geometric mean's own top ten.
- The counts under every test on GIFT-Eval (the paper's table of counts): `gift-eval --tests all`, with `--metric CRPS` or `--rule geometric-mean --order-by mean-rank` for the other two rows. The leading tiers of 3, 17 and 16 entries: `gift-eval` and `gift-eval --rule geometric-mean`.

## The bystander test

Each board is replayed up to the snapshot's commit.

| Board and rule | Command | Reversed pairs | Updates |
|---|---|---|---|
| GIFT-Eval, mean rank | `vernier replay --board gift-eval --until b61c48aa` | 20 (18 with `--ties average`) | 14 of 141 |
| fev-bench, win rate | `vernier replay --board fev-bench --until 9cf943c3` | 2 | 2 of 30 |
| Chronos Benchmark II, mean rank | `vernier replay --board chronos-bench-ii --until 9cf943c3` | 8 | 3 of 11 |
| TIME, MASE rank column | `vernier replay --board time --rule mean-rank --until 3bf103cb` | 13 | 12 of 44 |

## The history-length experiment

```bash
uv run python paper/code/history_length/effects.py paper/results/runs/history_length paper/results/runs/eligible_configurations.json
```

```text
MASE at 512 steps of history over MASE at the documented length, 70 eligible configurations
variant              units sources  effect  95% interval              p     Holm p
chronos_bolt_base       70      14   +2.7%   -0.9% to +6.2%      0.1146     0.1146
moirai2_small           70      14   +6.2%   +1.6% to +10.4%   0.005127    0.01025
timesfm25               70      14  +15.3%   +4.1% to +32.6%  0.0002441  0.0009766
toto__A__seed42         70      14  +15.3%   +4.0% to +27.1%  0.0002441  0.0009766
```

A configuration is eligible when its median series length is at least 1,024 steps. For each model, d = log(MASE at 512 steps / MASE at the documented length) on each configuration. The effect is exp(mean d) - 1. The p-value is exact over all sign flips of the sources' totals of d. The disclosure checklist that follows from this experiment is in [disclosure.md](disclosure.md).

## What was checked, and what was not

Checked on 2026-11-26 from a machine that reaches GitHub and not Hugging Face:

| Claim | How | Result |
|---|---|---|
| The snapshots of fev-bench, fev-bench-mini, Chronos Benchmark II and BOOM are the boards' files at the pinned commits | `just test-live` | passed |
| Each row above except the two of TIME | `just test` | passed |
| The replays of fev-bench and Chronos Benchmark II | `just test-live` | passed |
| The counts under the exact test with the t statistic on GIFT-Eval | `just test-slow` | passed |
| BOOM's snapshot gives the mean ranks the board publishes, to its three decimals | `just test` | passed |
| The report of each of these boards as it is now | `vernier report --board NAME` | ran |

Not checked, because the machine could not reach Hugging Face:

- **TIME.** No snapshot of TIME is in `src/vernier/pins/` yet, and its report and replay have not run on TIME's real files. The loader follows TIME's own `scripts/compute_local_leaderboard.py` and is tested on small files written in TIME's format. `just pin time` writes the snapshot, and `just test` then checks the two TIME rows above.
- **GIFT-Eval over the network.** The report and replay of the Hugging Face Space have not run from this machine. The same code ran on GIFT-Eval's GitHub repository, which holds almost the same files with another commit history.

`just test-live github,huggingface` runs each check of both lists.

## Folders

- `src/vernier/`: the tool. `src/vernier/pins/`: the paper's snapshot of each board as a table.
- `tests/`: the tool on small hand-made boards, on the pinned snapshots (the paper's numbers) and on each board's own repository (`just test-live`).
- `paper/code/`: code that produces a result of the paper and is not part of the tool. It holds the history-length analysis and its test.
- `paper/results/`: the files that this code reads: the two result tables of each model run, four records of a run's settings, and the two lists of configurations.

## What this tree does not replicate yet

The commands on this page give every number of the paper's board tables, of the bystander test and of the history-length experiment. These parts of the paper are not replicated from this tree: the calibration simulations, the checks against reference software, the removal and near-copy tests, the GIFT-Eval sub-boards, the model runs themselves and the three figures. Their scripts and stored outputs are public at two earlier commits of this repository: [`47595a5`](https://github.com/therealram18/vernier/tree/47595a571429962ee5635d21e29cb5a162dc4604) (scripts in `camera_ready/analysis/`, `camera_ready/history_length/` and `code/`) and [`3ab82ad`](https://github.com/therealram18/vernier/tree/3ab82adebd7d14aaa07924377b05268711b9f828) (outputs in `record/` and `runs/`, GIFT-Eval's raw files in `snapshot/`). Those scripts are not written on top of the tool, so they are not copied here.
