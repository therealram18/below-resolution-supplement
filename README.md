<h1 align="center">Vernier</h1>

<p align="center"><b>How small a difference can a forecasting leaderboard detect?</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.31619"><img alt="arXiv 2611.31619" src="https://img.shields.io/badge/arXiv-2611.31619-b31b1b"></a>
  <img alt="NeurIPS 2026, FMTS Workshop, oral" src="https://img.shields.io/badge/NeurIPS_2026-FMTS_Workshop_%C2%B7_Oral-68448b">
  <a href="https://pypi.org/project/vernier-leaderboard/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vernier-leaderboard"></a>
  <a href="https://github.com/therealram18/vernier/actions/workflows/test.yml"><img alt="tests" src="https://github.com/therealram18/vernier/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://github.com/therealram18/vernier/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache_2.0-blue"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.31619">Paper</a> ·
  <a href="#one-line">One line</a> ·
  <a href="#your-own-leaderboard">Your own leaderboard</a> ·
  <a href="https://github.com/therealram18/vernier/blob/main/docs/reference.md">Reference</a> ·
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/therealram18/vernier/main/docs/vernier.gif" alt="A new model joins a forecasting leaderboard and two other models swap places; the smallest gap that a leaderboard can detect; the result on five leaderboards" width="100%">
</p>

Vernier reads the scores that a leaderboard already publishes and reports the board's *resolution*: the smallest gap between two entries that the board's data can detect. On four of the five leaderboards in the paper, the resolution is 3.6 to 19 times the gap between neighbouring top-ten entries.

The paper: [What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models](https://arxiv.org/abs/2611.31619), by Arya Sikder and Sriram Pillutla.

## One line

```bash
uvx vernier-leaderboard report --board gift-eval
```

The line needs [uv](https://docs.astral.sh/uv/) and no install step. It fetches the board's own result files as they are today, repeats the board's own arithmetic and tests the order that the board displays.

| `--board` | Leaderboard | The board sorts by | Resolution | Gap | Resolution ÷ gap |
|---|---|---|---|---|---|
| `gift-eval` | [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) | mean rank | 6.80 | 0.91 | 7.5 |
| `fev-bench` | [fev-bench](https://github.com/autogluon/fev), 100 tasks | win rate | 7.51 | 2.08 | 3.6 |
| `fev-bench-mini` | fev-bench, 20 tasks | win rate | 15.3 | 2.18 | 7.0 |
| `chronos-bench-ii` | [Chronos Benchmark II](https://github.com/autogluon/fev) | mean rank | 4.24 | 0.22 | 19 |
| `time` | [TIME](https://huggingface.co/datasets/Real-TSF/TIME-Output) | geometric mean | 0.0238 | 0.0021 | 11 |
| `boom` | [BOOM](https://github.com/DataDog/toto) | mean rank | 0.196 | 0.312 | 0.63 |
| `all` | every board, one row each | | | | |

The numbers are those of the paper's snapshots, and `--pinned` reads a snapshot with no network. The gap is the median gap between neighbouring top-ten entries. A ratio above 1 means that the board cannot tell its neighbours apart.

## The report

`vernier report --board gift-eval --pinned` prints:

```text
GIFT-Eval, the paper's snapshot of 2026-09-07 (commit b61c48aa of https://huggingface.co/spaces/Salesforce/GIFT-Eval)
mean-rank on MASE, ties first: 128 entries listed with all 97 units from 28 sources
Values are rank places; a lower value is better
place  entry                                  mean  gap to next         95% interval          p, cr2         p, wild
    1  STRIDE w/ Synapse                   14.6186       4.2474   -2.3998 to 10.8947          0.1889          0.1783
    2  EXAONE-Forecast-Agent               18.8660       0.4433   -6.6844 to 7.5710           0.8952          0.8968
    3  LS-MoE                              19.3093       1.9381   -0.1473 to 4.0236           0.0665          0.0639
    4  LS-Agent                            21.2474       0.9072   -3.8242 to 5.6386           0.6909          0.6894
    5  Falcon-Agent                        22.1546       0.0722   -4.7538 to 4.8981           0.9752          0.9772
    6  CastStar                            22.2268       0.0103   -2.9771 to 2.9977           0.9943          0.9999
    7  limix_moe                           22.2371       1.4227   -4.1707 to 7.0160           0.5947          0.5872
    8  TIMEHEDGE                           23.6598       0.6495   -6.6299 to 7.9289           0.8496          0.8537
    9  RegLab-AH-1                         24.3093       1.3299    0.0945 to 2.5653           0.0364          0.0399
   10  RacineCast-1-NC                     25.6392
Resolution, the gap a 5% test detects 8 times in 10 (median over neighbours): 6.8010
Median gap between neighbours: 0.9072, so the resolution is 7.5 times the gap
Separated at 5%, and after Holm:  neighbouring pairs  all pairs           leading tier
  CR2 t-test                      1 of 9, 0           12 of 45, 0         3
  wild cluster bootstrap          1 of 9, 0           13 of 45, 1         3
Leading tier: the leader and the entries not separated from it among the first 40 places
  CR2 t-test: STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE
  wild cluster bootstrap: the same entries
```

- A *unit* is one evaluation task, and its *source* is the data that the task was cut from. Units of one source are correlated, so each test treats a source as one cluster.
- Each row compares a place with the next place: the gap, its 95% interval and the p-value of each test.
- The *leading tier* is the leader and the entries that the data cannot separate from it.

| To | Add |
|---|---|
| read the paper's snapshot, with no network | `--pinned` |
| read the board as of a commit | `--at COMMIT` |
| use another rule, score or test | `--rule win-rate`, `--metric CRPS`, `--tests all` |
| get JSON | `--json` |

All options and tests: [docs/reference.md](https://github.com/therealram18/vernier/blob/main/docs/reference.md).

## Your own leaderboard

Write one row per entry and unit, then give the file to the same command:

```csv
entry,unit,source,score
Auto_Arima,bitbrains_fast_storage/5T/long,bitbrains_fast_storage,1.14
Auto_Arima,bitbrains_fast_storage/5T/medium,bitbrains_fast_storage,1.22
```

```bash
vernier report scores.csv
```

A lower score is better; add `--higher-is-better` for accuracy and skill scores.

## Does the order depend on who else submitted?

`vernier replay` replays a board's history. It lists the top-ten pairs that swapped order in an update although neither entry's own results changed:

```bash
vernier replay --board fev-bench --until 9cf943c3
```

```text
fev-bench, win-rate on SQL: 30 updates examined, 2 reversed 2 pairs of top-ten entries whose own results did not change
date        update    placed higher before          placed lower before
2026-10-14  83d347de  ChakraTS                      timesfm-3
2026-10-15  4ae368ca  TiRex-2                       Toto-2.0-2.5B
```

Mean rank and win rate score an entry against the other entries, so new entries can reverse such pairs. Across four boards the paper counts 43 of these pairs.

## For agents

`report`, `replay` and `boards` take `--json`. The exit status is 0 when the command ran and 2 for a wrong option. It is 1 when the command could not run, and stderr says why in one line. [AGENTS.md](https://github.com/therealram18/vernier/blob/main/AGENTS.md) lists the fields, and [skills/vernier](https://github.com/therealram18/vernier/blob/main/skills/vernier/SKILL.md) is a skill that an agent can load.

## Install

```bash
uv tool install vernier-leaderboard
```

or `pip install vernier-leaderboard`. The command is `vernier`. It needs Python 3.10 or later, and git to read a board's repository. The newest source: `uv tool install git+https://github.com/therealram18/vernier`.

## Reproduce the paper

```bash
git clone https://github.com/therealram18/vernier && cd vernier
uv sync && uv run pytest
```

The tests recompute each number of the paper from the snapshots in `src/vernier/pins/`. [docs/paper.md](https://github.com/therealram18/vernier/blob/main/docs/paper.md) gives the command for each row of the paper's tables and says what has been checked.

## Citation

```bibtex
@inproceedings{sikder2026what,
  title     = {What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models},
  author    = {Arya Sikder and Sriram Pillutla},
  booktitle = {NeurIPS 2026 Workshop: Foundation Models for Temporal Systems -- From Forecasting to World Modeling},
  year      = {2026},
  eprint    = {2611.31619},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2611.31619}
}
```

Apache License 2.0. The tables in `src/vernier/pins/` are derived from the leaderboards' own repositories.
