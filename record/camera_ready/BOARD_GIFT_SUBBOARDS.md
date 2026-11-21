# GIFT-Eval sub-boards: audit of the tabs the leaderboard itself displays

Answers the reviewers' question on how the audit behaves with different dataset compositions and numbers of evaluation units. Script: `audit_gift_subboards.py`; outputs `boards/gift_subboards_groups.json`, `boards/gift_subboards_report.json`. Same shared code as the full-board audit (`vernier_board.report_from_matrix`), seed 20261111, known-shift control with 300 draws per pair.

## What the app displays (snapshot_2026-09-07)

Tabs, `app.py`:
- `app.py:187` `with gr.TabItem('🏅 Overall', ...)` and `:189` `leaderboard = init_leaderboard(overall_df, model_info_df, sort_val=['Test Leak.', 'MASE_Rank'])`
- `app.py:191-192` By Domain: `leaderboard = init_leaderboard(domain_df, model_info_df)`; `:195-196` By Frequency (`freq_df`); `:199-200` By Term Length (`term_length_df`); `:203-204` By Variate Type (`variate_type_df`). None passes `sort_val`.
- `:206` About is the last tab. There is no tab by dataset or source.

How configs are assigned to groups, `src/utils.py`:
- `:122` `all_results_df[['dataset', 'frequency', 'term_length']] = all_results_df['dataset'].str.split('/', expand=True)` : the config name `dataset/frequency/term` gives frequency and term length directly.
- `:124-135` and `:197-207`: the properties file `results/dataset_properties.csv` (columns dataset, domain, frequency, num_variates) is matched on the normalised dataset name and writes `domain` and `num_variates` onto every config of that dataset (frequency only if empty, which it never is here).
- `:62-83` `unify_freq`: strips digits, drops text after `-`, maps `T`->Minutely, `H`->Hourly, `D`->Daily, `W`->Weekly, `M`->Monthly, `Q`->Quarterly, `Y` and `A`->Yearly, `S`->Secondly. So 5T, 10T, 15T all become Minutely and 10S becomes Secondly.
- `:241` `df['univariate'] = df['num_variates'] == 1`; `:87-89` relabels True/False as `univariate`/`multivariate`.
- `:266` `for col_name in ["domain", 'term_length', 'frequency', 'univariate']: grouped_dfs[col_name] = group_by(df, col_name)`.

What each group tab shows and sorts by:
- `:329-331` `group_by`: `grouped_results = df.groupby([col_name, 'model'])[METRIC_CHOICES].agg(stats.gmean)` plus `grouped_results_rank = df.groupby([col_name, 'model'])[['rank']].mean()`. So per group each model gets: geometric mean of seasonal-naive-scaled MASE, geometric mean of scaled CRPS, and mean CRPS rank (not MASE rank). The scaling is `standardize_df` (`:273-325`, division by the Seasonal_Naive row of the same config).
- `app.py:192, 196, 200, 204`: the tab is built with no sort, so the rows come out of `pivot_table(index='model', ...)` (`utils.py:98`) in alphabetical model order; the groups sit side by side as columns. The user can sort by clicking a column. There is no default ranking in any sub-board tab.
- The per-config rank used by the app (`utils.py:218-222`) is `rank(method='first')` within (dataset, term_length, frequency).

So the statistic the tabs show is the geometric mean of scaled MASE (and CRPS, and CRPS mean rank). Mean MASE rank per group is not shown on any tab; it is what the task asked for and what the full board sorts by, so it is the primary analysis here. The geometric-mean statistic is analysed as well (log of scaled MASE, mean over the group's configs; ordering by that mean, leak-flagged entries pushed down). CRPS columns are not analysed.

## Groups (printed and saved before any result)

97 configs, 28 sources (base datasets). A group is analysed if it has at least 6 sources.

| group | configs | sources | analysed |
|---|---|---|---|
| term_length=long | 21 | 14 | yes |
| term_length=medium | 21 | 14 | yes |
| term_length=short | 55 | 28 | yes |
| domain=Econ/Fin | 6 | 6 | yes |
| domain=Energy | 32 | 4 | no |
| domain=Healthcare | 5 | 3 | no |
| domain=Nature | 15 | 4 | no |
| domain=Sales | 4 | 3 | no |
| domain=Transport | 15 | 3 | no |
| domain=Web/CloudOps | 20 | 5 | no |
| frequency=Daily | 15 | 15 | yes |
| frequency=Hourly | 31 | 13 | yes |
| frequency=Minutely | 30 | 10 | yes |
| frequency=Monthly | 5 | 5 | no |
| frequency=Quarterly | 1 | 1 | no |
| frequency=Secondly | 6 | 2 | no |
| frequency=Weekly | 8 | 8 | yes |
| frequency=Yearly | 1 | 1 | no |
| univariate=multivariate | 43 | 8 | yes |
| univariate=univariate | 54 | 20 | yes |

Not analysed (fewer than 6 sources): domains Energy (32 configs, 4 sources), Healthcare (5, 3), Nature (15, 4), Sales (4, 3), Transport (15, 3), Web/CloudOps (20, 5); frequencies Monthly (5, 5), Quarterly (1, 1), Secondly (6, 2), Yearly (1, 1). Reason: the cluster-based tests treat each source as one independent unit; with 5 or fewer sources the corrected tests have 1-4 degrees of freedom or a wild bootstrap with at most 32 distinct sign patterns, so the methods cannot give a usable interval. Domain Econ/Fin has exactly 6 sources with one config each for 5 of them, so it is at the edge and its numbers should be read with that in mind. Note that sub-boards overlap: a config is in one group per tab, and the same 28 sources appear across tabs, so rows are not independent of each other.

## Summary, statistic = mean MASE rank (as on the full board)

| group | configs | sources | resolution paper / corrected | median adjacent gap | ratio paper / corrected | adjacent separated of 9 (paper / CR2-icc / wild) | all 45 separated (paper / CR2-icc / wild) | all 45 after Holm (CR2-icc) | tier size (paper / CR2-icc) | corrected res x sqrt(sources) | top-10 overlap with full | leader |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | 97 | 28 | 6.209 / 6.801 | 0.9072 | 6.8 / 7.5 | 0 / 1 / 1 | 12 / 12 / 13 | 0 | 3 / 3 | 36 | 10/10 | STRIDE w/ Synapse |
| term_length=long | 21 | 14 | 11.82 / 13.44 | 0.6667 | 17.7 / 20.2 | 0 / 0 / 0 | 10 / 6 / 10 | 0 | 3 / 6 | 50.3 | 9/10 | STRIDE w/ Synapse |
| term_length=medium | 21 | 14 | 9.689 / 11.02 | 1.238 | 7.8 / 8.9 | 0 / 0 / 0 | 10 / 12 / 12 | 4 | 2 / 2 | 41.2 | 9/10 | STRIDE w/ Synapse |
| term_length=short | 55 | 28 | 7.784 / 8.381 | 1.218 | 6.4 / 6.9 | 0 / 0 / 0 | 5 / 6 / 6 | 0 | 10 / 11 | 44.4 | 9/10 | STRIDE w/ Synapse |
| domain=Econ/Fin | 6 | 6 | 15.5 / 21.25 | 0.5 | 31.0 / 42.5 | 0 / 0 / 0 | 11 / 4 / 2 | 0 | 6 / 12 | 52 | 7/10 | CastStar |
| frequency=Daily | 15 | 15 | 15.21 / 17.03 | 1.667 | 9.1 / 10.2 | 0 / 0 / 0 | 10 / 7 / 7 | 3 | 4 / 4 | 66 | 7/10 | EXAONE-Forecast-Agent |
| frequency=Hourly | 31 | 13 | 12.53 / 14.67 | 1.452 | 8.6 / 10.1 | 0 / 0 / 0 | 8 / 5 / 7 | 1 | 5 / 10 | 52.9 | 8/10 | STRIDE w/ Synapse |
| frequency=Minutely | 30 | 10 | 9.038 / 10.7 | 0.4 | 22.6 / 26.7 | 0 / 0 / 0 | 5 / 1 / 2 | 0 | 6 / 10 | 33.8 | 8/10 | STRIDE w/ Synapse |
| frequency=Weekly | 8 | 8 | 29.62 / 36.66 | 1.625 | 18.2 / 22.6 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 16 / 20 | 104 | 6/10 | LS-MoE |
| univariate=multivariate | 43 | 8 | 6.147 / 7.951 | 0.2093 | 29.4 / 38.0 | 0 / 0 / 0 | 2 / 1 / 1 | 0 | 13 / 16 | 22.5 | 8/10 | STRIDE w/ Synapse |
| univariate=univariate | 54 | 20 | 9.947 / 11.78 | 0.6481 | 15.3 / 18.2 | 0 / 0 / 0 | 16 / 10 / 13 | 1 | 2 / 3 | 52.7 | 9/10 | STRIDE w/ Synapse |

Full-board row reproduces the existing full-board report (`gift_mase_rank_report.json`, resolution 6.209 / 6.801). Tier size is capped at 40 (the code compares the leader with the next 39 entries); 40 means nobody was separated.

Known-shift control (detection rate when one entry is shifted by the board's own claimed resolution; target about 0.8; the last number is the false-alarm rate at shift 0, target 0.05), at the paper resolution / corrected resolution:

| group | paper rule | CR2-icc | wild |
|---|---|---|---|
| FULL | 0.75 / 0.79 / 0.05 | 0.71 / 0.76 / 0.05 | 0.71 / 0.76 / 0.05 |
| term_length=long | 0.81 / 0.87 / 0.09 | 0.73 / 0.81 / 0.07 | 0.74 / 0.81 / 0.08 |
| term_length=medium | 0.79 / 0.88 / 0.04 | 0.78 / 0.87 / 0.05 | 0.79 / 0.87 / 0.05 |
| term_length=short | 0.78 / 0.82 / 0.04 | 0.77 / 0.81 / 0.05 | 0.77 / 0.81 / 0.06 |
| domain=Econ/Fin | 0.89 / 0.98 / 0.16 | 0.69 / 0.86 / 0.08 | 0.68 / 0.79 / 0.07 |
| frequency=Daily | 0.80 / 0.84 / 0.07 | 0.75 / 0.80 / 0.05 | 0.75 / 0.81 / 0.05 |
| frequency=Hourly | 0.80 / 0.86 / 0.11 | 0.72 / 0.79 / 0.10 | 0.75 / 0.81 / 0.10 |
| frequency=Minutely | 0.78 / 0.88 / 0.09 | 0.68 / 0.81 / 0.07 | 0.68 / 0.80 / 0.07 |
| frequency=Weekly | 0.84 / 0.96 / 0.11 | 0.69 / 0.87 / 0.07 | 0.67 / 0.83 / 0.07 |
| univariate=multivariate | 0.63 / 0.80 / 0.06 | 0.57 / 0.72 / 0.07 | 0.55 / 0.68 / 0.07 |
| univariate=univariate | 0.80 / 0.89 / 0.06 | 0.73 / 0.84 / 0.06 | 0.75 / 0.85 / 0.09 |

## How each column changes with the number of sources (MASE rank)

Across the 10 sub-boards the sources range from 6 to 28 and the configs from 6 to 55. A fit of log corrected resolution on log sources gives slope -0.48 (correlation -0.48); on log configs the slope is -0.51 (correlation -0.85). So the numbers are consistent with 1/sqrt(n) but the count that tracks it better is the number of configs, not the number of sources, and the scatter is large.
- Resolution (paper and corrected): falls as groups get larger (short, 28 sources: 7.8 / 8.4; full board 6.2 / 6.8; 6-8 sources 15-37, except multivariate 6.1/8.0 with 43 configs). The product corrected resolution x sqrt(sources) is not constant: 22 to 104, median about 50, against 36 for the full board. Groups with few configs per source (Weekly 8 configs, Econ/Fin 6) have the largest products, and multivariate (43 configs from 8 sources) the smallest, again pointing to configs rather than sources.
- Median adjacent gap: no trend with sources; it is 0.2 to 1.7 rank points, set by how closely the leaders are bunched in that group (multivariate 0.21, Minutely 0.40, Daily 1.67).
- Ratio resolution / gap: 6.4 to 31 (paper) and 6.9 to 42.5 (corrected), always well above 1 and generally bigger in smaller sub-boards, because resolution grows and gaps do not. Full board 6.8 / 7.5.
- Adjacent separated: 0 of 9 on every sub-board under the paper rule; CR2-icc and wild also 0 everywhere (the full board has 1 under CR2-icc and wild). Nothing about this improves or worsens with the sub-board size.
- All 45 separated: 0 to 16 under the paper rule, 0 to 12 under CR2-icc; the larger the sub-board, the more pairs separate in general (short, 28 sources: 5 / 6; medium, 14 sources: 10 / 12), but not monotonically (long 10 / 6, univariate 16 / 10). The paper rule separates more pairs than CR2-icc on 7 of 10 sub-boards (equal on Weekly, lower on medium and short).
- All 45 after Holm (CR2-icc): 0 on every sub-board except medium (4), Daily (3), Hourly (1), univariate (1). The full board: 0.
- Tier size: 2 to 16 under the paper rule, with no trend; short (10), Weekly (16), multivariate (13) have big tiers; medium and univariate have 2.
- Top ten: see the overlap column; the leader is STRIDE w/ Synapse on 7 of 10 sub-boards; CastStar leads Econ/Fin, EXAONE-Forecast-Agent leads Daily, LS-MoE leads Weekly.

## Same audit on the statistic the tabs show (geometric mean of seasonal-naive-scaled MASE)

Units here are log ratios, so resolutions and gaps are not comparable with the rank table, only the ratio is.

| group | configs | sources | resolution paper / corrected | median adjacent gap | ratio paper / corrected | adjacent separated of 9 (paper / CR2-icc / wild) | all 45 separated (paper / CR2-icc / wild) | all 45 after Holm (CR2-icc) | tier size (paper / CR2-icc) | corrected res x sqrt(sources) | top-10 overlap with full | leader |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | 97 | 28 | 0.04429 / 0.04966 | 0.004471 | 9.9 / 11.1 | 1 / 1 / 1 | 15 / 9 / 11 | 0 | 9 / 17 | 0.263 | 10/10 | EXAONE-Forecast-Agent |
| term_length=long | 21 | 14 | 0.06763 / 0.07566 | 0.007026 | 9.6 / 10.8 | 0 / 0 / 0 | 5 / 2 / 4 | 0 | 9 / 12 | 0.283 | 7/10 | EXAONE-Forecast-Agent |
| term_length=medium | 21 | 14 | 0.0765 / 0.0861 | 0.006057 | 12.6 / 14.2 | 0 / 0 / 0 | 9 / 4 / 5 | 0 | 8 / 19 | 0.322 | 7/10 | EXAONE-Forecast-Agent |
| term_length=short | 55 | 28 | 0.03543 / 0.03883 | 0.003037 | 11.7 / 12.8 | 1 / 0 / 1 | 6 / 1 / 6 | 0 | 40 / 40 | 0.205 | 8/10 | TurkForecast-FM-Chronos2-LoRA-v1 |
| domain=Econ/Fin | 6 | 6 | 0.01135 / 0.01544 | 0.001526 | 7.4 / 10.1 | 0 / 0 / 0 | 12 / 0 / 2 | 0 | 10 / 20 | 0.0378 | 5/10 | Falcon-Agent |
| frequency=Daily | 15 | 15 | 0.157 / 0.1744 | 0.005947 | 26.4 / 29.3 | 0 / 0 / 0 | 11 / 7 / 7 | 0 | 4 / 4 | 0.676 | 4/10 | EXAONE-Forecast-Agent |
| frequency=Hourly | 31 | 13 | 0.04903 / 0.05703 | 0.00572 | 8.6 / 10.0 | 0 / 0 / 0 | 13 / 2 / 5 | 0 | 8 / 23 | 0.206 | 8/10 | EXAONE-Forecast-Agent |
| frequency=Minutely | 30 | 10 | 0.1039 / 0.123 | 0.001181 | 88.0 / 104.1 | 0 / 0 / 0 | 2 / 0 / 0 | 0 | 20 / 35 | 0.389 | 8/10 | EXAONE-Forecast-Agent |
| frequency=Weekly | 8 | 8 | 0.1094 / 0.1365 | 0.00545 | 20.1 / 25.0 | 0 / 0 / 0 | 1 / 0 / 0 | 0 | 40 / 40 | 0.386 | 6/10 | TurkForecast-FM-Chronos2-LoRA-v1 |
| univariate=multivariate | 43 | 8 | 0.02077 / 0.02649 | 0.001783 | 11.7 / 14.9 | 0 / 0 / 0 | 4 / 0 / 1 | 0 | 28 / 40 | 0.0749 | 7/10 | EXAONE-Forecast-Agent |
| univariate=univariate | 54 | 20 | 0.06083 / 0.07768 | 0.006667 | 9.1 / 11.7 | 2 / 1 / 2 | 11 / 6 / 7 | 0 | 4 / 22 | 0.347 | 9/10 | STRIDE w/ Synapse |

Known-shift control (paper / corrected resolution / zero shift):

| group | paper rule | CR2-icc | wild |
|---|---|---|---|
| FULL | 0.69 / 0.71 / 0.11 | 0.65 / 0.68 / 0.10 | 0.66 / 0.68 / 0.10 |
| term_length=long | 0.68 / 0.73 / 0.12 | 0.68 / 0.72 / 0.09 | 0.68 / 0.72 / 0.10 |
| term_length=medium | 0.69 / 0.73 / 0.09 | 0.66 / 0.69 / 0.07 | 0.67 / 0.70 / 0.07 |
| term_length=short | 0.66 / 0.69 / 0.08 | 0.66 / 0.68 / 0.08 | 0.66 / 0.67 / 0.08 |
| domain=Econ/Fin | 0.71 / 0.79 / 0.18 | 0.54 / 0.70 / 0.08 | 0.45 / 0.60 / 0.06 |
| frequency=Daily | 0.69 / 0.73 / 0.08 | 0.65 / 0.69 / 0.05 | 0.66 / 0.69 / 0.05 |
| frequency=Hourly | 0.63 / 0.66 / 0.12 | 0.53 / 0.59 / 0.08 | 0.56 / 0.61 / 0.09 |
| frequency=Minutely | 0.72 / 0.76 / 0.13 | 0.67 / 0.70 / 0.10 | 0.68 / 0.70 / 0.11 |
| frequency=Weekly | 0.75 / 0.82 / 0.19 | 0.62 / 0.74 / 0.12 | 0.67 / 0.73 / 0.13 |
| univariate=multivariate | 0.63 / 0.73 / 0.10 | 0.51 / 0.63 / 0.06 | 0.51 / 0.63 / 0.07 |
| univariate=univariate | 0.68 / 0.77 / 0.16 | 0.60 / 0.70 / 0.13 | 0.61 / 0.71 / 0.13 |

Scaling with sources on the geometric-mean statistic: log corrected resolution on log sources has slope +0.46 (correlation +0.28); on configs the slope is 0.0. There is no 1/sqrt(sources) relation here; the resolution is set by how spread out the per-source log ratios are in that group (Daily, Minutely and Weekly are spread widely, Econ/Fin and multivariate narrowly). The product res x sqrt(sources) is 0.04 to 0.68. The ratio resolution / gap is 7.4 to 88 (paper) and 10 to 104 (corrected). Adjacent pairs separated: 1 on the full board (all three methods); on sub-boards at most 2 (paper, univariate) and 1 (CR2-icc, univariate). Tiers are 4 to 40 under both rules (40 means nobody was separated from the leader); full board 9 under the paper rule and 17 under CR2-icc; CR2-icc tiers are larger than paper tiers on 7 of 10 sub-boards.

## Does the top ten differ between sub-boards (MASE rank)?

Overlap of the sub-board's top ten with the full board's displayed top ten, and the sub-board leader: see the `top-10 overlap` and `leader` columns of the first table (full list of names in the report JSON under `top10`). Overlap is 6 to 9 of 10 (Weekly 6; Econ/Fin 7; Daily 7; the rest 8-9). Leader differs from the full board's STRIDE w/ Synapse on Econ/Fin (CastStar), Daily (EXAONE-Forecast-Agent) and Weekly (LS-MoE). On the geometric-mean statistic the full-board leader is EXAONE-Forecast-Agent, overlap is 4 to 9 (Daily 4, Econ/Fin 5), and the leader is TurkForecast-FM-Chronos2-LoRA-v1 on short and Weekly, Falcon-Agent on Econ/Fin.

## Deviations, failures, things that surprised

- The per-config ranks use scipy `average` ties as in `test_vernier_board.py` (the published full-board numbers); the app itself ranks with `method='first'` (`utils.py:218-222`), and only its CRPS rank is a displayed column. The tab columns are the geometric means and the CRPS mean rank; no tab shows MASE mean rank. The sub-board "displayed order" is by the statistic as instructed; the real tabs have no default sort (alphabetical by model name).
- Eligible entries are those on the full board's displayable list (all 97 configs present and a config.json); this keeps the same entry set in every sub-board. The app's own fill of missing values with column means (`utils.py:281-283`) is not applied.
- Duplicate rows in `dataset_properties.csv` (one per frequency) are collapsed to one entry per dataset by the app's dictionary; domain and number of variates are the same on every row of a dataset, so this changes nothing (pandas warns "columns are not unique", harmless here).
- Sources are base datasets (the part of the config name before the first `/`), as in the full-board audit; not the coarser 19-source grouping.
- Corrected resolutions here are from CR2 with the intraclass-correlation degrees of freedom; with 6-8 sources the interval is very wide, and these rows are included only for completeness.
- Nothing failed. The full run took 6 min 10 s wall clock in one process (22 board analyses, known-shift control 300 per pair).
