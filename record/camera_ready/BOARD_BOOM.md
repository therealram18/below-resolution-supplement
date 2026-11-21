# BOOM leaderboard audit

Plan: `boards/PREREG_BOARDS.md` (2026-11-11). Code: `audit_boom.py`, which calls the shared `vernier_board.report_from_matrix` and `vernier_board.pool_tests` unchanged. All numbers: `boards/boom_report.json` (parts in `boards/boom_parts/`). BOOM files: DataDog/toto commit 1d3e1c9, 23 models.

## 1. Reproduction of the published table (done first)

The board's own functions (`boom/utils/leaderboard.py`: `load_model_results`, `separate_zero_inflated_data`, `process_benchmark_model_results`, `get_separate_zero_inflated_leaderboard`) were imported and run on the 23 result files.

- The split matches: 6,750 rows scaled by seasonal naive, 663 rows kept unscaled (253 series: 80 on the board's hard-coded low-variance list and 179 where seasonal naive's MASE is exactly 0; 6 are in both).
- The board's own output matches `boom/leaderboards/boom_leaderboard.csv` exactly, number for number, in all six columns, and gives the same row order.
- The 1e-6 relative tolerance in the plan cannot be applied: the board rounds every published number to three decimals (`format_number`), so the CSV holds only rounded values. I used the best test available instead: my unrounded values (shifted geometric means, and the mean of per-row ranks on the 6,750 and on the 663 rows) all lie within half a rounding unit (0.0005) of the published value. Largest differences: Rank-6750 0.000481, Rank-663 0.000483, MASE-6750 0.000468. So the means of ranks and the geometric means are reproduced to the published precision, not to 1e-6.
- Displayed order: the unrounded Rank-6750 ordering is identical to the published order for all 23 models, and the top ten are identical.
- Which column the table is sorted by: `boards/boom/app.py` line 61-62: `if "Rank" in merged_df.columns: merged_df = merged_df.sort_values(by=["Rank"], ascending=True)`. The `Rank` column of the space's `BOOM_leaderboard.csv` is the same numbers as `Rank-6750-scaled` (the 6,750-row set; the 663-row ranks are not shown there). `boards/boom/src/about.py` line 42 says Rank "determines overall performance and is used to order models on the leaderboard". The sort is on the rounded three-decimal values; no two models tie at that precision.
- Tie handling matters and I followed the board: `rank(method="first")` gives the earlier model in the concatenation order the better rank, and that order is `os.listdir` of the results folder (seasonalnaive moved last). In the scaled set, 2,286 of 6,750 rows have at least two models with identical quantile loss. Reordering the models alphabetically changes some mean ranks by up to 0.092 but leaves the displayed order of all 23 unchanged. Matching the published numbers to 0.0005 with the order this machine returns also shows the published table was made with the same order.

## 2. Units and clusters

Unit: one (series, horizon) row of the displayed 6,750-row scaled set. These rows belong to 2,554 series; 2,098 series appear at three horizons and 456 at one. All 2,554 series have an entry in `boom_properties.json` (the file has 2,807 keys; the other 253 are the series that went to the 663-row unscaled set). Keys in the file are the part of `dataset` before the first "/", for example `ds-860-T`. No series is missing properties.

| Cluster definition | Clusters | Rows per cluster: min / median / max |
|---|---|---|
| (a) series id (primary) | 2554 | 1 / 3 / 3 (456 clusters of 1 row, 2,098 of 3 rows) |
| (b) domain + type | 36 | 3 / 45.5 / 1840 |
| (c) domain alone (several-domain series keep the full combination as key) | 23 | 4 / 38 / 2988 |

Cluster sizes under (b) and (c) are very uneven: under (b) the largest cluster (Application Usage + rate) holds 1,840 of 6,750 rows; under (c) the single domain "Application Usage" holds 2,988 rows (44%), "Infrastructure" 1,593, "Database" 759. Under (c) 18 of the 23 clusters are multi-domain combinations (with their type) of 4 to 364 rows each; only five are single domains.

## 3. Results

Matrix: for each row, the rank of each of the 23 models on `mean_weighted_sum_quantile_loss` among the 23, ties first-come as the board does; the mean of a model's row is the displayed Rank. Differences between neighbours are in rank points. Median gap between displayed neighbours (top ten): 0.312.

Columns "paper / CR2-icc / wild" are, in order, the submitted paper's rule (both percentile bootstrap intervals, over rows and over clusters, exclude zero), the cluster-robust t-test with the pre-registered primary degrees-of-freedom method, and the wild cluster bootstrap. Resolution = smallest true gap detected 80% of the time (paper: 2.80 x bootstrap standard error; corrected: from the CR2 standard error and degrees of freedom).

| Clusters | n | rows per cluster min / median / max | Resolution paper / corrected | Median adjacent gap | Ratio resolution / gap, paper / corrected | Adjacent separated of 9: paper / CR2-icc / wild | All 45 after Holm: paper / CR2-icc / wild | Tier size (paper / CR2-icc / wild) | Known-shift detection at the resolution: paper / CR2-icc / wild |
|---|---|---|---|---|---|---|---|---|---|
| (a) series id | 2554 | 1 / 3 / 3 | 0.195 / 0.196 | 0.312 | 0.63 / 0.63 | 9 / 9 / 9 | 45 / 45 / 45 | 1 / 1 / 1 | 82% / 82% / 82% |
| (b) domain+type | 36 | 3 / 45.5 / 1840 | 0.267 / 0.420 | 0.312 | 0.86 / 1.35 | 7 / 6 / 6 | 42 / 36 / 40 | 1 / 1 / 1 | 65% / 63% / 79% |
| (c) domain | 23 | 4 / 38 / 2988 | 0.265 / 0.635 | 0.312 | 0.85 / 2.04 | 7 / 5 / 6 | 40 / 9 / 38 | 1 / 1 / 1 | 70% / 75% / 92% |

Reading the last column: a shift equal to the stated resolution was added to each of the 9 neighbour pairs; the share of resamples (clusters redrawn with replacement) in which the rule detects it. About 80% is the target. Paper rule is tested at the paper resolution, the other two at the corrected resolution.

More detail (adjacent pairs only for the degrees of freedom):

| Clusters | CR2-icc degrees of freedom min / median / max | Estimated within-cluster correlation of the neighbour differences | Sources needed (bootstrap / CR2) | All 45 before Holm: paper / CR2-icc / wild | Adjacent separated, dataset-means t-test |
|---|---|---|---|---|---|
| (a) | 2245.0 / 2264.2 / 2293.9 | 0.19 to 0.45 | 584 / 586 | 45 / 45 / 45 | 7 |
| (b) | 2.3 / 2.4 / 3.2 | 0.00 to 0.03 | 12 / 9 | 43 / 42 / 42 | 4 |
| (c) | 1.5 / 1.6 / 2.2 | 0.00 to 0.01 | 13 / 6 | 43 / 40 / 42 | 5 |

Known-shift control in full (detection at the paper resolution, at the corrected resolution, and false alarms at zero shift, for each of the three rules; the target false-alarm rate is 5%):

| Clusters | paper @paper res | paper @corrected res | paper zero shift | CR2-icc @paper res | CR2-icc @corrected res | CR2-icc zero shift | wild @paper res | wild @corrected res | wild zero shift | resamples per pair |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) | 82% | 82% | 5% | 82% | 82% | 4% | 81% | 82% | 4% | 80 |
| (b) | 65% | 86% | 9% | 50% | 63% | 5% | 55% | 79% | 8% | 200 |
| (c) | 70% | 97% | 9% | 45% | 75% | 6% | 57% | 92% | 6% | 200 |

Reading the control: under (a) every rule detects the shift at about 82% (target 80%) and false-alarms 4 to 5% (target 5%), so the stated resolution is accurate there. Under (b) and (c) it is not: the paper rule detects its own resolution only 65% (b) and 70% (c) of the time and false-alarms 9% at zero shift; CR2-icc detects at its corrected resolution 63% (b) and 75% (c); the wild bootstrap 79% (b) and 92% (c) with 8% and 6% false alarms. With 23 to 36 groups and a few very large ones, none of the three rules hits both targets. Resamples per pair are 200 (b, c) or 80 (a), so rates carry about 1 to 2 points of sampling error.

The leading tier is one model, `toto_2.0_2.5b`, under every rule and every cluster definition: it is separated from each of the other 39 models ranked 2 to 40 by all three rules (as the definition of the tier is used here: models not separated from the leader).

Which neighbours fail. Under (a) all 9 pairs are separated by every rule; the thinnest is 1 vs 2 (gap 0.088, CR2 p = 0.023). Under (b) and (c) the pairs that fail under the paper rule are 5 vs 6 (`toto` vs `toto_2.0_4m`, gap 0.234) and 6 vs 7 (`toto_2.0_4m` vs `chronos_2`, gap 0.217); the CR2-icc rule additionally loses 7 vs 8 (`chronos_2` vs `timesfm_2_5_200m`, gap 0.312) and, under (c), 9 vs 10 (`moirai_1.1_base` vs `moirai_1.1_large`); the wild bootstrap loses 7 vs 8 under (b) and 9 vs 10 under (c).

Secondary rule: ranking on MASE instead of the quantile loss gives a different top ten (the MASE-rank order puts `timesfm_2_0_500m` in the top ten in place of `moirai_1.1_large`). Run on the same 10 models and displayed order as the primary, with the MASE-rank differences, all 9 adjacent pairs and all 45 pairs are separated at series level under the paper rule (see `mase_rank_displayed_by_crps_order` in the JSON). The MASE-ordered top ten was not itself analysed.

## 4. What the three cluster levels mean

If every series is treated as independent evidence (a), the board separates every neighbour in the top ten and the leader from everyone: resolution 0.195 against a median gap of 0.312, and the corrected method gives the same answer because with 2,554 clusters the two methods differ by under 1%. This is the most favourable reading, and the one the board's own averaging assumes.

If series from the same domain and type move together (b, 36 groups) or series from the same domain do (c, 23 groups), the resolution rises to 0.267 (paper rule) or 0.420 (CR2) under (b) and 0.265 or 0.635 under (c), larger than the median gap between neighbours in the CR2 columns, and the number of separated neighbours falls from 9 to 6-7 (b) and 5-7 (c). The strictest rule at 23 groups (CR2-icc with Holm correction over 45 pairs) separates 9 of 45 pairs, against 45 of 45 at series level. So the claim that most of the order within the top ten is established rests on treating the 2,554 series as independent: neighbour pairs 1-2, 2-3, 3-4 and 8-9 are separated under every grouping and every rule when no multiple-comparison correction is applied (after Holm correction over 45 pairs at 23 groups, CR2-icc keeps none of the 9 neighbour pairs), but the order among the middle neighbours (`toto`, `toto_2.0_4m`, `chronos_2`, `timesfm_2_5_200m`) is not separated under (b) or (c) by any rule. These groupings are my reading of what could be shared; the files carry no field saying that same-domain series are correlated, and the estimated within-cluster correlation of the neighbour differences is 0.19 to 0.45 within a series (a) but only 0 to 0.03 within a domain group, so (b) and (c) are the pessimistic end. Even 0.02 is not negligible with groups of hundreds to thousands of rows (the variance is inflated by about 1 + (rows in group - 1) x correlation). The very small degrees of freedom (about 2.4 under (b), 1.6 under (c)) come from the uneven cluster sizes: one cluster holds 27% (b) to 44% (c) of the rows, so a handful of clusters carry most of the information.

## 5. Dependence on the pool of models

Using the board's own rule on the 6,750 scaled rows (mean rank, first-come ties in the board's model order), the top ten is toto_2.0_2.5b, toto_2.0_1b, toto_2.0_313m, toto_2.0_22m, toto, toto_2.0_4m, chronos_2, timesfm_2_5_200m, moirai_1.1_base, moirai_1.1_large.

- Removing any one of the 13 models outside the top ten: reorders the top ten in 0 of 13 cases.
- Adding a near-copy (1e-6 relative perturbation) of any of the 23 models: reorders the top ten in 6 of 23 cases (toto_2.0_4m, moirai_1.1_large, chronos_2, toto_2.0_313m, timesfm_2_5_200m, toto_2.0_1b).
- History replay: not meaningful, since only 4 commits touch the results, and the last two rewrite five existing files rather than add models:

| Date | Commit | Model files |
|---|---|---|
| 2026-05-12 | b3f3ffb | 0 added, 5 modified |
| 2026-05-11 | f67fbc3 | 0 added, 5 modified |
| 2026-04-23 | d0c2dc0 | 8 added, 0 modified |
| 2025-05-28 | fd75d06 | 15 added, 0 modified |

## 6. Overlap with GIFT-Eval

- Datasets: the BOOM dataset card (huggingface.co/datasets/Datadog/BOOM) says the data come from an internal Datadog deployment monitoring pre-production systems, 2,807 series, and names no public dataset. `boom_properties.json` holds only anonymous ids (`ds-<number>-<frequency>`) with domain, type, term, frequency and number of variates, so there is no name that could match any of GIFT-Eval's 28 dataset names (bitbrains, bizitobs, car_parts, covid_deaths, electricity, ett, hierarchical_sales, hospital, jena_weather, kdd_cup_2018, loop_seattle, m4, m_dense, restaurant, saugeen, solar, sz_taxi, temperature_rain, us_births). No overlap found; the card makes no explicit statement that there is none, and the files cannot rule out a hidden match.
- Models (BOOM name -> GIFT-Eval board name, matched by name only): exact: chronos_bolt_base, chronos_bolt_small, timesfm_2_0_500m, visionts. Same name with different case or separators: toto_2.0_1b / 2.5b / 22m / 313m / 4m -> Toto-2.0-1B / 2.5B / 22m / 313m / 4m; chronos_2 -> chronos-2; timesfm_2_5_200m -> TimesFM-2.5; autoarima, autoets, autotheta -> auto_arima, auto_ets, auto_theta; seasonalnaive -> seasonal_naive. Approximate, version not confirmed: toto -> Toto_Open_Base_1.0; moirai_1.1_small / base / large -> Moirai_small / base / large; moirai_2_small -> Moirai2. No match: time_moe_50M, time_moe_200M, timer (GIFT-Eval has Timer-s1, a different model). That is 20 of the 23 BOOM models with a GIFT-Eval row by exact or case/separator match (15 plus 5 approximate), 3 without.

## 7. Deviations and failures

- Reproduction tolerance: 1e-6 relative cannot be tested against a table rounded to three decimals; replaced by "within half a rounding unit" (all 6 columns pass) plus exact equality of the board function's own output with the published file.
- Bootstrap sizes: under (b) and (c) the full settings were used (20,000 bootstrap draws, 99,999 wild draws). Under (a) (2,554 clusters, about 0.6 s per CR2 degrees-of-freedom calculation) I used B_boot = 5,000 and B_wild = 19,999 to fit the time limit. Known-shift control: 200 resamples per adjacent pair under (b) and (c), 80 under (a), instead of 1,000. It was run as one process per pair by calling `vernier_board._known_shift`, the function `report_from_matrix` itself uses, with the resolution values from the same report; this is not a reimplementation but it is not the single call `report_from_matrix(n_control=...)` either. The detection rates are averages over the 9 pairs.
- The MASE-rank secondary analysis was run on the primary (quantile-loss) top ten and displayed order only, and only under definition (a).
- Wild bootstrap is exact only up to 16 clusters; under (c) with 23 clusters it is a random-draw bootstrap.
- The board's model order for tie-breaking is the operating system's folder listing order; it reproduced the published numbers on this machine and the displayed order does not depend on it, but individual mean ranks do (up to 0.092).
- Background jobs were started in the main session's shell; none are left running.
