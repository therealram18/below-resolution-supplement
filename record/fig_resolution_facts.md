# Figure 1 (`fig_resolution.pdf`): what is plotted and where each number comes from

Written by `figs/make_fig_resolution.py` in the same run that drew the figure, so the numbers below are the plotted numbers.

Command (one line, clean shell; the throwaway environment needs matplotlib, numpy, seaborn and scipy):

```
uv run --no-project --with matplotlib --with numpy --with seaborn --with scipy python fmts/camera_ready/paper/figs/make_fig_resolution.py
```

## What each panel plots

Both panels: for the nine neighbouring pairs of the displayed top ten (places 1-2 to 9-10), the marker is the gap between the two entries (positive means the higher-placed entry is ahead) and the whisker is its 95% interval from the CR2 method: a t interval, gap plus or minus t(0.975, df) times the cluster-robust CR2 standard error, with df from Satterthwaite's approximation. A filled marker means the interval excludes zero; a hollow marker means it does not. The orange line is the board's corrected resolution: for each pair, (t(0.975, df) + t(0.80, df)) times its CR2 standard error, the smallest true gap detected eight times in ten at the 5% level; the board's value is the median over the nine pairs.

Panel (a), GIFT-Eval, mean rank on MASE, snapshot of 2026-09-07. Gaps in rank places. Resolution 6.80 rank places; median neighbour gap 0.91.

Panel (b), fev-bench (100 tasks, 39 task families, from `reports.A_wr.n_units` and `n_sources`), win rate, the board as pinned on 2026-11-11 (autogluon/fev commit 9cf943c, 40 entries; unchanged at the 2026-11-13 re-pin). Gaps in win-rate points (0 to 100). Resolution 7.51 win-rate points; median neighbour gap 2.08.

## Counts

| Panel | Intervals that exclude zero (of 9) | Gaps that reach the resolution line (of 9) | Largest gap | Resolution |
|---|---|---|---|---|
| (a) GIFT-Eval | 1 (9-10) | 0 | 4.25 | 6.80 |
| (b) fev-bench | 0 (none) | 0 | 3.13 | 7.51 |

The zero counts are uncorrected (each pair tested at 5% on its own). After the Holm correction over the nine pairs no GIFT-Eval pair separates: the smallest corrected p-value is 0.33 (column `holm9_cr2` of `pair_table_MASE_Rank.csv`; places 9-10 has 0.33).

## Plotted numbers, panel (a) GIFT-Eval

Source: `camera_ready/pair_table_MASE_Rank.csv`, the row whose `places` is the pair; gap = `gap_theta`, interval = `ci95_lo`, `ci95_hi`; standard error = `cr2_se`; degrees of freedom = `df`. The script checks that the same row in `camera_ready/retest_gift.json` (`pairs.MASE_Rank[k].theta` and `pairs.MASE_Rank[k]["CR2-icc"]`, with k = [0, 9, 17, 24, 30, 35, 39, 42, 44] for the nine pairs in order) holds identical values, and asserts every value against the printed table to 2 decimals (passed).
Resolution line: `retest_gift.json` key `resolution.MASE_Rank.summary.corr_res_icc` = 6.8010. Median gap: `resolution.MASE_Rank.summary.median_gap` = 0.9072.

| Places | Entries (higher / lower) | Gap | 95% interval | CR2 standard error | Degrees of freedom | Interval excludes 0 | Gap reaches resolution |
|---|---|---|---|---|---|---|---|
| 1-2 | STRIDE w/ Synapse / EXAONE-Forecast-Agent | 4.25 | [-2.40, 10.89] | 3.048 | 11.89 | no | no |
| 2-3 | EXAONE-Forecast-Agent / LS-MoE | 0.44 | [-6.68, 7.57] | 3.299 | 12.97 | no | no |
| 3-4 | LS-MoE / LS-Agent | 1.94 | [-0.15, 4.02] | 0.989 | 17.07 | no | no |
| 4-5 | LS-Agent / Falcon-Agent | 0.91 | [-3.82, 5.64] | 2.243 | 17.07 | no | no |
| 5-6 | Falcon-Agent / CastStar | 0.07 | [-4.75, 4.90] | 2.288 | 17.07 | no | no |
| 6-7 | CastStar / limix_moe | 0.01 | [-2.98, 3.00] | 1.416 | 17.07 | no | no |
| 7-8 | limix_moe / TIMEHEDGE | 1.42 | [-4.17, 7.02] | 2.615 | 14.43 | no | no |
| 8-9 | TIMEHEDGE / RegLab-AH-1 | 0.65 | [-6.63, 7.93] | 3.353 | 12.42 | no | no |
| 9-10 | RegLab-AH-1 / RacineCast-1-NC | 1.33 | [0.09, 2.57] | 0.586 | 17.07 | yes | no |

## Plotted numbers, panel (b) fev-bench

Source: `camera_ready/boards/fev_family_report.json`, report `reports.A_wr` (fev-bench, 100 tasks, win rate; `n_entries` = 40). The nine pairs are the entries of `reports.A_wr.pairs` with `adjacent` true (indices [0, 9, 17, 24, 30, 35, 39, 42, 44], in place order); gap = `theta`, interval = `cr2_icc.ci95`, standard error = `cr2_icc.se`, degrees of freedom = `cr2_icc.df`. The file stores win rates as fractions, so gap, interval and standard error are multiplied by 100 to give points; nothing else is changed.
Resolution line: `reports.A_wr.resolution.corrected_icc` x 100 = 7.5133 (the summary `summaries.A_wr.resolution_corrected` = 7.513 agrees). Median gap: `reports.A_wr.resolution.median_adjacent_gap` x 100 = 2.0769.

| Places | Entries (higher / lower) | Gap | 95% interval | CR2 standard error | Degrees of freedom | Interval excludes 0 | Gap reaches resolution |
|---|---|---|---|---|---|---|---|
| 1-2 | TimeRSI-1.5 / ci-proto | 2.64 | [-2.32, 7.60] | 2.233 | 10.20 | no | no |
| 2-3 | ci-proto / timeh2o-v1 | 0.10 | [-5.22, 5.43] | 2.282 | 7.50 | no | no |
| 3-4 | timeh2o-v1 / Timestral-209M | 2.72 | [-1.47, 6.91] | 1.820 | 8.11 | no | no |
| 4-5 | Timestral-209M / PDM-Nano | 1.18 | [-4.84, 7.19] | 2.547 | 7.05 | no | no |
| 5-6 | PDM-Nano / TimeRSI | 0.31 | [-5.05, 5.67] | 2.404 | 9.98 | no | no |
| 6-7 | TimeRSI / t0-gamma | 2.08 | [-3.89, 8.05] | 2.697 | 10.50 | no | no |
| 7-8 | t0-gamma / Simons-297M | 2.62 | [-5.32, 10.55] | 3.385 | 7.32 | no | no |
| 8-9 | Simons-297M / timesfm-3 | 3.13 | [-2.30, 8.56] | 2.339 | 7.72 | no | no |
| 9-10 | timesfm-3 / ChakraTS | 0.44 | [-5.18, 6.05] | 2.443 | 8.18 | no | no |

## Checks run by the script (all passed)

- The nine GIFT-Eval values equal the printed table to 2 decimals; the CSV and `retest_gift.json` agree exactly.
- For both panels, each stored interval equals gap plus or minus t(0.975, df) times the stored standard error, and the stored resolution equals the median over the nine pairs of (t(0.975, df) + t(0.80, df)) times the standard error (differences below 1e-6). These are checks on stored numbers; the plotted values are the stored intervals.
- fev-bench: the stored count of neighbouring pairs separated under CR2 (`summaries.A_wr.adjacent_of_9.CR2-icc`) is 0, matching the nine plotted intervals.
- No text touches a marker, a whisker, a cap, the zero line or the resolution line (at least 1 pt of free space); no two texts overlap; no text has a background box; all nine pairs are labelled on each x axis; no text in the figure uses a shortened form of "configuration".
- In the 200 dpi preview every pixel column across each panel shows the resolution line's colour (panel (a): 404 columns, panel (b): 404 columns).
- The PDF embeds its font as TrueType (1 embedded font program; no Type 3 fonts).

## Sentences a caption can use

Gaps between neighbouring entries of each board's displayed top ten, with 95% intervals from the CR2 cluster-robust standard error (t intervals with Satterthwaite degrees of freedom); the orange line is the board's resolution, the smallest gap it can detect eight times in ten (6.80 rank places on GIFT-Eval, 7.51 win-rate points on fev-bench).
No gap reaches the line, and only one of the eighteen intervals excludes zero (GIFT-Eval places 9-10, filled marker, uncorrected; it does not survive the Holm correction over nine pairs).
