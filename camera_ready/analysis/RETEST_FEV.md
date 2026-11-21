# fev-bench re-test under small-sample-corrected methods

Data: 100 tasks, 39 task families (clusters), top-10 models by displayed win rate. Gaps are in win-rate points (or log-error-ratio units for skill). Wild bootstrap: Rademacher, 99,999 draws. 'Paper rule' = percentile bootstrap, tasks independent and families clustered, both intervals must exclude zero.

## 0. Regression check against the submitted paper

| quantity | paper | reproduced |
|---|---|---|
| adjacent gaps surviving both schemes (win rate) | 1 of 9 | 1 of 9 |
| pair 1-2 clustered interval | [0.65, 8.57] | [0.65, 8.57] |
| resolution (2.80 x median clustered bootstrap SE) | 9.52 | 9.52 |
| median adjacent gap | 2.17 | 2.17 |
| leaderboard max abs error, win rate / skill | exact | 4.26e-14 / 2.13e-14 |

This table shows whether the paper's win-rate numbers come back from the pinned files with the original random stream (B=4,000, seed 20260907).

## win rate

### Pairs counted as separated (adjacent = the 9 neighbours; Holm is over all 45 top-10 pairs)

| method | adjacent, uncorrected | adjacent, after Holm | all 45, uncorrected | all 45, after Holm |
|---|---|---|---|---|
| paper rule (bootstrap, both) | 1 | 0 | 27 | 15 |
| CR2, df from estimated ICC | 1 | 0 | 20 | 5 |
| CR2, df independent | 1 | 0 | 24 | 13 |
| CR1, t with 38 df | 1 | 0 | 27 | 14 |
| wild cluster bootstrap | 1 | 0 | 24 | 10 |
| family-means t-test | 1 | 0 | 29 | 20 |

This table shows how many pairs each method separates, with the paper rule's Holm step using the larger of its two bootstrap p-values (B=100,000, seed 20260907).

### Adjacent pairs

| pair | gap | paper clustered 95% interval (B=100k) | CR2-icc p | CR2-icc 95% interval | CR2-indep p | CR1 p | wild p | family-means p |
|---|---|---|---|---|---|---|---|---|
| 1-2 | 5.033 | [0.614, 8.624] | 0.0407 | [0.271, 9.795] | 0.0243 | 0.0186 | 0.0359 | 0.2636 |
| 2-3 | 3.200 | [-0.787, 7.467] | 0.1656 | [-1.688, 8.088] | 0.1366 | 0.1297 | 0.1209 | 0.2293 |
| 3-4 | 0.600 | [-8.217, 10.366] | 0.9135 | [-12.264, 13.464] | 0.9110 | 0.9066 | 0.9261 | 0.2099 |
| 4-5 | 0.700 | [-1.648, 2.852] | 0.5476 | [-1.796, 3.196] | 0.5409 | 0.5367 | 0.5396 | 0.2807 |
| 5-6 | 1.700 | [-0.696, 3.789] | 0.1562 | [-0.742, 4.142] | 0.1466 | 0.1392 | 0.1457 | 0.3231 |
| 6-7 | 4.533 | [-3.472, 11.820] | 0.3340 | [-5.797, 14.864] | 0.3112 | 0.2841 | 0.4163 | 0.8491 |
| 7-8 | 2.167 | [-5.188, 10.088] | 0.6399 | [-8.505, 12.838] | 0.6282 | 0.6091 | 0.7015 | 0.0516 |
| 8-9 | 1.800 | [-7.079, 9.531] | 0.7089 | [-9.424, 13.024] | 0.6995 | 0.6868 | 0.7457 | 0.4942 |
| 9-10 | 3.400 | [-3.127, 10.612] | 0.3583 | [-4.869, 11.669] | 0.3338 | 0.3271 | 0.3278 | 0.0160 |

This table shows each adjacent pair's gap with the paper's clustered interval and the corrected p-values and intervals.

### Resolution (smallest gap detected with 80% power, median over the 9 adjacent pairs)

| quantity | value |
|---|---|
| paper: 2.80 x median clustered bootstrap SE (B=4,000 stream) | 9.522 |
| paper formula at B=100k | 9.736 |
| corrected: mde80(CR2 SE, df from estimated ICC), median over pairs | 11.366 (median df 7.0) |
| median adjacent gap | 2.167 |
| resolution / median gap: paper, corrected | 4.39, 5.25 |
| families needed G* = 39 x (2.80 SE / gap)^2, median: paper SE, CR2 SE | 306, 313 |

This table shows the resolution and the number of families needed to resolve the median adjacent gap.

### Estimated within-family correlation and skewness of family means (adjacent pairs)

| pair | icc_hat | skewness |
|---|---|---|
| 1-2 | 0.208 | -0.68 |
| 2-3 | 0.285 | -0.12 |
| 3-4 | 0.363 | 0.37 |
| 4-5 | 0.128 | -0.19 |
| 5-6 | 0.080 | -1.49 |
| 6-7 | 0.285 | -1.14 |
| 7-8 | 0.357 | 0.48 |
| 8-9 | 0.384 | -0.20 |
| 9-10 | 0.337 | 0.11 |

This table shows the two quantities the correction depends on.

## skill score

### Pairs counted as separated (adjacent = the 9 neighbours; Holm is over all 45 top-10 pairs)

| method | adjacent, uncorrected | adjacent, after Holm | all 45, uncorrected | all 45, after Holm |
|---|---|---|---|---|
| paper rule (bootstrap, both) | 2 | 0 | 27 | 15 |
| CR2, df from estimated ICC | 2 | 0 | 17 | 0 |
| CR2, df independent | 2 | 0 | 20 | 7 |
| CR1, t with 38 df | 2 | 0 | 22 | 8 |
| wild cluster bootstrap | 2 | 0 | 20 | 10 |
| family-means t-test | 2 | 0 | 24 | 13 |

This table shows how many pairs each method separates, with the paper rule's Holm step using the larger of its two bootstrap p-values (B=100,000, seed 20260907).

### Adjacent pairs

| pair | gap | paper clustered 95% interval (B=100k) | CR2-icc p | CR2-icc 95% interval | CR2-indep p | CR1 p | wild p | family-means p |
|---|---|---|---|---|---|---|---|---|
| 1-2 | 0.026 | [0.008, 0.049] | 0.0205 | [0.005, 0.048] | 0.0168 | 0.0133 | 0.0097 | 0.0449 |
| 2-3 | 0.033 | [0.010, 0.061] | 0.0349 | [0.003, 0.063] | 0.0182 | 0.0143 | 0.0100 | 0.0221 |
| 3-4 | 0.020 | [-0.029, 0.077] | 0.5246 | [-0.051, 0.090] | 0.5072 | 0.4906 | 0.5052 | 0.1627 |
| 4-5 | -0.000 | [-0.006, 0.006] | 0.9391 | [-0.007, 0.006] | 0.9391 | 0.9389 | 0.9408 | 0.5300 |
| 5-6 | 0.004 | [-0.003, 0.011] | 0.3311 | [-0.005, 0.012] | 0.3090 | 0.3025 | 0.3060 | 0.2753 |
| 6-7 | 0.029 | [-0.006, 0.062] | 0.1690 | [-0.015, 0.073] | 0.1517 | 0.1278 | 0.1878 | 0.4179 |
| 7-8 | -0.006 | [-0.033, 0.023] | 0.7294 | [-0.042, 0.030] | 0.7254 | 0.7112 | 0.8158 | 0.3495 |
| 8-9 | 0.012 | [-0.020, 0.041] | 0.4876 | [-0.026, 0.051] | 0.4766 | 0.4541 | 0.5849 | 0.6684 |
| 9-10 | -0.016 | [-0.063, 0.028] | 0.5168 | [-0.074, 0.042] | 0.4942 | 0.4899 | 0.4994 | 0.5803 |

This table shows each adjacent pair's gap with the paper's clustered interval and the corrected p-values and intervals.

### Resolution (smallest gap detected with 80% power, median over the 9 adjacent pairs)

| quantity | value |
|---|---|
| paper: 2.80 x median clustered bootstrap SE (B=4,000 stream) | 0.042 |
| paper formula at B=100k | 0.041 |
| corrected: mde80(CR2 SE, df from estimated ICC), median over pairs | 0.051 (median df 8.8) |
| median adjacent gap | 0.016 |
| resolution / median gap: paper, corrected | 2.65, 3.17 |
| families needed G* = 39 x (2.80 SE / gap)^2, median: paper SE, CR2 SE | 508, 582 |

This table shows the resolution and the number of families needed to resolve the median adjacent gap.

### Estimated within-family correlation and skewness of family means (adjacent pairs)

| pair | icc_hat | skewness |
|---|---|---|
| 1-2 | 0.055 | 0.97 |
| 2-3 | 0.239 | 0.68 |
| 3-4 | 0.379 | 0.14 |
| 4-5 | 0.000 | -0.85 |
| 5-6 | 0.273 | 0.79 |
| 6-7 | 0.152 | 1.48 |
| 7-8 | 0.145 | -0.84 |
| 8-9 | 0.183 | 0.30 |
| 9-10 | 0.564 | -2.23 |

This table shows the two quantities the correction depends on.

## What changes relative to the submitted paper

- Adjacent gaps separated, win rate: 1 of 9 (paper rule) -> 1 (CR2-icc), 1 (CR2-indep), 1 (CR1), 1 (wild), 1 (family-means), uncorrected.
- Resolution, win rate: 9.52 points -> 11.37 points; ratio to median gap 4.4 -> 5.2.
- Families needed for the median adjacent gap: 306 -> 313.
