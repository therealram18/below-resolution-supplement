# Context-length experiment re-test (512 vs default context, 12 configs)

Each cell is one model under one aggregation. Per-config difference = value at 512 minus value at default (rank places, or log of scaled MASE). The 12 configs come from 12 different datasets, so each is its own cluster. Sign-flip null: the 12 differences are independent and each symmetric about zero; p is from all 4,096 sign patterns. t-test: paired, 11 df. Wild: studentized, Rademacher, exact over 4,096 patterns (singleton clusters).

## 0. Regression check against the submitted paper

| quantity | paper | reproduced |
|---|---|---|
| movements with paper interval excluding 0 (of 8) | 5 | 5 (moirai2_small/gmean, timesfm25/rank, timesfm25/gmean, toto/rank, toto/gmean) |
| survivors after Holm, paper rule (bootstrap p, B=100,000) | 4 | 4 |
| rank movement chronos / moirai / timesfm / toto | +6.67 / +4.08 / +26.58 / +16.58 | +6.67 / +4.08 / +26.58 / +16.58 |
| geometric-mean movement | -0.0189 / +0.0494 / +0.1629 / +0.1675 | -0.0189 / +0.0494 / +0.1629 / +0.1675 |
| percentage change | -2.0 / +6.8 / +22.2 / +20.6 | -2.0 / +6.8 / +22.2 / +20.6 |

This table shows whether the submitted numbers come back from the pinned files with the paper's bootstrap (B=4,000, seed 20260914).

## Results per cell

| cell | point (diff.) | paper 95% interval (B=4,000) | paper | sign-flip p | t-test p | t-test 95% interval | Holm: paper / sign-flip / t / wild |
|---|---|---|---|---|---|---|---|
| chronos_bolt_base rank | 6.6667 | [-4.9167, 19.0000] | includes 0 | 0.3418 | 0.3231 | [-7.5156, 20.8490] | no / no / no / no |
| chronos_bolt_base gmean | -0.0189 | [-0.1750, 0.0633] | includes 0 | 0.7539 | 0.7498 | [-0.1528, 0.1133] | no / no / no / no |
| moirai2_small rank | 4.0833 | [-6.9167, 14.0833] | includes 0 | 0.4829 | 0.4799 | [-8.2055, 16.3722] | no / no / no / no |
| moirai2_small gmean | 0.0494 | [0.0060, 0.1070] | excludes 0 | 0.0713 | 0.0638 | [-0.0045, 0.1367] | no / no / no / no |
| timesfm25 rank | 26.5833 | [8.8333, 44.4167] | excludes 0 | 0.0215 | 0.0177 | [5.5830, 47.5837] | yes / no / no / no |
| timesfm25 gmean | 0.1629 | [0.0402, 0.3416] | excludes 0 | 0.0176 | 0.0479 | [0.0023, 0.3983] | yes / no / no / no |
| toto rank | 16.5833 | [6.5000, 28.0000] | excludes 0 | 0.0107 | 0.0135 | [4.1579, 29.0088] | yes / no / no / no |
| toto gmean | 0.1675 | [0.0666, 0.3169] | excludes 0 | 0.0039 | 0.0154 | [0.0434, 0.3314] | yes / yes / no / yes |

This table shows, for each of the 8 cells, the paper's interval, the exact sign-flip and paired t results, and whether each survives Holm over 8.

| cell | wild p (exact) | paper bootstrap p (B=100,000) |
|---|---|---|
| chronos_bolt_base rank | 0.3418 | 0.2822 |
| chronos_bolt_base gmean | 0.7539 | 0.7801 |
| moirai2_small rank | 0.4829 | 0.4495 |
| moirai2_small gmean | 0.0713 | 0.0174 |
| timesfm25 rank | 0.0215 | 0.0041 |
| timesfm25 gmean | 0.0176 | 0.0015 |
| toto rank | 0.0107 | 0.0002 |
| toto gmean | 0.0039 | 0.0000 |

This table shows the wild studentized exact p-value and the bootstrap p-value used for the paper's Holm step.

| model | % change in geometric mean | t interval on log scale, exponentiated | paper bootstrap interval (pct_change.json) |
|---|---|---|---|
| chronos_bolt_base | -2.0% | [-14.2%, +12.0%] | [-13.2%, +8.9%] |
| moirai2_small | +6.8% | [-0.4%, +14.6%] | [+1.0%, +14.0%] |
| timesfm25 | +22.2% | [+0.2%, +48.9%] | [+5.2%, +45.6%] |
| toto | +20.6% | [+4.4%, +39.3%] | [+8.1%, +36.8%] |

This table shows the percentage change in the geometric mean with the corrected and the paper's intervals.

## Cells surviving (of 8)

| method | uncorrected (p < 0.05 or interval excludes 0) | after Holm over 8 |
|---|---|---|
| paper bootstrap | 5 | 4 |
| exact sign-flip | 4 | 1 |
| paired t-test | 4 | 0 |
| wild exact (12 clusters) | 4 | 1 |
| CR2 icc, 9 source clusters | 2 | 1 |
| wild exact, 9 source clusters | 3 | 0 |

This table counts survivors for each method.

## Sensitivity: 9 source clusters (3 bizitobs configs as one, ett1 + ett2 as one)

| cell | CR2 (icc df) p | CR2 df | wild exact p (G=9) |
|---|---|---|---|
| chronos_bolt_base rank | 0.1298 | 6.3 | 0.1523 |
| chronos_bolt_base gmean | 0.6214 | 6.3 | 0.9570 |
| moirai2_small rank | 0.2973 | 6.3 | 0.2656 |
| moirai2_small gmean | 0.0137 | 6.3 | 0.0352 |
| timesfm25 rank | 0.0726 | 5.7 | 0.0586 |
| timesfm25 gmean | 0.2005 | 5.3 | 0.0547 |
| toto rank | 0.0048 | 6.3 | 0.0156 |
| toto gmean | 0.0735 | 6.3 | 0.0156 |

This table shows the two small-sample-corrected p-values when the 12 configs are collapsed to 9 source clusters.

## What changes relative to the submitted paper

- Movements separated, uncorrected: 5 of 8 (percentile bootstrap) -> 4 (sign-flip), 4 (t-test), 4 (wild exact).
- After Holm: 4 of 8 (bootstrap) -> 1 (sign-flip), 0 (t-test), 1 (wild exact).
- 9-cluster sensitivity: uncorrected 2 (CR2) / 3 (wild exact); after Holm 1 / 0.
- chronos_bolt_base percentage change interval: [-13.2%, +8.9%] -> [-14.2%, +12.0%]
- moirai2_small percentage change interval: [+1.0%, +14.0%] -> [-0.4%, +14.6%]
- timesfm25 percentage change interval: [+5.2%, +45.6%] -> [+0.2%, +48.9%]
- toto percentage change interval: [+8.1%, +36.8%] -> [+4.4%, +39.3%]
