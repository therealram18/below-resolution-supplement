# Do new datasets vary like the current ones? A cross-board check

Question: the paper's "about 45 times as many datasets" figure assumes new datasets would vary the way the current 28 GIFT-Eval datasets do. fev-bench is a second board, and several open models are scored on both. For each pair of those models I take the difference in log MASE, average it within each data source (a GIFT-Eval base dataset, or a fev-bench task family), and compare the standard deviation of those source means on the two boards. Only fev-bench families that do not share data with GIFT-Eval are used for the main comparison.

## Models scored on both boards (same checkpoint only)

| Model | GIFT-Eval folder | fev-bench file | Why it is the same |
|---|---|---|---|
| Chronos-2 | chronos-2 | chronos-2 | same link amazon/chronos-2 |
| Chronos-Bolt-Base | chronos_bolt_base | chronos-bolt | same link amazon/chronos-bolt-base |
| TimesFM-2.5 | TimesFM-2.5 | timesfm-2_5 | same link google/timesfm-2.5-200m-pytorch |
| TimesFM-3 | TimesFM-3 | timesfm-3 | same link google/timesfm-3.0-pytorch |
| Moirai-2.0 | Moirai2 | moirai-2_0 | same link Salesforce/moirai-2.0-R-small |
| Toto-1.0 | Toto_Open_Base_1.0 | toto-1_0 | same link Datadog/Toto-Open-Base-1.0 |
| Toto-2.0-4m | Toto-2.0-4m | toto-2_0-4m | same link |
| Toto-2.0-22m | Toto-2.0-22m | toto-2_0-22m | same link |
| Toto-2.0-313m | Toto-2.0-313m | toto-2_0-313m | same link |
| Toto-2.0-1B | Toto-2.0-1B | toto-2_0-1b | same link |
| Toto-2.0-2.5B | Toto-2.0-2.5B | toto-2_0-2_5b | same link |
| Sundial-Base | sundial_base_128m | sundial-base | same link thuml/sundial-base-128m |
| FlowState | Granite-FlowState-r1.1 | flowstate | same link ibm-granite/granite-timeseries-flowstate-r1 (GIFT folder says r1.1, fev says r1; same URL, kept) |
| Seasonal naive | seasonal_naive | seasonal_naive | same definition, no weights |
| Naive | naive | naive | same definition, no weights |

Rejected or uncertain, not used:

- TiRex: GIFT uses NX-AI/TiRex-1.1-gifteval, fev uses NX-AI/TiRex: different checkpoints
- TiRex-2: GIFT uses TiRex-2-gifteval-zs/-pretrain, fev uses TiRex-2-fevbench: different checkpoints
- TabPFN-TS: GIFT: tabpfn-time-series repo; fev: Prior-Labs/TabPFN-v2-reg; wrapper/version not confirmed; fev has 99/100 tasks
- AutoARIMA / AutoETS: fev results incomplete (96 and 97 of 100 tasks)
- AutoTheta: complete on both, but the implementation and settings per board are not documented; uncertain
- TFT, DeepAR, PatchTST: trained per dataset with board-specific setups; not the same checkpoint
- t0-alpha (GIFT) vs t0-beta/gamma (fev): different versions
- TimesFM-2.0, Moirai small/base/large (GIFT only), Drift, CatBoost, LightGBM (fev only): not on both boards

15 models give 105 pairs. In each pair, fev-bench rows where either model is flagged as trained on that dataset are skipped (0 to 30 rows per pair, median 1; Moirai-2.0 alone has 28 flagged rows).

## fev-bench families that share data with GIFT-Eval (excluded from the main comparison)

A family is the task name with its last underscore-separated part removed (single-word names stay whole). Uncertain cases are treated as shared.

- ETT: ETT = ett1/ett2
- LOOP_SEATTLE: loop_seattle
- M_DENSE: m_dense
- SZ_TAXI: sz_taxi
- bizitobs_l2c: bizitobs_l2c
- hierarchical_sales: hierarchical_sales
- hospital: hospital
- jena_weather: jena_weather
- restaurant: restaurant
- solar: UNCERTAIN: GIFT solar (Alabama plants); source of fev solar not checked
- solar_with_weather: UNCERTAIN: solar-named
- hospital_admissions: UNCERTAIN: hospital-named
- uk_covid_nation: UNCERTAIN: GIFT covid_deaths
- uk_covid_utla: UNCERTAIN: GIFT covid_deaths

Non-shared (25 families): australian, boomlet, ecdc, entsoe, epf, ercot, favorita_stores, favorita_transactions, fred_md, fred_qd, gvar, hermes, kdd_cup_2022, m5, proenfo, redset, rohlik_orders, rohlik_sales, rossmann, uci_air_quality, us_consumption, walmart, world, world_co2, world_life.

## Results

Ratio = fev-bench (non-shared families) divided by GIFT-Eval, computed per model pair, then summarised over pairs. A ratio near 1 means new datasets vary like the current ones; above 1 means more, below 1 means less. The interval is a bootstrap (2,000 draws, seed 20261111) on the median, resampling sources within each board.

| Ratio | Pairs | Median (95% interval) | Quartiles | Min to max | Pooled (root mean square) |
|---|---|---|---|---|---|
| SD of source means, non-shared fev families / GIFT 28 datasets | 105 | 1.13 (0.85 to 1.72) | 0.83 to 1.92 | 0.24 to 3.52 | 0.89 |
| Same, all 39 fev families (reference) | 105 | 1.08 (0.82 to 1.60) | 0.78 to 1.69 | 0.24 to 3.08 | 0.78 |
| Per-source noise scale (CR2 standard error x sqrt(sources)), non-shared fev / GIFT | 105 | 1.25 (1.11 to 1.43) | 0.90 to 2.19 | 0.23 to 5.33 | 1.12 |

## Caveats

The two boards differ in horizons, frequencies and number of forecast windows per task, so a source is not the same kind of unit on both; MASE is used on both. The shared models are few and heavily related (five Toto sizes, two TimesFM versions, two naive baselines), so the pairs are strongly correlated and the interval on the median understates the real uncertainty. Spread of source means also includes sampling noise from the number of series per source, which differs by board; the CR2 row accounts for unit counts but not for these design differences.

## What this does and does not show

The spread across fev-bench datasets that GIFT-Eval does not contain is about the same as across GIFT-Eval's own 28: the median ratio is 1.13 (interval 0.85 to 1.72) and the pooled ratio is 0.89, so the 45 times figure is not contradicted, and if anything the noise per new dataset may be somewhat larger (the standard-error version gives a median of 1.25, interval 1.11 to 1.43, pooled 1.12, i.e. roughly 25 percent more noise, about 1.5 times as many datasets needed). The check cannot show that the real new datasets would match, because only 15 related models and 25 fev families enter, individual pairs range from 0.2 to 5 times, and the board designs differ, so it supports the assumption without settling it and the paper should state it with that caveat.
