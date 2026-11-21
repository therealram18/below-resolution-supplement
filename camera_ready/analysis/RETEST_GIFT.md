# GIFT-Eval re-test under small-sample-corrected methods

Pre-registered in DESIGN.md (Experiment A2, Revision 2 items 5 and 7). Script: `retest_gift.py` (numbers, `retest_gift.json`) and `retest_gift_md.py` (this file). Base seed 20261111. Runtime 281 s.

Deviation: see the end of this file.

## Step 0. Regression check against the published counts

Shows whether the paper's rule (percentile bootstrap, iid and cluster both exclude zero; B=100,000) reproduces the published counts on the displayed top-10.

| rule | adjacent of 9 (published / ours) | pairs of 45 (published / ours) | Holm of 45 (published / ours) | status |
|---|---|---|---|---|
| MASE_Rank | 0 / 0 | 12 / 12 | 2 / 2 | exact |
| CRPS_Rank | 0 / 0 | 16 / 16 | 0 / 0 | exact |
| gmean | 2 / 2 | 23 / 23 | 11 / 10 | Holm differs by 1 (nearest pair 1-5: p=0.00146 vs Holm threshold 0.00143, Monte-Carlo SE of p 0.00017) |

The gmean pair 1-5 sits 3e-5 from its Holm threshold, inside the bootstrap's own noise. Every uncorrected count matches exactly. gmean pairs here use the displayed MASE_Rank order and log ratios (as `load_gift_pairs.py` provides); the paper's gmean bootstrap may have been on the ratio scale, which could account for the one-pair difference (not checked).

## Step 1. Per-pair numbers

All 45 pairs per rule (theta, estimated within-dataset correlation, skewness of the 28 dataset means, p-values and 95% intervals under every method) are in `retest_gift.json` under `pairs`. The dataset-means t-test weights datasets equally, so it tests a slightly different quantity than the config-weighted mean. Summary of the shape of the data:

| rule | icc range over 45 pairs | median icc | median skewness of dataset means | median CR2 df (icc) |
|---|---|---|---|---|
| MASE_Rank | 0.00 - 0.51 | 0.14 | 0.44 | 13.8 |
| CRPS_Rank | 0.00 - 0.69 | 0.30 | 0.75 | 12.3 |
| gmean | 0.00 - 0.64 | 0.09 | 1.14 | 14.5 |

## Step 2. Counts of separated pairs

Shows how many displayed-top-10 pairs each method separates at 5%, without correction and after Holm over the 45 pairs (adjacent = 9 neighbouring pairs; all = 45).

**MASE_Rank**

| method | adjacent unc. (of 9) | all unc. (of 45) | adjacent Holm | all Holm (of 45) | adjacent pairs separated (uncorrected) |
|---|---|---|---|---|---|
| P-both | 0 | 12 | 0 | 2 | none |
| P-clu | 1 | 14 | 0 | 2 | 9-10 |
| CR2-icc | 1 | 12 | 0 | 0 | 9-10 |
| CR2-indep | 1 | 13 | 0 | 0 | 9-10 |
| CR1 | 1 | 13 | 0 | 1 | 9-10 |
| WCB-Rad | 1 | 13 | 0 | 1 | 9-10 |
| Means-t | 0 | 7 | 0 | 0 | none |

WCB with Webb weights: adjacent 1, all 13, Holm adjacent 0, Holm all 1.

**CRPS_Rank**

| method | adjacent unc. (of 9) | all unc. (of 45) | adjacent Holm | all Holm (of 45) | adjacent pairs separated (uncorrected) |
|---|---|---|---|---|---|
| P-both | 0 | 16 | 0 | 0 | none |
| P-clu | 0 | 16 | 0 | 0 | none |
| CR2-icc | 0 | 9 | 0 | 0 | none |
| CR2-indep | 0 | 10 | 0 | 0 | none |
| CR1 | 0 | 12 | 0 | 0 | none |
| WCB-Rad | 0 | 12 | 0 | 0 | none |
| Means-t | 0 | 7 | 0 | 0 | none |

WCB with Webb weights: adjacent 0, all 12, Holm adjacent 0, Holm all 0.

**gmean**

| method | adjacent unc. (of 9) | all unc. (of 45) | adjacent Holm | all Holm (of 45) | adjacent pairs separated (uncorrected) |
|---|---|---|---|---|---|
| P-both | 2 | 23 | 0 | 10 | 8-9, 9-10 |
| P-clu | 3 | 25 | 0 | 10 | 3-4, 8-9, 9-10 |
| CR2-icc | 0 | 10 | 0 | 0 | none |
| CR2-indep | 0 | 11 | 0 | 0 | none |
| CR1 | 0 | 13 | 0 | 0 | none |
| WCB-Rad | 0 | 19 | 0 | 4 | none |
| Means-t | 0 | 12 | 0 | 0 | none |

WCB with Webb weights: adjacent 0, all 19, Holm adjacent 0, Holm all 0.

Pair at positions 9-10 under MASE_Rank (mean rank difference 1.33 places):

| method | p-value | separated at 5% |
|---|---|---|
| P-iid (paper) | 0.0610 | False |
| P-clu (paper) | 0.0301 | True |
| P-both (paper rule) | 0.0610 | False |
| CR2-icc | 0.0364 | True |
| CR2-indep | 0.0364 | True |
| CR1 | 0.0303 | True |
| WCB-Rad | 0.0406 | True |
| WCB-Webb | 0.0396 | True |
| Means-t | 0.0730 | False |

## Step 3. Resolution

Shows the smallest gap each method could detect with 80% power, the median gap between displayed neighbours, and the number of datasets needed to resolve that gap. Paper = 2.80 x cluster-bootstrap SE (B=4000, resolution_budget.py's exact resamples). Corrected = (t_0.975 + t_0.80 at Satterthwaite df) x CR2 SE, median over the 9 adjacent pairs. Units: rank places (MASE_Rank, CRPS_Rank), log scaled-MASE (gmean). gmean uses the gmean's own displayed order, as resolution_budget.py does.

| rule | paper resolution | corrected (df=icc) | corrected (df=indep) | median adjacent gap | resolution / gap (paper / corrected) | datasets needed, paper SE (multiple of 28) | datasets needed, CR2 SE | median se_cr2 / bootstrap SE |
|---|---|---|---|---|---|---|---|---|
| MASE_Rank | 6.165 | 6.801 | 6.801 | 0.9072 | 6.8 / 7.5 | 1269 (45x) | 1342 (48x) | 1.022 |
| CRPS_Rank | 6.834 | 7.381 | 7.268 | 0.6701 | 10.2 / 11.0 | 3307 (118x) | 3319 (119x) | 1.011 |
| gmean (own order) | 0.044 | 0.04966 | 0.04858 | 0.004471 | 9.8 / 11.1 | 7093 (253x) | 7567 (270x) | 1.033 |

Per-pair se_cr2 / bootstrap SE ratios (9 adjacent pairs, in displayed order):

| rule | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 | 9-10 |
|---|---|---|---|---|---|---|---|---|---|
| MASE_Rank | 1.052 | 1.025 | 0.984 | 1.028 | 1.039 | 1.015 | 1.014 | 1.005 | 1.022 |
| CRPS_Rank | 1.033 | 1.022 | 1.002 | 1.004 | 1.011 | 1.000 | 1.005 | 1.014 | 1.021 |
| gmean_own | 1.073 | 1.059 | 1.005 | 0.989 | 1.033 | 1.010 | 1.006 | 1.040 | 1.044 |

## Step 4. Leading tier

Shows which of the next 39 displayed entries are not separated from the displayed leader by uncorrected tests (leak-flagged entries pushed down, complete entries only, as resolution_budget.py). Not a confidence set for the best entry.

**MASE_Rank** (leader: STRIDE w/ Synapse)

| method | tier size | members |
|---|---|---|
| P-both (paper B=4000) | 3 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE |
| P-both (B=100k) | 3 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE |
| CR2-icc | 3 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE |
| WCB-Rad | 3 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE |
| Means-t | 11 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE, LS-Agent, Falcon-Agent, CastStar, limix_moe, RegLab-AH-1, RacineCast-1-NC, RacineCast-1, Toto-2.0-FnF |

**CRPS_Rank** (leader: STRIDE w/ Synapse)

| method | tier size | members |
|---|---|---|
| P-both (paper B=4000) | 2 | STRIDE w/ Synapse, EXAONE-Forecast-Agent |
| P-both (B=100k) | 2 | STRIDE w/ Synapse, EXAONE-Forecast-Agent |
| CR2-icc | 3 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE |
| WCB-Rad | 2 | STRIDE w/ Synapse, EXAONE-Forecast-Agent |
| Means-t | 12 | STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE, CastStar, LS-Agent, TIMEHEDGE, Falcon-Agent, RegLab-AH-1, limix_moe, RacineCast-1-NC, RacineCast-1, Cobra-Agent |

**gmean_own** (leader: EXAONE-Forecast-Agent)

| method | tier size | members |
|---|---|---|
| P-both (paper B=4000) | 9 | EXAONE-Forecast-Agent, STRIDE w/ Synapse, TIMEHEDGE, LS-MoE, LS-Agent, CastStar, TurkForecast-FM-Chronos2-LoRA-v1, Falcon-2.0, TimesFM-3 |
| P-both (B=100k) | 9 | EXAONE-Forecast-Agent, STRIDE w/ Synapse, TIMEHEDGE, LS-MoE, LS-Agent, CastStar, TurkForecast-FM-Chronos2-LoRA-v1, Falcon-2.0, TimesFM-3 |
| CR2-icc | 17 | EXAONE-Forecast-Agent, STRIDE w/ Synapse, TIMEHEDGE, LS-MoE, LS-Agent, CastStar, TurkForecast-FM-Chronos2-LoRA-v1, limix_moe, ForecastMate, RegLab-AH-1, FastCastrage, Falcon-Agent, Falcon-2.0, TimesFM-3, Taichu-TimeSeries-Agent, RAES-Conductance-Ensemble-V, STRIDE (+Chronos-2) |
| WCB-Rad | 17 | EXAONE-Forecast-Agent, STRIDE w/ Synapse, TIMEHEDGE, LS-MoE, LS-Agent, CastStar, TurkForecast-FM-Chronos2-LoRA-v1, limix_moe, ForecastMate, RegLab-AH-1, Falcon-Agent, Falcon-2.0, TimesFM-3, Taichu-TimeSeries-Agent, RAES-Conductance-Ensemble-V, STRIDE (+Chronos-2), Toto-2.0-FnF |
| Means-t | 26 | EXAONE-Forecast-Agent, STRIDE w/ Synapse, TIMEHEDGE, LS-MoE, LS-Agent, CastStar, TurkForecast-FM-Chronos2-LoRA-v1, limix_moe, ForecastMate, RegLab-AH-1, FastCastrage, Falcon-Agent, Falcon-2.0, TimesFM-3, Taichu-TimeSeries-Agent, RAES-Conductance-Ensemble-V, RacineCast-1, RacineCast-1-NC, EXAONE-Forecast, TimeRouter, Toto-2.0-FnF, TSOrchestra, Cobra-Agent, Toto-2.0-2.5B-FT, HistRoute-CV, Prism |

## Step 5. Coarser sources (19 clusters)

Shows the same counts and resolution when related datasets share one cluster. Datasets in the snapshot (28): bitbrains_fast_storage, bitbrains_rnd, bizitobs_application, bizitobs_l2c, bizitobs_service, car_parts, covid_deaths, electricity, ett1, ett2, hierarchical_sales, hospital, jena_weather, kdd_cup_2018, loop_seattle, m4_daily, m4_hourly, m4_monthly, m4_quarterly, m4_weekly, m4_yearly, m_dense, restaurant, saugeen, solar, sz_taxi, temperature_rain, us_births. Merged: m4_* (6) -> m4; ett1, ett2 -> ett; bizitobs_* (3) -> bizitobs; bitbrains_* (2) -> bitbrains. Sources (19): bitbrains, bizitobs, car_parts, covid_deaths, electricity, ett, hierarchical_sales, hospital, jena_weather, kdd_cup_2018, loop_seattle, m4, m_dense, restaurant, saugeen, solar, sz_taxi, temperature_rain, us_births.

**MASE_Rank** (19 clusters)

| method | adjacent unc. | all unc. | adjacent Holm | all Holm | adjacent separated |
|---|---|---|---|---|---|
| P-clu | 2 | 13 | 0 | 3 | 3-4, 9-10 |
| CR2-icc | 1 | 11 | 0 | 0 | 9-10 |
| WCB-Rad | 1 | 11 | 0 | 0 | 9-10 |

**gmean** (19 clusters)

| method | adjacent unc. | all unc. | adjacent Holm | all Holm | adjacent separated |
|---|---|---|---|---|---|
| P-clu | 2 | 22 | 0 | 7 | 8-9, 9-10 |
| CR2-icc | 0 | 4 | 0 | 0 | none |
| WCB-Rad | 0 | 15 | 0 | 2 | none |

Pair 9-10 under MASE_Rank at 19 clusters, p-values: P-clu 0.0393, CR2-icc 0.0379, WCB-Rad 0.0455.

| rule | 2.80 x bootstrap SE (B=20,000) | corrected resolution (CR2, icc df) | median gap | resolution / gap | sources needed, bootstrap SE | sources needed, CR2 SE |
|---|---|---|---|---|---|---|
| MASE_Rank | 6.048 | 6.752 | 0.9072 | 7.4 | 764 (40x of 19) | 760 (40x of 19) |
| gmean (own order) | 0.04663 | 0.05609 | 0.004471 | 12.5 | 5168 (272x of 19) | 5111 (269x of 19) |

## Step 6. Leave one dataset out

Shows how far the corrected resolution (CR2, icc df) and the number of adjacent pairs separated at 5% (CR2-icc, uncorrected) move when each of the 28 datasets is removed in turn. For gmean this uses the gmean's own displayed order.

| rule | full-data resolution | range over 28 drops | dataset that moves it most | adjacent count, full data | drops that change the adjacent count (new count) |
|---|---|---|---|---|---|
| MASE_Rank | 6.801 | 6.169 - 7.43 | kdd_cup_2018 -> 6.169 | 1 | bitbrains_fast_storage (0); bitbrains_rnd (0); bizitobs_l2c (0); car_parts (0); electricity (0); kdd_cup_2018 (2); m4_yearly (2); solar (0) |
| gmean_own | 0.04966 | 0.04093 - 0.0541 | loop_seattle -> 0.04093 | 1 | none |

## Step 7. Stability under subsets

A stability check, not a test of extrapolation: subsets of a fixed 28 datasets. Shows the median (over 200 random subsets) of SE_CR2 x sqrt(G') relative to its value at 28, MASE_Rank adjacent pairs. A value near 1 means SE shrinks like 1/sqrt(G') within these 28 datasets.

| G' | median over 9 pairs | range over pairs |
|---|---|---|
| 7 | 0.873 | 0.75 - 0.97 |
| 14 | 0.991 | 0.94 - 1.00 |
| 21 | 1.003 | 0.99 - 1.02 |
| 28 | 1.000 | 1.00 - 1.00 |

## What changes relative to the submitted paper

- Counts that stay: paper rule reproduces MASE_Rank 0/9, 12/45, Holm 2/45; CRPS_Rank 0/9, 16/45, Holm 0/45; gmean 2/9, 23/45 (Holm 10 vs published 11, borderline pair 1-5).
- Adjacent pairs separated, MASE_Rank: paper rule 0 -> CR2-icc 1, WCB 1 (pair 9-10 only), CR1 1, Means-t 0.
- Adjacent pairs separated, gmean: paper rule 2 -> CR2-icc 0, WCB 0, CR1 0, Means-t 0.
- Adjacent pairs separated, CRPS_Rank: 0 -> 0 under every corrected method.
- All 45 pairs uncorrected, MASE_Rank: 12 -> CR2-icc 12, WCB 13, Means-t 7; CRPS_Rank: 16 -> CR2-icc 9, WCB 12, Means-t 7; gmean: 23 -> CR2-icc 10, WCB 19, Means-t 12.
- All 45 pairs after Holm: MASE_Rank 2 -> CR2-icc 0, WCB 1; CRPS_Rank 0 -> 0; gmean 11 (ours 10) -> CR2-icc 0, WCB 4.
- Resolution (paper -> corrected, df=icc): MASE_Rank 6.165 -> 6.801; CRPS_Rank 6.834 -> 7.381; gmean 0.044 -> 0.04966.
- Datasets needed (multiple of 28, paper SE -> CR2 SE): MASE_Rank 45x -> 48x; CRPS_Rank 118x -> 119x; gmean 253x -> 270x.
- Leading tier size (paper rule -> CR2-icc / WCB / Means-t): MASE_Rank 3 -> 3 / 3 / 11; CRPS_Rank 2 -> 3 / 2 / 12; gmean_own 9 -> 17 / 17 / 26.

## Deviations from DESIGN.md

- Step 0: gmean Holm count is 10, published 11. The pair 1-5 p-value (0.00146) is within Monte-Carlo noise of its Holm threshold (0.00143). The run was allowed to continue on that basis (the instruction said borderline pairs can flip); an earlier run stopped at this point and wrote the same numbers.
- gmean has two orders: counts (steps 0-2, 5) use the MASE_Rank top-10 and log ratios from `load_gift_pairs.py`; resolution, tiers, leave-one-out use gmean's own displayed order as in `resolution_budget.py`. The two top-10 lists differ (see JSON `gmean_own_order_top10`, `gmean_pairs_order_top10`).
- Step 3/4 data for entries outside the top-10 come from the same audit pipeline as `load_gift_pairs.py` (ties ranked 'first'), restricted to entries with all 97 configs, as resolution_budget.py. Paper-style bootstrap SEs use resolution_budget.py's own resamples (seed 20260907, B=4000, string-sorted configs) and reproduce its published numbers exactly.
- Step 4 'P-both (B=100k)' separation is two-sided CI exclusion; the paper-exact line uses resolution_budget.py's one-sided lower-bound test. Tier sizes match.
- Step 5 bootstrap SE for the 19-source resolution uses B=20,000 (own seeds), not the paper's 4000.
- Step 6 adjacent count is the uncorrected CR2-icc count (9 pairs), with the pair order fixed to the full-data order.
- Bootstrap p-values and percentile intervals use the paper's conventions (iid / dataset resampling, shared resamples across pairs, index rule from run_audit.percentile_ci); `vernier_stats.pct_bootstrap` returns no draws so was not used for them.