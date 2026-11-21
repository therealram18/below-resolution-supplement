# TIME ("It's TIME", arXiv 2602.12147): leaderboard audit

Files: results dataset Real-TSF/TIME-Output at commit 3bf103c (2026-10-31), leaderboard code from the space at commit d83188a. Plan: boards/PREREG_BOARDS.md. Code: audit_time.py. Numbers: boards/time_report.json, boards/time_units.csv. Statistics run with vernier_board.py and vernier_stats.py unchanged. Resolution = the smallest true difference between two neighbours the test would detect 80% of the time. All geometric-mean rules are analysed on log scale (a gap of 0.002 is about 0.2%).

## Summary

| Rule | Entries | Units | Sources | Depends on pool? | Resolution paper / corrected | Median adjacent gap | Ratio (resolution / gap) paper / corrected | Adjacent separated of 9: paper / CR2-icc / wild | All 45 after Holm: paper / CR2-icc / wild | Leading tier size: paper / CR2-icc / wild | Known-shift detection at corrected resolution: paper / CR2-icc / wild | Removals / near-copies that reorder the top ten | History replay reversals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASE geometric mean vs seasonal naive (displayed sort) | 43 | 98 | 39 | no | 0.01921 / 0.02378 | 0.002088 | 9.2 / 11.4 | 1 / 0 / 1 | 7 / 0 / 7 | 1 / 7 / 2 | 85% / 73% / 86% | 0 of 32; 0 of 42 | 0 of 28 transitions, 0 pairs |
| MASE_rank (mean rank on MASE) | 43 | 98 | 39 | yes | 3.388 / 4.008 | 0.3776 | 8.97 / 10.6 | 1 / 1 / 1 | 17 / 0 / 12 | 4 / 4 / 4 | 87% / 74% / 82% | 0 of 33; 6 of 43 | 5 of 28 transitions, 6 pairs |
| CRPS geometric mean vs seasonal naive | 43 | 98 | 39 | no | 0.02376 / 0.03066 | 0.004653 | 5.11 / 6.59 | 1 / 0 / 1 | 15 / 1 / 12 | 1 / 4 / 1 | 90% / 82% / 92% | 0 of 32; 0 of 42 | 0 of 28 transitions, 0 pairs |
| CRPS_rank (mean rank on CRPS) | 43 | 98 | 39 | yes | 3.054 / 3.642 | 0.8163 | 3.74 / 4.46 | 1 / 0 / 0 | 22 / 3 / 18 | 1 / 2 / 2 | 88% / 78% / 83% | 1 of 33; 4 of 43 | 4 of 28 transitions, 4 pairs |
| MASE geometric mean, 8 uncertain-overlap datasets removed (sensitivity) | 43 | 81 | 31 | no | 0.02916 / 0.04132 | 0.002622 | 11.1 / 15.8 | 1 / 0 / 1 | 13 / 0 / 8 | 2 / 9 / 3 | 93% / 86% / 93% | not run | not run |

Reading it: the displayed rule orders the top ten by gaps of about 0.0021 in log units, while the smallest difference it could resolve is about 0.019 (paper) to 0.024 (corrected), so 9.2 to 11.4 times larger than a typical neighbour gap. At 39 datasets, a typical neighbour gap would need about 4217 datasets to be resolved. Under the paper rule 1 of 9 adjacent pairs and 20 of 45 top-ten pairs are separated (uncorrected; 7 after Holm); under CR2-icc 0 of 9 and 11 of 45 (0 after Holm). Leading tier (entries not separated from the displayed leader by an uncorrected test): 1 under the paper rule (TimeRSI-1.5), 7 under CR2-icc, 2 under the wild bootstrap. 
Uncorrected pairs of 45 (paper / CR2-icc / wild): MASE geometric mean 20 / 11 / 18; MASE_rank 24 / 22 / 23; CRPS geometric mean 25 / 14 / 22; CRPS_rank 26 / 24 / 24.

Known-shift control (resample datasets, add a shift equal to the stated resolution to an adjacent pair, 1000 repeats; false-alarm = shift of zero), MASE geometric mean: at the paper resolution the detection rates are paper 76%, CR2-icc 58%, wild 75%; at the corrected resolution 85%, 73%, 86%. False alarms at zero shift: paper 10.8%, CR2-icc 4.6%, wild 9.4%. The paper-rule false-alarm rate above 5% means that rule is too eager on this board; CR2-icc keeps it near 5% but then detects the shift less often at the paper resolution (the reason the corrected resolution is larger).

## What the board shows

- 43 entries are shown (all model folders, including seasonal_naive, which is the normaliser and sits last at 1.000). The board's code has no exclusion, flag or leakage column; every folder with results is shown (searched time_lb/src for leak, flag, exclude, hide).
- Default sort: MASE (norm.) ascending: for each (dataset/frequency, horizon) the mean of the MASE array (all series, windows, variates) divided by seasonal_naive's, then the geometric mean over the 98 units. MASE_rank and CRPS_rank are mean ranks over the same units (rank method "first"; no exact ties exist in MASE or CRPS, so row order does not matter).
- Units: 98 (dataset/frequency, horizon) combinations. Sources: 39 dataset names (first path component). Counting dataset and frequency together gives 50 (the TIME paper says 50 datasets; its count is by dataset and frequency).
- NaN cells in the normalised tables: 0.

## Reproduction check

- Board code run on the pinned files: the displayed order of all 43 entries from my log-mean equals the board function's order: True (CRPS order: True). Geometric mean from log-mean vs scipy gmean: max difference 0.0; rounded value difference 0.0; rank columns match to 4.9e-04 (display rounding).
- Top ten (pinned files): TimeRSI-1.5, PDM-Nano, Timestral-209M, Timestral-123M, TimeRSI, PDM-Pico, t0_gamma, Timestral-28M, QiYao-M, ChakraTS.
- Hart-36M shipped summaries: full_results.csv has 98 rows, 98 matched my units; max difference in MASE 0.0, CRPS 0.0, MAE 0.0 (exact). average_results.csv MASE 1.024171 vs my plain mean over units 1.024171; CRPS 0.196371 vs 0.196371. That file is the plain average of raw scores, not the board's displayed number (Hart-36M's displayed MASE (norm.) is 0.666).
- Live board: the HuggingFace space config, fetched 2026-11-11 (about 14:40 UTC), embeds a table of 47 entries, 43 on the pinned files. The extra entries are Simons-297M, Simons-178M, Simons-60M, Simons-25M, not in the pinned files; presumably added to the results repository after the pinned commit (the dataset's last-modified time was 2026-11-11 14:06 UTC; I did not fetch those files, and the space's table is whatever it computed when it last started). **Top ten does not match**: live has TimeRSI-1.5, PDM-Nano, Timestral-209M, Timestral-123M, TimeRSI, PDM-Pico, t0_gamma, Simons-297M, Timestral-28M, QiYao-M; pinned has TimeRSI-1.5, PDM-Nano, Timestral-209M, Timestral-123M, TimeRSI, PDM-Pico, t0_gamma, Timestral-28M, QiYao-M, ChakraTS; the difference is Simons-297M at place 8. For the 43 shared entries, order is identical: True, and MASE (norm.) and CRPS (norm.) agree to the displayed 3 decimals (max difference 0.0, 0.0). Rank columns differ as expected because the pool differs. Reported, not fixed.

## Single-dataset dominance

For each of the 45 top-ten pairs, the share of the variance estimate (sum over datasets of S_g squared, S_g = the dataset's summed deviation from the pair mean) coming from the largest single dataset:

| Rule | Median share | Maximum share | Pairs with one dataset above one half | Pairs above one third |
|---|---|---|---|---|
| MASE geometric mean vs seasonal naive (displayed sort) | 0.47 | 0.88 | 18 of 45 | 32 of 45 |
| MASE_rank (mean rank on MASE) | 0.29 | 0.64 | 7 of 45 | 19 of 45 |
| CRPS geometric mean vs seasonal naive | 0.42 | 0.87 | 17 of 45 | 37 of 45 |
| CRPS_rank (mean rank on CRPS) | 0.26 | 0.48 | 0 of 45 | 13 of 45 |

## Pool dependence

- Displayed geometric-mean rule: each entry's aggregate computed from the whole pool equals the aggregate from that entry and seasonal_naive alone (max difference 0.0; CRPS 0.0). Removing any of the 32 entries outside the top ten and adding a near-copy (score times 1+1e-6) of any of the 42 non-baseline entries reorders the top ten in 0 and 0 cases. Seasonal_naive itself is not in the perturbed pool: it is the fixed normaliser, and if it were removed the board's code skips normalisation and averages raw MASE instead.
- Mean-rank columns (pool dependent): MASE_rank: removal 0 of 33, near-copy 6 of 43 (near-copies of: TimeRSI, TimeRSI-1.5, TimesFM-3, Timestral-123M, Timestral-28M, t0_gamma). CRPS_rank: removal 1 of 33 (removing ChakraTS), near-copy 4 of 43 (near-copies of: PDM-Pico, QiYao-M, Timestral-123M, t0_gamma). Here the pool includes seasonal_naive as an ordinary entry, as the board's ranks do.

## History replay

The results repository has 45 commits that touch results/. A state is the set of model folders with metrics.npz at that commit; a model's content hash is the hash of its sorted metrics.npz LFS identifiers (sha256 of the file; I also checked that the sha256 of all 4214 local files equals the identifiers at the head commit: 0 mismatches, 0 files that were not LFS pointers). Scores come from the current files, so a model can be scored at a commit only if all 98 of its files at that commit are identical to the head files.

- Model-states (model present at a commit): 1135; scored 854; could not be scored 281 (238 content differs from the head, 43 folder name no longer exists at head, 0 incomplete).
- Models whose content changed while present: Chronos-2, Chronos-bolt, Eidos, Kairos, Moirai2, Moirai_base, PatchTST-FM-R1, Tabby, TiRex, Timer-S1, TimesFM-1.0, TimesFM-2.0, TimesFM-2.5, Toto, Toto-2.0-1B, Toto-2.0-2.5B, Toto-2.0-22m, Toto-2.0-313m, Toto-2.0-4m, chronos2_base, chronos_bolt_base, seasonal_naive, sundial_base, visiontspp_base. Names that no longer exist at head: PatchTST-FM-TsMix10M, ToTo, chronos2_base, chronos_bolt_base, kairos, moirai2, moirai_base.
- seasonal_naive itself changed on 2026-06-02 (the Crypto/D and Global_Influenza/W refresh), so the geometric-mean rule cannot be computed for the first 16 commits (2026-02-09 to 2026-06-02 before c43961bb), and no state before that has even ten scoreable entries. 29 of 45 states are replayable, giving 28 transitions (2026-06-02 to 2026-10-31). The replayed pool is smaller than the board's real pool at the time wherever models could not be scored (22 of 22 present at 2026-06-02 start vs 13 scored), which matters for the mean-rank columns.

| Rule | Transitions examined | Transitions with a reversal | Reversed pairs among top-ten entries that did not change |
|---|---|---|---|
| MASE geometric mean vs seasonal naive (displayed sort) | 28 | 0 | 0 |
| MASE_rank (mean rank on MASE) | 28 | 5 | 6 |
| CRPS geometric mean vs seasonal naive | 28 | 0 | 0 |
| CRPS_rank (mean rank on CRPS) | 28 | 4 | 4 |

MASE_rank events (date of previous state -> date of new state; pair order is previous order):
- 2026-06-02 8ec9db10 -> 2026-07-11 2ffa8721: reversed TimesFM-2.5 vs PatchTST-FM-R1; arrived or changed: OmniScient
- 2026-07-11 2ffa8721 -> 2026-07-14 42495a47: reversed PatchTST-FM-R1 vs TimesFM-2.5; arrived or changed: litespecformer
- 2026-07-14 42495a47 -> 2026-08-05 73a177cb: reversed TimesFM-2.5 vs PatchTST-FM-R1; arrived or changed: TS-ICL
- 2026-10-05 246dc3f1 -> 2026-10-09 4b07d9e5: reversed ChakraTS vs TimesFM-3, PatchTST-FM-Extended vs Chronos-2; arrived or changed: TimeRSI
- 2026-10-16 3d63e2a1 -> 2026-10-16 9a5e670d: reversed Timestral-5M vs Toto-2.0-1B; arrived or changed: Timestral-28M

CRPS_rank events:
- 2026-06-02 d086573e -> 2026-06-02 3963cf80: reversed Timer-S1 vs TiRex; arrived or changed: PatchTST-FM-R1
- 2026-06-02 8ec9db10 -> 2026-07-11 2ffa8721: reversed TiRex vs Timer-S1; arrived or changed: OmniScient
- 2026-09-30 66eea9da -> 2026-09-30 5787f898: reversed Toto-2.0-1B vs Tabby; arrived or changed: Tabby-Pretrain
- 2026-09-30 5787f898 -> 2026-10-05 246dc3f1: reversed Tabby vs Toto-2.0-1B; arrived or changed: ChakraTS

Geometric-mean rules: no reversals in either metric (as expected, no entry's aggregate depends on the others).

## Overlap with GIFT-Eval

GIFT-Eval has 28 dataset names (bitbrains_fast_storage, bitbrains_rnd, bizitobs_application, bizitobs_l2c, bizitobs_service, car_parts, covid_deaths, electricity, ett1, ett2, hierarchical_sales, hospital, jena_weather, kdd_cup_2018, loop_seattle, m4_daily, m4_hourly, m4_monthly, m4_quarterly, m4_weekly, m4_yearly, m_dense, restaurant, saugeen, solar, sz_taxi, temperature_rain, us_births). TIME's 39 datasets: Australia_Solar, Auto_Production_SF, CPHL, Coastal_T_S, Commodity_Import, Commodity_Production, Crypto, ECDC_COVID, EWELD_Load, Finland_Traffic, Global_Influenza, Global_Price, Housing_Inventory, JOLTS, Job_Claims, MetroPT-3, NE_China_Wind, Oil_Price, Online_Retail_2_UCI, OpenElectricity_NEM, Port_Activity, SG_Carpark, SG_PM25, SG_Weather, Smart_Manufacturing, Supply_Chain_Customer, Supply_Chain_Location, US_Labor, US_Term_Structure, Uncertainty_1M, Vehicle_Sales, Vehicle_Supply, WUI_Global, Water_Quality_Darwin, azure2019_D, azure2019_I, azure2019_U, current_velocity, epf_electricity_price.

- Sharing a data source (names, aliases, providers named in the TIME paper's appendix B): none found. TIME was built from fresh data; its names do not match any GIFT-Eval name.
- Uncertain (same topic as a GIFT-Eval dataset, provider appears different; the TIME paper names the providers, I did not inspect raw data): ECDC_COVID ~ covid_deaths (GIFT: JHU case counts; TIME: ECDC); Australia_Solar ~ solar (GIFT: Alabama solar plants; TIME: PVOutput stations in Australia); EWELD_Load ~ electricity (GIFT: UCI hourly consumption; TIME: Liu et al. 2023 industrial/commercial users); epf_electricity_price ~ electricity (price series, not load); SG_Weather ~ jena_weather / temperature_rain (weather, different providers); azure2019_D ~ bitbrains_* (cloud VM traces, different provider); azure2019_I ~ bitbrains_* (cloud VM traces, different provider); azure2019_U ~ bitbrains_* (cloud VM traces, different provider).
- Because no definite overlap exists, no overlap rerun was required. As a sensitivity, the primary report was rerun with the 8 uncertain datasets removed (81 units, 31 sources; displayed order recomputed on those units; top ten there: TimeRSI-1.5, PDM-Nano, Timestral-209M, Timestral-123M, TimeRSI, PDM-Pico, Timestral-28M, t0_gamma, QiYao-M, TimesFM-3): resolution 0.02916 / 0.04132, median gap 0.002622, adjacent separated 1 / 0 / 1, all 45 after Holm 13 / 0 / 8 (paper / CR2-icc / wild).

## Deviations and things that went wrong

1. The board's loader (utils.load_time_results) refuses to read a unit unless predictions.npz exists. Predictions (3.9 GB) were not downloaded, so I ran the board's own get_all_datasets_results, compute_ranks, normalize_by_seasonal_naive and get_overall_leaderboard unchanged, after replacing only that loader by a version that reads metrics.npz alone. The repository lists predictions.npz for all 4214 units at the head commit, so the live board should have the same set. Packages not installed here (gradio, timebench) and src.about/src.hf_config (which download data at import) were replaced by empty stand-ins; none of their code is used in the reduction.
2. Row order of the board's table comes from os.listdir on this machine; it only affects rank ties, and there are none.
3. Seasonal_naive was left out of the perturbed pool in the geometric-mean pool test (reason above); in the history replay and in the mean-rank pool test it is an ordinary entry.
4. History replay: the first 16 states cannot be scored (see above), and unscoreable model-states are dropped from the replayed pool, so the mean-rank replay is not the board's table as displayed at the time.
5. Live table: pinned top ten differs from live by the four entries added later (see reproduction check).
6. vernier_board.py (shared, not modified) defines _display_order, pool_tests, history_replay and mean_rank_rule twice in a row and has one unreachable line after mean_rank_rule's return; harmless, but worth cleaning.
7. Secondary rules use the rank columns as the board computes them (ties impossible here) and are ordered by their own mean; the board's default order is the geometric mean.
8. Shared-code check required by the plan: test_vernier_board.py was run after the board analysis and before reporting (about 8 minutes; the plan says it should come first, so this is a deviation in order) and printed ALL OK (GIFT-Eval: 0 of 9 adjacent, 12 of 45, 2 after Holm, tier of 3, 28 of 121 removals, 42 of 128 near-copies, replay 141 / 18 / 14). Its paper resolution came out 6.209 against the published 6.16, inside the test's tolerance of 0.1; the test regenerates gift_mase_rank_report.json, which I restored from a copy afterwards.
