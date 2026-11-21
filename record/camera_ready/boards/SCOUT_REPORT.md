# Scout report: forecasting boards with per-(entry, unit) public scores

Scouted 2026-11-11. All counts below were computed from the downloaded files (python csv, wc, git log), except where marked "from API listing" or "uncertain".
Local root: fmts/camera_ready/boards/

Summary

| Board | Verdict | Entries | Units | Board sorts by | History |
|---|---|---|---|---|---|
| fev-bench (autogluon/fev) | FEASIBLE | 52 | 100 tasks (96 distinct dataset configs) | win_rate, descending | yes, 32 commits on results |
| fev-bench-mini | FEASIBLE (subset of above) | same 52 files, 20 tasks | 20 | win_rate | same repo |
| Chronos Benchmark II zero-shot (autogluon/fev) | FEASIBLE | 21 | 27 | no table shown by the repo (see entry) | yes, 14 commits on results |
| Chronos Benchmark I in-domain | NOT FEASIBLE | - | - | - | not in the repo, not in its history |
| BOOM (DataDog/toto repo, boom/) | FEASIBLE | 23 | 7413 rows (2807 series x terms) | mean per-unit Rank, ascending | only 4 commits |
| TIME (Real-TSF/TIME-Output + leaderboard space) | FEASIBLE with caveat (per-window arrays, not a ready table) | 43 | 98 (dataset/freq/term) | MASE (norm.) ascending | yes, 45 commits touching results, 50 total |
| OpenTS / TFB | NOT FEASIBLE | - | - | - | - |
| ProbTS | NOT FEASIBLE | - | - | - | - |
| FoundTS | NOT FEASIBLE (repo not found) | - | - | - | - |
| Time-Series-Library | NOT FEASIBLE | - | - | - | - |
| Monash archive (rakshitha123/TSForecasting) | NOT FEASIBLE | - | - | - | - |
| TS-Arena | NOT FEASIBLE under download budget (data exists) | - | - | - | - |
| LiveHouse-TS, Impermanent | NOT FOUND | - | - | - | - |

---------------------------------------------------------------------------

## 1. fev-bench (and fev-bench-mini)

- URLs: https://github.com/autogluon/fev ; hosted board https://huggingface.co/spaces/autogluon/fev-bench (the README also links /fev-leaderboard, which redirects there); paper arXiv 2509.26468.
- Downloaded: full-history clone, commit 9cf943c388e5cfaf352175ea3c69e408b197d997, dated 2026-11-11. 145 commits in total.
- Local path: fev/benchmarks/fev_bench/results/*.csv (one CSV per model); tasks: fev/benchmarks/fev_bench/tasks.yaml; mini subset: fev/benchmarks/fev_bench/tasks_mini.yaml (20 tasks, counted by `dataset_config` lines); model metadata with a `hidden: true` flag: fev/benchmarks/fev_bench/models.yaml (13 hidden entries).
- Format: CSV, one row per task per model. Header (first columns): `task_name,model_name,model_class,dataset_path,dataset_config,horizon,num_windows,initial_cutoff,window_step_size,min_context_length,max_context_length,seasonality,eval_metric,extra_metrics,quantile_levels,...,test_error,SQL,MASE,WAPE,WQL,num_forecasts,fev_version,training_time_s,inference_time_s,trained_on_this_dataset,dataset_fingerprint`.
  Sample rows (TimeRSI.csv): `proenfo_gfc17,TimeRSI,timersi,autogluon/fev_datasets,proenfo_gfc17,168,20,...,0.3653806669537396 (test_error=SQL),MASE 0.4611...,WAPE 0.0294...,WQL 0.0233...,num_forecasts 160,...,trained_on_this_dataset False` and `entsoe_15T,TimeRSI,...,96,20,...,SQL 0.3749851689896301,...,num_forecasts 120`.
- Entries: 52 CSV files. Coverage per file: 47 files cover all 100 tasks; 2 files cover 97, 2 cover 96, 1 covers 99 (missing tasks exist; the board's `missing_strategy` handles them).
- Evaluation unit: one task = a dataset config plus horizon, window count, covariate setup. 100 tasks, 96 distinct `dataset_config` values (so a few configs appear in several tasks).
- Independent sources: the `dataset_config` column and the `dataset_path` column identify the data source; tasks sharing a `dataset_config` come from one source. Beyond that, groupings by domain are in the paper, not in the files (uncertain whether the repo has a domain column; I did not find one in the results CSV).
- Metrics per unit: SQL (scaled quantile loss, the `eval_metric`/`test_error`), MASE, WAPE, WQL; also runtime and a `trained_on_this_dataset` leakage flag.
- Rule the board sorts by: `fev/src/fev/analysis.py`, function `leaderboard`, ends with `return result_df.sort_values(by="win_rate", ascending=False)`; the docstring says "Leaderboard sorted by `win_rate`". (Its first docstring line says "Models are ranked by skill score", which contradicts the code; the code wins.) The fev-bench figure script sets `SORT_COL = "win_rate"` (fev/scripts/generate_fev_bench_figures.py line 59), baseline "Seasonal Naive", leakage replacement model "Chronos-Bolt", metrics SQL and MASE. Win rate = fraction of pairwise comparisons won; skill score = 1 - geometric mean of errors relative to the baseline, relative error clipped to [0.01, 100].
- Aggregated table available for checking: not as a file in the repo. It is recomputed by `fev.leaderboard(...)` / `scripts/render_fev_bench_leaderboard.py`, and the live table is on the HF space. I did not download the space table. Reproduction check against the live page is possible; against a file, not.
- History: yes. 32 commits touch benchmarks/fev_bench/results (first 2025-10-01 "Add task definitions and results for fev_bench (#52)", latest 2026-11-11). File additions by date (git name-status; 56 file additions over time, 52 files at HEAD, so some were renamed or removed): 2025-10-01 1; 2025-10-15 14; 2025-10-20 1; 2026-02-19 2; 2026-05-19 4; then one to five per submission through 2026-11-11 (5 on that day).
- License: Apache-2.0 (fev/LICENSE).
- Verdict: FEASIBLE.

## 2. Chronos Benchmark II (zero-shot)

- Same repo/commit as above. Path: fev/benchmarks/chronos_zeroshot/results/*.csv ; definitions: fev/benchmarks/chronos_zeroshot/tasks.yaml (27 tasks).
- Format: CSV, one row per task. Header: `model_name,dataset_path,dataset_config,horizon,num_windows,initial_cutoff,window_step_size,min_context_length,max_context_length,seasonality,eval_metric,extra_metrics,quantile_levels,...,task_name,test_error,training_time_s,inference_time_s,dataset_fingerprint,trained_on_this_dataset,fev_version,MASE,WQL,instance_type`.
  Sample (auto_arima.csv): `auto_arima,autogluon/chronos_datasets_extra,ETTh,24,1,-24,24,...,ETTh,0.8618277884304452,,444.0,...,False,0.6.0,MASE 0.8618,WQL 0.0890,ml.m6i.4xlarge` and `...,ETTm,...,MASE 1.0822,WQL 0.1050`.
- Entries: 21 files (auto_arima, auto_ets, auto_theta, seasonal_naive, chronos_{tiny,mini,small,base,large}, chronos_bolt_{tiny,mini,small,base}, moirai_{small,base,large}, timesfm, timesfm-2.0, tirex, ttm-r2, timecopilot). Every file has all 27 tasks (checked in python).
- Units: 27 tasks. 25 from dataset_path autogluon/chronos_datasets, 2 (ETTh, ETTm) from autogluon/chronos_datasets_extra. Task names show families: 17 are Monash datasets (monash_* prefix), 3 are m4/m5-type (m4_quarterly, m4_yearly, m5), others ERCOT, exchange_rate, dominick, nn5, ETT. Same-family tasks (monash_m3 monthly/quarterly/yearly, m1 three ways, tourism three ways, m4 two ways) come from one source.
- Metrics: MASE, WQL (point and probabilistic), plus runtimes.
- Rule: the repo does not define a displayed sort for this board. README only says "benchmark definitions for the 27 datasets used in Benchmark II"; and root README line 127 says results are "available in benchmarks/chronos_zeroshot/". The Chronos paper (arXiv 2403.07815) reports aggregate relative score (geometric mean of errors relative to seasonal naive) and win rate; I did NOT verify the paper here. The library default for `fev.leaderboard` is win_rate (see above). So for this board the "displayed rule" has to be taken from the paper, uncertain.
- Aggregated table: none in the repo.
- History: 14 commits touch results; 21 file additions: 2024-12-09 1, 2024-12-10 16, 2025-03-05 1, 2025-05-15 1, 2025-05-28 1, 2025-09-10 1.
- License: Apache-2.0.
- Verdict: FEASIBLE (small: 21 x 27).

## 2b. Chronos Benchmark I (in-domain)

Not in autogluon/fev: only chronos_zeroshot, example and fev_bench exist under benchmarks/, and `git log --all -- 'benchmarks/chronos_indomain*'` returns nothing. NOT FEASIBLE from this source. Not searched elsewhere.

## 3. BOOM (Datadog)

- URLs: HF space https://huggingface.co/spaces/Datadog/BOOM (aggregates only), and the repo with per-model files https://github.com/DataDog/toto (folder boom/). Dataset https://huggingface.co/datasets/Datadog/BOOM (data only, no results).
- Downloaded: toto repo full clone, commit 1d3e1c97ccf615016d38fe910602c905ae31fbb7 (2026-10-26); space clone (full history) 2bc23d1a3eeebb7ac73a610afcb5db9470c72552 (2026-05-12).
- Local paths: toto_repo/boom/results/<model>/all_results.csv ; toto_repo/boom/leaderboards/{boom,boomlet}_leaderboard.csv ; toto_repo/boom/breakdown_tabels/ ; toto_repo/boom/boom_properties.json, boomlet_properties.json (domain, frequency, term, type, num_variates per series); space: boom/results/leaderboards/*.csv.
- Format: CSV, GIFT-Eval-like. Header: `dataset,model,checkpoint,context_length,num_samples,eval_metrics/MSE[mean],eval_metrics/MSE[0.5],eval_metrics/MAE[0.5],eval_metrics/MASE[0.5],eval_metrics/MAPE[0.5],eval_metrics/MAAPE[0.5],eval_metrics/sMAPE[0.5],eval_metrics/MSIS,eval_metrics/RMSE[mean],eval_metrics/NRMSE[mean],eval_metrics/ND[0.5],eval_metrics/mean_weighted_sum_quantile_loss,domain,num_variates`.
  Sample (toto): `ds-860-T/Short/short,toto,...,2048,256,17.97...,10.29...,0.115...,MASE 1.0606,...,mean_weighted_sum_quantile_loss 0.6757,['Application Usage'],1`.
- Entries: 23 model folders (verified 23 files each with 7413 rows). Includes seasonalnaive, which the leaderboard code asserts must be present.
- Units: one row = one (series id, term) such as `ds-860-T/Short/short`: 7413 rows per model; 2807 distinct series ids; term counts short 2807, medium 2303, long 2303.
- Independent sources: the series id `ds-N-...` is the series; the file `boom_properties.json` gives domain (Application usage, Database, Infrastructure, Networking, Security; a series may carry several), type (count, gauge, rate, distribution), term. All series come from one company's observability data, so groups at the domain level are not independent in the sense of separate datasets (uncertain; no finer group key exists in the files). A series appears up to three times (terms), so rows are not independent units. A 'boomlet' subset (boomlet_properties.json) is also defined.
- Metrics: MASE, CRPS-like `mean_weighted_sum_quantile_loss`, MAE, MSE, RMSE, MAPE, sMAPE, MSIS, ND.
- Rule the board sorts by: toto_repo/boom/utils/leaderboard.py, function `get_leaderboard`: `combined_df["rank"] = combined_df.groupby("full_dataset_name")[ranking_metric].rank(method="first", ascending=True)` with default `ranking_metric="eval_metrics/mean_weighted_sum_quantile_loss"`, then mean rank per model and `.sort_values(by="rank", ascending=True)`. The MASE and CRPS columns shown are shifted geometric means (epsilon 1e-5), scaled by the seasonal-naive results for non-zero-inflated series. Space app.py (`init_leaderboard`) sorts by the `Rank` column ascending; src/about.py says Rank "determines overall performance and is used to order models on the leaderboard".
  Note: `method="first"` breaks ties by row order, a detail that may matter for reproduction.
- Aggregated table: yes. toto_repo/boom/leaderboards/boom_leaderboard.csv (columns `model,MASE-6750-scaled,CRPS-6750-scaled,Rank-6750-scaled,eval_metrics/MAE[0.5]-663-unscaled,CRPS-663-unscaled,Rank-663-unscaled`; 6750 scaled and 663 unscaled/zero-inflated series are split, ranks are computed separately) and boomlet_leaderboard.csv. Reproduction must split the series the same way (see utils/leaderboard.py, `separate_zero_inflated_data`). Not yet attempted by me.
- History: weak. In toto repo 4 commits touch boom/results (2025-05-28 breakdown tables, 15 model files; 2026-04-23 8 files; 2026-05-11 5; 2026-05-12 5). The space repo has 41 commits but holds only aggregates.
- License: Apache-2.0 (toto LICENSE; the space card says apache-2.0).
- Verdict: FEASIBLE; entries 23 but only 4 history points.

## 4. TIME ("It's TIME", arXiv 2602.12147)

- URLs: leaderboard space https://huggingface.co/spaces/Real-TSF/TIME-leaderboard (clone: time_lb/, commit d83188a4c48fcf645dfa76c39d6a8236953f9c63, 2026-03-05, 7 commits); results dataset https://huggingface.co/datasets/Real-TSF/TIME-Output (commit 3bf103cb8c2b418c4a8b17bda21b425034dc869c, 2026-10-31); data https://huggingface.co/datasets/Real-TSF/TIME (cc-by-nc-4.0, not downloaded). Space code reads results from the dataset (src/hf_config.py).
- Local paths: time_output/results/<model>/<dataset>/<freq>/<term>/{metrics.npz,config.json}; per-model summary csv only exists for Hart-36M (results/Hart-36M/full_results.csv, average_results.csv). I downloaded only metrics.npz, config.json and README/csv/jsonl files (about 280 MB of metrics; skipped predictions.npz, about 3.9 GB). time_output_history/ is a no-checkout clone with full git history (LFS pointers only).
- Format: for each (model, dataset/freq/term): metrics.npz holds arrays shaped (num_series, num_windows, num_variates) per metric (shape from the leaderboard code, src/leaderboard.py around line 253; I could not open the npz because numpy is not installed in the system python). config.json lists `dataset_config` ("SG_Weather/D/short"), num_series, num_windows, num_variates, prediction_length, quantile_levels, freq, seasonality, context_length, metric_names (MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS). Hart-36M csv sample: `Water_Quality_Darwin/15T/short,Hart-36M,2.0827...,0.4305...,0.5170...,0.0535...,0.0507...,MASE 0.6187...,ND 0.0516...,CRPS 0.0410...`. The board has to aggregate windows, series and variates itself; the exact aggregation to a (model, unit) score is in time_lb/src/leaderboard.py and about.py (`get_all_datasets_results`); I have not replayed it.
- Entries: 43 model folders at HEAD (from the API tree listing, all 43 with 98 metrics.npz each). 
- Units: 98 (dataset/freq/term) combinations over 39 datasets; terms: short 2150, medium 1032, long 1032 files over all models (so 50 short, 24 medium, 24 long units per model, from 2150/43, 1032/43). Leaderboard unit is (dataset_id = dataset/freq, horizon).
- Independent sources: the dataset name (39 datasets); the 98 units are (dataset, freq, horizon) so units of the same dataset are strongly dependent. Readable straight from the path.
- Metrics per unit: MASE, CRPS, MAE, MSE, RMSE, MAPE, sMAPE, ND.
- Rule the board sorts by: time_lb/src/leaderboard.py `get_overall_leaderboard(df_datasets, metric="MASE")`: normalises MASE and CRPS by seasonal_naive per (dataset_id, horizon), aggregates by geometric mean, and `leaderboard.sort_values(by=sort_metric, ascending=True)` where sort_metric is "MASE (norm.)" by default. The tab text (src/tab.py ~line 739-740) says MASE/CRPS (norm.) are geometric means of normalised task results and MASE_rank / CRPS_rank are mean ranks across tasks.
- Aggregated table available: only through the live space (not downloaded); reproduction can be checked against the page.
- History: yes, good. The dataset repo has 50 commits, 45 touching results/. First-seen date per model from git: 13 models on 2026-02-09 (TiRex, TimesFM 1.0/2.0/2.5, ToTo, chronos2_base, chronos_bolt_base, kairos, moirai2, moirai_base, seasonal_naive, sundial_base, visiontspp_base), then additions through 2026-10-31 (PDM-Femto/Nano/Pico). Some names were renamed or dropped (git shows 47 names ever added, 43 at HEAD).
- License: TIME-Output apache-2.0 (card); TIME data cc-by-nc-4.0.
- Verdict: FEASIBLE with extra work (needs numpy to reduce arrays; 280 MB local).
- Download status: complete, 4214 metrics.npz and 4214 config.json on disk (43 models x 98 units), 287 MB.

## 5. Others checked

- OpenTS / TFB (decisionintelligence): the site repo github.com/decisionintelligence/OpenTS (cloned, then deleted; HEAD a67d77d, 2026-10-20) holds only leaderboard.js / table.js front-end code whose data comes from the backend https://opents.mozie.top (a POST API; GET on /rank returns 405). The TFB repo has no results files (only demo data). NOT FEASIBLE: no static per-(model, dataset) file.
- ProbTS (microsoft/ProbTS): no results CSV/JSON in the 317-file tree; results are only markdown tables in docs/benchmark. NOT FEASIBLE.
- FoundTS: github.com/decisionintelligence/FoundTS returns 404. NOT FEASIBLE (not found).
- Time-Series-Library (thuml): 368 files, no results file. NOT FEASIBLE.
- Monash archive code (rakshitha123/TSForecasting): 17 files, no results. The Monash per-dataset errors live in the paper/Zenodo, not checked. NOT FEASIBLE from this repo.
- TS-Arena (Paderborn, arXiv 2512.20761): live board https://ts-arena.live ; archive https://huggingface.co/datasets/DAG-UPB/TS-Arena-Archive (cc-by-4.0, about 3.6 GB of parquet, sha 72c6cb0f, 2026-06-18) and https://huggingface.co/datasets/DAG-UPB/ts-arena-context-mase-splitted (about 1.6 GB). Per-forecast MASE exists but far over the 500 MB budget and not inspected. NOT FEASIBLE under the budget; could be revisited with a partial download (uncertain whether a per-model score file exists inside).
- LiveHouse-TS, Impermanent: web search and GitHub/HF probes found nothing. NOT FOUND.

## 6. Workshop camera-ready deadline

From https://fmts-workshop.github.io/ (index page, Important dates), verbatim table cell: "Camera-ready November 20, 2026 11:59 pm AoE · upload via OpenReview". The cfp.html page repeats the same line. Submission deadline listed as September 16, 2026 11:59 am UTC. (Workshop date December 11, 2026.)
