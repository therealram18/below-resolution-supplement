# Result-1 audit — Day 1 (snapshot commit b61c48aa3197a1a111e405c0e6f600128a1efee3, 2026-09-07)

This is a summary of numbers only. No interpretation beyond what is computed.

## Displayed top-10 (sorted: Test Leak. then MASE_Rank, matching the live app)

All 10 are agentic-type models with no test-data leakage flag.

| # | model | MASE_Rank | gmean | replication code |
|---|---|---|---|---|
| 1 | STRIDE w/ Synapse | 14.62 | 0.625 | No |
| 2 | EXAONE-Forecast-Agent | 18.87 | 0.610 | No |
| 3 | LS-MoE | 19.31 | 0.650 | No |
| 4 | LS-Agent | 21.25 | 0.655 | No |
| 5 | Falcon-Agent | 22.15 | 0.666 | No |
| 6 | CastStar | 22.23 | 0.655 | No |
| 7 | limix_moe | 22.24 | 0.659 | No |
| 8 | TIMEHEDGE | 23.66 | 0.650 | No |
| 9 | RegLab-AH-1 | 24.31 | 0.665 | No |
| 10 | RacineCast-1-NC | 25.64 | 0.672 | Yes |

## Headline survival counts (95% bootstrap CI excludes 0)

- MASE_Rank, iid bootstrap: 1 of 9 adjacent gaps survive; 15 of 45 all-pairs gaps survive.
- MASE_Rank, cluster bootstrap: 1 of 9 adjacent gaps survive; 14 of 45 all-pairs gaps survive.
- gmean, iid bootstrap: 3 of 9 adjacent gaps survive; 28 of 45 all-pairs gaps survive.
- gmean, cluster bootstrap: 3 of 9 adjacent gaps survive; 25 of 45 all-pairs gaps survive.

Under the paper's pre-committed rule (a gap is real only if it clears BOTH schemes):

- **MASE_Rank (the displayed ranking): 0 of 9 adjacent gaps; 12 of 45 all pairs.** The one iid survivor (#1→#2, lower bound 0.04) fails clustered; the one clustered survivor (#9→#10) fails iid (lower bound −0.08) — different pairs, neither clears both.
- gmean: 2 of 9 adjacent (TIMEHEDGE→RegLab-AH-1, RegLab-AH-1→RacineCast-1-NC); 23 of 45 all pairs.

Independent verification (main session, 2026-09-07): #9→#10 pair reproduced from raw CSVs with a separate implementation — gap 1.33, clustered CI [0.11, 2.41] vs the script's [0.14, 2.42].

## Reproducibility census

- All 130 models: replication code "No" for 55; leakage "Yes" for 17; agentic: 31 (deep-learning:10, fine-tuned:6, pretrained:35, statistical:6, zero-shot:42).
- Displayed top-10: replication code "No" for 9; leakage "Yes" for 0; agentic: 10 of 10.

## Aggregation disagreement (MASE_Rank vs gmean)

- Top-10 overlap: 8 of 10 models in common between the two aggregation rules.
- 3 largest rank swings across all displayed models (rank under MASE_Rank vs rank under gmean):
  - TurkForecast-FM-Chronos2-LoRA-v1: 43 vs 7 (diff 36)
  - Falcon-X: 54 vs 33 (diff 21)
  - ForecastMate: 24 vs 9 (diff 15)

## Data-quality notes

- 130 model directories; 129 appear on the displayed leaderboard (the app drops any model whose config.json name doesn't match its results-file name).
- Fewer than 97 configs, excluded from pairwise resampling: GestaltCog/Zeus-100M (14/97), Zeus (83/97).
- Dropped by name mismatch: ['GestaltCog/Zeus-100M', 'iTransformer'] had no matching config; ['i_transformer'] had no matching results.
