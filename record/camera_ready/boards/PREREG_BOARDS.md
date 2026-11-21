# Boards, units, sources and checks, fixed before analysis (2026-11-11)

Written after the scout listed what each board publishes (file layouts and column names
only) and before any score on these boards was analysed. A later change needs a dated note.

## Boards analysed (every board whose per-entry, per-unit files could be obtained)

| Board | Files (commit) | Entries | Units | Sources (clusters) | Rule the board sorts by | Depends on the pool? |
|---|---|---|---|---|---|---|
| GIFT-Eval (submitted paper) | snapshot b61c48aa | 130 | 97 configs | 28 datasets | mean rank on MASE | yes |
| fev-bench, current | autogluon/fev 9cf943c | models with complete results that the board does not hide | 100 tasks | task families | win rate | yes |
| fev-bench-mini | same, `tasks_mini.yaml` | same models | 20 tasks | families among the 20 | win rate | yes |
| Chronos Benchmark II | same repo, `chronos_zeroshot` | 21 | 27 tasks | families (below) | no table in the repo; see below | depends on rule |
| BOOM | DataDog/toto 1d3e1c9, `boom/` | 23 | 7,413 (series, horizon) rows | see below | mean rank on the probabilistic loss | yes |
| TIME | Real-TSF/TIME-Output 3bf103c | 43 | 98 (dataset, frequency, horizon) | 39 datasets | geometric mean of MASE scaled by seasonal naive | no |

Excluded because no per-entry, per-unit file is published: Chronos Benchmark I, OpenTS/TFB
(scores served by a private API), ProbTS (markdown tables only), FoundTS (repository gone),
Time-Series-Library, Monash archive repository. TS-Arena's archive exists but is 3.6 GB and
was not inspected. LiveHouse-TS and Impermanent could not be found. No board was or will be
dropped because of what its numbers show.

## Sources (cluster definitions)

- **fev-bench and fev-bench-mini:** family = task name with its last underscore-separated
  token removed. This is the definition already used in the submitted paper (39 families on
  the 100 tasks).
- **Chronos Benchmark II:** tasks that are the same collection at different frequencies form
  one source: `monash_m1_*` -> m1, `monash_m3_*` -> m3, `monash_tourism_*` -> tourism,
  `m4_*` -> m4, `ETTh` and `ETTm` -> ETT. Every other task is its own source. The exact map
  is written from the task names alone and saved before scores are read.
- **BOOM:** every series comes from one company's monitoring data, and the files carry no
  dataset key. Three levels, all reported: (a) the series (a series appears under up to
  three horizons, so the horizons of one series are one cluster); this is the most
  favourable reading, treating each series as independent evidence; (b) domain-and-type
  combination from `boom_properties.json`; (c) domain alone (a series with several domains
  uses the full combination as its key). Level (a) is primary because it is the unit the
  board itself averages over; (b) and (c) show how much rests on treating series as
  independent.
- **TIME:** the dataset name (first path component of the unit), 39 datasets.

## Rules

- A board's own sort rule is primary and is reproduced with the board's own code or an exact
  reimplementation. Reproduction tolerance: identical order for the displayed top ten, and
  aggregate values within 1e-6 relative where a published table file exists (BOOM's
  `boom_leaderboard.csv`). Where the only published table is a live page (fev-bench, TIME),
  the board's own code is run on the pinned files and the live page is compared if it can be
  fetched; a mismatch is reported, not hidden.
- **BOOM** splits series into 6,750 scaled and 663 zero-inflated ones and ranks them
  separately; the split and the displayed rank column follow `boom/utils/leaderboard.py` and
  the space's `app.py` exactly. The primary analysis uses the rank column the space sorts by.
- **Chronos Benchmark II** has no displayed table in the repository. Three rules are
  reported, none treated as "the" displayed one: geometric mean of error relative to seasonal
  naive (MASE, and WQL), mean rank, and win rate (the fev library default).
- **Secondary rules** reported where the board publishes them: skill score (fev-bench), mean
  rank columns (TIME), MASE-based rank (BOOM).

## What is computed for every board (same code for all)

1. Entries, units, sources; reproduction check of the displayed order.
2. Resolution: the paper's 2.80 x clustered bootstrap standard error, and the corrected
   version; the median gap between displayed neighbours; their ratio; sources needed.
3. Adjacent pairs separated (of 9) and all top-ten pairs (of 45), uncorrected and after Holm,
   under the paper's rule and the corrected methods; leading tier.
4. Known-shift control: resample sources, add a shift equal to the board's resolution to an
   adjacent pair, report the detection rate (about 80% expected).
5. Pool dependence: removing one entry from outside the top ten; adding a near-copy of an
   entry; history replay where a version history of per-entry files exists (fev-bench: 32
   commits; TIME: 45; Chronos Benchmark II: 14; BOOM has only 4, reported as such).
6. Overlap with GIFT-Eval's 28 datasets, by name, and the same report on the non-overlapping
   units.

The shared code is first checked against the submitted paper's GIFT-Eval numbers
(resolution 6.16, median gap 0.91, 0 of 9, 12 of 45, leading tier of 3, 28 of 121 removals
and 42 of 128 near-copies reordering the top ten). If that check fails, nothing else runs.
