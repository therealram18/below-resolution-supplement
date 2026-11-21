# Camera-ready additions

Everything outside this folder is the supplement of the submitted paper, unchanged (see `../README.txt`).
This folder holds what the camera-ready version added, in three rounds of work:

1. Small-sample corrections and other boards: the statistics tests were rerun with corrections for few datasets,
   and the same report was run on further leaderboards (fev-bench, BOOM, TIME, GIFT-Eval sub-boards).
2. Exact tests, reference checks and the geometric-mean test: exact sign-flip tests over all sign patterns,
   checks of our formulas against reference packages (R `dfadjust`, Python `wildboottest`), and a test of the
   geometric mean of scaled error.
3. The history-length experiment: models run on 70 configurations with the documented history length and with
   512 steps, including a reproduction of Toto.

## Folder map

| Folder | What it holds |
|---|---|
| `paper/` | `main.tex` (the camera-ready source), the two bibliographies, the style file, the three figure PDFs, and `figs/make_fig_resolution.py` with its fact sheet `figs/fig_resolution_facts.md` |
| `analysis/` | Scripts (`*.py`, `refcheck.R`), result files (`*.json`, `*.csv`, `*.npz`) and the write-ups (`*.md`) of rounds 1 and 2 |
| `analysis/boards/` | Our reports on the other boards: the pre-registration (`PREREG_BOARDS.md`), the scouting report, the second pin (`PIN_2026-11-13.md`) and our result `*.json` files, including intermediate parts (`boom_parts/`, `fev_family_parts/`, `s7_parts/`, `_time_cache/report_*.json`) and our summaries of the second pin (`pin_2026-11-13/*.json`). The boards' own repositories and data are not included; the pins name the commits used. |
| `history_length/` | Round 3: design and review (`DESIGN_ROUND3.md`, `REVIEW_ROUND3.md`), the analysis script and its tests, the final results (`analysis_final/`), the reviewers' check scripts (`review_scratch/`), the run harness (`harness/`: scripts, job lists, pinned requirements, data checksums) and the raw per-run results (`pulled/harness/results/`, one CSV and one `.run_meta.json` per run, plus `.failed.run_meta.json` for failed attempts; `pulled/harness/status.jsonl` logs every attempt) |

## Where each table and figure comes from

Mapping made from the file names and write-ups; paths are relative to this folder. "Submitted supplement" means the
folders outside `camera_ready/`.

| Label in `paper/main.tex` | Values | Made by |
|---|---|---|
| `tab:report` (Vernier on five boards) | `analysis/BOARDS_TODAY.md`, `analysis/retest_gift.json`, `analysis/retest_fev.json`, `analysis/boards/*_report.json` | `analysis/vernier_board.py`, `analysis/vernier_stats.py`, `analysis/retest_*.py`, `analysis/audit_*.py` |
| `fig:gaps` (adjacent gaps, GIFT-Eval and fev-bench) | `paper/figs/fig_resolution_facts.md` | `paper/figs/make_fig_resolution.py` |
| `tab:protocol` (geometric mean at 512 steps vs documented) | `history_length/analysis_final/GPU_ROUND_RESULTS.md` and `.json` | `history_length/analyze_gpu_round.py` |
| `tab:fields` (five disclosure fields) | definitions, no data file | none |
| `tab:calib` (false-difference rate of each method) | `analysis/CALIBRATION.md`, `analysis/calibration_results.json`, `analysis/calib_d4_data.npz` | `analysis/calibration_sim.py` |
| `tab:gmeanlevel` (tests of the geometric mean on 45 pairs) | `analysis/GMEAN_LEVEL_CHECK.md`, `analysis/gmean_level_check.json`, `analysis/gmean_level_recheck_main.json`, `analysis/gmean_level_data.npz` | `analysis/gmean_level_check.py` |
| `tab:adjacent` (nine neighbouring pairs) | `analysis/PAIR_TABLES.md`, `analysis/pair_table_*.csv` | `analysis/make_pair_tables.py` |
| `tab:counts`, `tab:resolution`, `tab:tiers` (GIFT-Eval by rule and test) | `analysis/RETEST_GIFT.md`, `analysis/retest_gift.json`, `analysis/exact_gift.json` | `analysis/retest_gift.py`, `analysis/run_exact_gift.py`, `analysis/retest_gift_md.py`, `analysis/exact_signflip.py` |
| `tab:replay` (20 reversed untouched pairs) | `analysis/reversals_exact_tiebreak.csv`, `analysis/reversals_ties_averaged.csv`, `analysis/verify_pin2_reversal.json`, `analysis/boards/pin_2026-11-13/gift_replay_after_b61c48aa.json` | `analysis/list_reversals_exact.py`, `analysis/verify_pin2_reversal.py`, `analysis/pin2_replay.py` |
| `fig:timeline` (GIFT-Eval updates that reversed pairs) | submitted supplement | `../code/figures/make_timeline.py` |
| `tab:perturb` (three entries with and without three pool members) | not traced to a file in this folder; see `../README.txt` | see `../README.txt` |
| `tab:boards` (the report on every board) | `analysis/BOARDS_TODAY.md`, `analysis/CROSS_BOARD_SPREAD.md`, `analysis/boards/*_report.json`, `analysis/boards/cross_board_spread.json` | `analysis/audit_*.py`, `analysis/pin2_boards.py`, `analysis/cross_board_spread.py` |
| `tab:timehist` (TIME over its whole history) | `analysis/BOARD_TIME_FULL_HISTORY.md`, `analysis/boards/time_full_history.json` | `analysis/time_prerefresh_*.py` |
| `tab:subboards` (GIFT-Eval sub-boards) | `analysis/BOARD_GIFT_SUBBOARDS.md`, `analysis/boards/gift_subboards_report.json`, `analysis/boards/gift_subboards_groups.json` | `analysis/audit_gift_subboards.py` |
| `tab:g1`, `tab:g2`, `tab:toto` (history-length experiment, newer models, Toto) | `history_length/analysis_final/GPU_ROUND_RESULTS.md` and `.json`, raw runs in `history_length/pulled/harness/results/` | `history_length/analyze_gpu_round.py`; runs by `history_length/harness/run_grid.py` and `dispatch.py` |
| `tab:grid` (submitted version's runs, kept as a record) | submitted supplement (`../results/`) | see `../README.txt` |
| `tab:effctx`, `fig:received` (documented vs received history length) | `analysis/received_runs.json` | `analysis/recompute_received_check.py`; figure by `../code/figures/make_received.py` |
| `tab:census` (fields each entry declares) | `../code/audit/audit_results.json` | see `../README.txt` |
| `tab:changes` (what changed since the submitted version) | `analysis/CAMERA_READY_EXPERIMENTS.md` | none |

## Running the checks

The scripts were run inside the author's project folder; file paths in them start with `fmts/`, which stands for
that project's root. Put the files at those paths, or edit the path constants at the top of each script.

- Statistics tests: `python3 -m pytest analysis/test_exact_signflip.py analysis/test_gmean_level_check.py history_length/test_analyze_gpu_round.py`;
  `analysis/test_vernier_stats.py` and `analysis/test_vernier_board.py` run as plain scripts (`python3 <file>`).
- Reference checks: `Rscript analysis/refcheck.R` (needs the R packages `dfadjust` and `clubSandwich`) and `python3 analysis/refcheck_wild.py`
  (needs `wildboottest`); results are in `analysis/refcheck_results.json` and `analysis/REFCHECK.md`.
- Figure: `python3 paper/figs/make_fig_resolution.py` (needs numpy, scipy, matplotlib, seaborn) writes `fig_resolution.pdf`.
- History-length runs: `history_length/harness/` holds the job lists (`jobs/`), pinned package versions
  (`requirements/`) and data checksums (`data_manifest.md5`); they need a GPU and the GIFT-Eval data, which are not included.
