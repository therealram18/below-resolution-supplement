# How the model grid was actually run

This grid is 4 models x 2 context lengths x 12 dataset-configs = 96 cells
(one cell = one model, one context length, one dataset-config, written as
one row in the matching `all_results.csv`). All 96 ran on one GPU, in three
passes, all on the same day, 2026-09-12, within roughly 70 minutes (pass 1
started around 13:50 UTC, pass 3 finished before about 15:00 UTC). This
file explains why the numbers in
`results/grid_run_log.txt` alone look like 41 failures when the CSVs next to
it are complete with no gaps and no duplicate rows — the short version is
that two more passes after the first one filled in every failure, and this
file plus the logs below are the record of that.

## Pass 1 — the original grid (`results/grid_run_log.txt`)

96 invocations were attempted. 41 failed:

- **40 failures were a naming bug**, not a data or model problem — and it
  was actually two different bugs, not one. Of the five dataset-configs
  that failed:
  - Two failed on directory capitalization: the leaderboard config list
    names them lowercase-with-underscores (`m_dense`, `loop_seattle`), but
    the on-disk dataset directories use different capitalization
    (`M_DENSE`, `LOOP_SEATTLE`).
  - Three failed on a frequency sub-directory that the config name implies
    but the dataset directory does not actually have (`m4_daily`,
    `bizitobs_application`, `bizitobs_service`).
  Each of the five affected dataset-configs failed for all 4 models and
  both context lengths (5 configs x 4 models x 2 context lengths = 40).
- **1 failure was a CUDA out-of-memory** on Chronos-Bolt-base at the
  default context length, on `electricity/H/long`.

## Pass 2 — the missing-config rerun (`results/rerun_missing_log.txt`)

Before rerunning anything, the dataset-name lookup was patched so the
leaderboard-style names resolve to the correct on-disk directory names.
The patch is `resolve_storage_name()` in `code/harness/run_grid.py`; that
function, and the lowercase-to-on-disk name mapping it uses, are unchanged
from what actually ran this pass.

The rerun script then went through the 40 naming failures plus the 1 OOM
cell (41 cells total) and reran each one. The script skips any config whose
result row already exists in `all_results.csv`, so this pass only ever adds
a row for a cell that had none — it never recomputed or overwrote a cell
that had already succeeded in pass 1.

Of those 41: 39 succeeded. 2 hit a fresh CUDA out-of-memory — both were
Chronos-Bolt-base at the default context length again, on the two largest
series in the grid (`electricity/H/long` and `loop_seattle/5T/long`).

## Pass 3 — the OOM rerun (`results/rerun_oom_log.txt`)

Those last 2 cells needed two rerun attempts before they succeeded.

The first attempt set the environment variable `CHRONOS_BATCH_SIZE=64`,
intending it to cap how many series get batched to the GPU during
inference. It didn't work: `evaluate_model()`, a function from the
`gluonts` forecasting library that the harness calls to run each model,
does not forward its `batch_size` argument into `predict()` (the function
that actually runs the model on the GPU) — so the environment variable
never actually reached the predictor, and both cells hit the same CUDA
out-of-memory error as in pass 2.

The second attempt added an actual cap inside the predictor itself:
`ChronosPredictor.max_batch_size`, a class attribute added in
`code/harness/run_grid.py` (an ~11-line change), which does reach
`predict()`. This doesn't change the model, the input data, or the
forecasts — only how many series are batched together on the GPU. Both
cells succeeded on this second attempt.

Both attempts — the failing one and the succeeding one — are visible in
`results/logs_per_run/oom_retry.log`: for each of the two cells, that file
shows the OOM failure followed by the successful rerun, all in one file.
`results/rerun_oom_log.txt` (this pass's master log) only records the
final successful outcome, not the failed first attempt.

## Final state

96 of 96 cells are present, each computed exactly once. No cell was ever
recomputed after it first succeeded — pass 2 and pass 3 only ever filled in
cells that were still missing.

## Code version across passes

Three versions of `code/harness/run_grid.py` ran across the three passes:
the original version (pass 1); with `resolve_storage_name()` added (pass
2, described above); and with the Chronos batch cap added
(`ChronosPredictor.max_batch_size`, pass 3's second attempt, described
above). Only the final version — with both patches applied — is shipped in
this zip. Those two patches are the only differences between the code
that ran in pass 1 and the code shipped here.

## Where the per-run detail is

- `results/grid_run_log.txt` — master log, pass 1 (all 96 invocations
  attempted, including the 41 failures).
- `results/rerun_missing_log.txt` — master log, pass 2 (the 41 reruns after
  the naming fix).
- `results/rerun_oom_log.txt` — master log, pass 3 (the final 2 reruns with
  the smaller batch size).
- `results/logs_per_run/` — the full stderr/stdout for every individual
  invocation across all three passes (one file per model x context x pass,
  plus rerun-specific files), for anyone who wants to check a specific cell
  beyond what the master logs summarize.
- `results/<model>/ctx_*/run_meta.json` — a small metadata file written by
  the harness next to each `all_results.csv`.

## A disclosed gap: `run_meta.json` is not a full per-pass record

Each `run_meta.json` was meant to record every config run in that pass, but
the harness script overwrote it after every single invocation instead of
appending — so each `run_meta.json` on disk only shows the LAST config
processed in whichever pass last touched that model/context combination,
not the full list. This is a logging bug, not a data problem: the actual
forecasts and metrics in `all_results.csv` are unaffected, and the complete
per-invocation record lives in `results/logs_per_run/` and the three master
logs above. We are disclosing this gap rather than papering over it.
