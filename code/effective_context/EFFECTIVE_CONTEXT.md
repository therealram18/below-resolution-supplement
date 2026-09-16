# Effective context length, all 96 grid cells

What this measures: for every (model, config, context setting) cell, how many
past time steps the model actually got to look at, after its own truncation
rule ran into how much history was actually available for that series and
that rolling test window — as opposed to the "nominal" number (2048, 4000,
15360, 4096, or 512) that just labels the setting. CPU only: no model was
run. Everything below comes from arithmetic over the on-disk series lengths
plus the exact truncation code in `harness/run_grid.py`'s per-model builders
(read, not guessed) and gift-eval's own `Dataset` class (`prediction_length`,
`windows`, `_min_series_length`).

Script: `compute_effective_context.py`. Output: `effective_context.csv` (96
rows, one per model x config x setting). Deterministic — no randomness
anywhere in the computation — confirmed by running it twice: identical file,
md5 `743b0533f2dd5016d9492a49013aa1b9` both times.

## How "available history" is computed

Gift-eval's `Dataset.test_data` builds rolling test windows with
`generate_instances(prediction_length=P, windows=W, distance=P)` off a split
at `offset=-P*W`. For a series of length `L`, this produces `W` windows whose
input (context) length grows by `P` each time:

    available(window i) = L - P*(W - i),   i = 0 .. W-1

so the earliest window sees the least history and the last window the most
— matching the task's note that later windows have more history. `P`
(prediction_length) and `W` (windows) both come straight from the `Dataset`
object, not reimplemented.

Per-model truncation, applied to that `available` number:

| model | default setting | 512 setting |
|---|---|---|
| chronos_bolt_base | checkpoint has a fixed internal cap of 2048 (can't be raised); `min(2048, available)` | client slices `arr[-512:]` before calling the model; `min(512, available)` |
| moirai2_small | `context_length=4000` is passed to the model; `min(4000, available)` | `context_length=512`; `min(512, available)` |
| timesfm25 | no per-series slicing happens in code, but the compiled shape can never exceed the model's 15360 cap, and a series can never be truncated below its own history unless it exceeds that cap (proof: the per-batch compile target is a `max()` over the batch, which is always ≥ any one series' own length, so it never shrinks what a given series gets) — net effect: `min(15360, available)` | client slices `arr[-512:]`; `min(512, available)` |
| toto | a *dataset-level* cap is computed once per config: `ctx = min(4096, shortest_series_length - (windows+1)*prediction_length)` (or 4096 if that subtraction is ≤ 0), then applied per series/window: `min(ctx, available)` | same formula with 512 in place of 4096: `min(ctx512, available)` |

Toto's cap is the odd one out: it's driven by the *shortest* series in the
whole config, not by the individual series/window being scored, so every
series in a config can get clamped to whatever the worst-case series allows
— explained below.

## Q1 — default-setting cells running below nominal

**13 of the 48 default-setting cells have `eff_median < nominal`** (median
taken over all series x windows in that config's test set). The critique's
claim of "at least 9" is confirmed — all 9 named cells are among the 13:

- `us_births/W/short` — below nominal for **all four models** (confirmed)
- `bizitobs_l2c/H/long` — below nominal for **all four models** (confirmed)
- `bizitobs_application/10S/medium` — below nominal for **timesfm25** (confirmed)

4 more cells fall below nominal that the critique didn't name: `timesfm25` on
`bizitobs_service/10S/short`, and all three of `moirai2_small`, `timesfm25`,
`toto` on `m4_daily/D/short`.

Full list (model, config, effective median vs nominal):

| model | config | eff_median | nominal |
|---|---|---:|---:|
| timesfm25 | bizitobs_service/10S/short | 8355 | 15360 |
| moirai2_small | m4_daily/D/short | 2940 | 4000 |
| timesfm25 | m4_daily/D/short | 2940 | 15360 |
| toto | m4_daily/D/short | 79 | 4096 |
| chronos_bolt_base | us_births/W/short | 983 | 2048 |
| moirai2_small | us_births/W/short | 983 | 4000 |
| timesfm25 | us_births/W/short | 983 | 15360 |
| toto | us_births/W/short | 923 | 4096 |
| timesfm25 | bizitobs_application/10S/medium | 7934 | 15360 |
| chronos_bolt_base | bizitobs_l2c/H/long | 1944 | 2048 |
| moirai2_small | bizitobs_l2c/H/long | 1944 | 4000 |
| timesfm25 | bizitobs_l2c/H/long | 1944 | 15360 |
| toto | bizitobs_l2c/H/long | 1224 | 4096 |

Note on reading this table fairly: chronos_bolt_base (nominal 2048) and toto
(nominal 4096) have fixed, comparatively modest nominal values, so they only
fall short on genuinely short configs (2-3 of 12). moirai2_small (nominal
4000) falls short on 2 of 12. timesfm25's nominal (15360) is a *ceiling*, not
a target the paper claims the model reaches — it's "whatever history exists,
up to 15360" — so it is expected to show up as "below nominal" on any config
whose available history per window is under ~15360, which is 5 of the 12
configs here. That's not a bug in timesfm25's setup, just a property of
what "nominal" means for a model whose default context is data-dependent by
design.

## Q2 — toto/m4_daily no-op cell

**Confirmed: 79 effective context at both settings.**
`toto,m4_daily/D/short,default` → eff_min = eff_median = eff_max = 79.
`toto,m4_daily/D/short,512` → eff_min = eff_median = eff_max = 79.

Why: m4_daily's shortest series has only 107 points (`_min_series_length =
107`), m4-family configs always use `windows = 1`, and `prediction_length`
for daily data is 14. Toto's dataset-level cap is
`107 - (1+1)*14 = 107 - 28 = 79`, which is smaller than both the 4096
(default) and 512 settings, so both settings get clamped to the same 79 for
every series in the dataset — the 512-vs-default contrast is a genuine
no-op for toto on this config, exactly as flagged.

## Q3 — 512-setting cells where effective < 512

**One cell, using the same eff_median convention as Q1: `toto,
m4_daily/D/short, 512`** (eff_median = 79, well under the 512 it's nominally
set to). This is the same m4_daily no-op from Q2 — toto's dataset-wide cap
of 79 makes 512 unreachable regardless of setting.

Every other 512-setting cell has `eff_median = 512` exactly, i.e. the median
series/window comfortably has ≥512 points of history to give the 512 setting
its full effect.

One nuance worth flagging even though it doesn't change the cell-level
answer: for `m4_daily/D/short`, `chronos_bolt_base`, `moirai2_small`, and
`timesfm25` all show `eff_min = 93` at the 512 setting — 1,283 of that
config's 4,227 series/window pairs (30.4%) actually have fewer than 512
points of real history, so the 512 setting is silently a no-op for nearly a
third of that config's test instances even though the *median* instance
still gets the full 512. m4_daily is the one config in this grid with a
short enough minimum series length (107) that this matters; no other config
shows any eff_min below 512.

## Other things worth noting

- **timesfm25's default context genuinely varies with the data**, unlike the
  other three models' default settings, which are almost always pinned at a
  flat value. E.g. `bizitobs_service/10S/short`: eff_min 7935, eff_median
  8355, eff_max 8775 — a real spread, reflecting that later rolling windows
  see more history, none of it clipped by the 15360 cap. Chronos, moirai2,
  and toto instead mostly sit exactly at their nominal value whenever there's
  enough data (2048 / 4000 / 4096 flat across min/median/max for 8 of 12
  configs each) — their "effective context" curve is flat until the config
  runs short, then it drops straight to the raw available-history number.
- **`bizitobs_l2c/H/long` is a thin config**: `windows = 1` and (after
  accounting for its 7 variates) `n_series_windows = 7` for the
  variate-split models and just 1 for toto (native multivariate) — the
  smallest test set of any of the 12 configs. Worth keeping in mind when
  reading model-quality numbers alongside these context numbers: a single
  series/window pair carries a lot of weight in the average there.
- Sanity checks that passed: for every config, the minimum of the raw
  per-series length array matched `Dataset._min_series_length` exactly (a
  built-in assertion in the script); `windows=1` was confirmed for
  `m4_daily` (the "m4" special case in gift-eval's `Dataset.windows`); toto's
  formula reproduced the known 79 independently from the general
  `min(cap, available)` arithmetic used everywhere else.

## Files

- `compute_effective_context.py` — the computation, reusing
  `harness/run_grid.py`'s `resolve_storage_name`, per-model constants, and
  `to_univariate_for`, plus gift-eval's own `Dataset` class. No model
  imports, no GPU, no network calls (data loaded from local disk only).
- `effective_context.csv` — 96 rows, columns: model, config, setting,
  nominal, eff_min, eff_median, eff_max, n_series_windows. md5
  `743b0533f2dd5016d9492a49013aa1b9`.
