# Review of DESIGN_ROUND3.md (GPU round for the camera-ready)

Written 2026-11-19, before any run of this round. Read-only review: nothing in the design or the
harness was changed. Every number below comes from a file and line I read, from arithmetic shown
here, or from a small script in `review_scratch/` (each script says in its first lines what it
does and how to run it; its numbers are in the `.json` file of the same name). I used only
synthetic data, the first round's 12-configuration results, the GIFT-Eval data files (series
lengths only, no model) and the 2026-09-07 board snapshot (only to compute the yardstick S, which
the design itself takes from the board).

Words used below. A **sign-flip test** asks: if each base dataset's total change were equally likely
to have come out positive or negative, how often would a random choice of signs give a total at
least as large as the real one? Its **level** is how often it claims an effect when there is none;
"5%" should mean 5 claims in 100 such cases. **Holm's correction** is a rule that raises the bar
when several models are tested at once. A distribution is **skewed** when a few large values sit on
one side and the many small values on the other.

## Summary

Must fix before running (four points; the first two cost no GPU time, but they must be written
into the design before any result exists, or they become choices made after seeing data):

1. **Test groups.** Three base datasets come from one source (bizitobs), two from another (ett)
   and two from a third (bitbrains). If datasets of one source move together, the test on 18
   base datasets claims an effect 12.5% of the time when there is none; on 14 sources it is 5.0%.
   In the first round the three bizitobs datasets gave the three largest TimesFM changes.
2. **"Level 5%" is not true as written.** With 18 groups and changes shaped like the first
   round's, the sign-flip test claims an effect 5% to 10% of the time when the true mean change
   is zero, and up to 9.7% in one direction where 2.5% is intended. A stricter cut-off or a
   studentized statistic does not repair it. The design must say so now and fix how the paper
   reports it.
3. **Setting A is not yet the notebook's setting.** It leaves out two lines of the notebook
   (TF32 matrix arithmetic, and the deterministic-algorithms switch with its cuBLAS variable),
   and it does not say what Toto gets at 512 steps. `torch.compile` in the notebook has no effect
   on the forecasts.
4. **Toto's time and cost under setting A were never estimated.** The approved estimate used the
   first round's Toto speed, which made one forecast per window; setting A makes 256. A rough
   projection puts one configuration (`bitbrains_fast_storage/5T/short`) at about 83 minutes even
   on the notebook's own GPU, and Toto's documented-length runs on the 70 at about 6 hours. The
   90-minute rule would then drop Toto's heaviest configurations by stopwatch.

Should fix: the 58 "new" configurations are not new data (point 7a); the headline interval is not
named (7c); S should be written down now (1d); the gate with Holm over the models that reach T2
lets false T2 claims reach 3.5% (1c); the G3 fall-back variants cannot explain a gap (2c); the 1%
criterion should be read against the seed spread that G4 measures anyway (3); the machine check
should run under the same conditions as G1 and its halt rule needs a size threshold (5, 6); pins,
logs, a frozen analysis script and a results-freeze time are missing (8).

Sound: see the last section.

---

## 1. Are T1 and T2 valid as written?

### 1a. The groups should be sources, not base datasets. [must fix before running]

The design groups the 70 configurations by base dataset because "configurations of one base
dataset share series" (DESIGN_ROUND3.md lines 13-15). The same reason applies one level up. Of the
18 base datasets, `bizitobs_application`, `bizitobs_service` and `bizitobs_l2c` come from one
source; `ett1` and `ett2` from one; `bitbrains_fast_storage` and `bitbrains_rnd` from one. The
round-2 review already said that "independence across the 28 datasets is itself doubtful (6 M4
subsets, 2 ETT, 3 bizitobs, 2 bitbrains)" (REVIEW_ROUND2.md lines 226-227).

The first round shows these families move together. Of the 12 configurations, the three largest
TimesFM changes are the three bizitobs ones (+148.4%, +108.7%, +32.4%); the two largest Toto changes
are bizitobs (+103.0%, +67.3%); the three largest Chronos-Bolt changes are bizitobs (-43.1%,
+39.5%, -17.9%) (`review_scratch/source_grouping_sim.json`, from RESULT2_GRID.md section 8).

What happens if each source shares one effect: I drew a symmetric effect per source (so the true
mean is zero and every single test should be exact) and ran T1 both ways, 20,000 times.

| grouping | T1 claims an effect when there is none |
|---|---|
| 18 base datasets (as designed) | 12.5% |
| 14 sources (bizitobs, ett, bitbrains merged) | 5.0% |

(`review_scratch/source_grouping_sim.py`.) Real data are somewhere between "no sharing" (5%) and
"full sharing" (12.5%); the first-round pattern suggests closer to the second for bizitobs.

Options:
- Group by the 14 sources (sizes 12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1). The smallest
  possible two-sided p-value becomes 2 / 2^14 = 0.00012, still far below Holm's smallest bar of
  0.0125. Leads to a test that is exact whenever sources are independent and symmetric.
- Keep 18 base datasets as the test of record and report the 14-source test beside it, both
  fixed now. Leads to two numbers that may disagree, with no rule yet for which one the paper
  states.
- Keep 18 groups only. A reviewer who reads the round-2 note can say the test overstates its
  evidence.

The same applies to G2 on the 12 configurations: they come from 12 base datasets but 9 sources,
so 512 sign patterns, not 4,096 (smallest two-sided p 0.0039, still below Holm's 0.025 for two
models).

### 1b. The sign-flip test does not hold 5% when the changes are skewed. [must fix before running]

The test is exact only if each group's total is symmetric about the null value. REVIEW_ROUND2.md
lines 221-225 already said this: under the weaker null "the mean change is zero" with skewed
dataset effects, "neither test is exact". The authors' own level check found the exact sign-flip
test on dataset means rejecting 8.51% of true nulls and failing (GMEAN_LEVEL_CHECK.md, candidate
1). The design now makes that kind of test the deciding one (lines 85-88) and calls it "Level 5%".

I checked it in the real layout: 18 groups of sizes 1 to 7 from `config_median_lengths.json`. The
null data are the first round's 12 values of d_c per model, shifted so their mean is exactly zero
and resampled, 20,000 times per row (`review_scratch/level_sim_t1t2.py`). "Shared" means every
configuration of a base dataset gets the same value (they move together); "independent" means each
configuration gets its own. "False '512 worse'" is a claim of a positive effect (the direction the
paper cares about), nominal 2.5%.

| shape of the null | grouping of values | T1 two-sided | false "512 worse" | false "512 better" |
|---|---|---|---|---|
| symmetric, heavy tails (check) | shared | 4.98% | 2.45% | 2.53% |
| TimesFM's 12 values, centred | shared | 9.79% | 0.55% | 9.24% |
| TimesFM's 12 values, centred | independent | 5.43% | 1.44% | 3.99% |
| Toto's 12 values, centred | shared | 7.98% | 0.64% | 7.34% |
| Moirai's 12 values, centred | shared | 6.26% | 0.97% | 5.29% |
| Chronos-Bolt's 12 values, centred | shared | 9.59% | 9.04% | 0.55% |
| all 48 values, centred | shared | 8.31% | 0.57% | 7.74% |
| all 48 values, centred | half shared | 6.43% | 0.94% | 5.49% |

How it happens, with TimesFM's values: after centring, two of the 12 values are large and positive
(+0.74 and +0.57 in log terms) and the rest are small and negative. A draw of 18 datasets misses
both large values with probability (10/12)^18 = 3.8%; then every dataset total is negative, the real
total is the most extreme of all sign patterns, and the test "finds" a negative effect. The excess
always falls on the side opposite the long tail.

Two repairs do not work (`review_scratch/level_sim_fixes.py`, 10,000 draws per row): the
studentized statistic (`exact_signflip.exact_student_p`) is no better (10.67% two-sided against
9.89% in the TimesFM shared row), and no stricter cut-off fixes the worst row: at a nominal 1%
two-sided, the worst one-direction rate is still 5.29%, against the 2.5% wanted. With 18 groups and
a mean carried by a few large values, no test of the mean can hold its level; the data hold too
little information about the mean.

What this means for the paper. For TimesFM and Toto, the first round's values have a long tail of
large increases. If the new values look the same, a false claim of "512 worse" is rarer than 2.5%,
so a positive T1 or T2 result is if anything conservative. For Chronos-Bolt the tail runs the other
way, and a false "512 worse" claim reaches 9%. The direction is known only after the run.

Options (all text, no GPU):
- Keep T1 and T2, drop "Level 5%" as an unqualified claim, and fix now one diagnostic: after the
  run, rerun `level_sim_t1t2.py` on each model's realized d_c (centred, same seeds, shared and
  independent grouping) and print the false-claim rate in the claimed direction next to each
  p-value. Leads to an honest statement whose wording is fixed before the data.
- Make the signed-rank test on group means co-deciding: claim "512 worse" only if both tests
  reject in that direction. Under these skewed nulls the signed-rank test rejects even more often
  (up to 24.5%), because it tests whether the typical dataset moved, and in a skewed world with mean
  zero the typical dataset did move; requiring both does not cap the error. Leads to fewer claims,
  not a valid level.
- Leave as is. Leads to a "5%" that the authors' own earlier check contradicts.

Also fix now what the paper writes when T1 and the signed-rank test disagree (lines 101-104 report
the signed-rank test but say nothing about disagreement).

### 1c. The gate with Holm over the models that reach T2 lets T2 errors exceed 2.5%. [should fix]

As written (lines 89-95), Holm at T2 is applied only to the models that passed T1 with a positive
effect, so the bar depends on how many passed. When only one passes, it is tested at the full
2.5%. I simulated four models with correlated, symmetric errors (so every single test is exact)
and true effects exactly at the T2 boundary for some models (`review_scratch/gate_holm_sim.py`,
20,000 draws per row, S = 11.76%).

| true effects | T1 power, model 1 | false T2 claim, as designed | false T2 claim, Holm over all four |
|---|---|---|---|
| two at the boundary, two at zero | 21% to 81% | 2.3% to 3.4% | 1.2% to 1.3% |
| four at the boundary | 24% to 88% | 2.4% to 3.5% | 2.2% to 2.4% |
| one at the boundary, one at half, two at zero | 20% to 79% | 1.7% to 2.2% | 0.6% to 0.7% |

Fix: Holm over all four models at T2, giving p = 1 to a model that did not pass the gate. This
can only lower each model's chance of a T2 claim and keeps the T2 family at 2.5%. Also say in the
design that the two families have separate error rates (5% two-sided for T1, 2.5% one-sided for
T2): together a false claim of some kind reached 6.0% in the first row.

The gate itself is harmless. For one model, T2 rejecting (the lower end of the one-sided 97.5%
bound above ln(1 + S) > 0) almost implies T1 rejecting in the positive direction.

### 1d. S is from the board, not from the model's results; write its value down now. [should fix]

S does not depend on the tested model's runs, so a statement conditional on these configurations
("the change is larger than the spread of the top ten on the same configurations") is defensible.
It can be computed today. With the definition of ANSWERS.md N11 (`poolcheck/resolution_budget.py`
lines 41 and 55-60; top ten by mean MASE rank on all 97, leakage entries last):

| configurations | S | ln(1 + S) |
|---|---|---|
| 70 | 11.76% | 0.1112 |
| 58 new | 12.71% | 0.1196 |
| 12 first-round | 9.34% (the paper's 9.3%, reproduced) | 0.0893 |

(`review_scratch/spread_S.py`.) S itself moves with the configuration set: leaving one base dataset
out of the 70 gives 8.51% (without ett1) to 13.22% (without jena_weather). The design's "Limits"
paragraph says S is treated as fixed; give this range there.

Also write the shift exactly: the shifted group total is s_g - n_g * ln(1 + S), where n_g is the
group's number of configurations. "the dataset sums of d_c - ln(1 + S)" (lines 94-95) can be misread as
subtracting ln(1 + S) once per group. And note that T2 is the same as "the lower end of the
inverted 95% interval is above ln(1 + S)", which is the easiest way to explain it in the paper.

---

## 2. Is setting A the notebook's setting?

I read `gift-eval/notebooks/toto.ipynb` cell by cell (cell 5 is identical to the copy in
`vendor/toto/toto/evaluation/gift_eval/toto.ipynb`). Differences that can change a score:

### 2a. Two notebook settings are missing from setting A. [must fix before running]

- **TF32 matrix arithmetic.** `evaluate_dataset_with_model` calls
  `torch.set_float32_matmul_precision("high")` before every task (cell 5). On A100 and H100 cards
  this switches matrix products to TF32, a faster format that keeps about 3 decimal digits instead
  of about 7. It changes every forecast slightly and the speed by a large factor (point 4). The
  first round did not set it (`run_grid.py` has no such call), and setting A (design lines 43-49)
  does not list it.
- **Deterministic algorithms.** The same function calls `torch.use_deterministic_algorithms(True)`,
  and cell 1 sets `CUBLAS_WORKSPACE_CONFIG=":4096:8"` before torch is imported. The switch without
  the variable makes the first cuBLAS call raise an error. The notebook's
  `try_prediction_with_config` catches every exception and returns `None`, and the caller then
  writes a row of NaN (cell 5). Copying the switch without the variable, or copying the error
  handling, can give rows of NaN that look like finished runs. Count a NaN row as a failed run.

### 2b. Other differences, and what each means. [should fix: write them into the design]

- **`torch.compile` does nothing here.** The notebook compiles the outer `Toto` module (cell 5,
  `evaluate_tasks`), but the forecaster is built from `prediction_net.model`
  (`gluonts_predictor.py` line 210), which reaches through the compiled wrapper to the inner,
  uncompiled backbone (`toto/model/toto.py` line 51 sets `self.model = TotoBackbone(...)`). So
  the compiled forward is never called. The contingency in line 47 is moot, and compile is not a
  candidate cause of the gap.
- **Attention kernel.** `toto/model/attention.py` lines 22-34 use xFormers' attention if the package
  is installed and PyTorch's otherwise. The first round had no xFormers (`setup.log` has no match).
  The board's run is unknown. Pin it (state "xFormers not installed") and record it.
- **Toto code version.** The design pins `c079fd7`, dated 2026-09-08 (the vendored clone has only
  this one commit). The board entry was made with whatever code existed when Datadog submitted it.
  The notebook imports `from inference.gluonts_predictor` and `from model.toto` (cell 1), an older
  import path than the installed package's. This is a candidate cause that the G3 fall-backs do not
  test; say so in the report.
- **History at 512 steps.** Setting A specifies only the documented length (line 46). Write the 512
  arm: min(512, the notebook's maximal length), at least 64, batch size from the same formula at 512.
- **Batch size.** The formula uses the card's total memory (cell 5, `get_total_gpu_memory`) with a
  safety factor of 0.01. On an 80 GB card I get 1 or 2 series per batch at 4,096 steps and 11 to 16
  at 512 (`review_scratch/toto_time_projection.json`; this assumes 12 layers of width 768, which I
  could not confirm on disk). Batch size only changes which random numbers go to which series, not
  the expected score.
- **Seeds and why draws cannot be matched.** The notebook re-seeds at the start of every task, so a
  task's draws do not depend on the long process (the reason given in lines 48-49 is not the real
  one). They cannot be matched because the batch size depends on the card's memory and because
  PyTorch's random numbers on a GPU depend on the card. After an out-of-memory error the notebook
  retries without re-seeding; copy that.
- **Minimum history and the cap.** I recomputed Toto's history on all 70 configurations under the
  notebook's rule and under the first round's code (`review_scratch/toto_context_70.py`). They agree
  on all 70; the 64-step minimum never applies. Only `m4_daily/D/short` gets the same history (79
  steps) at both lengths.
- **The forecast before scoring.** With 256 samples, MASE uses the median of the samples and
  MSE[mean] uses their mean (`forecaster.py` line 201). Nothing else is done to the forecast. The
  first round scored the mean path as a single sample (`gluonts` `evaluate_model` calls
  `model.predict(test_data.input)` with no sample count, evaluation.py line 260;
  `gluonts_predictor.py` lines 161-173). This confirms the design's reading.

### 2c. The two fall-back variants cannot explain a gap. [should fix]

If setting A misses, the design runs "A with the cache off" and "A with a fixed batch size of 16"
(lines 56-57). The key-value cache is an exact shortcut: it stores intermediate results instead of
recomputing them, so turning it off changes only rounding. The batch size only reassigns random
draws. Neither can move the expected score, so either variant can only show noise.

Options:
- Replace them by variants that can move the score: TF32 off (the first round's precision), and
  the Toto code at the board-era commit if it can be identified. Leads to a real test of two
  candidate causes.
- Drop them and use G4's seeds 43 and 44 as the noise yardstick. Leads to "the remaining gap is
  larger or smaller than seed-to-seed noise", at no cost.
- Keep them. Leads to two runs whose outcome is known in advance.

Also word the success case carefully: setting A changes several things at once (samples, cache,
TF32, batch size, seeds), so meeting the criterion shows the notebook's setting reproduces the
board; that the sample count is the cause rests on the other changes being neutral, as argued here.

---

## 3. Is the 1% criterion reasonable? [should fix]

The noise comes from the median of 256 draws. For a bell-shaped forecast the median of 256 draws
has a spread of about 1.25 / 16 = 0.078 times the forecast's own spread; if forecasts are about as
good as they claim, each forecast point's absolute error moves by about 10% of its size, in a random
direction. A configuration averages this over its forecast values. The 12 include very small ones:
`bizitobs_application/10S/medium` is 1 series with 2 variates and 2 windows,
`bizitobs_l2c/H/long` 1 series with 7 variates and 1 window, `us_births/W/short` 14 windows of
8 steps (`review_scratch/toto_context_70.json`).

| assumption | spread of the 12-configuration geometric mean, one run | spread of the difference from the board's run | chance a correct setting misses 1% |
|---|---|---|---|
| all steps of a window move together | 0.59% | 0.84% | 23% |
| every step independent | 0.08% | 0.11% | about 0% |

(`review_scratch/noise_12.py`.) The truth is in between and likely nearer the first line for long
horizons, because a sample path that starts high tends to stay high. So 1% is neither safely loose
nor tight. G4 already runs seeds 42, 43 and 44 on the same 12 configurations at the documented
length, which measures this noise directly.

Fix (no new run): add now that the gap is also compared with the seed spread: "within noise" if
0.7699 lies within the range of the three seeds' geometric means widened by half that range;
otherwise "not explained by sampling". Report the per-configuration differences (already planned).

Note: the "0.46%" for Chronos-Bolt (line 53) is the median per-configuration difference. Its
geometric-mean gap was 0.27% (0.9650 against 0.9676, RESULT2_GRID.md section 7). Compare like with
like.

---

## 4. Does the 90-minute rule bias G1? [must fix before running]

Setting A makes 256 forecasts per window where the first round made one: the forecaster copies each
input 256 times before running the model (`forecaster.py` lines 414-418). With the memory formula,
a batch holds 1 or 2 series at 4,096 steps against 11 to 16 at 512 (point 2b). The documented-length
arm is therefore the slow one.

I projected run times from the notebook's own printed timings (cell 7 output: `bizitobs_l2c/H/short`,
6 batches at 2.22 per second; `bizitobs_l2c/H/long`, 1 batch at 1.10 per second) and the instance
counts of each configuration (`review_scratch/toto_time_projection.py`; rough, two data points,
GPU of the notebook not stated, TF32 on as in the notebook):

| configuration | batches at 4,096 | minutes at 4,096 | minutes at 512 |
|---|---|---|---|
| bitbrains_fast_storage/5T/short | 22,500 | 83 | 10 |
| bitbrains_rnd/5T/short | 9,000 | 33 | 4 |
| electricity/15T/long | 3,700 | 26 | 14 |
| electricity/15T/medium | 3,700 | 21 | 9 |
| loop_seattle/5T/medium | 3,230 | 19 | 8 |
| all 70 | | about 6.0 hours | about 1.7 hours |

If the A100 is slower than the notebook's card, or TF32 is off, the first one or two pass 90 minutes
and others come close. The heaviest configurations are concentrated in electricity (7 of the 70) and
bitbrains (6). The approved estimate (GPU_COST_ESTIMATE.md lines 11 and 58) used the first round's
Toto speed (median 12.5 s per run) and gives 7 hours for G1 for all four models together.

Does the rule bias G1? It does not look at the scores, so it is not selection on results. But it
is predictable now, it hits Toto only and only through the documented arm, and it removes the
configurations with the most series and the longest histories, where losing history could matter
most. Toto's effect would then be measured on a different set from the other three models.

"S on the model's own remaining configurations" (line 113) is right for T2, because both sides of
that comparison then use the same configurations. It does not fix the comparison between models or
the change of what Toto's number means.

Options:
- Before G1 starts, project each configuration's Toto time from the G3 runs (which include
  `loop_seattle/5T/long`, `electricity/H/long` and `solar/10T/short` at setting A), write the list
  of Toto configurations and its time limit into the design, and send the owner a revised cost.
  Leads to a fixed set decided without results, and a budget the owner has seen.
- Raise the limit for Toto's documented arm only (for example 4 hours) and run the heaviest ones
  first on their own cards. Leads to all 70, at a cost that must be approved.
- Keep the stopwatch. Leads to drop-outs decided during the run, a Toto estimate on fewer
  configurations, and possibly G2 not running at all because it is last in the order (line 143).

Whatever is chosen, fix now that the cross-model table also reports every model on the
configurations all four finished.

---

## 5. Can two processes per GPU change the three deterministic models' numbers? [should fix]

The design does not mention two processes per GPU (line 144 says only that ready work runs in
parallel on free cards). In itself, sharing a card does not change a kernel's result: each process
has its own context and the same kernels run on the same inputs. The risk is memory. The first round
already ran out of memory for Chronos-Bolt at the documented length on a whole 80 GB card and was
rerun with a smaller batch (`CHRONOS_BATCH_SIZE`, run_grid.py lines 282-285 and 319-323, and
GPU_COST_ESTIMATE.md line 48). With two processes sharing 80 GB this becomes more likely, and a
smaller batch is where numbers can move. For TimesFM-2.5 each batch is padded to its own longest
series and the model is recompiled for that shape (run_grid.py lines 396-405), so a different batch
size changes padding and shapes; in exact arithmetic the masked padding should not matter, but
rounding can.

Fix: run the machine check under exactly the G1 conditions (two processes per card, the same batch
sizes and environment variables); record each run's batch size and every out-of-memory retry; keep
TimesFM's batch at 1,024 (one process on the card if needed).

---

## 6. A100 instead of H100: is the 1e-4 check enough, and what if it fails? [should fix]

1e-4 is generous for these models. In the first round, Moirai-2.0 and TimesFM-2.5 matched the
board's published rows to under 1e-6 (RESULT2_GRID.md section 7), although the board ran on other
machines, so card-to-card rounding is far smaller than 1e-4. The effects that matter are around
1e-2. A failure would therefore most likely mean a software, weights or data difference, not the
card. Those are not pinned (point 8): the first round installed `torch>=2.1` for Chronos-Bolt
(h100_setup.sh line 18) and got torch 2.14.0, and downloaded the GIFT-Eval data with no revision
(line 33).

The halt rule (line 34, "nothing else is analysed until the cause is found") has no size threshold
and no time limit, with the camera-ready due 2026-11-20 11:59 pm AoE (STILL_NEEDED.md line 4), that
is 2026-11-21 17:29 IST. G1 reruns all 70 on the new cards, so G1 does not depend on the old 12;
only the sentence "same numbers on a different card" and the "first 12 alone" comparison do.

Fix (options, one to choose now):
- Below 1e-4: pass. From 1e-4 to 1e-3: continue, report the size, and run one configuration twice
  on one card to show repeat runs agree. Above 1e-3, or any d_c changing sign: stop and check
  versions, weights and data before anything else. Leads to a rule that cannot stall the round on
  rounding.
- Keep the halt rule as written. Leads to a possible stop with no deadline for finding the cause.

Also keep `set_float32_matmul_precision("high")` inside the Toto builder only. If it is placed at
the top of the shared `run_grid.py`, it switches the other three models to TF32 as well and the
machine check fails for that reason.

---

## 7. What could make the result unusable, or look like a choice made after seeing data

### 7a. The 58 are not new data, and the 70 contain the 12 that suggested the effect. [should fix]

The design says the 58 "were not used for any earlier claim, so they are a repeat of the first round
on new data" (lines 106-107). Of the 58, 43 come from a base dataset already among the 12, and 20 use
the very same series as one of the 12 (same dataset and frequency, a different forecast horizon);
the 58 span 17 base datasets, 11 of them shared with the 12 (computed from
`selected_configs.json` and `config_median_lengths.json`). The effects of 20% that motivate T2 were
seen on the 12, and the analysis of record (the 70) includes them.

Options:
- Make the 58 the confirming test (T1 and T2 on the 58, grouped by source) and the 70 the estimate
  of record. Leads to a claim not tested on the data that suggested it, with less power.
- Keep the 70 as the test of record and replace the "new data" sentence by the overlap counts
  above. Leads to an honest description; a reviewer may still ask about reuse.

### 7b. Decisions that are already sound in this respect. [note]

The 70 are every configuration that passes a rule written on 2026-09-12 before any model was run;
setting A is fixed for Toto whatever it gives (line 59) and replaces the abstract's "more than 20%"
whatever the number is (lines 59-61); the sum statistic matches the board's geometric mean, which
counts every configuration equally; S is mechanical. None of these was chosen after seeing results.

### 7c. Say now which interval and which numbers the paper prints. [should fix]

Two intervals are planned (lines 99-100): the inverted sign-flip interval and the CR2 interval. The
authors' own level check found CR2 rejecting 15.37% of true nulls on geometric-mean pairs
(GMEAN_LEVEL_CHECK.md, set i). Name the inverted sign-flip interval as the one in the abstract and
the main table, now. Also say which numbers replace "0.8126" and "+20.6%" in the paper: the
12-configuration setting A numbers in the table of the first round, and the 70 (or 58) for the
abstract, so that the choice is not made after the numbers are in.

### 7d. Only one score. [note]

The first round reported two aggregations (mean insertion rank and the geometric mean,
RESULT2_GRID.md section 1). The design reports the geometric mean only. If the camera-ready table
keeps both, say now that the rank aggregation on the 70 is descriptive only.

---

## 8. What is missing and cannot be added afterwards [should fix]

- **Software pins.** Torch per environment as in the first round (`run_meta.json`: 2.14.0 for
  Chronos-Bolt and TimesFM-2.5, 2.4.1 for Moirai-2.0, 2.7.0 for Toto), the CUDA wheels from
  `setup.log`, `transformers` and `gluonts` versions, xFormers absent, the Hugging Face model
  revisions the first round recorded (for example Toto `0411ceb2`, Chronos-Bolt `5d9f166d`,
  Moirai `30f43ff0`, TimesFM `1d952420`), and the GIFT-Eval data checked against the first round's
  files (`harness/verify_data.py`) or pinned by dataset revision.
- **Logs per run.** Card model, versions, TF32 and deterministic switches, attention kernel, batch
  size, final samples per batch after any out-of-memory retry, history used, wall time, seed. Without
  these the claim "setting A is the notebook's setting" cannot be shown after the fact.
- **The analysis script.** Write it now (one-sided count for T2, Holm as fixed above, interval
  inversion with a stated grid, CR2, signed-rank, leave-one-out, the level diagnostic of 1b), test
  it on synthetic data, and record its checksum in the design before the first result is copied
  off the pod.
- **A results-freeze time.** The budget timer (line 145) is about money, not the paper. Set a time
  (IST) after which nothing new enters the camera-ready, and the sentence the paper uses for each
  request not finished by then.
- **Toto at identical histories.** For `m4_daily/D/short` both lengths give Toto the same 79 steps.
  Fix now that d_c is set to 0 there by rule (and the second run skipped), instead of relying on
  two runs drawing identical random numbers.

---

## What is sound

- The definition of the effect: d_c = ln(MASE at 512 / MASE documented), averaged and reported as
  exp(mean) - 1, equals the change in the geometric mean of scaled MASE (the Seasonal Naive
  scaling cancels); same computation as `audit/pct_change.py`.
- Exact enumeration of sign patterns (no random-draw error) with the existing, tested
  `exact_signflip.exact_sum_p`; the sum statistic matches the board's configuration-weighted
  geometric mean. Under symmetry it is exact in this layout (4.98% in the check row above).
- Holm over the four models at T1, which are fixed in advance.
- Setting A fixed for Toto's G1 and G4 before any result, and the commitment to replace the
  first round's Toto numbers whatever they become.
- Not repeating Chronos-Bolt, Moirai-2.0 and TimesFM-2.5 across seeds: all three output quantiles
  directly; nothing is drawn at random at inference.
- Toto's history cap: the notebook's rule and the first round's code give the same history on all
  70 configurations, so the first round's context handling was not a cause of the gap.
- Reporting the 58 and the 12 separately and the leave-one-out range; keeping d_c = 0 rows.
- The 1e-4 tolerance itself (point 6), the checksum copy of results, and the money timer.
- Running G3 and G4 before G1, which gives a measured Toto speed before the large run, if it is used
  as in point 4.

## Scripts (all in `review_scratch/`)

| script | what it checks | key output |
|---|---|---|
| `level_sim_t1t2.py` | T1 and one-sided T2 level under symmetric and skewed nulls, 18-group layout | 4.98% symmetric; up to 9.82% skewed; up to 9.27% in one direction |
| `level_sim_fixes.py` | studentized statistic and stricter cut-offs | studentized no better; 1% nominal still 5.29% in one direction |
| `source_grouping_sim.py` | 18 base datasets against 14 sources when sources share effects | 12.5% against 5.0% |
| `gate_holm_sim.py` | T2 error with Holm over the models that pass, against Holm over all four | up to 3.5% against at most 2.4% |
| `spread_S.py` | S on 70, 58 and 12 from the board snapshot | 11.76%, 12.71%, 9.34% |
| `toto_context_70.py` | Toto's history under the notebook's and the first round's rules, instance counts | identical on all 70; m4_daily the only tie |
| `noise_12.py` | sampling noise of the 12-configuration geometric mean with 256 samples | 0.11% to 0.84% for the difference |
| `toto_time_projection.py` | Toto run time under setting A per configuration | 83 min for the heaviest; about 6 h documented arm |
