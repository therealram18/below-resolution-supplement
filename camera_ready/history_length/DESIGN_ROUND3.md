# Camera-ready, GPU round: design (written 2026-11-19, before any of these runs)

Source of the requests: `STILL_NEEDED.md` section 4, items G1 to G4. This file fixes what is run,
with which settings, and which tests are applied, before any run starts. Nothing here may change
after results are seen without a dated note.

No result of this round exists at the time of writing. What is already known and used here: the
first round's results on 12 configurations (`grid_results_h100/`, `audit/pct_change.json`) and the
board's published rows.

## Shared definitions

- **Configuration**: one GIFT-Eval `dataset/frequency/term`. **Base dataset**: the part before the
  first slash. Base datasets are the groups in every test below, because configurations of one
  base dataset share series.
- **Lengths**: "documented" is the history length the model's own documentation or evaluation
  notebook uses; "512" gives every model the last 512 steps. Same two settings as the first round
  (`--context default` and `--context 512`).
- **Score**: MASE at the median forecast (`eval_metrics/MASE[0.5]`), as in `audit/pct_change.py`.
- **Effect of a model**: for each configuration c, d_c = ln(MASE at 512 / MASE at documented).
  The effect is the mean of d_c over the configurations, reported as a percentage,
  exp(mean) - 1. This equals the change in the geometric mean of scaled MASE that the paper
  prints (the scaling by Seasonal Naive cancels).
- **Code**: a copy of the first round's harness in `camera_ready/gpu_round/harness/`. The first
  round's files are not edited. Pins unchanged: gift-eval 9e88bb6, Toto repository c079fd7,
  chronos-forecasting 2.3.2, uni2ts 2.0.0, timesfm 3.0.2.
- **Hardware**: A100 80 GB cards on RunPod (the first round used one H100 80 GB). Timing is not a
  result of this round.

## Check of the new machine (before anything else counts)

Chronos-Bolt, Moirai-2.0 and TimesFM-2.5 are run at both lengths on the 12 first-round
configurations. Each of the 72 MASE values must equal the first round's value to a relative
difference of 1e-4. If any does not, nothing else is analysed until the cause is found. This
check is also the evidence that these three models give the same numbers on every run.

## G3. Toto's reproduction gap

Known from reading the code (not yet from a run): the first round called Toto without a sample
count, so it returned one point forecast. The board entry's notebook
(`gift-eval/notebooks/toto.ipynb`) scores 256 samples.

**Setting A (the notebook's setting)**: 256 samples; 256 samples per batch, halved after an
out-of-memory error as the notebook does; key-value cache on; seed 42 set for torch and numpy at
the start of each run; batch size from the notebook's memory formula (at most 16); minimum
history equal to the model's patch stride; history length min(4,096, the notebook's maximal
length). `torch.compile` as in the notebook if it runs on the pod; if it does not, it is turned
off and that is recorded. The notebook seeds once per dataset task inside one long process, so
its random draws cannot be matched run for run; the check below is on the summary.

**Criterion**: on the 12 configurations at the documented length, the geometric mean of scaled
MASE under setting A is within 1% of the board's 0.7699. (The first round accepted Chronos-Bolt
at 0.46%.) The per-configuration differences are reported next to it.

- If it is met: the cause is the missing sample count.
- If it is not met: two more variants are run on the 12 configurations, A with the cache off and
  A with a fixed batch size of 16. The remaining gap is reported. No further search.

**Either way, setting A is Toto's setting for G1 and G4**, because it is the documented one. The
first round's Toto numbers (0.8126, and +20.6% from the documented length to 512) are replaced in the paper by
setting A's numbers, whatever they are. This includes the "more than 20%" of the abstract.

## G4. Repeat runs

Toto, setting A, seeds 42, 43 and 44, on the 12 configurations at both lengths (72 runs; seed 42
at the documented length is also the G3 run). Reported: the geometric mean of scaled MASE at each
length for each seed, the effect for each seed, and the range over the three seeds. Seed 42 is
the value of record.

The other three models are not repeated: they have no random step at inference. The evidence is
the machine check above (same numbers on a different card, two months apart).

## G1. The history-length test on every eligible configuration

**Configurations**: the 70 that pass the rule written on 2026-09-12
(`selection/SELECTION_RULE.md`, version 2: median series length at least 1,024). They come from
18 base datasets. 12 of them are the first round's; 58 are new. All 70 are run on the new
machine.

**Models**: Chronos-Bolt, Moirai-2.0, TimesFM-2.5 with the first round's settings; Toto with
setting A, seed 42.

**Tests, for each model, in this fixed order.**

- **T1, effect different from zero.** Exact sign-flip test: s_g is the sum of d_c over the
  configurations of base dataset g; the two-sided p-value is the share of the 2^18 sign patterns
  with |sum of w_g s_g| at least |sum of s_g| (the sum statistic of
  `camera_ready/exact_signflip.py`). Holm correction over the four models. Level 5%.
- **T2, effect larger than the spread of the top ten. Only for a model whose T1 rejects after
  Holm with a positive effect.** S is the spread of the board's displayed top ten on the same
  configurations: for each of the ten entries of the 2026-09-07 snapshot (commit b61c48aa), the
  geometric mean over those configurations of MASE divided by Seasonal Naive's MASE; S is the
  largest divided by the smallest, minus 1 (the definition behind the paper's 9.3% on the 12).
  One-sided exact sign-flip test of the null "mean of d_c is at most ln(1 + S)", on the dataset
  sums of d_c - ln(1 + S), at level 2.5%. Holm over the models that reach T2.

**Reported with the tests, not deciding anything:**

- a 95% interval for the effect by inverting the two-sided sign-flip test, and the CR2 interval
  with Satterthwaite degrees of freedom (the first round's method, `vernier_stats.py`);
- the signed-rank sign-flip test on base-dataset means (the test that held its level in
  `GMEAN_LEVEL_CHECK.md`), because tests of a mean of log ratios rejected too often there when
  one dataset carried most of the spread. It tests the typical dataset, with datasets counted
  equally;
- the effect with each base dataset left out in turn (smallest and largest);
- the same numbers on the 58 new configurations alone and on the first 12 alone. The 58 were not
  used for any earlier claim, so they are a repeat of the first round on new data.

The 70 configurations are the analysis of record.

**Rules fixed in advance for incomplete data.** A run that takes more than 90 minutes is stopped.
A configuration enters a model's analysis only if both lengths finished; configurations that drop
out are listed by name, and S for that model is computed on the configurations that remain. A
configuration where both lengths give the model the same history (d_c = 0 exactly) stays in.

**Limits, stated with the result.** S is used as a fixed yardstick although it is itself computed
from data. Eighteen groups is few: the smallest two-sided p-value possible is 2 / 2^18, but a
single large dataset can still carry a sum. The leave-one-out range is there for that reason.

## G2. Two newer models

TimesFM-3 (`google/timesfm-3.0-pytorch`) and Timestral-209M (`mistralai/Timestral-209M`), on the
12 first-round configurations, at the documented length and at 512.

- **Documented length.** TimesFM-3: what the board entry's notebook
  (`gift-eval/notebooks/timesfm3.ipynb`) passes. Timestral-209M: up to 32,768 steps, the maximum
  on its model card (it keeps the most recent steps). All other settings come from the notebook
  (TimesFM-3) or the model card's defaults (Timestral), are written into `g2_models.py` before
  any run, and are listed in the report.
- **Reproduction check.** Geometric mean of scaled MASE on the 12 configurations at the
  documented length against the published rows: TimesFM-3 in the 2026-09-07 snapshot;
  Timestral-209M at the second pin (GIFT-Eval commit 42beef3e), since it was not on the board in
  September. Criterion 1%, as for Toto. A model that misses it is reported with its effect and
  with the miss stated.
- **Test.** T1 as above on the 12 configurations (12 base datasets, 4,096 sign patterns), Holm
  over the two models, with the interval. T2 with the paper's 9.3%. Twelve configurations give
  wide intervals; this is said next to the numbers.
- If a model cannot be made to run inside the budget, it is reported as NOT MEASURED with the
  reason.

## Order and budget

Order: machine check, then Toto on the 12 configurations (G3 and G4), then G1, then G2. Work that
is ready runs in parallel on the free cards. A timer on the pod stops the pod at the amount the
owner approved. Whatever has not finished by then is reported as not run. Results are copied off
the pod as they land and compared by checksum before the pod is stopped.

## What is not in this round

No retraining, no new board, no change to the 2026-09-07 analysis of record. No test is added
after results are seen; anything added later is labelled as added later.

---

# Revision 1 (2026-11-19, 23:00 IST, after one adversarial review, before any run)

Review: `REVIEW_ROUND3.md` (scripts in `review_scratch/`). No pod existed and no model had been
run when this was written. Where this revision and the text above disagree, this revision holds.
I checked these review points myself before accepting them: the notebook's two switches (lines 437
and 440 of its code), that the compiled module is never called (`gluonts_predictor.py` line 210),
and the source sizes (they add up to 70).

## Groups

The groups of record are **sources**, not base datasets: the three bizitobs datasets are one
source, the two ETT datasets one, the two bitbrains datasets one (the grouping the paper already
uses for its 19 sources). The 70 configurations come from 14 sources (sizes 12, 12, 7, 6, 6, 6,
6, 3, 3, 3, 2, 2, 1, 1); the 12 come from 9. Every test below uses sources. The same tests on
the 18 base datasets are printed beside them, not used for any claim. Reason: in the review's
simulation, datasets of one source that move together make the 18-group test claim an effect
12.5% of the time when there is none, against 5.0% with sources.

## What the tests can and cannot promise

The sign-flip test is exact only when, with no true effect, a source's total is as likely to be
positive as negative. With changes shaped like the first round's (a few large values on one
side), the review's simulation has it claim an effect 5% to 10% of the time, with the excess on
the side opposite the long tail. "Level 5%" and "level 2.5%" above are therefore the nominal
values. Fixed now:

- After the run, for each model the simulation of `review_scratch/level_sim_t1t2.py` is repeated
  on that model's own d_c values, centred to mean zero, in the 14-source layout, with values
  shared inside a source and with values independent, 20,000 draws, seed 20261119. The rate of
  false claims in the direction of the model's observed effect is printed beside its p-values.
  The paper states that rate with the p-value.
- If T1 and the signed-rank test disagree, both are reported; the text follows T1, because it
  tests the board's own statistic, and says that the test of the typical source does not agree.
- The interval the paper prints is the one obtained by inverting the two-sided sign-flip test on
  sources. The CR2 interval stays in the appendix table only.

## T2

- Holm at T2 is over all four models; a model that did not pass T1 with a positive effect
  enters with p = 1. The two families have separate error rates (5% two-sided for T1, 2.5%
  one-sided for T2).
- The shifted total of a source is s_g - n_g x ln(1 + S), with n_g its number of
  configurations. T2 rejecting is the same as the lower end of the inverted 95% interval lying
  above S.
- S is computed from the board snapshot and does not depend on our runs. Its values, written
  down now (`review_scratch/spread_S.py`): 11.76% on the 70, 12.71% on the 58, 9.34% on the 12.
  Leaving one base dataset out of the 70 moves it between 8.51% and 13.22%; this range is
  printed with the result.

## The 70, the 58 and the 12

The sentence that the 58 are "new data" is withdrawn. Of the 58, 43 come from a base dataset that
is already among the 12 and 20 use the same series as one of the 12 at another forecast horizon.
The 70 stay the estimate and the test of record, because they are every configuration that
passes a rule written before any model was run. The paper may say the effect is confirmed on
configurations that did not suggest it only if T1 also rejects on the 58 (Holm over the four
models), and it gives the overlap counts above.

Every table that compares models also reports each model on the configurations that all four
finished.

## Setting A, corrected

Added, as in the notebook: `torch.set_float32_matmul_precision("high")` (faster matrix products
that keep about 3 decimal digits; set inside the Toto builder only, so the other models are
untouched), `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set
before torch is imported. After an out-of-memory retry the seed is not set again (as in the
notebook). `torch.compile` is left out: the notebook compiles a wrapper whose forward is never
called, so it cannot change a forecast.

- **At 512 steps**: history min(512, the notebook's maximal length), batch size from the same
  formula at that length.
- **m4_daily/D/short**: both lengths give Toto the same 79 steps. d_c is 0 by rule and the 512
  run is skipped.
- A result row with a missing or non-finite MASE is a failed run.
- Recorded per run: card, torch and CUDA versions, the two switches, whether xFormers is
  installed (it must not be; the first round had none), batch size, samples per batch after any
  retry, history used, seed, wall time, checksum of `run_grid.py`.
- Not testable here: the board entry was made with Toto's code as it was when Datadog submitted
  it; our clone has one commit (c079fd7). This stays a possible cause of any remaining gap.

## G3 criterion and fall-back

- The gap is read two ways: against 1% (as above; the first round's Chronos-Bolt gap on the same
  measure was 0.27%, not 0.46%), and against sampling noise: "within noise" if 0.7699 lies inside
  the range of the three seeds' values (G4) widened by half that range on each side.
- If setting A meets the 1% criterion, the statement is "the notebook's setting reproduces the
  board". That the sample count is the cause rests on the other changes (cache, batch size)
  being unable to move the expected score.
- The two fall-back variants above are dropped; neither can move the expected score. If setting
  A misses both readings, one variant is run on the 12: setting A with the faster matrix products
  off. Nothing else.

## Machine check

Run under the same conditions as G1 (two processes per card, same batch sizes). Chronos-Bolt
uses batch size 64 on every run (the first round used 1,024, and 64 on the two runs that ran out
of memory). Largest relative difference from the first round: below 1e-4, pass; from 1e-4 to
1e-3, continue, report the size, and run one configuration twice on one card to show repeat runs
agree; above 1e-3, or any d_c changing sign, stop and check versions, weights and data first.
The number of processes per card (2 or 3) is a speed setting and may be raised after the check.
The downloaded data are compared by checksum with the first round's files, and each model's
Hugging Face revision is compared with the one the first round recorded.

## Time limits and Toto's list

- Time limit per run: 90 minutes for Chronos-Bolt, Moirai-2.0, TimesFM-2.5 and the G2 models;
  4 hours for Toto.
- Toto's runs on the 12 (seed 42) finish first. From their run times, each of Toto's remaining
  runs is projected (time per forecast batch times the number of batches). **Before any Toto run
  on the 58 starts, a dated note is added here** with the projection and Toto's list. If the
  projection fits in what is left of the approved amount with a margin of 25%, all 58 are run.
  If not, things are removed in this order until it fits: G2, then seed 44, then seed 43, then
  whole sources for Toto only, the source with the largest projected time first. The note names
  what was removed. No score is looked at for this decision.

## Other fixed points

- Only the geometric mean of scaled MASE is analysed on the 70. The mean-rank column of the
  first round's table is not computed for the 70.
- Numbers that change in the paper: in the first round's table (12 configurations), Toto's row
  becomes setting A, seed 42; the abstract's figure comes from the 70.
- The analysis script is `analyze_gpu_round.py`; its checksum is added below before the first
  result is copied off the pod.
- **Freeze**: nothing that finishes after 2026-11-20 12:00 IST enters the camera-ready. A
  request not finished by then is described in the paper as not run.

## Analysis script frozen (2026-11-19T17:41Z)

`analyze_gpu_round.py` sha256 `f1303124851dfc6cf7a9d7bce872a52f5a32d93da0709f8820b96f53082b20e5` (38 tests pass). Recorded before any result was copied off
the pod; runs had started on the pod at 2026-11-19T17:38Z and no score had been read. Readings
the script fixes where the text above left room: the level check always uses the 14-source
layout of the 70 (9 sources for G2), also when a model drops configurations; G2's T2 uses Holm
over its two models; in the machine check a change to or from exactly zero counts as a change of
sign; "T2 rejects = the interval's lower end is above S" holds before Holm only. Runs that finish
after the freeze time are listed by the script and are left out of the paper by hand.

## Note before Toto's runs on the 58 (2026-11-19T18:47Z; no score has been read)

Toto's 23 runs on the 12 configurations (setting A, seed 42) finished at 18:10Z with no failure
and took 3,363 seconds in all, one run per card. The longest were loop_seattle/5T/long at the
documented length (1,416 s), electricity/H/long (518 s) and solar/10T/short (336 s).

Projection for the 58 (`review_scratch/toto_time_projection.json` scaled to these run times,
`review_scratch/toto_calibration_from_12.json`): 8.6 card-hours with the overall scale (1.30 at
the documented length, 0.80 at 512), 10.7 card-hours with the largest scale seen on a heavy run
(1.40 and 1.78). The longest single run is bitbrains_fast_storage/5T/short at the documented
length, about 110 to 117 minutes, under the 4-hour limit.

What is left of the approved amount at 18:50Z is 34.5 card-hours (4 cards until the timer at
03:28Z). Still to run: the three other models on the 58 (about 3.4 card-hours), Toto on the 58
(10.7), seeds 43 and 44 (1.9), the two newer models (about 1.5). That is 17.5 card-hours, 21.9
with the 25% margin. It fits. **Toto's list is all 58 configurations; nothing is removed.**

The three other models are running three processes per card and are about 2.8 times slower per
run than the weights assumed, because the card is the limit, not the processor. Toto runs one
process per card.
