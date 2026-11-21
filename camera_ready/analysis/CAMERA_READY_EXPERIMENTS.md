# Camera-ready experiments for the two reviewer questions

Run on 2026-11-11. Everything is CPU-only re-analysis of public files; nothing was pushed or
posted anywhere, and the paper itself has not been edited. Camera-ready is due
2026-11-20, 11:59 pm AoE (workshop site; not yet re-checked on OpenReview).

## Short answers

**Reviewer 1 (28 datasets is few; is the bootstrap too narrow?).** Yes, a little. The
paper's rule declares a difference about 6 to 10% of the time when there is none, where 5%
is intended. A standard few-cluster correction (CR2 standard error with Satterthwaite
degrees of freedom) brings this back to 5%. Under the correction:

- "No neighbouring pair in the top ten separates" holds after correction for multiple
  comparisons, under every method tried.
- The resolution widens from 6.2 to 6.8 rank places. "45 times as many datasets" becomes 48.
- The positive counts shrink. Under mean rank, 12 of 45 pairs still separate, but none
  survives the multiple-comparison correction (the paper says 2). Under the geometric mean,
  23 of 45 becomes 10, and 11 after correction becomes 0.
- The context-length experiment weakens most. The size of the effect is unchanged (TimesFM-2.5
  +22.2%, Toto +20.6%), but only one of the eight movements survives correction under an
  exact test, where the paper says four.

**Reviewer 2 (does it generalize beyond GIFT-Eval?).** The same report was run on four more
boards and on ten sub-boards of GIFT-Eval. The pattern holds on every board built from a few
dozen datasets, and the one board with thousands of units behaves differently for a reason
the report makes visible:

- On GIFT-Eval, fev-bench, fev-bench-mini, Chronos Benchmark II and TIME, the smallest
  difference the board can detect is 3 to 20 times the typical gap between neighbours, and
  at most one of nine neighbouring pairs separates under the corrected methods.
- On BOOM, all nine neighbouring pairs separate, provided each of its 2,554 series counts as
  independent evidence. Grouped by domain, five do.
- Rules that rank an entry against the other entries reversed untouched top-ten pairs on
  every board that has a usable history: GIFT-Eval (18 pairs with tied ranks averaged, 20
  under the board's exact tie-break), fev-bench (2), Chronos Benchmark II (8), TIME's rank
  column (6). Rules that depend only on an entry's own scores
  never did, on any board.

One result contradicts the submitted paper: fev-bench is not the stable contrast the paper
describes. With its history now available, two untouched top-ten pairs reversed in October
2026.

## How this was run

1. The design was written before any run ([DESIGN.md](DESIGN.md)), attacked by a strict
   reviewer, and revised before any run (Revision 2 in the same file). The review confirmed
   the correction formulas by brute force and found that three planned checks could not have
   come out against the paper; those were replaced.
2. The outside boards, their units, what counts as an independent source on each, and the
   reproduction tolerances were fixed in [boards/PREREG_BOARDS.md](boards/PREREG_BOARDS.md)
   before any of their scores was analysed.
3. Every analysis first had to reproduce published numbers (the paper's own counts, or the
   board's own displayed order).
4. All statistics come from one tested module (`vernier_stats.py`) and one shared report
   function (`vernier_board.py`), so the boards are comparable.
5. The main session re-derived the load-bearing numbers with separate code: the
   degrees-of-freedom values, the wild bootstrap against a brute-force enumeration, the
   calibration rates, the GIFT-Eval counts and corrected resolution, the context-experiment
   p-values, the BOOM and TIME headline rows, and both fev-bench reversal events (with the
   board's own leaderboard function).

Methods, in plain terms:

| Name here | What it is |
|---|---|
| Paper rule | Percentile bootstrap, once resampling configs and once resampling whole datasets; a gap counts only if both intervals exclude zero |
| CR2 | Cluster-robust standard error with the Bell-McCaffrey adjustment, t interval with Satterthwaite degrees of freedom (about 11 to 17 on GIFT-Eval). Primary corrected method, chosen by a rule fixed in advance |
| Wild bootstrap | Wild cluster bootstrap-t with the null imposed: every dataset's differences get one random sign |
| Means t-test | t-test on the 28 dataset means. Weights datasets equally, so it answers a slightly different question; shown as a sensitivity only |

---

## Reviewer 1: validation with small-sample corrections

### 1.1 Is the paper's rule too lenient, and what fixes it?

Simulated data with GIFT-Eval's exact layout (28 datasets contributing 1 to 8 configs) and
no true difference. Entries are the percentage of cases in which a difference was wrongly
declared; the target is 5%. Simulated rows use 4,000 simulated boards each (Monte-Carlo error
about 0.35 points); the two real-data rows pool 45 pairs with 1,000 resamples of the 28
datasets each.

| Simulated data | Paper rule | CR2 | Wild bootstrap | Means t-test |
|---|---|---|---|---|
| Configs independent | 4.4 | 5.0 | 5.0 | 4.9 |
| Mild within-dataset correlation (0.1) | 6.0 | 5.0 | 4.6 | 4.9 |
| Moderate correlation (0.3) | 7.0 | 4.8 | 4.7 | 5.5 |
| Moderate correlation, heavy tails | 8.1 | 4.4 | 4.8 | 4.7 |
| Moderate correlation, strongly lopsided dataset effects | 8.4 | 5.8 | 5.9 | 5.3 |
| Moderate correlation, effect grows with dataset size | 8.5 | 5.8 | 5.4 | 4.9 |
| Strong correlation (0.5), strongly lopsided | 9.5 | 6.8 | 7.0 | 5.4 |
| **Real GIFT-Eval pairs, resampled (mean rank)** | **6.1** | **5.2** | **5.4** | **6.0** |
| Real GIFT-Eval pairs, resampled (geometric mean) | 16.2 | 15.7 | 15.5 | 8.4 |

The real mean-rank pairs have within-dataset correlation of 0 to 0.51 (median 0.14), so the
mild and moderate rows are the relevant ones.

- The paper's rule is too lenient whenever configs within a dataset move together, by one to
  five points. Its independent-configs scheme alone rejects 10 to 35% of the time; requiring
  both schemes is what keeps the rule near 6 to 9%.
- CR2 and the wild bootstrap are close to 5% except under strongly lopsided data, where both
  drift to 6 to 7%.
- By the rule fixed in advance, CR2 is the primary corrected method: 5.2% on the resampled
  real mean-rank pairs, inside the 4 to 6% band.
- The geometric-mean row is a separate problem, covered in 1.4.

**Does the "resolution" deliver the 80% detection it claims?** Two checks on resampled real
pairs. With a true difference of 2.80 times each pair's own standard error (the definition
behind the resolution), the paper's rule detects it 79% of the time and CR2 75%. With a
difference of exactly 6.2 rank places on the nine neighbouring pairs, the paper's rule
detects it 74% of the time and CR2 71%; at the corrected 6.8, 79% and 75%. In simulation CR2
reaches 79 to 93% at its own corrected resolution. The claim is roughly right for the
paper's own rule, partly because that rule is lenient, and a few points short of 80% on the
real data for every method.

### 1.2 The headline counts under the corrected methods (GIFT-Eval)

The paper's published counts were reproduced first. Cells are: neighbouring pairs separated
(of 9) / all top-ten pairs separated (of 45) / after Holm correction over the 45.

| Rule | Paper rule | CR2 | Wild bootstrap | Means t-test |
|---|---|---|---|---|
| Mean rank on MASE (what the board sorts by) | 0 / 12 / 2 | 1 / 12 / 0 | 1 / 13 / 1 | 0 / 7 / 0 |
| Mean rank on CRPS (the earlier rule) | 0 / 16 / 0 | 0 / 9 / 0 | 0 / 12 / 0 | 0 / 7 / 0 |
| Geometric mean of scaled MASE | 2 / 23 / 10 | 0 / 10 / 0 | 0 / 19 / 4 | 0 / 12 / 0 |

- After Holm correction (computed over all 45 top-ten pairs), no method separates any
  neighbouring pair under any rule. The same holds correcting over the nine neighbours
  alone: the smallest neighbouring p-value under the corrected methods is 0.036, against a
  threshold of 0.0056.
- The one neighbouring pair that separates uncorrected under the corrected methods is the
  pair at positions 9 and 10 under mean rank: p = 0.036 (CR2), 0.041 (wild bootstrap), 0.073
  (means t-test). The paper already reports this pair as clearing the dataset-resampling
  scheme alone; the paper's rule returned 0 of 9 because it also required the
  independent-configs scheme.
- The geometric mean loses both of its neighbouring pairs and all of its corrected pairs
  under CR2.
- As in the paper, the geometric-mean counts here are for the displayed top ten (the
  mean-rank order). The geometric-mean resolution and tier in 1.3 use that rule's own order,
  as the paper's resolution script does, so the two describe different sets of ten.

### 1.3 Resolution, tiers and robustness

| Rule | Resolution, paper | Resolution, corrected | Median gap between neighbours | Datasets needed, paper → corrected |
|---|---|---|---|---|
| Mean rank on MASE | 6.16 rank places | 6.80 | 0.91 | 45 times → 48 times as many |
| Mean rank on CRPS | 6.83 | 7.38 | 0.67 | 118 → 119 |
| Geometric mean (log scale) | 0.0440 | 0.0497 | 0.0045 | 253 → 270 |

The corrected standard error is only 1 to 3% larger than the bootstrap one. The resolution
grows 8 to 13% because 28 unequal clusters call for a t multiplier of about 3.0 instead of
2.8.

- **Leading tier** (entries not separated from the displayed leader by uncorrected tests):
  mean rank on MASE stays at 3 entries under the paper rule, CR2 and the wild bootstrap;
  mean rank on CRPS goes from 2 to 3 under CR2; the geometric mean goes from 9 to 17 under
  both CR2 and the wild bootstrap. TimesFM-3's place in that tier is no longer a narrow
  call. The dataset-means t-test, which has less power, gives larger tiers still (11, 12
  and 26).
- **Coarser sources.** Six of the 28 datasets are M4 subsets, two are ETT, three are
  bizitobs, two are bitbrains. Grouped into 19 sources: one neighbouring pair uncorrected
  (9-10), none after correction; resolution 6.75; about 40 times as many sources needed.
- **Leaving out one dataset at a time:** the corrected resolution stays between 6.17 and
  7.43. The largest change comes from dropping `kdd_cup_2018`.
- **Subsets of datasets** (a stability check, not a test of the extrapolation): standard
  error times the square root of the number of datasets is flat from 14 to 28 datasets
  (0.99, 1.00, 1.00) and 0.87 at 7.

### 1.4 The geometric mean rests on one dataset

Found while calibrating, and new relative to the paper. On resampled real data, every
method (corrected or not) wrongly declares a difference about 16% of the time for
geometric-mean pairs, and above 10% for 31 of the 45 pairs.

The cause: a geometric-mean comparison between two top-ten entries usually rests on a single
dataset. In 28 of 45 pairs one dataset supplies more than half of the variance estimate
(median share 61%, maximum 93%), most often `solar`, which contributes 8 configs. Under mean
rank the median share is 31% and only 3 pairs exceed one half, because ranks are bounded.
TIME shows the same thing on its own data (18 of 45 pairs for its geometric mean, 7 for its
mean rank).

Consequence: pair counts under the geometric mean cannot be trusted in either direction, and
the paper's observation that this rule is "the sharper of the three" does not hold.

### 1.5 fev-bench (the September table used in the paper)

| | Paper rule | CR2 | Wild bootstrap |
|---|---|---|---|
| Neighbouring pairs separated, win rate (of 9) | 1 | 1 | 1 |
| All 45 after Holm, win rate | 15 | 5 | 10 |
| Resolution, win-rate points | 9.52 | 11.37 | |

The one separated pair (positions 1 and 2) survives CR2 at p = 0.041. Median gap between
neighbours 2.17; families needed for it 306 → 313 of 39.

### 1.6 The context-length experiment (12 configs)

The concern is sharper here than on the 28 datasets: with 12 units, the paper's bootstrap
declares a false difference 8 to 13% of the time in simulation. An exact sign-flip test (all
4,096 sign patterns; assumes the 12 differences are independent and symmetric about zero)
and a paired t-test (11 degrees of freedom) replace it.

| Model | Change at 512 steps | Paper interval | t interval | Exact p | t-test p |
|---|---|---|---|---|---|
| TimesFM-2.5 | +22.2% | +5.2 to +45.6 | +0.2 to +48.9 | 0.018 | 0.048 |
| Toto | +20.6% | +8.1 to +36.8 | +4.4 to +39.3 | 0.004 | 0.015 |
| Moirai-2.0 | +6.8% | +1.0 to +14.0 | -0.4 to +14.6 | 0.071 | 0.064 |
| Chronos-Bolt | -2.0% | -13.2 to +8.9 | -14.2 to +12.0 | 0.75 | 0.75 |

Movements separated, out of the eight (four models, two aggregation rules):

| Method | Uncorrected | After Holm over the eight |
|---|---|---|
| Paper bootstrap | 5 | 4 |
| Exact sign-flip | 4 | 1 (Toto, geometric mean) |
| Paired t-test | 4 | 0 |
| With the 12 configs grouped into 9 sources (CR2 / wild) | 2 / 3 | 1 / 0 |

Moirai's movement is no longer distinguishable from zero. TimesFM-2.5 and Toto remain
significant uncorrected under both rules; only Toto's geometric-mean movement survives
correction. Even the corrected tests run at about 10% false alarms on strongly lopsided data
at 12 units, so this experiment supports "a large effect, measured imprecisely" and no more.

---

## Reviewer 2: other boards, compositions and sizes

### 2.1 Which boards

Every forecasting board whose per-entry, per-unit scores could be obtained was analysed.

| Board | Source (commit) | Entries | Units | Independent sources | Rule it sorts by |
|---|---|---|---|---|---|
| GIFT-Eval | paper's snapshot | 130 (128 complete and displayable) | 97 configs | 28 datasets | mean rank on MASE |
| fev-bench, current | autogluon/fev 9cf943c | 40 shown | 100 tasks | 39 task families | win rate |
| fev-bench-mini | same | 40 | 20 tasks | 15 families | win rate |
| Chronos Benchmark II | same repository | 21 | 27 tasks | 19 sources | none fixed; three rules reported |
| BOOM | DataDog/toto 1d3e1c9 | 23 | 6,750 series-horizons | 2,554 series from one company | mean rank on the probabilistic loss |
| TIME | Real-TSF/TIME-Output 3bf103c | 43 | 98 | 39 datasets | geometric mean of scaled MASE |

Not included, because no per-entry, per-unit file is published: Chronos Benchmark I,
OpenTS/TFB (scores behind a private API), ProbTS (markdown tables only), FoundTS (repository
gone), Time-Series-Library, the Monash archive repository. TS-Arena's archive (3.6 GB) was not
inspected. LiveHouse-TS and Impermanent could not be found. No board was dropped because of
its results.

Reproduction: fev-bench, mini and Chronos Benchmark II match the board's own leaderboard
function exactly; BOOM matches its published table in all six columns at the three decimals
it publishes, and in order for all 23 models; TIME matches the board's own function for all
43 entries.

### 2.2 The same report on every board

"Resolution" is the corrected one (CR2). "Ratio" is resolution divided by the median gap
between displayed neighbours: above 1 means the board displays finer distinctions than it
can detect. Neighbouring pairs separated are out of 9, uncorrected, shown as paper rule /
CR2 / wild bootstrap.

| Board and rule | Depends on the other entries? | Resolution | Median neighbour gap | Ratio | Neighbours separated | Untouched top-ten pairs reversed over the board's history |
|---|---|---|---|---|---|---|
| GIFT-Eval, mean rank on MASE | yes | 6.80 rank places | 0.91 | 7.5 | 0 / 1 / 1 | 18 in 14 of 141 updates (20 under the board's exact tie-break) |
| GIFT-Eval, geometric mean | no | 0.0497 (log) | 0.0045 | 11 | 2 / 0 / 0 | 0 in 141 |
| fev-bench, win rate | yes | 7.51 points | 2.08 | 3.6 | 0 / 0 / 0 | 2 in 2 of 30 updates |
| fev-bench, skill score | no | 3.20 | 0.83 | 3.9 | 0 / 0 / 0 | 0 in 30 |
| fev-bench-mini, win rate | yes | 15.3 points | 2.18 | 7.0 | 1 / 1 / 1 | not replayed |
| Chronos Benchmark II, mean rank | yes | 4.24 rank places | 0.22 | 19 | 1 / 0 / 0 | 8 in 3 of 11 updates |
| Chronos Benchmark II, geometric mean | no | 9.29 | 0.59 | 16 | 0 / 0 / 0 | 0 in 11 |
| TIME, geometric mean (displayed) | no | 0.0238 (log) | 0.0021 | 11 | 1 / 0 / 1 | 0 in 28 |
| TIME, mean rank column | yes | 4.01 rank places | 0.378 | 11 | 1 / 1 / 1 | 6 in 5 of 28 updates (replayed on a smaller pool than the board showed; see limits) |
| BOOM, each series independent | yes | 0.196 rank points | 0.312 | 0.63 | 9 / 9 / 9 | history too short (4 commits) |
| BOOM, grouped by domain and type (36 groups) | yes | 0.420 | 0.312 | 1.35 | 7 / 6 / 6 | |
| BOOM, grouped by domain (23 groups) | yes | 0.635 | 0.312 | 2.04 | 7 / 5 / 6 | |

fev-bench skill and Chronos geometric-mean values are log error ratios times 100. On Chronos
Benchmark II, win rate and mean rank give the same ordering by arithmetic, so only mean rank
is listed.

What the table shows:

1. **Resolution.** Every board built from 15 to 39 independent sources displays neighbours
   3 to 20 times closer together than it can detect, and separates at most one of nine
   under the corrected methods. This holds under mean rank, win rate, skill score and
   geometric mean alike.
2. **A board that does resolve its order.** BOOM's resolution is finer than its typical gap
   and all nine neighbours separate, so the report is not one that calls every board
   unresolved. That result rests on treating 2,554 series from one company's monitoring data
   as independent. Grouped by domain, 5 of 9 separate and the corrected tests have about 2
   degrees of freedom, because one group holds 44% of the rows.
3. **Pool dependence.** On all four boards with a usable history, the rule that ranks an
   entry against the others reversed top-ten pairs whose own results had not changed, and
   the rule that uses only an entry's own scores reversed none. The second half is true by
   construction; the first half is the measurement.
4. **Fewer units, coarser board.** fev-bench-mini (20 tasks) has twice the resolution of the
   full board (15.3 against 7.5 points) with the same gaps.

### 2.3 fev-bench's history: a correction to the submitted paper

The paper could not replay fev-bench and reported instead that removing any single entry
from its September table reverses no top-ten pair. Its repository now has 32 commits of
results. Replaying them under win rate, with the board's own function:

| Update | What arrived | Untouched pair that reversed |
|---|---|---|
| 2026-10-13 → 10-14 | timeh2o-v1-tiny | ChakraTS (85.39 → 84.66) and TimesFM-3 (85.26 → 84.91): second and third place swapped |
| 2026-10-14 → 10-15 | TimeRSI | TiRex-2 (73.22 → 71.58) and Toto-2.0-2.5B (73.06 → 71.67): eighth and ninth swapped |

Both events were re-derived by the main session. Under skill score there were none. On the
current 40-entry board, removing one entry from outside the top ten reorders it in 2 of 30
cases and adding a near-copy does so in 8 of 40. The current board also separates 0 of 9
neighbours, where the September table separated 1.

So fev-bench is not a stable contrast to GIFT-Eval. Its rate is lower (2 reversed pairs in
30 updates against 18 to 20 in 141), and it has grown more crowded at the top since
September.

### 2.4 GIFT-Eval's own sub-boards

The app shows tabs by horizon, frequency, domain and variate type. Ten of these groups have
at least 6 sources and were analysed (the full board is the first row). The app's tabs show
geometric means and a CRPS mean rank per group and have no default sort, so the order used
here, mean rank on MASE within the group, is constructed for comparability with the full
board and is not something the app displays.

| Group | Configs / sources | Corrected resolution | Median neighbour gap | Ratio | Neighbours separated (paper / CR2 / wild) | Leader |
|---|---|---|---|---|---|---|
| Full board | 97 / 28 | 6.80 | 0.91 | 7.5 | 0 / 1 / 1 | STRIDE w/ Synapse |
| Short horizon | 55 / 28 | 8.38 | 1.22 | 6.9 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Medium horizon | 21 / 14 | 11.0 | 1.24 | 8.9 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Long horizon | 21 / 14 | 13.4 | 0.67 | 20 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Univariate | 54 / 20 | 11.8 | 0.65 | 18 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Multivariate | 43 / 8 | 7.95 | 0.21 | 38 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Hourly | 31 / 13 | 14.7 | 1.45 | 10 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Minutely | 30 / 10 | 10.7 | 0.40 | 27 | 0 / 0 / 0 | STRIDE w/ Synapse |
| Daily | 15 / 15 | 17.0 | 1.67 | 10 | 0 / 0 / 0 | EXAONE-Forecast-Agent |
| Weekly | 8 / 8 | 36.7 | 1.63 | 23 | 0 / 0 / 0 | LS-MoE |
| Econ/Fin domain | 6 / 6 | 21.3 | 0.50 | 43 | 0 / 0 / 0 | CastStar |

No sub-board separates a neighbouring pair under any method. Resolution gets coarser as the
group gets smaller (slope about -0.5 against the number of configs on a log scale, which is
what an inverse-square-root law predicts, with wide scatter). The leader changes on three of
ten groups. Ten more groups have fewer than 6 sources and were not analysed, and Econ/Fin,
with exactly 6, is at the edge of what any of these methods can support.

### 2.5 Would new datasets behave like the current ones?

"45 times as many datasets" assumes new datasets vary like the existing 28. Fifteen entries
are scored on both GIFT-Eval and fev-bench with the same checkpoint (13 pretrained models
and the two naive baselines), giving 105 pairs. For each pair, the source-to-source spread of their log error difference was computed
on GIFT-Eval's 28 datasets and on the 25 fev-bench families that share no data source with
GIFT-Eval.

| Quantity, fev-bench non-shared families divided by GIFT-Eval | Median over 105 pairs | 95% interval | Range |
|---|---|---|---|
| Spread of source means | 1.13 (pooled over pairs: 0.89) | 0.85 to 1.72 | 0.24 to 3.52 |
| Per-source noise scale used by the "times as many" calculation | 1.25 (pooled: 1.12) | 1.11 to 1.43 | 0.23 to 5.33 |

On the first measure new datasets vary about as much as the current ones; on the second,
which is the one the calculation uses, somewhat more. The extrapolation is supported but not
settled, and if anything it is too small (a ratio of 1.25 would mean about 1.5 times as many
datasets again). The intervals are optimistic: the 15 models are closely
related (five Toto sizes, two TimesFM versions), and horizons and window counts differ
between the boards.

### 2.6 Overlap between boards

- **fev-bench:** 9 of its 39 families share a data source with GIFT-Eval by name (ETT,
  LOOP_SEATTLE, M_DENSE, SZ_TAXI, bizitobs_l2c, hierarchical_sales, hospital, jena_weather,
  restaurant), and 5 more are uncertain. With all 14 removed (67 tasks, 25 families): 0 of 9
  neighbours separated, resolution 10.4 points against a 1.4-point median gap.
- **Chronos Benchmark II:** 7 of 27 tasks share a source with GIFT-Eval (M4, ETT, three
  Monash sets); 2 more uncertain.
- **TIME:** no dataset shares a source by name; 8 are the same topic from a different
  provider. With those removed: 0 of 9 under CR2, ratio 16.
- **BOOM:** Datadog's internal monitoring data; no public dataset named.
- Many entries are shared: 15 of BOOM's 23 models match GIFT-Eval entries by name and 5
  more match with the version unconfirmed, and at least 15 of fev-bench's entries are on
  GIFT-Eval. The boards are independent in their data far more than in their entries.

---

## What this changes in the submitted paper

| Paper says | Result now | Suggested change |
|---|---|---|
| Resolution 6.2 rank places | 6.8 with the small-sample correction | Report 6.8, or both, and name the correction |
| No neighbouring pair separates under either rank rule | Holds after correction for multiple comparisons under every method; pair 9-10 clears 5% uncorrected (p about 0.04) | Keep the claim; say "after correction for multiple comparisons" and keep the 9-10 disclosure |
| 12 of 45 pairs separate, two after Holm | 12 of 45, none after Holm (CR2); one under the wild bootstrap | Replace "two" |
| Earlier rank rule: 16 of 45, none after Holm | 9 of 45, none after Holm | Replace 16 |
| Geometric mean: 2 of 9, 23 of 45, 11 after Holm; "the sharper of the three" | 0 of 9, 10 of 45, 0 after Holm; no method holds its level on this rule | Cut the sharpness sentence; add that geometric-mean comparisons rest on one dataset in 28 of 45 pairs |
| Leading tier under the geometric mean holds nine entries, TimesFM-3 narrowly | 17 entries under the corrected methods | Drop "narrowly"; the point gets stronger |
| Roughly 45 times as many datasets | 48 with the corrected standard error; new datasets look at least as variable | "roughly 50 times, and probably more" |
| fev-bench: removing any single entry reverses no pair; fev-bench as the stable control | 2 untouched top-ten pairs reversed in 30 updates; current board separates 0 of 9 | Rewrite: the control is the rule that does not depend on the pool (zero reversals on four boards); fev-bench shows the same effect at a lower rate |
| Table 1, fev-bench column | Based on the September table | Update to the current board, or state the date |
| Context experiment: both movements survive correction across the eight tests; Table 2 intervals | One survives under the exact test (Toto, geometric mean), none under the t-test; TimesFM +0.2 to +48.9, Toto +4.4 to +39.3, Moirai -0.4 to +14.6 | Replace intervals and the sentence; say the effect is large and imprecisely measured |
| Appendix: "a wild cluster bootstrap is the natural next check" | Done | Replace with the result |
| (not in the paper) | Four more boards, ten sub-boards, BOOM as the board that does resolve | New table (2.2) and one paragraph |

The reversal count on GIFT-Eval (20 under the board's exact tie-break, 18 with ties
averaged), the census, the effective-context findings and the sizes of the context effect
are untouched by any of this.

---

## Deviations, failures and limits

- **Geometric-mean Holm count reproduced as 10, published as 11.** One borderline pair
  (p = 0.00146 against a threshold of 0.00143), within bootstrap noise. The paper's
  bootstrap may also have used the ratio scale where this round uses log ratios; that was
  not checked.
- **BOOM reproduction tolerance.** The plan asked for agreement within one part in a
  million; BOOM publishes three decimals, so "within half a rounding unit" was used. Its
  ranks have 2,286 tied rows broken by folder order; changing that order moves mean ranks
  by up to 0.09 but not the displayed order.
- **BOOM bootstrap sizes** were reduced at the series level (5,000 bootstrap draws, 19,999
  wild draws, 80 known-difference resamples per pair) to fit the run time.
- **fev-bench pool.** The plan said "models with complete results"; the board itself fills
  15 missing cells with the baseline and shows those 5 models, so the analysis follows the
  board. With complete models only (35 entries): same top-ten set, still 0 of 9.
- **fev-bench live page not compared.** Its tables are generated when the site is built and
  are not in its repository. The site's code also selects models differently from the
  repository's script (45 entries and a different top ten), so which pool the live page
  shows is uncertain.
- **TIME live page has four entries more** than the pinned files (the Simons models), one of
  them in the top ten. The 43 shared entries match exactly.
- **TIME history only partly replayable.** A data refresh on 2026-06-02 changed the
  baseline's files and those of the models then present, and the earlier contents were not
  downloaded, so the first 16 of 45 commits cannot be scored: 29 states, 28 transitions.
  Within those, entries whose files later changed could not be scored either, so the
  mean-rank replay ran on a smaller pool than the board showed at the time (13 of 22 entries
  at the start). Its 6 reversals show the rule's behaviour on that pool; they are not a
  record of what the live page displayed. The geometric-mean result (zero) does not depend
  on the pool.
- **TIME loader.** The board's own code needs a 3.9 GB predictions file that was not
  downloaded; only its loader was replaced, to read the metrics files. All 43 entries then
  match the board's function.
- **Chronos Benchmark II.** 74 cells flagged as trained on the test dataset were kept as
  published, with no sensitivity run. The board's function fails on this board with its
  default arguments and was called with task names as the key.
- **Known-difference control below 80% in places.** At the corrected resolution, CR2 detects
  75% on GIFT-Eval and 73 to 82% on TIME. On BOOM grouped by domain and type, detection is
  65% (paper rule), 63% (CR2) and 79% (wild bootstrap), with false alarms of 9% and 8% for
  the paper rule and the wild bootstrap. Grouped by domain: 70%, 75% and 92%, with false
  alarms of 9%, 6% and 6%. With a few dozen very unequal groups, no method is reliable.
- **BOOM secondary rule.** Its MASE-based rank was analysed only with each series as its own
  source, on the primary top ten.
- **CR2 is not uniformly best.** On fev-bench's layout with strong within-family correlation
  it is too cautious (2 to 3% where 5% is intended); the wild bootstrap is better there.
  CR2 stays primary because the rule for choosing it was fixed in advance.
- **Wild bootstrap gives p-values only**; no intervals were computed from it.
- **Shared code file had a duplicated block** from a bad write (the second, correct copy
  was the one in effect). Removed after the board analyses finished; the regression test was
  re-run afterwards (see the last paragraph of this file).
- **Process slips.** One agent ran the shared-code regression test after its analysis
  instead of before (it passed). One agent stopped a test process that may have belonged to
  another; every analysis still completed and was checked.
- **Not done.** The planned subset curves on fev-bench (10, 20, 30 and 39 families) were not
  run; the review had already shown that such curves cannot fail, and the GIFT-Eval one is
  reported only as a stability check. Uncertainty from within each config (rolling windows)
  and repeat runs of the models need the GPU and were not asked for. Chronos Benchmark I and
  TS-Arena were not analysed.

## Files

All under `papers/fmts/camera_ready/`.

| File | Contents |
|---|---|
| `DESIGN.md`, `boards/PREREG_BOARDS.md` | The plan, its revision after review, and the board definitions (hashes `ac208f04…`, `9ebd0364…`) |
| `vernier_stats.py`, `vernier_board.py` and their tests | Shared, tested methods and report code |
| `CALIBRATION.md` | Full false-alarm and detection tables (44 simulated settings, plus resampled real data) |
| `RETEST_GIFT.md`, `RETEST_FEV.md`, `RETEST_CONTEXT.md` | Reviewer 1 re-tests in full |
| `BOARD_FEV_FAMILY.md`, `BOARD_BOOM.md`, `BOARD_TIME.md`, `BOARD_GIFT_SUBBOARDS.md` | Reviewer 2 boards in full, with event lists |
| `CROSS_BOARD_SPREAD.md` | The new-datasets check |
| `boards/SCOUT_REPORT.md` | What each board publishes, with commits |
| `*.json` beside each | Every number above, per pair |

Regression check after the code cleanup (2026-11-11): `test_vernier_board.py` ends "ALL OK"
on the submitted paper's GIFT-Eval numbers (0 of 9, 12 of 45, 2 after Holm, leading tier of
3, 28 of 121 removals, 42 of 128 near-copies, replay 141 / 18 / 14), and its output matches
the pre-cleanup output to 15 significant digits.
