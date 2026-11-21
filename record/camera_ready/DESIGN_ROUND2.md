# Camera-ready, second round: design (written 2026-11-13, before any of these runs)

Source of the requests: `STILL_NEEDED.md` (2026-11-12), sections 2 and 3. Section 2 asks for
values that already exist; they need no design. This file fixes the new analyses of section 3.
Everything is CPU-only re-analysis of public files. Nothing here may change after results are
seen without a dated note.

Shared conventions, unchanged from `DESIGN.md` and `boards/PREREG_BOARDS.md`: the pairs are the
45 pairs of the displayed top ten; d_c is the per-configuration difference (lower-placed minus
higher-placed; a log ratio for the geometric mean); 28 datasets are the clusters on GIFT-Eval;
Holm correction is applied over the 9 neighbouring pairs and, separately, over all 45; seed
20261111.

## S1. Exact p-values by full enumeration

For a pair, let s_g be the sum of d_c over the configurations of dataset g and n_g their number
(G = 28, N = 97). A sign pattern w is one value of +1 or -1 per dataset. There are 2^28
patterns. Two exact tests are computed, both two-sided, both counting the observed pattern:

- **Sum statistic (the test asked for).** p = share of patterns with
  |sum_g w_g s_g| >= |sum_g s_g|. Computed by splitting the datasets into two halves of 14,
  listing the 2^14 partial sums of each, sorting one, and counting.
- **Studentized statistic (what the wild cluster bootstrap in `vernier_stats.wild_cluster_p`
  approximates by random draws).** For each pattern, t*(w) = mean*(w) / se*(w) with
  mean* = A/N, A = sum_g w_g s_g, and se*^2 = (G/(G-1)) (Q - 2 mean* C + mean*^2 R) / N^2,
  where C = sum_g w_g n_g s_g, Q = sum_g s_g^2, R = sum_g n_g^2. p = share of patterns with
  |t*| >= |t_obs|. Computed exactly over all 2^14 x 2^14 combinations of the two halves.

The two statistics give the same test only when all datasets have the same number of
configurations. They do not on GIFT-Eval, so the two p-values can differ by more than random
error. Both are reported.

Ties: a pattern counts if its statistic is at least the observed one minus a relative
tolerance of 1e-9.

Check before use: on a random layout with 12 clusters, both routines must equal brute force over
all 4,096 patterns to 1e-12; on one real pair the studentized exact p must lie within 4 Monte
Carlo standard errors of the stored 99,999-draw value.

Reported for each of the three rules: neighbours separated (of 9) and pairs separated (of 45),
uncorrected and after Holm, under each exact test; the largest absolute difference between the
exact studentized p and the stored wild-bootstrap p (random error only), and between the exact
sum-statistic p and the stored wild-bootstrap p (random error plus the change of statistic).

BOOM: the same on the displayed top ten with the 23 domain groups (both statistics) and the 36
domain-and-type groups (sum statistic exact; the studentized statistic by 999,999 random draws,
because 2^36 combinations of a two-number statistic are too many to list).

## S2. Cross-check against reference software

On the 45 GIFT-Eval pairs under MASE_Rank:

- CR2 standard error and degrees of freedom with independent-errors working model: R package
  `clubSandwich` (`coef_test(lm(d ~ 1), vcov = "CR2", cluster = dataset, test = "Satterthwaite")`).
  Pass: relative difference below 1e-8 for the standard error and 1e-6 for the degrees of
  freedom, on all 45.
- Degrees of freedom under the working model with within-dataset correlation: R package
  `dfadjust` (Imbens and Kolesar), which estimates the correlation its own way. Two comparisons:
  (i) its degrees of freedom against ours when our formula is given its correlation estimate
  (pass: 1e-6); (ii) its degrees of freedom and p-values as shipped against ours, reported as a
  sensitivity, with the counts (neighbours of 9, pairs of 45, Holm) under its version.
- Wild cluster bootstrap p-values: Python package `wildboottest` (null imposed, Rademacher
  weights, 99,999 draws) or R `fwildclusterboot` if it installs. Pass: every pair within 4 Monte
  Carlo standard errors of the exact studentized p-value of S1.

If a reference package cannot be installed, the item is reported as not done, with the reason.

## S3. A test for the geometric mean that holds its level

Problem from the first round: on resampled real geometric-mean pairs every method rejects a
true null about 16% of the time (8.4% for the t-test on dataset means), because one dataset
often carries most of the variance.

**What is being tested changes with the test, and this is stated wherever a result is quoted.**
The board's statistic weights each configuration equally. Each candidate below tests a
different summary of the same per-configuration log ratios.

Candidates, tried in this order. The first that passes is the one reported as "the test"; all
are reported.

1. **Sign-flip on dataset means.** m_g = mean of d_c in dataset g. Exact sign-flip test (S1,
   sum statistic) on the 28 values m_g - theta_0. Summary tested: the equal-weight mean of
   dataset means.
2. **The same after winsorizing at 5% per tail.** Within the pair, the 97 values d_c are clipped
   to their own 5th and 95th percentiles (numpy `percentile`, linear interpolation); then as 1.
   Summary tested: the equal-weight mean of dataset means of the clipped values.
3. **The same after winsorizing at 10% per tail** (10th and 90th percentiles).
4. **Sign-flip on dataset-level signed ranks.** Rank |m_g - theta_0| over the 28 datasets
   (average ranks for ties); statistic = sum of sign(m_g - theta_0) x rank; exact two-sided
   p-value over all 2^28 sign patterns. Summary tested: the centre of the dataset means in the
   sense of the median of all pairwise averages (m_g + m_h)/2, g <= h.

**Level check (same real-data resampling as the first round, arm D4).** For each of the 45
geometric-mean pairs used in the first round's D4 arm (`calib_d4_data.npz`), 1,000 replicates:
draw 28 datasets with replacement (a dataset drawn twice counts as two datasets), and test the
null that the summary equals its value on the original 28 datasets. For candidates 2 and 3 the
"value on the original data" uses clipping points computed on the original 97 values; the
replicate computes its own clipping points. For candidate 4 it is the median of the 28 x 29 / 2
pairwise averages of the original dataset means. Rejection at 5%.

**Pass:** pooled rejection rate over the 45 pairs in [4%, 6%]. (Monte Carlo standard error
about 0.1 points.) Also reported, not deciding: the range of the per-pair rates, and the same
check on the MASE_Rank pairs.

**Power, so that a test cannot pass by never rejecting.** In the same replicates, add a constant
to every d_c equal to the first round's corrected resolution for this rule (0.0497 in log units)
and report the detection rate, next to CR2's on the same replicates.

**If a candidate passes,** on the real data report: neighbours separated of 9, pairs separated
of 45, Holm counts, and the leading tier with its members (entries not separated from the
displayed leader by uncorrected tests), for the geometric-mean order's own top ten and the 39
entries the first round used for tiers. **If none passes,** that is the result, and the paper
keeps "at least 17".

Known limit, stated with the result: resampling the 28 observed datasets cannot produce a
dataset more extreme than those observed.

## S4. Per-pair table under CR2

For all 45 pairs and the three rules, from `retest_gift.json` and S1: gap, CR2 standard error,
degrees of freedom, 95% t interval, p (CR2), p (wild bootstrap, 99,999 draws), exact p (both
statistics), Holm-adjusted p over 45 (and over 9 for neighbours), per-pair resolution. One CSV
and one markdown table per rule. No new statistics.

## S5. The boards today

The 2026-09-07 snapshot (commit b61c48aa) stays the analysis of record. Second pin: for each of
GIFT-Eval, fev-bench and TIME, the newest commit of the same repository at the time of fetching
on 2026-11-13, recorded before any score is read. Same code as the first round
(`vernier_board.report_from_matrix`, `history_replay`), same units, sources and rules as
`boards/PREREG_BOARDS.md`.

Reported: (a) GIFT-Eval updates after 2026-09-07 and the top-ten pairs they reversed without
touching either entry, under both tie conventions used so far (tied ranks averaged; the board's
exact tie-break); (b) the first-round report on the new pin; (c) for the entry named
Timestral-209M (or the closest name on the board, stated), its place on each board and the
tests against the entry displayed directly below it (paper rule, CR2, wild bootstrap, exact).
If it is not on a board, that is reported.

## S6. TIME before its data refresh

Find whether the per-entry files from before 2026-06-02 can be obtained (the first round could
score only the 29 later states). If they can, replay every update on the pool the board showed,
with the board's own rule and the rank columns; if they cannot, say exactly what is missing.

## S7. Smaller checks

- **Chronos Benchmark II without results on datasets a model was trained on (74 cells).**
  Primary: whatever the `fev` library's own leaderboard code does with results flagged
  `trained_on_this_dataset` (read the code first). Secondary: drop every task that has at least
  one flagged cell, for all entries. Same report as the first round.
- **Pin of the 40-entry fev-bench pool:** commit and date, from the first round's files.
- **TSFM-Bench and TS-Arena:** establish from file layouts and column names only whether
  per-entry, per-unit scores are published. If yes for a board, its units, sources and displayed
  rule are added to `boards/PREREG_BOARDS.md` with a dated note before any score is analysed,
  then the first-round report is run on it.

## Not covered here

Section 4 of `STILL_NEEDED.md` (GPU runs) needs a cost estimate and the owner's yes; nothing in
it runs in this round.

---

# Revision 1 (2026-11-13, after one adversarial review, before S3 was run)

Review: `REVIEW_ROUND2.md`. By the time it arrived S1, S2 and S4 to S7 had been run as written
above; none of them had design choices that depended on results. S3 had not been run in any
form. The changes below apply to S3 only.

What the review showed, with simulations on synthetic data (no S3 result was computed on the
real data):

- Clipping at percentiles of the same data (candidates 2 and 3) makes the dataset means move
  together, and a sign-flip test then rejects too often even on well-behaved data.
- No test of a mean can hold 5% in this check on a pair where one dataset carries most of the
  spread, because a replicate leaves that dataset out 36% of the time.
- The earlier sentence that the resampling check "leans optimistic" is wrong for tests of a mean
  on such pairs; for them it is harsh.

Candidates 1 to 3 are therefore expected to fail. They stay in the list, in the order the
request gave, so that the expectation is tested and not assumed.

Changes:

1. **Candidate 4, zeros.** A dataset whose mean equals the null value exactly keeps its rank
   but gets sign zero (Pratt's rule). Ties in absolute value get average ranks.
2. **Pass rule, two parts, both required.** (a) Pooled rejection rate over the pairs in
   [4%, 6%], as before. (b) Per pair: at most 2 of the 45 pairs above 7.5%, and none above 10%.
   Each pair gets its own independent stream of 1,000 resamples.
3. **Which pairs.** The check is run on three sets and a real-data result is quoted only if its
   own set passes: (i) the 45 pairs of the first round's D4 arm (the MASE_Rank top ten compared
   under the geometric mean), which decides whether a candidate "passes" and the pair counts on
   those pairs; (ii) the 45 pairs of the geometric mean's own top ten, for the counts on that
   top ten; (iii) the 39 leader-against-entry pairs of the first round's tier, for the tier.
4. **Power, with a bar.** For each candidate, and for CR2 on the same replicates, the detection
   rate when every d_c is shifted by 0.5, 1, 1.5, 2 and 3 times the first round's corrected
   resolution (0.0497 log units), with the candidate's own false-rejection rate printed next to
   it. A candidate that passes the level check but detects less than half the time at twice
   that resolution (0.0994) is reported as holding its level but too weak to settle pairs, and
   the paper then keeps "at least 17".
5. **Stated with any result:** candidate 4 tests the typical dataset-level difference (the
   median of pairwise averages of dataset means), with every dataset counted equally. It is
   not a test of the board's configuration-weighted geometric mean.

Other review points and how they were already met: the S2 agent found independently that
`dfadjust` returns a covariance and keeps negative values, and converted it; S7 named the
replacement model because the library has no default; S5 matched the entry name exactly and
tested both neighbours; S1 compared all 135 stored p-values with the exact ones, and the real
MASE_Rank differences (multiples of one half, many ties) were checked against a direct listing
of all 2^28 patterns. Not done: the exact test at 19 sources.
