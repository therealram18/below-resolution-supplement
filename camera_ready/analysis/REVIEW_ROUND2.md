# Review of DESIGN_ROUND2.md (strict statistical review, 2026-11-13)

Scope: S3 first, then S1, S2, S5 to S7. Nothing in the design was run on real data. The only computations on the
real differences are shapes: skewness of dataset means, the largest single-dataset share of
sum_g (m_g - mean)^2, how much of the dominant dataset config-level clipping reaches, ties, and which pairs overlap.
No p-value, rejection rate or pair count was computed for any S3 candidate on real data.

Scripts and their printed output are in `review_scratch/round2/` (run with the chronos_bolt_base python, from
`camera_ready/`):

| script | what it shows | output |
|---|---|---|
| `shapes.py`, `shapes_reported_pairs.py` | real-data shapes only | `shapes.out`, `shapes_reported_pairs.out` |
| `sim_one_pair.py` | one dominant dataset; signed-rank test under skew; clipping in the level check | `sim_one_pair.out` |
| `sim_clip.py` | why clipping at percentiles over-rejects | `sim_clip.out` |
| `sim_worlds.py` | a synthetic 10-entry, 45-pair population tuned to the real shapes: true rejection rates vs the design's level check | `sim_worlds.out`, `sim_worlds_sensitivity.out` |
| `check_s1_s2.py` | S1 algebra; dfadjust's degrees of freedom ported and compared | `check_s1_s2.out` |

In `sim_worlds.py` the phrase "an observed sample" means one draw of 28 datasets from the synthetic population. It
plays the part of the 28 GIFT-Eval datasets we happen to have. The design's level check is then run on it exactly as
written: 28 datasets drawn with replacement, the null set at the value on the observed 28. The synthetic population
reproduces the real shapes: median largest share 0.37 to 0.44 against 0.44 real; median skewness of dataset means
0.8 to 1.0 against 1.14. It also reproduces the first round's real-data level check for the t-test on dataset means:
8.0 to 9.7% against 8.4%.

---

## Objections, most important first

### 1. Candidates 2 and 3 over-reject by construction, even on clean normal data

**What is wrong.** The clipping points are percentiles of the same 97 values that are then tested. The sign-flip
test treats the clipped values as if they were the data. But the clipping points vary from sample to sample, and
when they move, every clipped dataset mean moves the same way. A sign-flip test reads a shift shared by all datasets
as evidence against the null. Its reference distribution is therefore too narrow. This has nothing to do with the
dominant dataset; it happens with symmetric, well-behaved data.

**Evidence.** All synthetic, all run through the design's level check.
- `sim_clip.py`, normal data, no scale differences between datasets: clip points recomputed per replicate (as the
  design specifies) reject 7.5% (5% tails) and 11.8% (10% tails). With the same clip points held fixed, the rates are
  4.8% and 5.1%. With a constant clip level fixed in advance, they are 4.7% and 5.1%. The spread of the clipped
  summary over replicates is 1.11 and 1.24 times the spread the sign-flip test assumes. With the fixed versions the
  ratio is 1.00.
- The same at a within-dataset correlation of 0.10: 6.9% and 8.2% recomputed, against 5.0% and 4.7% fixed. With
  skewed data and unequal dataset scales: 9.5 to 15.3% (5% tails) and 13.5 to 18.4% (10% tails).
- `sim_worlds.py`, real-world rate (28 datasets drawn from the population, each test at the population value of its
  own summary): candidate 2 rejects 10.2 to 10.6% and candidate 3 rejects 14.0 to 15.7%, across three population
  settings.
- The design is right to recompute the clip points in each replicate. That is how the real-data test works.
  Holding them at the original values would have reported about 5% (`sim_one_pair.py` Part C: 5.7 to 5.8% fixed,
  against 12.9% and 19.2% recomputed). It would have passed a test whose true rate is 10 to 16%.

**How sure.** High. The mechanism is simple, it shows on normal data, and it reproduces in every setting tried.

**Smallest fix.** Drop candidates 2 and 3 as written. If clipping is wanted, clip each d_c at a constant fixed now
in log units, |d_c| <= c. Choose c on substantive grounds before looking at the data. The level check then uses the
same c in every replicate, and a fixed-c check is the correct check for it. In `sim_worlds.py` such a test rejects
5.1 to 5.2% in truth and 5.1 to 5.2% in the check, and all observed samples land in [4%, 6%]. Two caveats go with
it. First, a constant clip only limits a dominant dataset if c is near the spread of the other datasets' means;
at a loose c such as log 1.5 it may do nothing. Second, it is not shift-equivariant: a constant added to every d_c
moves the clipped summary by less than that constant (see objection 4).

### 2. A test of a mean cannot pass this check on dominated pairs, and the check over-states the rate of even a correctly sized mean test

**What is wrong.** A replicate misses a given dataset with probability (27/28)^28 = 0.361. Suppose the dominant
dataset holds a share f of sum_g (m_g - mean)^2. When it is missing, the other datasets are centred away from the
null value. The offset, in standard errors of the mean of 28, is

  delta = sqrt(28 f / (27 - 28 f)).

Those replicates are clean samples from a distribution whose mean is not the null value. So for any test whatever:

  rejection rate >= 0.361 x (the test's power against a shift of delta standard errors in clean data).

A level of 5% therefore forces power of at most 13.8% at delta. A t-test has that power at delta = 0.89, which is
f = 0.43. Above f of about 0.43, a test of the mean either rejects more than 5% in this check or has less power than
a t-test against a 0.9-standard-error shift. On the real level-check pairs the share has median 0.44, upper
quartile 0.53 and maximum 0.88, and 14 of 45 pairs are above 0.5 (`shapes.out`).

**Evidence.**
- `sim_one_pair.py` Part A, exact sign-flip test on dataset means (candidate 1). The rejection rate is 5.4% at
  f = 0.30, 6.6% at 0.44, 7.7% at 0.53, 12.4% at 0.70 and 30.3% at 0.88. The bound with t-test power gives 5.2, 6.7,
  12.6 and 31.6%. Almost all rejections come from replicates without the dominant dataset: 11 to 84% rejected,
  against at most 3.5% when the dataset is present. The first round's dataset-means t-test (8.4% pooled, 28.1% on
  the worst pair) fits this, so candidate 1's failure is predictable now.
- `sim_worlds.py` gives a t-test on dataset means a critical value set from the known population, so its true
  rate is exactly 5%. In the design's check it rejects 6.4%, 7.7% and 8.7% on average in the three population
  settings, with an SD across observed samples of 1.7 to 2.8 points. It lands in [4%, 6%] for only 44%, 25% and 8%
  of observed samples. Candidate 1 is 7.1 to 7.6% in truth and 8.1 to 9.9% in the check. CR2 is 7.0 to 7.9% in
  truth and 9.3 to 10.4% in the check. So for tests of a mean the check fails tests that are correctly sized, and
  the "known limit" sentence (resampling "leans optimistic") has the wrong direction for them.
- The signed-rank test (candidate 4) does not have this problem. It is 4.9 to 5.0% in truth and 5.1 to 5.3% in the
  check, with an SD of 0.3 points, and it is in [4%, 6%] for 97 to 100% of observed samples. In Part A it stays at
  4.6 to 5.2% for every f up to 0.88, and so does a dataset-level clip at a constant fixed in advance.

**How sure.** High for the bound and the exact one-pair simulation. Medium for the size of the over-statement,
which depends on the synthetic population, although that population reproduces the first round's 8.4%.

**Smallest fix.** Write down now that no test of the mean (of dataset means or of configurations) is expected to
pass, with the bound above as the reason. Make the signed-rank test (or a constant-level clip from objection 1) the
first, deciding candidate. Report candidate 1 as a known failure. In the paper, say that the geometric mean's own
pair differences have no reliable test at 28 datasets when one dataset holds more than about 43% of the spread of
dataset means, and give that count for the reported pairs (a shape count, not a test). Change the known-limit
sentence: the check can mislead in both directions, and for tests of a mean the simulation shows it over-states
the rate.

### 3. The pass rule pools over pairs, and the stated uncertainty is far too small

**What is wrong.**
- A pooled rate cannot tell "5% on every pair" from "2% on most pairs and 25% on a few". The few are the dominated
  pairs, the very problem S3 is meant to solve. In `sim_worlds.py` the worst pair, for mean-type tests, has a median
  of 17 to 32% across observed samples while the pooled rate is 6 to 10%. Per-pair ranges are reported but do not
  decide.
- The stated Monte Carlo SE of about 0.1 points holds only if each pair gets its own independent resamples (as in
  the first round's `d4_task`). The design does not say. With the same resamples reused across pairs, my simulation
  gives 0.3 to 0.5 points at 500 replicates.
- The larger uncertainty is not Monte Carlo error. The check's result depends on which 28 datasets were observed,
  and all 45 pairs share those datasets. Across observed samples from one population, the pooled rate has an SD of
  1.7 to 2.9 points for mean-type tests. That is wider than the whole [4%, 6%] window, so for those tests pass or
  fail is mostly the luck of the sample. For the signed-rank test and the constant clip the SD is 0.3 points, and
  the window is informative.
- "First that passes" is a real commitment and leaves no choice afterwards. But the order 1, 2, 3, 4 tries first
  the candidates that pass only by luck (1) or are wrong by construction (2, 3), and tries last the only one that
  is reliably near 5%.

**How sure.** High.

**Smallest fix.** Add a per-pair condition, fixed now, and state that each pair gets its own independent
resamples. With 1,000 independent replicates the per-pair Monte Carlo SE at 5% is 0.69 points, and a pair truly at
5% exceeds 7.5% with probability 2.6e-4, and the largest of 45 such pairs exceeds it with probability 0.012;
a pair truly at 6.5% exceeds it with probability 0.09. A workable rule: pooled rate in [4%, 6%], at most 2 of 45 pairs
above 7.5%, none above 10%. Reorder as in objection 2.

### 4. The power check cannot fail, and its comparison with CR2 is not like for like

**What is wrong.**
- No minimum is stated. The detection rate is only "reported", so a test that rarely rejects is not stopped.
- CR2 rejects 15.7% of true nulls on these pairs in the first round. Its detection rate at the shift therefore
  includes false rejections, and a calibrated test placed next to it looks weaker than it is.
- Adding the same constant to every d_c is fair in one sense. Each design candidate's summary moves by exactly that
  constant: the mean of dataset means; the clipped means, because percentiles move with the data; and the
  pseudo-median. But a uniform shift is the alternative that suits a rank test best. A difference concentrated in a
  few datasets moves the mean and barely moves the pseudo-median. The power check cannot show this; it is the
  "different summary" point in another form. In `sim_worlds.py`, at a shift of 2.8 times the median CR2 standard
  error: signed rank 91 to 92%, constant clip 93 to 96%, correctly sized mean test 68 to 70%, CR2 72 to 73%.
- A constant-level clip (objection 1) moves by less than the shift, so its detection rate is understated.

**How sure.** High (reading of the design, plus the algebra of shift-equivariance).

**Smallest fix.** Fix a minimum now, for example pooled detection at 0.0497 of at least 50%, or at least that of
the dataset-means t-test. Report CR2's null rejection rate beside its detection rate, or size-adjust it: use the
95th percentile of CR2's |t| in the null replicates as its critical value. State in the paper that the power check
uses a uniform shift.

### 5. The level is checked on different pairs from those reported

**What is wrong.** The check uses the 45 pairs of the MASE_Rank top ten (`calib_d4_data.npz`; `load_gift_pairs`
builds geometric-mean pairs in that order). The real-data report uses the geometric-mean order's own top ten, which
shares 8 entries and so 28 of 45 pairs. It also uses the 39 leader-vs-entry pairs for the tier, of which at most 9
are level-check pairs.

**Evidence (shapes).** The reported pairs are milder than the checked ones. Largest share: own top ten has median
0.34 and maximum 0.74, with 9 pairs above 0.5; the tier pairs have median 0.28 and maximum 0.33, with none above 0.5;
the checked pairs have median 0.44 and maximum 0.88 (`shapes_reported_pairs.out`).

**How sure.** High on the fact, low that it changes the decision. The checked pairs are the harsher set.

**Smallest fix.** Run the same level check on the 45 own-order pairs and the 39 tier pairs as well, and report it.
This does not look at any real p-value.

### 6. Candidate 4: zero differences are not defined, and the test is of symmetry rather than of the pseudo-median

**What is wrong.**
- The design gives no rule for a dataset mean exactly equal to the null value. On the MASE_Rank pairs (the
  reported, non-deciding check), 5 of 45 pairs have a dataset mean exactly equal to the g <= h median, so every
  replicate that contains that dataset has a zero. Dropping zeros (Wilcoxon) and keeping them in the ranking with
  sign 0 (Pratt) give different statistics. On the geometric-mean pairs there are no tied means and no zeros in the
  original data; ties created by resampling are covered by "average ranks".
- The signed-rank test is exact for "symmetric about the null value". As a test of the pseudo-median under skew it
  is close but not exact. With 28 independent draws tested at the true pseudo-median (`sim_one_pair.py` Part B(a)),
  it rejects 5.0% at skewness 1, 5.7% at 2, 6.3% at 4 and 5.4% for a lognormal. The real skewness of dataset means
  reaches 4.37, so the most skewed pairs may sit at 6 to 7%.

**How sure.** High.

**Smallest fix.** Fix the zero rule now; Pratt's is the natural one for the sign-flip form. Word the claim as "a
test of symmetry about zero, read as a test of the pseudo-median".

### 7. S2: dfadjust is not used the way the design assumes

**What is wrong** (source read: CRAN dfadjust 1.1.0, published 2024-12-18, file `R/ols.R`).
- The function is `dfadjustSE`, with defaults `IK = TRUE` and `rho0 = FALSE`.
- The returned `rho` is not a correlation. It is the between-dataset covariance of the Moulton model, in squared
  units of d: (sum_g (sum of residuals in g)^2 - sum of squared residuals) / (sum_g n_g^2 - N). The returned `sig`
  is max(SSR/N - rho, 0). The correlation is rho / (rho + sig).
- With `rho0 = FALSE`, negative values are kept. With no true clustering, the estimate is negative in 60% of
  synthetic draws; `icc_hat` floors these at 0.
- The variance is LZ2, which is CR2 with adjustment matrices built from the design matrix alone (an identity
  working model). The degrees of freedom use the Moulton model. That structure matches
  `vernier_stats.satterthwaite_df`.

**Evidence** (`check_s1_s2.py`, dfadjust's code ported line by line for `lm(d ~ 1)`). With rho converted to
rho / (rho + sig), our formula equals dfadjust's IK degrees of freedom to a relative 1.3e-15 over 300 draws, 21 of
them with negative correlation. Passing `rho` as if it were a correlation gives a wrong answer: with d in units like
rank places, 9.45 against the correct 11.03.

**How sure.** High.

**Smallest fix.** In (i), convert as above. In (ii), say that "as shipped" means `rho0 = FALSE`. Also report
`rho0 = TRUE`, so that the effect of the floor at zero is separated from the effect of the different estimator.

### 8. S1: the checks miss ties, and "exact" needs its null stated

**What is wrong.**
- The 12-cluster brute-force check uses continuous random data, so the tie rule (relative tolerance 1e-9) is never
  exercised. On integer-valued data, as in MASE_Rank, exact ties are common: in one synthetic example 10.6% of the
  4,096 patterns tie the observed |sum| exactly. A wrong tie rule moves p by that much.
- The one-pair check against the stored 99,999-draw p-value, within 4 Monte Carlo SEs, would pass an error of
  about 0.0025 at p = 0.04.
- "Exact" holds under one null only. The p-value of either statistic is exact if the 28 dataset sums of
  d_c - theta_0 are independent across datasets and each is symmetric about zero. More generally, their joint
  distribution must be unchanged when the sign of any subset is flipped. A plausible version: on every dataset the
  two entries' results are exchangeable (swapping the two entries' labels leaves that dataset's joint distribution
  unchanged), and datasets are independent. That is a "no difference on any dataset" null, much stronger than "the
  mean difference is zero". Under the weaker null with skewed dataset effects, neither test is exact; the first
  round's resampling shows 15.5% for the random-draw version on geometric-mean pairs. Independence across the 28
  datasets is itself doubtful (6 M4 subsets, 2 ETT, 3 bizitobs, 2 bitbrains; the first round grouped them into 19
  sources).

**How sure.** High.

**Smallest fix.** Add an integer-valued layout to the brute-force check. Compare all 135 stored p-values, not one;
the z-scores should look standard normal. State the exact null in the paper. Run the exact test at 19 sources as
well (2^19 patterns, trivial).

### 9. S7: fev's leaderboard leaves the flagged cells unchanged by default

**What is wrong** (source read: `src/fev/analysis.py` on fev's main branch, fetched 2026-11-13). `leaderboard()`
and `pairwise_comparison()` have `leakage_imputation_model: str | None = None`. Flagged results are replaced by that
model's scores only when a model is named ("Applied before `missing_strategy`"). With the defaults, "whatever the
fev library's own leaderboard code does" changes nothing, and the primary analysis would equal the first round's.

**How sure.** Medium-high. This is the main branch; the version tied to the pinned pool may differ.

**Smallest fix.** Name the imputation model (the one the published Chronos Benchmark II or fev leaderboard uses, if
stated) and pin the fev version. For the secondary rule, report how many tasks remain. With few tasks every pair is
unresolved by construction; say so rather than letting it read as a finding.

### 10. S5: two choices are left open until after the board is seen

**What is wrong.**
- "Or the closest name on the board" lets the entry be chosen after looking.
- Only the entry directly below is tested.

**Smallest fix.** Fix the matching rule now (for example, case-insensitive substring "timestral"; if several match,
report all). Test against both neighbours, or state why only the one below.

---

## Checked and found correct

- **S1 studentized formula.** It equals `vernier_stats._cr1_t`: the maximum absolute difference is 1.3e-15 over
  5,000 sign patterns on the GIFT layout. By algebra, w_g^2 = 1 gives
  sum_g (w_g s_g - n_g mean*)^2 = Q - 2 mean* C + mean*^2 R.
- **"Same test only with equal sizes."** Correct. With equal sizes,
  t^2 = ((G-1)/G) A^2 / (Q - A^2/G), which increases with |A|, so the two p-values agree; in 200 skewed draws the
  maximum difference is 0. With unequal sizes they differ: median |difference| 0.006, maximum 0.065, and they
  disagree at 5% in 2% of draws.
- **Meet-in-the-middle count.** It equals brute force on 12 clusters.
- **S3 target values.** Candidate 1's (the mean of the 28 dataset means) is exactly the resampling world's value.
  Candidate 4's median of g <= h pairwise averages differs from that world's exact pseudo-median by a median
  0.004 to 0.015 standard deviations of the estimate, and the rejection rates agree within 0.1 points. Candidates 2 and 3: using
  the original clip points for the target and recomputing them in the replicate is the right way to check the test
  as specified. With clip points held fixed, the estimator's median bias is 0.005 to 0.008 SD; recomputed, 0.06 to 0.07 SD (too
  small to explain objection 1, which is about spread, not bias).
- **Shift-equivariance.** Each of the four design summaries moves by exactly the added constant.
- **clubSandwich 0.7.0** (CRAN source). For an unweighted `lm`, `vcovCR.lm` sets `inverse_var = TRUE` and the
  working model to the identity. `coef_test` defaults to `test = "Satterthwaite"`, with degrees of freedom
  2 E^2 / V from the working model, which is (trace)^2 / (sum of squares) as in the design. The CR2 adjustment for
  an intercept-only model reduces to `se_cr2`'s formula. The planned call and pass thresholds are consistent.
- **dfadjust's structure.** CR2 with identity adjustment matrices, degrees of freedom from the Moulton model. It
  matches ours exactly once rho is converted (objection 7).

## Not checked

- `wildboottest` and `fwildclusterboot` defaults (p-value convention, enumeration rule).
- S6 beyond reading. One gap: the design covers "files can" and "files cannot" be obtained, but not partial
  availability.
