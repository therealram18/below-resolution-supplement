# Camera-ready experiments: design (written 2026-11-11, before any run)

Two reviewer questions drive this. Everything here is CPU-only re-analysis of public files.
Nothing in this file may be changed after results are seen without a dated note saying what
changed and why.

## Reviewer question 1

> The resolution estimate relies on bootstrap resampling over only 28 underlying datasets, and
> the clustered percentile bootstrap may produce intervals that are too narrow with so few
> clusters. Validate the main conclusions with a wild cluster bootstrap or another
> small-sample correction.

### Setup and notation

For two entries A and B under one aggregation rule, the board's statistic is an equal-weight
mean over configs. Let d_c be the per-config difference (B minus A), c = 1..N, and g(c) the
dataset of config c, with G datasets and n_g configs in dataset g. The estimate is
theta_hat = (1/N) * sum_c d_c. On GIFT-Eval N = 97, G = 28, with n_g distributed as:
11 datasets with 1 config, 1 with 2, 4 with 3, 5 with 4, 1 with 6, 2 with 7, 4 with 8.

### Methods compared

- P-iid: percentile bootstrap, resampling configs (paper).
- P-clu: percentile bootstrap, resampling whole datasets (paper).
- P-both: the paper's reporting rule, a gap counts only if P-iid and P-clu both exclude zero.
- CR2-t: cluster-robust standard error with the Bell-McCaffrey (CR2) adjustment and
  Satterthwaite degrees of freedom, used in a t interval. For this mean-only model:
  S_g = sum over c in g of (d_c - theta_hat);
  Var_CR2 = (1/N^2) * sum_g S_g^2 / (1 - n_g/N);
  degrees of freedom nu = (trace M)^2 / trace(M^2), where M is the G x G matrix with
  M_gh = (1/N^2) * k_g * k_h * (n_g * [g = h] - n_g * n_h / N) and k_g = (1 - n_g/N)^(-1/2)
  (working model: independent equal-variance errors). Also report nu under a working model with
  within-dataset correlation equal to the estimated intraclass correlation, as a sensitivity.
- CR1-t(G-1): the common default, Var_CR1 = (G/(G-1)) * (1/N^2) * sum_g S_g^2, t with G-1 df.
- WCB: wild cluster bootstrap-t with the null imposed. Each draw multiplies every dataset's
  differences (after subtracting the null value) by one random sign (Rademacher), recomputes the
  mean and its CR1 standard error, and forms t*. The p-value is the share of |t*| >= |t|.
  Webb six-point weights are run as a variant.
- Means-t: one-sample t-test on the G dataset means (G-1 df). This answers a slightly
  different question (each dataset weighted equally, not each config), so it is reported as a
  sensitivity, not as a replacement.

### Experiment A1: calibration by simulation (is the paper's interval too narrow, and what fixes it)

Simulate differences with the real cluster layout and a true mean of zero, apply every method,
and record how often each rejects at the nominal 5% level (the target is 5%).

- D1 normal: d_gc = a_g + e_gc, a_g ~ N(0, tau^2), e_gc ~ N(0, sigma^2), with intraclass
  correlation rho = tau^2/(tau^2 + sigma^2) in {0, 0.1, 0.3, 0.5, 0.7}.
- D2 heavy tails: same, with a_g and e_gc drawn from a t distribution with 3 df (rescaled).
- D3 unequal spread: dataset-specific noise scale, sigma_g drawn log-normally (sd 0.5 on the
  log scale), rho as in D1.
- D4 real data ("plasmode"): take the real per-config differences of each of the 45 top-10
  pairs under MASE_Rank. Draw 28 datasets with replacement from the real ones to make a new
  sample; the true value is the mean of the original data. Record whether each method's 95%
  interval (or test of the true value) covers it.
- Also report the intraclass correlation estimated from the real 45 pairs, so the reader can
  see which rho values are relevant.
- Sizes: 4,000 simulated datasets per cell, 999 bootstrap draws inside each (seed fixed).
  Monte-Carlo error on a 5% rejection rate is about 0.35 percentage points.
- The same study is repeated for the fev-bench layout (39 task families) and for 12
  independent units (the context-length experiment).

Pre-stated reading: a method is "adequately calibrated" if its rejection rate lies in
[4%, 6%] across the rho values that bracket the real data. Whatever the outcome for the
paper's rule, it is reported.

### Experiment A2: every headline count under the corrected methods (GIFT-Eval)

Using exactly the paper's per-config differences (the existing audit code is imported, and the
paper's published counts must be reproduced first as a check):

- For MASE_Rank, CRPS_Rank and the geometric mean of scaled MASE: the 9 adjacent gaps and all
  45 top-10 pairs, counted as separated at 5% uncorrected and after Holm, under P-both (paper),
  CR2-t, WCB (99,999 draws) and Means-t.
- Resolution: the paper's 2.8 x clustered bootstrap SE (median over adjacent pairs) against
  (t_{0.975,nu} + t_{0.80,nu}) x CR2 standard error, and the implied multiple of datasets
  needed to resolve the median adjacent gap.
- Leading tier (entries not separated from the leader) under each method.
- The one single-scheme survivor (pair 9-10 under the clustered scheme) under each method.

### Experiment A3: the same for fev-bench and for the context-length experiment

- fev-bench: 9 adjacent gaps under win rate and skill score, 39 task families.
- Context-length experiment (12 configs from 12 datasets, 4 models, 2 aggregation rules):
  exact sign-flip test (all 4,096 sign patterns) and a paired t interval (11 df) on the
  per-config differences, against the paper's percentile bootstrap (5 of 8 uncorrected, 4 of 8
  after Holm).

## Reviewer question 2

> Most conclusions rest on GIFT-Eval, with fev-bench secondary. How well does Vernier
> generalize to other forecasting benchmarks with different aggregation rules, dataset
> compositions, or numbers of evaluation units?

### Experiment B1: the report on additional boards

A scout first establishes which forecasting boards publish a score for every (entry,
evaluation unit). Candidates: Chronos Benchmark II (zero-shot) and I (in-domain) as published
in the fev repository; BOOM (Datadog); TIME; ProbTS; TFB/OpenTS; Monash archive results;
long-horizon forecasting tables; live boards. Every board whose files can be obtained is
analysed; a board is excluded only if the per-unit files do not exist, and exclusions are
listed. No board is dropped for its result.

For each board, fixed before looking at its numbers:
1. Units, and the independent sources they group into, taken from the board's own
   documentation.
2. The displayed rule, reproduced from the files and checked against the published table.
3. Resolution, median gap between displayed neighbours, adjacent pairs established, and the
   leading tier: once with the paper's method and once with the method that Experiment A1
   shows to be calibrated.
4. Pool dependence: whether the displayed rule depends on the other entries; the single-entry
   removal test and the near-copy test; and, where a version history of per-entry files
   exists, the history replay (including fev-bench, which the paper did not replay).

### Experiment B2: composition and number of units, inside the boards we already have

- GIFT-Eval sub-boards the site itself displays: by horizon (short 55, medium 21, long 21
  configs), by domain, by frequency group. Same report on each.
- Resolution against number of datasets: draw random subsets of G' datasets (G' = 7, 14, 21,
  28 on GIFT-Eval; 10, 20, 30, 39 families on fev-bench), 200 draws each, and compare the
  resolution with the 1/sqrt(G') scaling that the paper's "45 times as many datasets"
  extrapolation assumes.

## Output

`CAMERA_READY_EXPERIMENTS.md` in this folder: plan, what ran, every result table, what
changed relative to the submitted paper, and what failed. Scripts and JSON outputs sit beside
it; every table is produced by a script, with fixed seeds.

---

# Revision 2 (2026-11-11, after one adversarial review, still before any run)

A strict reviewer checked the formulas numerically and they hold (brute-force CR2 matches;
degrees of freedom 17.07 with independent errors and 14.4 / 12.3 / 11.4 / 11.0 at within-dataset
correlation 0.1 / 0.3 / 0.5 / 0.7; independently recomputed by the main session with the same
result). The review also showed that three planned checks could not have come out against the
paper. The changes below fix that. The review's scratch scripts looked only at shapes of the
real data (skewness, correlation), not at p-values or pair counts.

1. **Skewed simulation arms.** D1-D3 are symmetric, which makes the wild bootstrap exact by
   construction. Added: D5, dataset effect drawn from a centred gamma with skewness 1 and with
   skewness 2; D6, the size of the effect grows with the number of configs in the dataset.
2. **Primary method fixed in advance.** Primary corrected method: CR2 standard error with
   Satterthwaite degrees of freedom computed under the working model with the pair's estimated
   within-dataset correlation (floored at zero). Fallback: if on the real-data arm (D4) the
   primary's rejection rate falls outside [4%, 6%] while another candidate (CR2 with
   independent-errors degrees of freedom, or the wild cluster bootstrap) falls inside, the
   candidate closest to 5% becomes primary, ties going to the more conservative. The
   dataset-means t-test is never primary because it weights datasets rather than configs. All
   methods are reported regardless; "primary" only decides which single corrected number is
   quoted for the resolution and used on other boards.
3. **Real-data arm (D4) details.** A dataset drawn twice counts as two separate clusters. Each
   method is scored against its own true value (config-weighted mean for all methods except
   the dataset-means t-test, whose true value is the equal-weight mean of dataset means). D4
   cannot produce a dataset more extreme than the 28 observed, so it leans optimistic; this is
   stated wherever D4 is quoted.
4. **Power, and a known-shift control.** Size alone is not enough, since "not resolved" is
   easy for a conservative rule. For every arm, shift the true mean by the claimed resolution
   and record the detection rate (target about 80%). The same check runs on every board in
   Experiment B1: add a shift equal to the board's resolution to one entry and report the
   detection rate. A report that calls everything unresolved must still detect a difference of
   the size it says it can detect.
5. **The subsampling curve is renamed a stability check**, since subsets of a fixed 28 give a
   near-flat curve for any data. Added: leave-one-dataset-out, reporting how far the
   resolution and the pair counts change when each dataset is removed.
6. **Boards, units and sources are fixed in a dated, hashed file before any board is
   analysed** (`boards/PREREG_BOARDS.md` with the unit-to-source maps and the tolerance for
   reproducing each board's displayed table). Each board's overlap with GIFT-Eval's datasets
   is reported, and the report is rerun on the non-overlapping units.
7. **Coarser sources on GIFT-Eval.** Six of the 28 datasets are M4 subsets, two are ETT, three
   are bizitobs and two are bitbrains; grouping these leaves 19 sources. One sensitivity row
   at 19 clusters.
8. **Do new datasets look like the current ones?** The "45 times as many datasets" statement
   assumes new datasets vary like the existing 28. For open models scored on both GIFT-Eval
   and fev-bench, compare the between-source spread of per-source log error-ratio differences
   on fev-bench's non-shared families with the spread on GIFT-Eval's datasets.

Smaller points adopted: the wild bootstrap p-value is (1 + count)/(B + 1); for the geometric
mean the tests run on per-config log ratios; "datasets needed" uses the large-sample
multiplier 2.80 at the target size, G* = G x (2.80 x SE / gap)^2, while the resolution at the
current size uses the t multiplier; the exact sign-flip test's null is stated (differences
independent and symmetric about zero); the leading tier is described as "entries not
separated from the displayed leader by uncorrected tests", not as a confidence set for the
best entry.
