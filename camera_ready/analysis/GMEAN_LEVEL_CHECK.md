# S3: a test for the geometric mean that holds its level

Design: DESIGN_ROUND2.md, S3 plus Revision 1. Code: gmean_level_check.py, tests: test_gmean_level_check.py, numbers: gmean_level_check.json (partial files in gmean_level_parts/). 1,000 resamples per pair, 28 datasets drawn with replacement, own random stream per pair, rejection at p < 0.05.

Sanity checks: (a) normal data, real layout, 2,000 draws: candidate 1 rejects 5.45%, candidate 4 4.60% (both inside 4 to 6%). (b) candidate 4 equals scipy's exact Wilcoxon p-value on 10 random vectors (largest difference 0). (c) CR2 on set (i) pooled rate 15.37% against 15.72% in the first round (within 1.5 points).

## Level (rate of rejecting a true null; pass = pooled in 4 to 6%, at most 2 pairs above 7.5%, none above 10%)

Set i: 45 MASE_Rank-top-ten pairs under geometric mean (D4)

| test | pooled % | pairs above 7.5% | pairs above 10% | worst pair (%) | pass |
|---|---|---|---|---|---|
| 1 mean of dataset means | 8.51 | 23 | 8 | 8_9 (30.0) | no |
| 2 clip at 5%/95% | 12.13 | 42 | 33 | 2_9 (20.3) | no |
| 3 clip at 10%/90% | 17.52 | 45 | 44 | 0_9 (27.2) | no |
| 4 signed ranks (Pratt) | 5.56 | 0 | 0 | 1_6 (7.5) | yes |
| CR2 (reference) | 15.37 | 39 | 31 | 8_9 (37.0) | - |

Set ii: 45 pairs of geometric mean's own top ten

| test | pooled % | pairs above 7.5% | pairs above 10% | worst pair (%) | pass |
|---|---|---|---|---|---|
| 1 mean of dataset means | 6.86 | 12 | 4 | 3_4 (18.2) | no |
| 2 clip at 5%/95% | 11.65 | 42 | 33 | 3_9 (17.8) | no |
| 3 clip at 10%/90% | 17.96 | 45 | 44 | 3_7 (27.8) | no |
| 4 signed ranks (Pratt) | 5.40 | 0 | 0 | 4_7 (6.9) | yes |
| CR2 (reference) | 10.00 | 29 | 24 | 1_2 (25.6) | - |

Set iii: 39 leader-against-entry pairs of the geometric-mean tier

| test | pooled % | pairs above 7.5% | pairs above 10% | worst pair (%) | pass |
|---|---|---|---|---|---|
| 1 mean of dataset means | 5.78 | 2 | 0 | 0_36 (8.5) | yes |
| 2 clip at 5%/95% | 10.49 | 39 | 23 | 0_17 (14.1) | no |
| 3 clip at 10%/90% | 17.15 | 39 | 39 | 0_1 (23.5) | no |
| 4 signed ranks (Pratt) | 5.85 | 2 | 0 | 0_29 (9.0) | yes |
| CR2 (reference) | 8.30 | 23 | 10 | 0_18 (12.3) | - |

Only candidate 4 passes on set (i), the deciding set. Candidate 1 fails on set (i) and set (ii) and passes on set (iii) only; it does not count as passing. Candidates 2 and 3 fail everywhere, as the revision expected.

## Power (pooled detection %, shift = k x 0.0497 added to every configuration difference, tested against the original value of each test's summary)

| test | set | level | k=0.5 | k=1 | k=1.5 | k=2 | k=3 |
|---|---|---|---|---|---|---|---|
| 1 mean of dataset means | 1 | 8.5 | 61.8 | 88.4 | 96.0 | 99.0 | 100.0 |
| 2 clip at 5%/95% | 1 | 12.1 | 67.3 | 92.4 | 98.3 | 99.4 | 100.0 |
| 3 clip at 10%/90% | 1 | 17.5 | 77.3 | 95.3 | 99.5 | 99.9 | 100.0 |
| 4 signed ranks (Pratt) | 1 | 5.6 | 88.3 | 98.4 | 99.5 | 99.8 | 99.9 |
| CR2 (reference) | 1 | 15.4 | 44.9 | 74.5 | 85.5 | 92.9 | 99.5 |
| 1 mean of dataset means | 2 | 6.9 | 49.0 | 73.0 | 80.1 | 83.7 | 87.5 |
| 2 clip at 5%/95% | 2 | 11.7 | 56.3 | 78.2 | 84.1 | 86.7 | 90.8 |
| 3 clip at 10%/90% | 2 | 18.0 | 65.7 | 83.7 | 89.6 | 92.3 | 95.7 |
| 4 signed ranks (Pratt) | 2 | 5.4 | 72.9 | 82.7 | 86.7 | 89.2 | 92.9 |
| CR2 (reference) | 2 | 10.0 | 37.8 | 61.9 | 71.8 | 81.1 | 92.6 |
| 1 mean of dataset means | 3 | 5.8 | 13.5 | 45.4 | 76.7 | 91.9 | 97.8 |
| 2 clip at 5%/95% | 3 | 10.5 | 16.5 | 59.6 | 87.5 | 95.0 | 98.4 |
| 3 clip at 10%/90% | 3 | 17.1 | 21.3 | 69.7 | 92.2 | 96.8 | 99.0 |
| 4 signed ranks (Pratt) | 3 | 5.9 | 55.8 | 86.5 | 93.5 | 96.3 | 98.4 |
| CR2 (reference) | 3 | 8.3 | 8.7 | 14.7 | 31.5 | 59.8 | 96.3 |

Detection at k = 2 (0.0994) for candidate 4 is 99.8% (set i), 89.2% (set ii), 96.3% (set iii), above the 50% bar. Detection rates of tests that fail the level check are inflated by their excess rejection and are not evidence of strength. Adding a constant before clipping moves the clipped summary by exactly the shift (largest numerical deviation 2e-16), so the shift is exact for all four tests and CR2.

## Real data (theta_0 = 0, tests applied to the original 97 differences)

Candidate 4 passed its level check on set (i), so its set (i) numbers are the result; its set (ii) and (iii) numbers are labelled by whether it passed on that set (it did on all three). Other candidates are shown for information, level check failed, unless noted.

| test | set i: neighbours of 9 | set i: pairs of 45 | set i Holm (9 / 45) | set ii: neighbours of 9 | set ii: pairs of 45 | set ii Holm (9 / 45) | tier size (set iii) |
|---|---|---|---|---|---|---|---|
| 1 mean of dataset means (info, level failed) | 1 | 16 | 0 / 2 | 1 | 12 | 1 / 1 | 26 |
| 2 clip at 5%/95% (info, level failed) | 1 | 18 | 0 / 4 | 1 | 15 | 1 / 4 | 13 |
| 3 clip at 10%/90% (info, level failed) | 1 | 23 | 0 / 6 | 3 | 20 | 1 / 4 | 10 |
| 4 signed ranks (Pratt) | 1 | 12 | 0 / 1 | 2 | 11 | 1 / 2 | 16 |
| CR2 (reference) | 0 | 10 | 0 / 0 | 1 | 9 | 0 / 0 | 17 |

Stored first round (retest_gift.json), set (i): CR2 0 of 9, 10 of 45, Holm 0 / 0; wild bootstrap 0 of 9, 19 of 45, Holm 0 / 4; tier of the geometric mean's own order 17 (CR2 and wild bootstrap). Set (ii) counts were not stored; the CR2 row above for set (ii) is recomputed here.

Leading tier under candidate 4 (leader plus entries not separated at 0.05, 16 entries): EXAONE-Forecast-Agent; STRIDE w/ Synapse; TIMEHEDGE; LS-MoE; LS-Agent; CastStar; TurkForecast-FM-Chronos2-LoRA-v1; limix_moe; RegLab-AH-1; Falcon-Agent; Falcon-2.0; TimesFM-3; RAES-Conductance-Ensemble-V; RacineCast-1; RacineCast-1-NC; Toto-2.0-FnF.

What the passing test does and does not test: candidate 4 tests whether the typical dataset-level difference (the median of the pairwise averages of the 28 dataset means, every dataset counted equally) is zero. It is not a test of the board's geometric mean, which weights every configuration equally. The tier of 16 is smaller than CR2's 17 by one entry; the paper's wording "at least 17" is therefore not changed by a stronger test, and the two tests answer different questions. Known limit: resampling the 28 observed datasets cannot produce a dataset more extreme than those observed, and on pairs where one dataset carries most of the spread the check is harsh for any test of a mean.

## Deviations and not done
- Set (ii) and (iii) data were loaded once through retest_gift.top_entries (imported, not edited) and saved to gmean_level_data.npz, which workers read.
- Set (iii) pair differences are entry minus leader; sign does not affect any p-value.
- Candidate 4 pass on set (i): the worst pair is exactly 7.5% (not above it).
- Sanity check (c) was verified from the full run, not from a separate small run.
- Real-data tiers use uncorrected p < 0.05 on each leader-entry pair with the same candidate.
- Nothing was skipped. Total compute: about 2.6 hours of single-core time, about 17 minutes of wall time on 10 cores (three commands of about 5 to 6 minutes each; the harness moved the commands to the background after 3 minutes, no nohup or & was used).
