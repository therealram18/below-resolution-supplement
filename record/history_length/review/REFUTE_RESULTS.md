# Attempt to break the GPU-round readings (2026-11-20)

Scripts used, all in this folder: refute_check.py (selection and Toto against the board), refute_check2.py (d_c, history lengths, run records; writes refute_dc70.json), refute_tests.py (brute-force sign-flip p-values, Holm, intervals, seeds), refute_check5.py (the two newer models, pins), refute_level_sim.py (level under five nulls, 4,000 draws; output refute_level_sim.json). All read per_config/*.csv and the .run_meta.json beside each file directly, not the analysis output.

## 1. Toto reproduction: HOLDS WITH A CHANGED WORDING
The numbers hold. Gap on the 12 is -0.016%, on the 70 -0.008%, on the 58 -0.007%. The largest single difference is +0.486% (us_births/D/short), then -0.433% (jena_weather/H/medium).
- The numbers were computed, not copied. None of the 70 MASE values equals the board's bit for bit (median relative difference 1.5e-4), and no other column matches exactly either. MSE[mean] differs by up to 7,855 times on one configuration, which fits independent sampling. Nothing in run_grid.py, g2_models.py or merge_results.py reads a board file. Rows come only from evaluate_model (run_grid.py lines 918-963).
- It is the right board entry. Toto_Open_Base_1.0's model column is "Toto_Open_Base_1.0" on all 97 rows. The other Toto entries give 0.6982 to 0.7885 on the 12.
- Setting A does what the notebook does. It matches the notebook code on: the seed and switches set per task, history min(4096, maximal) then floored at the patch stride, the batch formula computed before that floor, the wrapper that halves the sample batch after running out of memory, evaluate_model batch_size = num_samples, and to_univariate=False.
- The run records agree. All 139 run_meta files show 256 samples, cache on, seed 42, deterministic on, matmul "high", xFormers absent, evaluate batch 256, revision 0411ceb, and run_grid sha 7a5d198c on both lengths. One gap: gift_eval_commit is "unknown" in every run_meta.
- The objection is to the causal sentence. "came from the first round scoring one point forecast, not from data, model or library drift" says more than DESIGN_ROUND3.md lines 241-243 allow: "the statement is 'the notebook's setting reproduces the board'. That the sample count is the cause rests on the other changes ... being unable to move the expected score."
  - Setting A differs from the first round in four other ways besides samples: cache on, matmul "high", deterministic algorithms, and the formula batch size instead of 16.
  - The first round's point setting was not rerun on the new card. Its 0.8126 comes from grid_results_h100.
  - The first round's run_meta has model_revision None (grid_results_h100/results/toto/ctx_default/run_meta.json), so the records cannot rule out model drift.
- Wording: "With the board notebook's setting (256 samples, median scored), our Toto run reproduces the published Toto_Open_Base_1.0 rows (-0.02% on the 12, -0.01% on the 70, at most 0.49% on one configuration). The first round differed mainly in scoring one point forecast (the mean) rather than the median of 256 samples. We attribute its 0.8126 to this because the other differences (cache, batch size, matrix-product precision) should not move the expected score. We did not rerun the first round's setting to isolate it."

## 2. The 70 and the tests: HOLDS
- The 70 are exactly the configurations with median length at least 1,024 (smallest included 1,043, largest excluded 1,008, of 97). They come from 18 base datasets. The 14 source sizes are 12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1. The 12 are in 9 sources. The merges are bizitobs, ett and bitbrains; M4 is the only other merge among the paper's 19 sources, and only m4_daily is eligible.
- The brute-force p-values on 2^14 patterns match:

  | Model | p | Holm |
  |---|---|---|
  | Chronos-Bolt | 0.1146 | 0.1146 |
  | Moirai-2.0 | 0.00513 | 0.0103 |
  | TimesFM-2.5 | 2.44e-4 | 9.77e-4 |
  | Toto | 2.44e-4 | 9.77e-4 |

  The effects are +2.71%, +6.24%, +15.33% and +15.26%. A grid inversion gives the same intervals to within 0.1 point.
- Documented lengths, from run_meta:

  | Model | Documented | 512 arm |
  |---|---|---|
  | Chronos-Bolt | 2048 | 512 |
  | Moirai-2.0 | 4000 | 512 |
  | TimesFM-2.5 | 15360 recorded | 512 |
  | Toto | min(4096, maximal) | min(512, maximal) |

  - TimesFM-2.5's 15360 is the cap. run_grid.py line 471 records the nominal cap, and the history actually used is min(15360, longest series in the batch). Do not print 15,360 as "the history used".
  - Toto gets 4096 on 59 configurations. Its smallest documented lengths are 79 (m4_daily/D/short) and then 923 (us_births/W/short).
- Same-length arms. Only Toto's m4_daily/D/short has equal arms, since its maximal history (79) is at most 512. No other configuration has a median series whose history at the latest window is 512 or less. No d_c is exactly 0 apart from that rule. Values with |d_c| < 1e-3: Chronos-Bolt 3, the others 0.
- Retries. status.jsonl has 697 lines for 651 jobs. There are 46 failed attempts (Chronos-Bolt 9, Moirai-2.0 9, TimesFM-2.5 14, TimesFM-3 8, Timestral 6). Every failed run_meta error is "OutOfMemoryError: CUDA out of memory". Each job has exactly one ok, there are no timeouts, and the last run ended at 2026-11-20T00:14Z, before the freeze. The final run_meta of all 140 runs per model shows one evaluate batch size each: Chronos-Bolt 64 (env 64), Moirai-2.0 512, TimesFM-2.5 1024. Revision and run_grid sha are also the same everywhere.

## 3. What the paper may say: HOLDS WITH A CHANGED WORDING
The four test outcomes are right. The design requires items the reading leaves out:
- With "confirmed on the 58", it must give the overlap counts (DESIGN lines 210-211): 43 of the 58 come from a base dataset among the 12, and 20 use the same series as one of the 12 at another horizon. "New data" was withdrawn (line 206).
- S must come with its leave-one-out range, 8.51% to 13.22% (lines 201-202).
- Each p-value must come with its false-claim rate in the direction of the effect (line 185). See reading 4 for which rate.
- The abstract's figure comes from the 70 (line 276), and the first round's +20.6% is replaced (lines 59-61). So "more than 20%" goes: Toto is +15.3% on the 70. The +22% on the 12 must not take its place.
- Chronos-Bolt: do not offer its rejection on the 58 (p 0.035) as partial evidence. The design allows "confirmed" only if T1 "also" rejects, meaning on the 70 first. On top of that, the test over-claims rises for this model (reading 4).
- Wording: "Cutting the history to 512 steps raises the error of Moirai-2.0 (+6.2%, p 0.010 after Holm), TimesFM-2.5 (+15.3%, 0.001) and Toto (+15.3%, 0.001) on the 70 eligible configurations. False claims of a rise occur in at most 1% of simulated draws for these three. The rise is also found on the 58 configurations outside the first round, although 43 of them share a base dataset with the 12 and 20 share series. For Chronos-Bolt no effect is established. No model's effect is shown to exceed the 11.8% spread of the top ten (8.5% to 13.2% leaving one dataset out). On the first 12 alone, no effect survives the Holm correction."

## 4. Level check: REFUTED as a general reading (it holds for three models, with a changed wording)
These reruns use 4,000 draws, same seed. The paper's method reproduces: one value per source gives 6.0% / 4.5%, 12.4% / 0.28%, 19.0% / 0.03%, 13.5% / 0.15%.
- "A claim that error rises is conservative" is false for Chronos-Bolt. The yardstick for one direction of a two-sided 5% test is 2.5%. Chronos-Bolt makes false rise claims in 4.75% of draws (one value per source) and 4.26% (one per configuration). Redrawing whole sources with their real sizes gives 6.05%. The reason is that its d_c are skewed the other way: skewness -1.53, against +2.41, +2.80 and +2.23 for the other three.
- For the other three, positive claims stay conservative under two other nulls:

  | Null | Moirai-2.0 | TimesFM-2.5 | Toto |
  |---|---|---|---|
  | Whole sources with real sizes | 0.28% | 0.00% | 0.10% |
  | Source means into the fixed layout | 0.82% | 0.05% | 0.38% |

- The mirror-image null (same values, signs reversed) gives positive false claims of 12.2%, 19.0% and 13.4%. So the conservativeness rests entirely on the long tail lying on the side of the rise: 69-73% of the centred values are below zero. The data support that shape, but it should be said.
- The reading quotes the smaller of the two printed rates. The one-value-per-configuration way gives 0.96%, 0.77% and 0.99% (GPU_ROUND_RESULTS.md lines 100-102). State the larger.
- A permutation of the 70 centred values without replacement is degenerate: the sum is always 0, so it never rejects. It was not used.
- Wording: "The two-sided test does not hold 5% on changes this skewed (6.0% to 19.5%). For Moirai-2.0, TimesFM-2.5 and Toto, whose changes have a long tail on the side of a rise, false claims of a rise occur in at most 1.0% of draws, under every null we tried that keeps that shape. For Chronos-Bolt they occur in 4.3% to 6.1%, above the 2.5% one direction should have."

## 5. TimesFM-3 and Timestral-209M: HOLDS
- The match is computed, not copied. None of the 12 MASE values is bit-identical, and the largest relative difference is 1.5e-7 (TimesFM-3) and 1.3e-7 (Timestral). For example 2.0906880800 against 2.0906879013. The other columns differ at 1e-7 to 5e-5. Both are deterministic, so this is believable.
- Documented length. TimesFM-3 gets the full history, which the package caps at 15,360 per batch; the longest history given was up to 104,400. Timestral gets up to 32,768 of the most recent steps, and 5 of the 12 hit that cap. The 512 arm gives 512 for both.
- Holm: Timestral 0.0039 x 2 = 0.0078, TimesFM-3 max(0.0078, 0.0156) = 0.0156.
- Mixing the pins does not matter here:
  - Seasonal Naive is identical between the pins on the 12. The analysis asserts a difference below 1e-12.
  - TimesFM-3's rows are identical between the pins (largest difference 0).
  - On the 12, the same ten entries at the November pin keep the same maximum and minimum (0.6846 / 0.6261; only limix_moe moves, 0.6711 to 0.6675). So S is 9.34% either way.
  - T2 does not reject for either model in any case.
- The design requires (line 137) that "Twelve configurations give wide intervals" be said next to the numbers. Each model's false-claim rate in the direction of the rise (one value per source; one per configuration) also goes with its p-value: 0.10% and 0.19% for TimesFM-3, 0.38% and 0.56% for Timestral.

## 6. Three Toto seeds: HOLDS
Recomputed from the files: +21.98%, +22.05%, +21.67%. The run_meta seeds are {42}, {43}, {44}. At the documented length, seed 42 and seed 43 differ by up to 1.1% on single configurations, so the seed took effect. The m4_daily rule is applied to each seed (no 512 file). This is on the 12, not the 70; it is not the abstract's figure.
