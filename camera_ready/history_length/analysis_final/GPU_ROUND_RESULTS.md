# GPU round: results of the checks fixed in DESIGN_ROUND3.md

Freeze, from DESIGN_ROUND3.md: nothing that finishes after 2026-11-20 12:00 IST enters the camera-ready. A request not finished by then is described in the paper as not run.

Computed by `analyze_gpu_round.py` from the results folder `fmts/camera_ready/gpu_round/pulled/harness/results`. Every number here is also in GPU_ROUND_RESULTS.json, next to the files it was computed from (path and md5 under `files_read`).

The effect of a model is computed from d_c = ln(MASE at 512 / MASE at the documented length) for each configuration c: it is exp(mean of d_c) - 1, in percent. A positive effect means the model's error is higher when it is given only the last 512 steps than with its documented history length. Scaled MASE is MASE divided by Seasonal Naive's MASE from the 2026-09-07 snapshot. The groups of every test of record are sources: the three bizitobs datasets are one source, the two ETT datasets one, the two bitbrains datasets one, and every other base dataset is its own source (14 sources on the 70 configurations, 9 on the 12).

Reference values computed from the board before any result: S, the spread of the displayed top ten, is 11.76% on the 70 configurations, 12.71% on the 58 and 9.34% on the 12 (the design wrote down 11.76%, 12.71% and 9.34%: all match). Leaving one base dataset out of the 70 moves S between 8.51% (without ett1) and 13.22% (without jena_weather).

## Check of the new machine

Outcome PASS: 72 of 72 MASE values compared with the first round, 0 missing; largest relative difference 2.08e-05 (Chronos-Bolt, documented length, jena_weather/10T/medium); 0 at or above 1e-4, 0 above 1e-3; 0 d_c with a different sign than in the first round. Below 1e-4 is PASS, from 1e-4 to 1e-3 is CONTINUE, above 1e-3 or any change of sign is STOP.

| Model | Length | Values compared | Missing | Largest relative difference | At or above 1e-4 | Above 1e-3 |
|---|---|---|---|---|---|---|
| Chronos-Bolt | documented | 12 | 0 | 2.08e-05 | 0 | 0 |
| Chronos-Bolt | 512 | 12 | 0 | 8.27e-06 | 0 | 0 |
| Moirai-2.0 | documented | 12 | 0 | 5.10e-07 | 0 | 0 |
| Moirai-2.0 | 512 | 12 | 0 | 1.27e-05 | 0 | 0 |
| TimesFM-2.5 | documented | 12 | 0 | 6.46e-07 | 0 | 0 |
| TimesFM-2.5 | 512 | 12 | 0 | 7.16e-07 | 0 | 0 |


The machine check passed.

## G3. Toto's reproduction gap

The board's value, the geometric mean of scaled MASE of the published Toto rows (Toto_Open_Base_1.0) on the 12 configurations, is 0.7699 (the paper prints 0.7699: matches). The gap is read two ways at the documented length: (a) within 1% of the board's value; (b) within noise, meaning the board's value lies inside the range of the three seeds' values widened by half that range on each side.

### Reading (a): within 1%

Toto, setting A, seed 42: geometric mean of scaled MASE 0.7698 on 12 configurations, against 0.7699 from the published rows (Toto_Open_Base_1.0 in the 2026-09-07 snapshot) on the same configurations; relative gap -0.02%; within 1%: yes; verdict PASS.

| Configuration | Our MASE | Published MASE | Relative difference |
|---|---|---|---|
| bizitobs_application/10S/medium | 2.3045 | 2.3039 | +0.03% |
| bizitobs_l2c/H/long | 0.7978 | 0.7972 | +0.08% |
| bizitobs_service/10S/short | 0.7885 | 0.7886 | -0.01% |
| electricity/H/long | 1.2358 | 1.2354 | +0.03% |
| ett1/H/medium | 1.2717 | 1.2718 | -0.01% |
| ett2/15T/long | 0.8716 | 0.8714 | +0.02% |
| jena_weather/10T/medium | 0.5977 | 0.5977 | +0.00% |
| loop_seattle/5T/long | 0.8479 | 0.8478 | +0.00% |
| m4_daily/D/short | 3.3122 | 3.3122 | +0.00% |
| m_dense/H/short | 0.8784 | 0.8786 | -0.01% |
| solar/10T/short | 1.0324 | 1.0329 | -0.04% |
| us_births/W/short | 1.2311 | 1.2346 | -0.28% |

Toto, setting A with the faster matrix products off, seed 42: not run (input file absent).

### Reading (b): within noise

Seeds 42, 43, 44 on 12 configurations: 0.7698, 0.7692, 0.7709; range 0.7692 to 0.7709, widened 0.7684 to 0.7717; the board's value on the same configurations is 0.7699, inside; verdict PASS.

Setting A meets the 1% criterion. The statement allowed by DESIGN_ROUND3.md is "the notebook's setting reproduces the board"; that the sample count is the cause rests on the other changes being unable to move the expected score.

Setting A is Toto's setting for G1 and G4 whatever this check shows (DESIGN_ROUND3.md).

The first round's point setting (one point forecast, no sample count; the paper prints 0.8126): geometric mean of scaled MASE 0.8126 on 12 configurations, against 0.7699 from the published rows (Toto_Open_Base_1.0 in the 2026-09-07 snapshot) on the same configurations; relative gap +5.54%; within 1%: no; verdict FAIL.

| Configuration | Our MASE | Published MASE | Relative difference |
|---|---|---|---|
| bizitobs_application/10S/medium | 2.6894 | 2.3039 | +16.73% |
| bizitobs_l2c/H/long | 0.9303 | 0.7972 | +16.70% |
| bizitobs_service/10S/short | 0.7865 | 0.7886 | -0.27% |
| electricity/H/long | 1.2551 | 1.2354 | +1.59% |
| ett1/H/medium | 1.2606 | 1.2718 | -0.89% |
| ett2/15T/long | 0.8964 | 0.8714 | +2.88% |
| jena_weather/10T/medium | 0.6314 | 0.5977 | +5.63% |
| loop_seattle/5T/long | 1.0899 | 0.8478 | +28.55% |
| m4_daily/D/short | 3.2956 | 3.3122 | -0.50% |
| m_dense/H/short | 0.8807 | 0.8786 | +0.25% |
| solar/10T/short | 1.0330 | 1.0329 | +0.02% |
| us_births/W/short | 1.2369 | 1.2346 | +0.19% |

## G4. Repeat runs of Toto, setting A

| Seed | Configurations (documented / 512 / both) | Geometric mean of scaled MASE, documented | Geometric mean of scaled MASE, 512 | Effect | Dropped |
|---|---|---|---|---|---|
| 42 | 12 / 12 / 12 | 0.7698 | 0.9391 | +22.0% | none |
| 43 | 12 / 12 / 12 | 0.7692 | 0.9388 | +22.1% | none |
| 44 | 12 / 12 / 12 | 0.7709 | 0.9380 | +21.7% | none |

Range over seeds 42, 43, 44: effect +21.7% to +22.1% (0.38 percentage points); geometric mean at the documented length 0.7692 to 0.7709; at 512 0.9380 to 0.9391. Seed 42 is the value of record. For m4_daily/D/short the 512 value is the documented value by rule (d_c = 0).

## G1. The history-length test on every eligible configuration

The 70 configurations are the estimate and the test of record; the 58 and the 12 are reported beside them. T1 is the exact two-sided sign-flip test of a zero effect on source totals of d_c, Holm over the four models, nominal level 5%. T2 is the exact one-sided sign-flip test of the null "mean of d_c is at most ln(1 + S)" on the shifted source totals s_g - n_g x ln(1 + S), Holm over the four models, nominal level 2.5%; a model that did not pass T1 after Holm with a positive effect enters T2's Holm correction with p = 1. The interval of record is the inverted sign-flip interval on sources. S is the spread of the board's displayed top ten on the configurations the model used. A model with no result enters each Holm correction with p = 1.

The level check repeats the review's simulation on each model's own d_c on the 70, centred to mean zero, drawn into the 14-source layout of the 70 (20,000 draws, seed 20261119): the share of draws in which T1 claims an effect although there is none, in either direction and in the direction of the model's observed effect. The paper states that rate with the p-value.

### All 70 eligible configurations (estimate and test of record)

Tests of record, on sources:

| Model | Configurations used | Sources | Effect | 95% interval (inverted sign-flip test) | T1 p, two-sided (sign patterns) | T1 p after Holm over the four models | T1 rejects at 5% | Level check, one value per source: false claims, two-sided; in the effect's direction | Level check, one value per configuration: same | S (leaving one base dataset out) | T2 p, one-sided | T2 p after Holm over the four models | T2 rejects at 2.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 70 of 70 | 14 | +2.7% | -0.9% to +6.2% | 0.1146 (1878 of 16,384) | 0.1146 | no | 6.01%; 4.75% | 5.72%; 4.26% | 11.76% (8.51% to 13.22%) | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no |
| Moirai-2.0 | 70 of 70 | 14 | +6.2% | +1.6% to +10.4% | 0.0051 (84 of 16,384) | 0.0103 | yes | 11.71%; 0.27% | 6.26%; 0.96% | 11.76% (8.51% to 13.22%) | 0.9929 | 1.0000 | no |
| TimesFM-2.5 | 70 of 70 | 14 | +15.3% | +4.1% to +32.6% | 2.44e-04 (4 of 16,384) | 9.77e-04 | yes | 19.48%; 0.07% | 7.18%; 0.77% | 11.76% (8.51% to 13.22%) | 0.4977 | 1.0000 | no |
| Toto, setting A, seed 42 | 70 of 70 | 14 | +15.3% | +4.0% to +27.1% | 2.44e-04 (4 of 16,384) | 9.77e-04 | yes | 13.63%; 0.15% | 6.41%; 0.99% | 11.76% (8.51% to 13.22%) | 0.4165 | 1.0000 | no |

Reported beside the tests, on sources, deciding nothing:

| Model | Signed-rank sign-flip p, source means | 95% interval, CR2 with Satterthwaite df | Leave one source out: smallest | largest | Leave one base dataset out: smallest | largest |
|---|---|---|---|---|---|---|
| Chronos-Bolt | 0.0785 | -1.0% to +6.6% (df 8.5) | +1.9% (without solar) | +4.1% (without bizitobs) | +1.7% (without bizitobs_l2c) | +4.0% (without bizitobs_service) |
| Moirai-2.0 | 0.0107 | +1.1% to +11.6% (df 7.2) | +4.1% (without bizitobs) | +6.9% (without solar) | +5.3% (without bizitobs_service) | +6.9% (without solar) |
| TimesFM-2.5 | 2.44e-04 | -4.5% to +39.3% (df 5.3) | +6.7% (without bizitobs) | +17.7% (without ett) | +12.5% (without bizitobs_service) | +16.7% (without ett1) |
| Toto, setting A, seed 42 | 2.44e-04 | +0.9% to +31.7% (df 5.5) | +9.3% (without bizitobs) | +16.7% (without ett) | +12.8% (without bizitobs_service) | +16.1% (without jena_weather) |

The same tests on base datasets, not used for any claim:

| Model | Base datasets | T1 p | T1 p after Holm | T1 rejects at 5% | T2 p | T2 p after Holm | T2 rejects at 2.5% | 95% interval, inverted sign-flip test | 95% interval, CR2 | Signed-rank p, base-dataset means |
|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 18 | 0.2468 | 0.2468 | no | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no | -2.4% to +6.7% | -1.8% to +7.5% (df 10.8) | 0.1964 |
| Moirai-2.0 | 18 | 3.59e-04 | 7.17e-04 | yes | 0.9963 | 1.0000 | no | +2.8% to +9.9% | +2.7% to +9.9% (df 12.6) | 6.71e-04 |
| TimesFM-2.5 | 18 | 1.53e-05 | 6.10e-05 | yes | 0.2349 | 0.7878 | no | +5.3% to +27.3% | +4.7% to +27.0% (df 10.4) | 1.53e-05 |
| Toto, setting A, seed 42 | 18 | 1.53e-05 | 6.10e-05 | yes | 0.1970 | 0.7878 | no | +6.5% to +24.0% | +6.9% to +24.2% (df 10.5) | 1.53e-05 |

No configuration was dropped.

Every model finished the same 70 configurations, so the tables above are already on the configurations all of them finished.

### The 58 configurations not among the first round's 12 (reported beside)

Tests of record, on sources:

| Model | Configurations used | Sources | Effect | 95% interval (inverted sign-flip test) | T1 p, two-sided (sign patterns) | T1 p after Holm over the four models | T1 rejects at 5% | Level check, one value per source: false claims, two-sided; in the effect's direction | Level check, one value per configuration: same | S (leaving one base dataset out) | T2 p, one-sided | T2 p after Holm over the four models | T2 rejects at 2.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 58 of 58 | 13 | +3.7% | +0.3% to +6.9% | 0.0354 (290 of 8,192) | 0.0354 | yes | only on the 70 | only on the 70 | 12.71% (9.42% to 14.34%) | 0.9999 | 1.0000 | no |
| Moirai-2.0 | 58 of 58 | 13 | +6.1% | +1.3% to +11.3% | 0.0085 (70 of 8,192) | 0.0171 | yes | only on the 70 | only on the 70 | 12.71% (9.42% to 14.34%) | 0.9901 | 1.0000 | no |
| TimesFM-2.5 | 58 of 58 | 13 | +14.0% | +4.7% to +29.3% | 2.44e-04 (2 of 8,192) | 9.77e-04 | yes | only on the 70 | only on the 70 | 12.71% (9.42% to 14.34%) | 0.4990 | 1.0000 | no |
| Toto, setting A, seed 42 | 58 of 58 | 13 | +13.9% | +4.4% to +24.3% | 2.44e-04 (2 of 8,192) | 9.77e-04 | yes | only on the 70 | only on the 70 | 12.71% (9.42% to 14.34%) | 0.4586 | 1.0000 | no |

Reported beside the tests, on sources, deciding nothing:

| Model | Signed-rank sign-flip p, source means | 95% interval, CR2 with Satterthwaite df | Leave one source out: smallest | largest | Leave one base dataset out: smallest | largest |
|---|---|---|---|---|---|---|
| Chronos-Bolt | 0.0479 | +0.4% to +7.1% (df 7.2) | +2.9% (without solar) | +4.4% (without bizitobs) | +2.9% (without solar) | +4.4% (without bizitobs_application) |
| Moirai-2.0 | 0.0034 | +0.2% to +12.4% (df 7.0) | +3.6% (without bizitobs) | +6.9% (without solar) | +4.8% (without bizitobs_l2c) | +6.9% (without solar) |
| TimesFM-2.5 | 2.44e-04 | -2.2% to +32.8% (df 6.0) | +7.0% (without bizitobs) | +16.0% (without ett) | +11.6% (without bizitobs_l2c) | +15.0% (without ett1) |
| Toto, setting A, seed 42 | 2.44e-04 | +2.3% to +26.9% (df 6.2) | +9.2% (without bizitobs) | +15.1% (without ett) | +11.7% (without bizitobs_service) | +14.6% (without kdd_cup_2018) |

The same tests on base datasets, not used for any claim:

| Model | Base datasets | T1 p | T1 p after Holm | T1 rejects at 5% | T2 p | T2 p after Holm | T2 rejects at 2.5% | 95% interval, inverted sign-flip test | 95% interval, CR2 | Signed-rank p, base-dataset means |
|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 17 | 0.0583 | 0.0583 | no | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no | -0.2% to +6.7% | +0.3% to +7.2% (df 10.9) | 0.1594 |
| Moirai-2.0 | 17 | 0.0010 | 0.0021 | yes | 0.9962 | 1.0000 | no | +2.5% to +10.4% | +2.1% to +10.2% (df 12.3) | 2.90e-04 |
| TimesFM-2.5 | 17 | 1.53e-05 | 6.10e-05 | yes | 0.3889 | 1.0000 | no | +5.8% to +24.0% | +5.1% to +23.5% (df 10.7) | 1.53e-05 |
| Toto, setting A, seed 42 | 17 | 1.53e-05 | 6.10e-05 | yes | 0.3701 | 1.0000 | no | +6.6% to +21.7% | +6.6% to +21.7% (df 10.6) | 1.53e-05 |

No configuration was dropped.

Every model finished the same 58 configurations, so the tables above are already on the configurations all of them finished.

- Chronos-Bolt: T1 on sources also rejects on the 58 after Holm over the four models (effect +3.7% on the 58, +2.7% on the 70).
- Moirai-2.0: T1 on sources also rejects on the 58 after Holm over the four models (effect +6.1% on the 58, +6.2% on the 70).
- TimesFM-2.5: T1 on sources also rejects on the 58 after Holm over the four models (effect +14.0% on the 58, +15.3% on the 70).
- Toto, setting A, seed 42: T1 on sources also rejects on the 58 after Holm over the four models (effect +13.9% on the 58, +15.3% on the 70).

Of the 58, 43 come from a base dataset that is already among the 12 and 20 use the same series as one of the 12 at another forecast horizon. The paper may say the effect is confirmed on configurations that did not suggest it only if T1 also rejects on the 58 (Holm over the four models).

### The first round's 12 configurations (reported beside)

Tests of record, on sources:

| Model | Configurations used | Sources | Effect | 95% interval (inverted sign-flip test) | T1 p, two-sided (sign patterns) | T1 p after Holm over the four models | T1 rejects at 5% | Level check, one value per source: false claims, two-sided; in the effect's direction | Level check, one value per configuration: same | S (leaving one base dataset out) | T2 p, one-sided | T2 p after Holm over the four models | T2 rejects at 2.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 12 of 12 | 9 | -2.0% | -8.1% to +4.6% | 0.9570 (490 of 512) | 0.9570 | no | only on the 70 | only on the 70 | 9.34% (7.79% to 10.54%) | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no |
| Moirai-2.0 | 12 of 12 | 9 | +6.8% | +0.8% to +12.1% | 0.0352 (18 of 512) | 0.1055 | no | only on the 70 | only on the 70 | 9.34% (7.79% to 10.54%) | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no |
| TimesFM-2.5 | 12 of 12 | 9 | +22.2% | -0.1% to +55.9% | 0.0547 (28 of 512) | 0.1094 | no | only on the 70 | only on the 70 | 9.34% (7.79% to 10.54%) | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no |
| Toto, setting A, seed 42 | 12 of 12 | 9 | +22.0% | +1.8% to +45.2% | 0.0156 (8 of 512) | 0.0625 | no | only on the 70 | only on the 70 | 9.34% (7.79% to 10.54%) | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no |

Reported beside the tests, on sources, deciding nothing:

| Model | Signed-rank sign-flip p, source means | 95% interval, CR2 with Satterthwaite df | Leave one source out: smallest | largest | Leave one base dataset out: smallest | largest |
|---|---|---|---|---|---|---|
| Chronos-Bolt | 0.4961 | -10.6% to +7.5% (df 6.3) | -2.8% (without jena_weather) | +2.1% (without bizitobs) | -5.1% (without bizitobs_l2c) | +3.0% (without bizitobs_service) |
| Moirai-2.0 | 0.0391 | +1.9% to +12.0% (df 6.3) | +5.3% (without loop_seattle) | +7.8% (without us_births) | +5.0% (without bizitobs_service) | +7.9% (without ett1) |
| TimesFM-2.5 | 0.0977 | -13.6% to +72.7% (df 5.3) | +5.4% (without bizitobs) | +26.5% (without ett) | +14.5% (without bizitobs_application) | +25.4% (without us_births) |
| Toto, setting A, seed 42 | 0.0156 | -4.0% to +55.1% (df 6.3) | +9.9% (without bizitobs) | +24.5% (without us_births) | +15.3% (without bizitobs_application) | +24.5% (without us_births) |

The same tests on base datasets, not used for any claim:

| Model | Base datasets | T1 p | T1 p after Holm | T1 rejects at 5% | T2 p | T2 p after Holm | T2 rejects at 2.5% | 95% interval, inverted sign-flip test | 95% interval, CR2 | Signed-rank p, base-dataset means |
|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 12 | 0.7539 | 0.7539 | no | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no | -14.2% to +10.3% | -14.2% to +12.0% (df 11.0) | 0.6772 |
| Moirai-2.0 | 12 | 0.0713 | 0.1426 | no | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no | -0.6% to +14.8% | -0.4% to +14.6% (df 11.0) | 0.1763 |
| TimesFM-2.5 | 12 | 0.0176 | 0.0527 | no | not computed (T1 does not reject after Holm; enters Holm with p = 1) | 1.0000 | no | +2.3% to +50.1% | +0.2% to +48.9% (df 11.0) | 0.0269 |
| Toto, setting A, seed 42 | 12 | 0.0029 | 0.0117 | yes | 0.0830 | 0.3320 | no | +5.7% to +43.4% | +4.1% to +43.0% (df 11.0) | 0.0029 |

No configuration was dropped.

Every model finished the same 12 configurations, so the tables above are already on the configurations all of them finished.

Limits stated with the result (DESIGN_ROUND3.md): S is used as a fixed yardstick although it is itself computed from data. Fourteen sources is few: the smallest two-sided p-value possible is 2 / 2^14, and a single large source can still carry a sum; the leave-one-out ranges are there for that reason. The sign-flip test is exact only when, with no true effect, a source's total is as likely to be positive as negative; the level check above shows how far that holds for each model.

## G2. Two newer models on the 12 first-round configurations

The board's value for TimesFM-3 on the 12 configurations, from the published rows of TimesFM-3 in the 2026-09-07 snapshot, is 0.6750.
The board's value for Timestral-209M on the 12 configurations, from the published rows of Timestral-209M at the November pin, GIFT-Eval commit 42beef3e, is 0.6521.

Reproduction check, TimesFM-3: geometric mean of scaled MASE 0.6750 on 12 configurations, against 0.6750 from the published rows (TimesFM-3 in the 2026-09-07 snapshot) on the same configurations; relative gap +0.00%; within 1%: yes; verdict PASS.

| Configuration | Our MASE | Published MASE | Relative difference |
|---|---|---|---|
| bizitobs_application/10S/medium | 2.0907 | 2.0907 | +0.00% |
| bizitobs_l2c/H/long | 0.5486 | 0.5486 | +0.00% |
| bizitobs_service/10S/short | 0.6754 | 0.6754 | -0.00% |
| electricity/H/long | 1.0756 | 1.0756 | +0.00% |
| ett1/H/medium | 1.2695 | 1.2695 | +0.00% |
| ett2/15T/long | 0.8667 | 0.8667 | -0.00% |
| jena_weather/10T/medium | 0.5596 | 0.5596 | +0.00% |
| loop_seattle/5T/long | 0.7504 | 0.7504 | -0.00% |
| m4_daily/D/short | 3.2500 | 3.2500 | +0.00% |
| m_dense/H/short | 0.6908 | 0.6908 | +0.00% |
| solar/10T/short | 0.8940 | 0.8940 | +0.00% |
| us_births/W/short | 0.9954 | 0.9954 | -0.00% |

Reproduction check, Timestral-209M: geometric mean of scaled MASE 0.6521 on 12 configurations, against 0.6521 from the published rows (Timestral-209M at the November pin, GIFT-Eval commit 42beef3e) on the same configurations; relative gap +0.00%; within 1%: yes; verdict PASS.

| Configuration | Our MASE | Published MASE | Relative difference |
|---|---|---|---|
| bizitobs_application/10S/medium | 2.1173 | 2.1173 | -0.00% |
| bizitobs_l2c/H/long | 0.5359 | 0.5359 | +0.00% |
| bizitobs_service/10S/short | 0.6583 | 0.6583 | +0.00% |
| electricity/H/long | 1.0460 | 1.0460 | -0.00% |
| ett1/H/medium | 1.2564 | 1.2564 | +0.00% |
| ett2/15T/long | 0.8345 | 0.8345 | -0.00% |
| jena_weather/10T/medium | 0.5387 | 0.5387 | -0.00% |
| loop_seattle/5T/long | 0.7327 | 0.7327 | -0.00% |
| m4_daily/D/short | 3.1936 | 3.1936 | +0.00% |
| m_dense/H/short | 0.6389 | 0.6389 | +0.00% |
| solar/10T/short | 0.7907 | 0.7907 | +0.00% |
| us_births/W/short | 0.9735 | 0.9735 | +0.00% |

Tests of record, on sources:

| Model | Configurations used | Sources | Effect | 95% interval (inverted sign-flip test) | T1 p, two-sided (sign patterns) | T1 p after Holm over the two models | T1 rejects at 5% | Level check, one value per source: false claims, two-sided; in the effect's direction | Level check, one value per configuration: same | S (leaving one base dataset out) | T2 p, one-sided | T2 p after Holm over the two models | T2 rejects at 2.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TimesFM-3 | 12 of 12 | 9 | +31.8% | +5.3% to +69.4% | 0.0156 (8 of 512) | 0.0156 | yes | 16.32%; 0.10% | 13.39%; 0.19% | 9.34% (7.79% to 10.54%) | 0.1035 | 0.1094 | no |
| Timestral-209M | 12 of 12 | 9 | +28.9% | +6.3% to +54.1% | 0.0039 (2 of 512) | 0.0078 | yes | 10.12%; 0.38% | 8.20%; 0.56% | 9.34% (7.79% to 10.54%) | 0.0547 | 0.1094 | no |

Reported beside the tests, on sources, deciding nothing:

| Model | Signed-rank sign-flip p, source means | 95% interval, CR2 with Satterthwaite df | Leave one source out: smallest | largest | Leave one base dataset out: smallest | largest |
|---|---|---|---|---|---|---|
| TimesFM-3 | 0.0195 | -6.3% to +85.3% (df 6.3) | +13.6% (without bizitobs) | +37.7% (without ett) | +20.0% (without bizitobs_application) | +35.5% (without m4_daily) |
| Timestral-209M | 0.0039 | +0.8% to +64.9% (df 6.3) | +16.4% (without bizitobs) | +33.9% (without ett) | +21.7% (without bizitobs_application) | +32.1% (without ett1) |

The same tests on base datasets, not used for any claim:

| Model | Base datasets | T1 p | T1 p after Holm | T1 rejects at 5% | T2 p | T2 p after Holm | T2 rejects at 2.5% | 95% interval, inverted sign-flip test | 95% interval, CR2 | Signed-rank p, base-dataset means |
|---|---|---|---|---|---|---|---|---|---|---|
| TimesFM-3 | 12 | 0.0034 | 0.0034 | yes | 0.0291 | 0.0479 | no | +8.9% to +66.9% | +4.3% to +66.6% (df 11.0) | 0.0049 |
| Timestral-209M | 12 | 0.0015 | 0.0029 | yes | 0.0239 | 0.0479 | no | +9.5% to +54.2% | +8.2% to +53.6% (df 11.0) | 0.0015 |

T2 uses S computed on the configurations the model used; on all 12 it is 9.34% (the paper's 9.3%). The level check uses the 9-source layout of the 12. Twelve configurations from nine sources give wide intervals.

No configuration was dropped.

Every model finished the same 12 configurations, so the tables above are already on the configurations all of them finished.

## Run records

| Variant | Runs | Cards | torch | CUDA | Hugging Face revision | Batch size passed to evaluate_model | Peak memory (largest) | Wall time, total |
|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt | 140 ok, 9 failed | NVIDIA A100-SXM4-80GB (149) | 2.14.0+cu130 (149) | 13.0 (149) | 5d9f166d69f47aef3401367a7b842e78fe97b121 (140), not recorded (9) | 64 (140), not recorded (9) | peak_gpu_memory_gb 7.279217664 | 8.57 h |
| Moirai-2.0 | 140 ok, 9 failed | NVIDIA A100-SXM4-80GB (149) | 2.4.1+cu121 (149) | 12.1 (149) | 30f43ff08c8494f4943ae1521e9d4e94a0fbb389 (140), not recorded (9) | 512 (140), not recorded (9) | peak_gpu_memory_gb 44.479368704 | 3.59 h |
| TimesFM-2.5 | 140 ok, 14 failed | NVIDIA A100-SXM4-80GB (154) | 2.14.0+cu130 (154) | 13.0 (154) | 1d952420fba87f3c6dee4f240de0f1a0fbc790e3 (140), not recorded (14) | 1024 (140), not recorded (14) | peak_gpu_memory_gb 25.627358208 | 5.12 h |
| Toto, setting A, seed 42 | 139 ok, 0 failed | NVIDIA A100-SXM4-80GB (139) | 2.7.0+cu126 (139) | 12.6 (139) | 0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5 (139) | 256 (139) | peak_gpu_memory_gb 84.281571328 | 8.70 h |
| Toto, setting A, seed 43 | 23 ok, 0 failed | NVIDIA A100-SXM4-80GB (23) | 2.7.0+cu126 (23) | 12.6 (23) | 0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5 (23) | 256 (23) | peak_gpu_memory_gb 84.281571328 | 0.90 h |
| Toto, setting A, seed 44 | 23 ok, 0 failed | NVIDIA A100-SXM4-80GB (23) | 2.7.0+cu126 (23) | 12.6 (23) | 0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5 (23) | 256 (23) | peak_gpu_memory_gb 84.281571328 | 0.90 h |
| TimesFM-3 | 24 ok, 8 failed | NVIDIA A100-SXM4-80GB (32) | 2.14.0+cu130 (32) | 13.0 (32) | google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2 (24), not recorded (8) | 16 (12), 64 (12), not recorded (8) | peak_gpu_memory_gb 27.00477696 | 0.96 h |
| Timestral-209M | 24 ok, 6 failed | NVIDIA A100-SXM4-80GB (30) | 2.14.1+cu130 (30) | 13.0 (30) | mistralai/Timestral-209M@9874d252ab0aae16f7f7c999f98f98d6ad7d8365 (24), not recorded (6) | 512 (24), not recorded (6) | peak_gpu_memory_gb 83.068296192 | 0.69 h |

| Toto variant | Matmul precision | Deterministic | xFormers installed | Samples per batch after retries | Batch size |
|---|---|---|---|---|---|
| Toto, setting A, seed 42 | high (139) | true (139) | false (139) | 256 (133), 32 (3), 64 (3) | 11 (12), 3 (19), 16 (34), 1 (42), 2 (26), 8 (1), 7 (2), 5 (1), 4 (1), 12 (1) |
| Toto, setting A, seed 43 | high (23) | true (23) | false (23) | 256 (22), 64 (1) | 11 (2), 3 (3), 16 (6), 1 (7), 2 (4), 12 (1) |
| Toto, setting A, seed 44 | high (23) | true (23) | false (23) | 256 (22), 64 (1) | 11 (2), 3 (3), 16 (6), 1 (7), 2 (4), 12 (1) |

## How missing or odd values were handled

- Groups of record are sources (DESIGN_ROUND3.md, Revision 1). The same tests on base datasets are printed beside them and are not used for any claim.
- A configuration enters a model's analysis only if both lengths are present; the ones that drop out are named, and S for that model is computed on the configurations that remain.
- Toto (every setting-A variant): for m4_daily/D/short the 512 value is the documented value by rule (d_c = 0), whenever the variant's 512 results file exists. A 512 row for it, if present, is replaced.
- A MASE value that is missing, not finite, or not positive counts as a missing configuration (a failed run), and so does a configuration listed twice in one file with different values; both are named under files_read -> problems.
- T1: Holm over a fixed family (four models in G1, two in G2); a model with no result enters with p = 1. T2: Holm over the same family; a model that did not pass T1 after Holm with a positive effect, or has no result, enters with p = 1. In G1 each of the 70, the 58 and the 12 has its own Holm correction.
- Machine check: largest relative difference below 1e-4 is PASS; from 1e-4 to 1e-3 is CONTINUE; above 1e-3, or any d_c of the three models with a different sign (positive, zero, negative) than in the first round, is STOP. Only STOP holds the other parts (unless --analyse-despite-machine-check is given). The outcome is computed on the values present; missing values are counted and named.
- A reproduction check (G3, G2) with configurations missing is computed on the configurations present and marked INCOMPLETE instead of PASS or FAIL; G3 reading (b) is NOT RUN if a seed's file is absent.
- Level check: the model's d_c on the configurations it used, centred to mean zero, drawn with replacement into the fixed source layout (14 sources of the 70 for G1, 9 sources of the 12 for G2), one value shared inside a source or one value per configuration; a fresh generator with the design's seed for each model and each way.

- m4_daily/D/short rule, per Toto variant: toto__A__seed42: applied; toto__A__seed43: applied; toto__A__seed44: applied; toto__A_tf32off__seed42: not applied: the 512 results file is absent.

