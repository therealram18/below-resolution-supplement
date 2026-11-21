# Result 2 -- context-length grid, final numbers

Our 4 measured models, each run at its own documented default context length and at a fixed 512, on the 12 pre-registered configs (`selection/selected_configs.json`). Two aggregations per model x context: the mean rank our measured MASE would get if inserted into the real leaderboard's per-config score pool, and the geometric mean of our MASE divided by seasonal-naive's MASE. Sampling-noise bands are read from `audit/result2_bands.json`, not recomputed here (except for the robustness check in section 10, which does recompute them).

Snapshot commit `b61c48aa3197a1a111e405c0e6f600128a1efee3`. `selection/selected_configs.json` md5: `32d6a021400e97e7040e0963449761f9`. `audit/result2_bands.json` md5: `8472fd71b201c2bf36abacf919776f88`.

All 8 grid CSVs (4 models x 2 contexts) contain exactly the 12 selected configs, no more, no fewer: True.

## 1. Model x context table (both aggregations)

| model | context | mean insertion rank | gmean scaled MASE |
|---|---|---|---|
| chronos_bolt_base | default | 80.33 | 0.9650 |
| chronos_bolt_base | 512 | 87.00 | 0.9461 |
| moirai2_small | default | 63.33 | 0.7233 |
| moirai2_small | 512 | 67.42 | 0.7727 |
| timesfm25 | default | 62.25 | 0.7348 |
| timesfm25 | 512 | 88.83 | 0.8977 |
| toto | default | 82.08 | 0.8126 |
| toto | 512 | 98.67 | 0.9801 |

## 2. Movement from default to 512 -- paired bootstrap CI (primary), band comparison (coarser, secondary)

Delta = value at 512 minus value at default. Paired CI: bootstrap the difference itself, resampling the 12 configs with replacement (B=4000, seed 20260914), the same resampled config set used for both contexts on every replicate. "excludes zero" is the primary verdict. The band-half-width comparison is a coarser yardstick: movement vs the config-selection spread of the score level (how much this model's own score moves around just from which 12 configs get picked, at a fixed context length). The paired bootstrap CI on the difference itself is the primary test; the band comparison is reported alongside it, secondary.

| model | aggregation | default | at 512 | Delta | paired 95% CI | excludes zero (primary) | band half-width (coarser) | \|Delta\|/half-width | exceeds band (coarser)? |
|---|---|---|---|---|---|---|---|---|---|
| chronos_bolt_base | mean insertion rank | 80.33 | 87.00 | +6.67 | [-4.92, +19.00] | False | 17.83 | 0.37x | False |
| chronos_bolt_base | gmean scaled MASE | 0.9650 | 0.9461 | -0.0189 | [-0.1750, +0.0633] | False | 0.3296 | 0.06x | False |
| moirai2_small | mean insertion rank | 63.33 | 67.42 | +4.08 | [-6.92, +14.08] | False | 18.21 | 0.22x | False |
| moirai2_small | gmean scaled MASE | 0.7233 | 0.7727 | +0.0494 | [+0.0060, +0.1070] | True | 0.1023 | 0.48x | False |
| timesfm25 | mean insertion rank | 62.25 | 88.83 | +26.58 | [+8.83, +44.42] | True | 11.87 | 2.24x | True |
| timesfm25 | gmean scaled MASE | 0.7348 | 0.8977 | +0.1629 | [+0.0402, +0.3416] | True | 0.1068 | 1.53x | True |
| toto | mean insertion rank | 82.08 | 98.67 | +16.58 | [+6.50, +28.00] | True | 11.83 | 1.40x | True |
| toto | gmean scaled MASE | 0.8126 | 0.9801 | +0.1675 | [+0.0666, +0.3169] | True | 0.0775 | 2.16x | True |

## 3. Leave-one-out: does the exceeds-band verdict survive dropping any one of the 12 configs?

Recomputes the point delta and the (coarser, band-based) exceeds-band verdict on the remaining 11 configs, once per dropped config; "flips" = verdict differs from the full-12 verdict.

| model | aggregation | flips (of 12) | flipping configs |
|---|---|---|---|
| chronos_bolt_base | mean insertion rank | 0 | none |
| chronos_bolt_base | gmean scaled MASE | 0 | none |
| moirai2_small | mean insertion rank | 0 | none |
| moirai2_small | gmean scaled MASE | 0 | none |
| timesfm25 | mean insertion rank | 0 | none |
| timesfm25 | gmean scaled MASE | 1 | bizitobs_application/10S/medium |
| toto | mean insertion rank | 0 | none |
| toto | gmean scaled MASE | 0 | none |

## 4. Direction consistency: per-config MASE, 512 vs default

Count of the 12 configs where the model's MASE is worse (higher) at 512 than at its own default.

| model | worse at 512 | better at 512 | tied |
|---|---|---|---|
| chronos_bolt_base | 7/12 | 5/12 | 0/12 |
| moirai2_small | 7/12 | 5/12 | 0/12 |
| timesfm25 | 9/12 | 3/12 | 0/12 |
| toto | 9/12 | 2/12 | 1/12 |

## 5. Ordering of the 4 models, default vs 512

**Mean insertion rank** (best to worst):
- default: timesfm25 > moirai2_small > chronos_bolt_base > toto
- 512: moirai2_small > chronos_bolt_base > timesfm25 > toto
  - rank flip: chronos_bolt_base vs timesfm25 -- default 80.33 vs 62.25 (gap 18.08), 512 87.00 vs 88.83 (gap 1.83)
  - rank flip: moirai2_small vs timesfm25 -- default 63.33 vs 62.25 (gap 1.08), 512 67.42 vs 88.83 (gap 21.42)

**Gmean scaled MASE** (best to worst):
- default: moirai2_small > timesfm25 > toto > chronos_bolt_base
- 512: moirai2_small > timesfm25 > chronos_bolt_base > toto
  - rank flip: chronos_bolt_base vs toto -- default 0.9650 vs 0.8126 (gap 0.1524), 512 0.9461 vs 0.9801 (gap 0.0340)

## 6. Moirai check (pre-registered question)

moirai2_small's MASE at context length 4000 (default) is different from its MASE at context length 512, across the 12 pre-registered configs (max relative difference 29.8228%, median relative difference 3.7220%).

| config | MASE at default (4000) | MASE at 512 | relative diff |
|---|---|---|---|
| bizitobs_service/10S/short | 0.8739 | 1.1345 | 29.82% |
| m4_daily/D/short | 3.0710 | 3.0264 | 1.45% |
| m_dense/H/short | 0.7930 | 0.8142 | 2.68% |
| solar/10T/short | 0.6575 | 0.7074 | 7.58% |
| us_births/W/short | 0.8714 | 0.8427 | 3.29% |
| bizitobs_application/10S/medium | 2.4790 | 2.4622 | 0.68% |
| ett1/H/medium | 1.3043 | 1.2553 | 3.75% |
| jena_weather/10T/medium | 0.6510 | 0.7176 | 10.23% |
| bizitobs_l2c/H/long | 0.6108 | 0.5882 | 3.69% |
| electricity/H/long | 1.2622 | 1.2863 | 1.91% |
| ett2/15T/long | 0.9591 | 1.1952 | 24.62% |
| loop_seattle/5T/long | 0.9282 | 1.1648 | 25.48% |

## 7. Reproduction check: our default-context MASE vs the snapshot's published rows, all 4 models

**chronos_bolt_base** (snapshot key `chronos_bolt_base`, `testdata_leakage: Yes`): median relative diff 0.46%, max 2.53% (at `loop_seattle/5T/long`). Our default-context mean insertion rank 80.33 vs the published entry's 79.58 on the same 12 configs; our gmean scaled MASE 0.9650 vs published 0.9676.

**moirai2_small** (snapshot key `Moirai2`, `testdata_leakage: No`): median relative diff 0.00%, max 0.00% (at `bizitobs_application/10S/medium`). Our default-context mean insertion rank 63.33 vs the published entry's 62.83 on the same 12 configs; our gmean scaled MASE 0.7233 vs published 0.7233.

**timesfm25** (snapshot key `TimesFM-2.5`, `testdata_leakage: No`): median relative diff 0.00%, max 0.00% (at `bizitobs_service/10S/short`). Our default-context mean insertion rank 62.25 vs the published entry's 61.75 on the same 12 configs; our gmean scaled MASE 0.7348 vs published 0.7348.

**toto** (snapshot key `Toto_Open_Base_1.0`, `testdata_leakage: No`): median relative diff 1.24%, max 28.55% (at `loop_seattle/5T/long`). Our default-context mean insertion rank 82.08 vs the published entry's 69.08 on the same 12 configs; our gmean scaled MASE 0.8126 vs published 0.7699.

Toto's reproduction gap is the largest of the 4: median relative difference 1.24%, max 28.55% at `loop_seattle/5T/long`, and its own mean insertion rank comes out 82.08 (ours) vs 69.08 (published) on the same 12 configs -- moirai2_small and timesfm25 reproduce far tighter (both medians and maxes under 1e-6 relative difference).

| model | config | ours (default ctx) | snapshot published | relative diff |
|---|---|---|---|---|
| chronos_bolt_base | bizitobs_service/10S/short | 3.2986 | 3.3162 | 0.53% |
| chronos_bolt_base | m4_daily/D/short | 3.1985 | 3.1985 | 0.00% |
| chronos_bolt_base | m_dense/H/short | 0.7754 | 0.7754 | 0.00% |
| chronos_bolt_base | solar/10T/short | 0.9907 | 0.9907 | 0.00% |
| chronos_bolt_base | us_births/W/short | 1.0875 | 1.0875 | 0.00% |
| chronos_bolt_base | bizitobs_application/10S/medium | 9.5763 | 9.7203 | 1.48% |
| chronos_bolt_base | ett1/H/medium | 1.3615 | 1.3748 | 0.96% |
| chronos_bolt_base | jena_weather/10T/medium | 0.6079 | 0.6102 | 0.39% |
| chronos_bolt_base | bizitobs_l2c/H/long | 0.5652 | 0.5557 | 1.72% |
| chronos_bolt_base | electricity/H/long | 1.2424 | 1.2375 | 0.40% |
| chronos_bolt_base | ett2/15T/long | 0.9456 | 0.9400 | 0.60% |
| chronos_bolt_base | loop_seattle/5T/long | 1.2073 | 1.2386 | 2.53% |
| moirai2_small | bizitobs_service/10S/short | 0.8739 | 0.8739 | 0.00% |
| moirai2_small | m4_daily/D/short | 3.0710 | 3.0710 | 0.00% |
| moirai2_small | m_dense/H/short | 0.7930 | 0.7930 | 0.00% |
| moirai2_small | solar/10T/short | 0.6575 | 0.6575 | 0.00% |
| moirai2_small | us_births/W/short | 0.8714 | 0.8714 | 0.00% |
| moirai2_small | bizitobs_application/10S/medium | 2.4790 | 2.4790 | 0.00% |
| moirai2_small | ett1/H/medium | 1.3043 | 1.3043 | 0.00% |
| moirai2_small | jena_weather/10T/medium | 0.6510 | 0.6510 | 0.00% |
| moirai2_small | bizitobs_l2c/H/long | 0.6108 | 0.6108 | 0.00% |
| moirai2_small | electricity/H/long | 1.2622 | 1.2622 | 0.00% |
| moirai2_small | ett2/15T/long | 0.9591 | 0.9591 | 0.00% |
| moirai2_small | loop_seattle/5T/long | 0.9282 | 0.9282 | 0.00% |
| timesfm25 | bizitobs_service/10S/short | 0.7261 | 0.7261 | 0.00% |
| timesfm25 | m4_daily/D/short | 3.2974 | 3.2974 | 0.00% |
| timesfm25 | m_dense/H/short | 0.7688 | 0.7688 | 0.00% |
| timesfm25 | solar/10T/short | 1.0857 | 1.0857 | 0.00% |
| timesfm25 | us_births/W/short | 1.0866 | 1.0866 | 0.00% |
| timesfm25 | bizitobs_application/10S/medium | 2.1551 | 2.1552 | 0.00% |
| timesfm25 | ett1/H/medium | 1.3476 | 1.3476 | 0.00% |
| timesfm25 | jena_weather/10T/medium | 0.6271 | 0.6271 | 0.00% |
| timesfm25 | bizitobs_l2c/H/long | 0.5939 | 0.5939 | 0.00% |
| timesfm25 | electricity/H/long | 1.1440 | 1.1440 | 0.00% |
| timesfm25 | ett2/15T/long | 0.9372 | 0.9372 | 0.00% |
| timesfm25 | loop_seattle/5T/long | 0.8448 | 0.8448 | 0.00% |
| toto | bizitobs_service/10S/short | 0.7865 | 0.7886 | 0.27% |
| toto | m4_daily/D/short | 3.2956 | 3.3122 | 0.50% |
| toto | m_dense/H/short | 0.8807 | 0.8786 | 0.25% |
| toto | solar/10T/short | 1.0330 | 1.0329 | 0.02% |
| toto | us_births/W/short | 1.2369 | 1.2346 | 0.19% |
| toto | bizitobs_application/10S/medium | 2.6894 | 2.3039 | 16.73% |
| toto | ett1/H/medium | 1.2606 | 1.2718 | 0.89% |
| toto | jena_weather/10T/medium | 0.6314 | 0.5977 | 5.63% |
| toto | bizitobs_l2c/H/long | 0.9303 | 0.7972 | 16.70% |
| toto | electricity/H/long | 1.2551 | 1.2354 | 1.59% |
| toto | ett2/15T/long | 0.8964 | 0.8714 | 2.88% |
| toto | loop_seattle/5T/long | 1.0899 | 0.8478 | 28.55% |

## 8. Per-config detail (appendix): MASE at default, at 512, relative change

| config | model | MASE default | MASE 512 | relative change % |
|---|---|---|---|---|
| bizitobs_service/10S/short | chronos_bolt_base | 3.2986 | 1.8772 | -43.09% |
| bizitobs_service/10S/short | moirai2_small | 0.8739 | 1.1345 | +29.82% |
| bizitobs_service/10S/short | timesfm25 | 0.7261 | 1.5151 | +108.67% |
| bizitobs_service/10S/short | toto | 0.7865 | 1.3157 | +67.29% |
| m4_daily/D/short | chronos_bolt_base | 3.1985 | 3.1493 | -1.54% |
| m4_daily/D/short | moirai2_small | 3.0710 | 3.0264 | -1.45% |
| m4_daily/D/short | timesfm25 | 3.2974 | 3.2882 | -0.28% |
| m4_daily/D/short | toto | 3.2956 | 3.2956 | +0.00% |
| m_dense/H/short | chronos_bolt_base | 0.7754 | 0.7838 | +1.08% |
| m_dense/H/short | moirai2_small | 0.7930 | 0.8142 | +2.68% |
| m_dense/H/short | timesfm25 | 0.7688 | 0.8199 | +6.66% |
| m_dense/H/short | toto | 0.8807 | 0.9302 | +5.61% |
| solar/10T/short | chronos_bolt_base | 0.9907 | 1.0570 | +6.69% |
| solar/10T/short | moirai2_small | 0.6575 | 0.7074 | +7.58% |
| solar/10T/short | timesfm25 | 1.0857 | 1.1660 | +7.40% |
| solar/10T/short | toto | 1.0330 | 1.3647 | +32.10% |
| us_births/W/short | chronos_bolt_base | 1.0875 | 1.1215 | +3.13% |
| us_births/W/short | moirai2_small | 0.8714 | 0.8427 | -3.29% |
| us_births/W/short | timesfm25 | 1.0866 | 0.9950 | -8.43% |
| us_births/W/short | toto | 1.2369 | 1.2003 | -2.96% |
| bizitobs_application/10S/medium | chronos_bolt_base | 9.5763 | 7.8640 | -17.88% |
| bizitobs_application/10S/medium | moirai2_small | 2.4790 | 2.4622 | -0.68% |
| bizitobs_application/10S/medium | timesfm25 | 2.1551 | 5.3533 | +148.40% |
| bizitobs_application/10S/medium | toto | 2.6894 | 5.4590 | +102.98% |
| ett1/H/medium | chronos_bolt_base | 1.3615 | 1.2932 | -5.02% |
| ett1/H/medium | moirai2_small | 1.3043 | 1.2553 | -3.75% |
| ett1/H/medium | timesfm25 | 1.3476 | 1.2729 | -5.54% |
| ett1/H/medium | toto | 1.2606 | 1.2521 | -0.67% |
| jena_weather/10T/medium | chronos_bolt_base | 0.6079 | 0.6524 | +7.34% |
| jena_weather/10T/medium | moirai2_small | 0.6510 | 0.7176 | +10.23% |
| jena_weather/10T/medium | timesfm25 | 0.6271 | 0.6594 | +5.16% |
| jena_weather/10T/medium | toto | 0.6314 | 0.6741 | +6.76% |
| bizitobs_l2c/H/long | chronos_bolt_base | 0.5652 | 0.7885 | +39.50% |
| bizitobs_l2c/H/long | moirai2_small | 0.6108 | 0.5882 | -3.69% |
| bizitobs_l2c/H/long | timesfm25 | 0.5939 | 0.7864 | +32.42% |
| bizitobs_l2c/H/long | toto | 0.9303 | 1.0972 | +17.94% |
| electricity/H/long | chronos_bolt_base | 1.2424 | 1.2854 | +3.46% |
| electricity/H/long | moirai2_small | 1.2622 | 1.2863 | +1.91% |
| electricity/H/long | timesfm25 | 1.1440 | 1.2486 | +9.14% |
| electricity/H/long | toto | 1.2551 | 1.3058 | +4.04% |
| ett2/15T/long | chronos_bolt_base | 0.9456 | 1.0138 | +7.21% |
| ett2/15T/long | moirai2_small | 0.9591 | 1.1952 | +24.62% |
| ett2/15T/long | timesfm25 | 0.9372 | 1.0423 | +11.21% |
| ett2/15T/long | toto | 0.8964 | 1.1225 | +25.22% |
| loop_seattle/5T/long | chronos_bolt_base | 1.2073 | 1.1795 | -2.30% |
| loop_seattle/5T/long | moirai2_small | 0.9282 | 1.1648 | +25.48% |
| loop_seattle/5T/long | timesfm25 | 0.8448 | 1.0792 | +27.74% |
| loop_seattle/5T/long | toto | 1.0899 | 1.3792 | +26.55% |

## 9. Toto / m4_daily: no-op grid cell

toto's recipe caps context length at `min(setting, shortest_series_length - (windows+1)*prediction_length)`; for m4_daily that cap lands on the same value under both the default and the 512 setting, so the two runs are byte-identical rows and that cell measures nothing (toto/m4_daily/D/short). This biases toto's measured movement toward zero for m4_daily specifically, which makes toto's overall exceeds-band verdict conservative (if anything, an understatement of toto's real movement), not an inflation of it.

## 10. Mixed-provenance robustness note

The headline band comparison in section 2 is mixed-provenance: the band half-widths come from the snapshot's own published rows, while the 512-vs-default movement comes from our measurements. Recomputing those same bands from our own default-context per-config values instead (same bootstrap: seed 20260913, 4000 resamples of the 12 configs) changes the half-widths by at most 1.46 rank places / 0.0040 gmean across the 4 models, and flips no exceeds-band verdict.
