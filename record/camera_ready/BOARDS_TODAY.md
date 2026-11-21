# Boards today (second pin, 2026-11-13). Spec: DESIGN_ROUND2.md S5

Pins: boards/PIN_2026-11-13.md. Numbers below come from boards/pin_2026-11-13/ (gift_replay_after_b61c48aa.json, gift_pin_summary.json, time_pin_summary.json, fev_pin_summary.json, report_*.json). Scripts: pin2_fetch_time.py, pin2_replay.py, pin2_boards.py, pin2_syncdiff.py. Seed 20261111, 20,000 bootstrap draws, 99,999 wild-bootstrap draws.

| Board | Pin | Commits since first-round pin |
|---|---|---|
| GIFT-Eval | 42beef3e (2026-11-13) | 39 since b61c48aa (38 touch results/, one is a docs note) |
| fev-bench | 9cf943c (2026-11-11) | 0, so not re-analysed (the entry tests in section 4 still use it) |
| TIME | 3731cc5 (2026-11-11) | 4 since 3bf103c: Simons-25M, -60M, -178M, -297M added; nothing else changed |

## 1. GIFT-Eval updates after b61c48aa and the reversals

38 updates, each compared with the one before (all had the same 97 configs, so all 38 were replayed). 30 of them added entries (41 names; one of these, HCR, is the rename of HistRoute-CV; the Simons family of 4 arrived on the last day), 5 changed an existing entry's results (PatchFM, Zeus, limix_moe, Prism, and the 24-file sync of 2026-09-28), and 4 changed only a label or config (iTransformer, IK-QuadTSFM, Tabby, Tabby-Pretrain). The 2026-09-28 "sync validated submissions" commit rewrote 24 results files but no MASE value differs (checked), so those entries count as "changed" in the replay by file hash, as in the first round. Full list with dates: gift_replay_after_b61c48aa.json, key "updates".

| Convention | Transitions | With a reversal | Reversed pairs |
|---|---|---|---|
| Tied ranks averaged (vernier_board.history_replay) | 38 | 8 | 8 |
| Board's exact tie-break (poolcheck code, ranks "first" in alphabetical model order) | 38 | 8 | 8 |

The two conventions give the same 8 pairs (no exact tie between any two untouched top-ten entries was met). Pair order is the order before the update:

| Update | What arrived | Pair that swapped |
|---|---|---|
| 2026-10-16 | Timestral family (4) | limix_moe over EXAONE-Forecast-Agent became the reverse |
| 2026-10-17 | Granite-timeseries-ensemble-r1 (2) | EXAONE-Forecast-Agent / limix_moe, back again |
| 2026-10-23 | PCFM | PERSONA-AI-T4 / Timestral-209M |
| 2026-10-23 | RegLab-AH-2 | limix_moe / EXAONE-Forecast-Agent |
| 2026-10-23 | TimeH2O-v1 and -tiny | EXAONE-Forecast-Agent / limix_moe |
| 2026-10-29 | HART-36M | limix_moe / EXAONE-Forecast-Agent |
| 2026-10-30 | iAmTime | EXAONE-Forecast-Agent / limix_moe |
| 2026-10-30 | PDM-Femto, -Nano, -Pico | limix_moe / EXAONE-Forecast-Agent |

So limix_moe and EXAONE-Forecast-Agent changed places 7 times during these updates, none of which touched either entry. Together with the first-round count (18 averaged, 20 exact over 141 updates), the history is now 26 and 28 reversals over 179 updates.

## 2. First-round report on the new GIFT-Eval pin (displayed rule MASE_Rank)

Entries counted: 170 on the pin, all displayed (the first round displayed 128 of 131 names; the three not displayed were GestaltCog/Zeus-100M, Zeus and iTransformer, whose names disagreed between the results and config files until 2026-09-28). Regression check: my loader and report code on the 2026-09-07 files gives the stored first-round report exactly (top ten, resolution 6.209, all counts, leading tier). The paper's 6.16 comes from its own script; the first-round code gives 6.209.

| | 2026-09-07 (b61c48aa) | 2026-11-13 (42beef3e) |
|---|---|---|
| Entries displayed | 128 | 170 |
| Resolution, paper / corrected | 6.21 / 6.80 | 15.32 / 17.28 |
| Median gap between neighbours | 0.907 | 0.485 |
| Resolution / gap, paper | 6.8 | 31.6 |
| Neighbours separated of 9: paper, CR2, wild | 0, 1, 1 | 0, 0, 0 |
| Pairs of 45 separated, uncorrected: paper / CR2 / wild | 12 / 12 / 13 | 9 / 8 / 8 |
| After Holm: paper / CR2 / wild | 2 / 0 / 1 | 1 / 0 / 0 |
| Leading tier: paper / CR2 / wild | 3 / 3 / 3 | 2 / 3 / 3 |
| Removal of one entry outside the top ten reorders the top ten | 28 of 121 | 67 of 160 |
| Near-copy of an entry reorders the top ten | 42 of 128 | 53 of 170 |

Leading tier now (paper rule): Taichu-TimeSeries-Agent-v2 and STRIDE w/ Synapse. Resolution is in mean-rank units, which grow with the pool (170 ranks per unit instead of 128), so the larger resolution is partly a change of scale.

Top ten side by side:

| # | 2026-09-07 | 2026-11-13 |
|---|---|---|
| 1 | STRIDE w/ Synapse | Taichu-TimeSeries-Agent-v2 |
| 2 | EXAONE-Forecast-Agent | STRIDE w/ Synapse |
| 3 | LS-MoE | RegLab-AH-2 |
| 4 | LS-Agent | HCR |
| 5 | Falcon-Agent | TSRouter |
| 6 | CastStar | EXAONE-Forecast-Agent |
| 7 | limix_moe | limix_moe |
| 8 | TIMEHEDGE | OpenTSLM-TeeMoE |
| 9 | RegLab-AH-1 | TW3Cast |
| 10 | RacineCast-1-NC | PDM-Nano |

## 3. TIME at the new pin (board's own rule: MASE normalised by seasonal naive, geometric mean)

47 entries, 98 units, 39 datasets. The 43 first-round entries reproduce the old normalised MASE to 2e-16, and my order equals the board code's own order of all 47 entries. This rule does not depend on the pool, so adding 4 entries moved nothing for the others.

| | 2026-10-31 pin (43) | 2026-11-11 pin (47) |
|---|---|---|
| Resolution, paper / corrected | 0.0192 / 0.0238 | 0.0192 / 0.0235 |
| Median neighbour gap | 0.00209 | 0.00210 |
| Neighbours separated of 9: paper / CR2 / wild | 1 / 0 / 1 | 1 / 0 / 1 |
| Pairs of 45: paper / CR2 / wild | 20 / 11 / 18 | 20 / 11 / 18 |
| After Holm: paper / CR2 / wild | 7 / 0 / 7 | 8 / 0 / 8 |
| Leading tier: paper / CR2 / wild | 1 / 7 / 2 | 1 / 7 / 2 |

Top ten, old: TimeRSI-1.5, PDM-Nano, Timestral-209M, Timestral-123M, TimeRSI, PDM-Pico, t0_gamma, Timestral-28M, QiYao-M, ChakraTS. New: same, with Simons-297M inserted at place 8 (Timestral-28M 9, QiYao-M 10, ChakraTS drops out). This matches the live-board top ten noted in BOARD_TIME.md.

Secondary rule MASE_rank (pool dependent), new pin: top ten Timestral-209M, TimeRSI-1.5, Timestral-123M, PDM-Nano, PDM-Pico, Timestral-28M, TimeRSI, Simons-297M, TimesFM-3, QiYao-M (old: same order without Simons-297M); resolution 3.74 / 4.66, median gap 0.520; adjacent separated 1 / 1 / 1; pairs 24 / 22 / 23. The history replay of TIME was not redone.

## 4. Timestral-209M and TimesFM-3

Exact name found on all three boards: "Timestral-209M" (TIME and GIFT-Eval and fev-bench; on fev-bench it is shown, while the other three Timestral sizes are hidden). TimesFM-3 appears as "TimesFM-3" (GIFT, TIME) and "timesfm-3" (fev-bench). Gaps are the lower-placed entry's score minus the higher-placed one's (positive = the higher-placed entry is better); units are mean-rank points (GIFT), log of the normalised MASE (TIME), win-rate fraction or log skill (fev-bench).

| Board, rule | Timestral-209M place | TimesFM-3 place | Pair | Gap | Paper rule: iid CI; cluster CI; p | CR2: SE, df, p | Wild p |
|---|---|---|---|---|---|---|---|
| GIFT-Eval, MASE_Rank (170) | 11 | 28 | above: PDM-Nano (10) | 0.320 | [-6.25, 6.70]; [-9.00, 9.30]; 0.93 | 4.79, 12.8, 0.948 | 0.946 |
| | | | below: LS-MoE (12) | 0.227 | [-6.79, 7.36]; [-10.73, 10.12]; 0.99 | 5.44, 12.5, 0.967 | 0.966 |
| TIME, MASE (norm.) geometric mean (47) | 3 | 13 | above: PDM-Nano (2) | 0.0021 | [-0.0091, 0.0150]; [-0.0096, 0.0172]; 0.76 | 0.0066, 4.5, 0.765 | 0.768 |
| | | | below: Timestral-123M (4) | 0.0024 | [-0.0007, 0.0057]; [-0.0025, 0.0062]; 0.31 | 0.0022, 4.8, 0.326 | 0.308 |
| TIME, MASE_rank (secondary) | 1 | 9 | below: TimeRSI-1.5 (2) | 0.041 | [-1.33, 1.41]; [-1.68, 1.56]; 0.97 | 0.81, 5.1, 0.962 | 0.955 |
| fev-bench, win rate (40) | 4 | 9 | above: timeh2o-v1 (3) | 0.0272 | [-0.0049, 0.0605]; [-0.0056, 0.0652]; 0.108 | 0.0182, 8.1, 0.173 | 0.134 |
| | | | below: PDM-Nano (5) | 0.0118 | [-0.0254, 0.0479]; [-0.0422, 0.0549]; 0.67 | 0.0255, 7.1, 0.657 | 0.650 |
| fev-bench, skill score (secondary) | 5 | 8 | above: timeh2o-v1 (4) | 0.0090 | [-0.0007, 0.0193]; [-0.0019, 0.0219]; 0.111 | 0.0059, 6.3, 0.177 | 0.127 |
| | | | below: PDM-Nano (6) | 0.0012 | [-0.0127, 0.0147]; [-0.0197, 0.0175]; 0.92 | 0.0098, 6.7, 0.904 | 0.901 |

No difference is separated by any method. (The "exact" sign-flip test named in S5 was not run; the task list for this run did not include it.)

Timestral-209M arrived on GIFT-Eval with the Timestral family on 2026-10-16. The replay shows it inside the GIFT-Eval top ten on 2026-10-23 (it swapped with PERSONA-AI-T4 when PCFM arrived); at the new pin it is 11th, just outside.

## Deviations

- Known-shift control (1000 resampling repeats) not run (not in the task list); the report was run with n_control=0, which leaves every other number unchanged because that step uses the random stream last.
- test_vernier_board.py not run (it overwrites gift_mase_rank_report.json and its 141-update replay exceeds the time limit); replaced by a regression of the same loader and report code on the 2026-09-07 snapshot, which reproduced the stored report exactly.
- Imports from downloaded folders: TIME's own leaderboard code from the first-round clone boards/time_lb (same stubs as audit_time.py); the fev library from the first-round clone boards/fev (same commit 9cf943c as the new fev clone, so no difference). Nothing was run from the new clones. The TIME leaderboard space has no new commit (d83188a).
- TIME clone is blobless with no checkout; the 392 metrics/config files of the 4 new entries were fetched into boards/pin_2026-11-13/time_results/ and the 43 old entries were taken from the first-round boards/time_output (unchanged by the four new commits). The combined folder boards/pin_2026-11-13/time_res_all/ holds symlinks.
- fev-bench: not re-analysed beyond Timestral-209M and TimesFM-3 (no new commit).
