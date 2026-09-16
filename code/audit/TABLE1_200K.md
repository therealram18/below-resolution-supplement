# Table 1 refresh at B=200,000 (snapshot commit `b61c48aa3197a1a111e405c0e6f600128a1efee3`)

Seed 20260907 for computation 1 (rank/gmean/win-rate); seed 20260907 for computation 2 (CRPS_Rank, matches critic_checks.py's SEED_MAIN -- same value, different name upstream). Deterministic: run twice and md5 table1_200k.json to confirm.

## 1. The nine adjacent gaps, recomputed at B=200000 (win rate at B=200000)

| Pair | Δ MASE_Rank | independent 95% | clustered 95% | Δ gmean | win rate (95%) |
|---|---|---|---|---|---|
| 1-2 | 4.25 | [-0.01, 8.51] | [-1.42, 10.40] | -0.0155 | 0.62 [0.53, 0.71] |
| 2-3 | 0.44 | [-4.55, 5.40] | [-6.20, 6.63] | 0.0401 | 0.51 [0.40, 0.61] |
| 3-4 | 1.94 | [-0.38, 4.40] | [-0.09, 3.82] | 0.0048 | 0.63 [0.54, 0.72] |
| 4-5 | 0.91 | [-3.72, 5.57] | [-3.75, 4.90] | 0.0110 | 0.51 [0.40, 0.61] |
| 5-6 | 0.07 | [-4.94, 5.16] | [-4.22, 4.49] | -0.0106 | 0.52 [0.41, 0.62] |
| 6-7 | 0.01 | [-3.30, 3.24] | [-2.49, 3.01] | 0.0042 | 0.53 [0.42, 0.63] |
| 7-8 | 1.42 | [-2.75, 5.59] | [-3.80, 6.17] | -0.0097 | 0.49 [0.39, 0.60] |
| 8-9 | 0.65 | [-3.88, 5.32] | [-5.46, 7.48] | 0.0149 | 0.51 [0.40, 0.61] |
| 9-10 | 1.33 | [-0.05, 2.70] | [0.13, 2.41] | 0.0073 | 0.58 [0.47, 0.67] |

## Headline counts, unchanged at 200k?

- MASE_Rank both schemes: 0 of 9 adjacent (paper: 0), 12 of 45 all pairs (paper: 12).
- gmean both schemes: 2 of 9 adjacent (paper: 2), 23 of 45 all pairs (paper: 23).
- **Unchanged at 200k: True**

## Endpoint sanity checks vs. the critic's B=200,000 run

- Pair 1-2 independent: ours [-0.01, 8.51], critic [0.0, 8.49]
- Pair 9-10 independent: ours [-0.05, 2.7], critic [-0.05, 2.71]
- Pair 9-10 clustered: ours [0.13, 2.41], critic [0.13, 2.41]
- Sanity-check only, not asserted exact: the critic's run used its own seed, and MASE_Rank moves in steps of 1/97 =~ 0.0103, so a near-zero bound can land a step either side between runs (paper text: the pair-1-2 independent lower bound 'sits at zero' and the pair-9-10 clustered lower bound 'holds near 0.13 across three seeds').

## Ready-to-paste LaTeX table body

```latex
1-2  & 4.25 & [$-$0.01, 8.51] & [$-$1.42, 10.40] & $-$0.0155 & 0.62 [0.53, 0.71] \\
2-3  & 0.44 & [$-$4.55, 5.40] & [$-$6.20, \phantom{1}6.63] & \phantom{$-$}0.0401 & 0.51 [0.40, 0.61] \\
3-4  & 1.94 & [$-$0.38, 4.40] & [$-$0.09, \phantom{1}3.82] & \phantom{$-$}0.0048 & 0.63 [0.54, 0.72] \\
4-5  & 0.91 & [$-$3.72, 5.57] & [$-$3.75, \phantom{1}4.90] & \phantom{$-$}0.0110 & 0.51 [0.40, 0.61] \\
5-6  & 0.07 & [$-$4.94, 5.16] & [$-$4.22, \phantom{1}4.49] & $-$0.0106 & 0.52 [0.41, 0.62] \\
6-7  & 0.01 & [$-$3.30, 3.24] & [$-$2.49, \phantom{1}3.01] & \phantom{$-$}0.0042 & 0.53 [0.42, 0.63] \\
7-8  & 1.42 & [$-$2.75, 5.59] & [$-$3.80, \phantom{1}6.17] & $-$0.0097 & 0.49 [0.39, 0.60] \\
8-9  & 0.65 & [$-$3.88, 5.32] & [$-$5.46, \phantom{1}7.48] & \phantom{$-$}0.0149 & 0.51 [0.40, 0.61] \\
9-10 & 1.33 & [$-$0.05, 2.70] & [\phantom{$-$}0.13, \phantom{1}2.41] & \phantom{$-$}0.0073 & 0.58 [0.47, 0.67] \\
```

## 2. CRPS_Rank Holm correction at B=100,000

- Top-10 by CRPS_Rank: ['STRIDE w/ Synapse', 'EXAONE-Forecast-Agent', 'LS-MoE', 'CastStar', 'LS-Agent', 'TIMEHEDGE', 'Falcon-Agent', 'RegLab-AH-1', 'limix_moe', 'RacineCast-1-NC']
- Raw both-schemes survivors (95% CI excludes zero under iid AND cluster): 16/45 (matches critic_checks.json's cached figure: True)
- Raw significant (p_both <= 0.05, uncorrected): 16/45
- **Holm-corrected survivors: 0/45** (critic's external claim: 0/45, matches: True)

