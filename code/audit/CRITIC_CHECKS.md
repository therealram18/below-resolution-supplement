# Critic claim checks

Snapshot commit `b61c48aa3197a1a111e405c0e6f600128a1efee3`. All bootstraps use fixed seeds (main 20260907; B=200000 stability check [20260907, 20260908, 20260909]; Result-2 paired 20260914), so this script is deterministic: run twice and md5 critic_checks.json to confirm.

## Claim 1: CRPS_Rank audit

- Same top-10 SET as MASE_Rank: **True**. CRPS_Rank top-10: ['STRIDE w/ Synapse', 'EXAONE-Forecast-Agent', 'LS-MoE', 'CastStar', 'LS-Agent', 'TIMEHEDGE', 'Falcon-Agent', 'RegLab-AH-1', 'limix_moe', 'RacineCast-1-NC']
- CastStar position under CRPS_Rank: 4 (critic claims 4th: True)
- TIMEHEDGE position under CRPS_Rank: 6 (critic claims 6th: True)
- Both-schemes adjacent survivors at B=4000: 0/9; at B=100000: 0/9
- Both-schemes all-45 survivors at B=4000: 16/45; at B=100000: 16/45
- LS-MoE vs LS-Agent (B=100000): point_gap=2.2474, iid CI=[0.22680412371134295, 4.319587628865978], cluster CI=[0.43434343434343603, 4.200000000000003], survives both: True
- STRIDE vs EXAONE (CRPS_Rank, B=100000): point_gap=5.8660, iid CI=[1.144329896907216, 10.835051546391751] (survives=True), cluster CI=[-1.2735849056603783, 14.216981132075471] (survives=False)

## Claim 2: B=200000 stability

- Paper (B=4000): pair 1-2 iid lower bound 0.04, pair 9-10 cluster lower bound 0.14
- seed 20260907: pair 1-2 iid CI=[-0.010309278350515427, 8.50515463917526] (survives=False); pair 9-10 cluster CI=[0.1264367816091969, 2.4141414141414153] (survives=True)
- seed 20260908: pair 1-2 iid CI=[0.0, 8.50515463917526] (survives=False); pair 9-10 cluster CI=[0.13888888888888573, 2.4204545454545467] (survives=True)
- seed 20260909: pair 1-2 iid CI=[-0.020618556701030855, 8.49484536082474] (survives=False); pair 9-10 cluster CI=[0.13043478260869534, 2.410256410256409] (survives=True)
- pair 1-2 iid survives at all 3 seeds: False
- pair 9-10 cluster survives at all 3 seeds: True

## Claim 3: Holm correction

- (a) MASE_Rank, 45 pairs: raw 12/45 -> Holm 2/45 (critic: 12/45 -> 2/45)
- (b) gmean, 45 pairs: raw 23/45 -> Holm 11/45 (critic: 23/45 -> 10/45)
- (c) Result-2, 8 movements: raw 5/8 -> Holm 4/8 (critic: 5/8 -> 4/8)
  - Moirai gmean p-value: 0.01744 (critic: approx 0.016); Holm-significant: False

## Claim 4: Leave-one-out, Moirai gmean (primary paired interval)

- Full-12 CI: [0.0059896737906618425, 0.10701415659555336] (excludes zero: True)
- Drops that cross zero: 3/12
  - drop bizitobs_service: CI=[-0.002379271606501532, 0.08807232138381871], excludes zero: False
  - drop ett2: CI=[-0.0020731714604045326, 0.09121984654803117], excludes zero: False
  - drop loop_seattle: CI=[-0.002009213617531813, 0.0934847665714501], excludes zero: False
- All 3 highlighted drops cross zero: True

## Claim 5: Leakage push-down relevance

- Top-12 by raw MASE_Rank all unflagged: **True**
  - [('STRIDE w/ Synapse', 'No'), ('EXAONE-Forecast-Agent', 'No'), ('LS-MoE', 'No'), ('LS-Agent', 'No'), ('Falcon-Agent', 'No'), ('CastStar', 'No'), ('limix_moe', 'No'), ('TIMEHEDGE', 'No'), ('RegLab-AH-1', 'No'), ('RacineCast-1-NC', 'No'), ('RacineCast-1', 'No'), ('FastCastrage', 'No')]

## Claim 6: gmean first-place gap (EXAONE vs STRIDE)

- point_gap=-0.0155, iid CI=[-0.04164509930897409, 0.009535808106711507] (includes zero: True), cluster CI=[-0.07024699892146669, 0.030684770703729103] (includes zero: True)

## Claim 7: Toto like-for-like baseline

- (a) Original mismatched gap (insertion-rank ours vs within-pool board): 13.00
  - Like-for-like (both insertion ranks into pool-minus-Toto): board avg 69.08, our avg 81.33, gap 12.25 (critic: 12.0)
- (b) >5% reproduction-error configs: ['bizitobs_application/10S/medium', 'bizitobs_l2c/H/long', 'jena_weather/10T/medium', 'loop_seattle/5T/long']
  - matches critic's list exactly: True
  - num_variates per config: {'bizitobs_service/10S/short': 2, 'm4_daily/D/short': 1, 'm_dense/H/short': 1, 'solar/10T/short': 1, 'us_births/W/short': 1, 'bizitobs_application/10S/medium': 2, 'ett1/H/medium': 7, 'jena_weather/10T/medium': 21, 'bizitobs_l2c/H/long': 7, 'electricity/H/long': 1, 'ett2/15T/long': 7, 'loop_seattle/5T/long': 1}
  - multivariate configs (num_variates>1): ['bizitobs_application/10S/medium', 'bizitobs_l2c/H/long', 'bizitobs_service/10S/short', 'ett1/H/medium', 'ett2/15T/long', 'jena_weather/10T/medium']
  - >5%-error set equals multivariate set: False
- (c) Movement using board's published per-config values as baseline: 29.58 (paper's movement vs our own default-context baseline: 16.58; critic claims approx 28.58)

## Claim 8: Falcon-X swing under rank method 'average'

- method='first' (paper): {'model': 'Falcon-X', 'rank_MASE_Rank': 54, 'rank_gmean': 33, 'abs_diff': 21}
- method='average': {'model': 'Falcon-X', 'rank_MASE_Rank_avg': 54, 'rank_gmean': 33, 'abs_diff': 21}
- 21 becomes 20 under 'average': **False**
- configs with at least one MASE tie: 96/97

