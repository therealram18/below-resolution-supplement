# Result 2 -- sampling-noise reference bands

This is the noise yardstick for Result 2: how much a model's score moves around just from which 12 dataset-configs happen to get picked, before we ever change context length. It comes only from the pre-registered 12 configs (not the full 97), because that is the same set the context-length comparison will use.

Snapshot commit `b61c48aa3197a1a111e405c0e6f600128a1efee3`. Bootstrap: seed 20260913, 4000 resamples of the 12 configs, 95% band from the 2.5/97.5 percentiles.

`selection/selected_configs.json` md5: `32d6a021400e97e7040e0963449761f9`.

The 12 selected configs come from 12 distinct base datasets, so resampling configs with replacement and resampling dataset-clusters with replacement are the same operation here -- no separate iid/cluster scheme is used (contrast with the 97-config universe in run_audit.py, where several configs share a base dataset).

## The 4 target models

| model | snapshot dir | n configs | mean rank (point) | rank 95% band | rank half-width | gmean scaled MASE (point) | gmean 95% band | gmean half-width |
|---|---|---|---|---|---|---|---|---|
| chronos_bolt_base | chronos_bolt_base | 12/12 | 79.58 | [61.92, 97.58] | 17.83 | 0.9676 | [0.7081, 1.3673] | 0.3296 |
| Moirai-2.0-R-small | Moirai2 | 12/12 | 62.83 | [43.75, 80.17] | 18.21 | 0.7233 | [0.6258, 0.8303] | 0.1023 |
| TimesFM-2.5 (200m) | TimesFM-2.5 | 12/12 | 61.75 | [49.58, 73.33] | 11.87 | 0.7348 | [0.6293, 0.8428] | 0.1068 |
| Toto-Open-Base-1.0 | Toto_Open_Base_1.0 | 12/12 | 69.08 | [57.42, 81.08] | 11.83 | 0.7699 | [0.6938, 0.8488] | 0.0775 |

## Reference only: leaderboard's displayed top-3 (sanity context, not part of the noise yardstick)

| model | snapshot dir | n configs | mean rank (point) | rank 95% band | rank half-width | gmean scaled MASE (point) | gmean 95% band | gmean half-width |
|---|---|---|---|---|---|---|---|---|
| STRIDE w/ Synapse | STRIDE w/ Synapse | 12/12 | 10.33 | [5.92, 15.92] | 5.00 | 0.6261 | [0.5373, 0.7295] | 0.0961 |
| EXAONE-Forecast-Agent | EXAONE-Forecast-Agent | 12/12 | 25.92 | [16.33, 37.00] | 10.33 | 0.6353 | [0.5477, 0.7394] | 0.0959 |
| LS-MoE | LS-MoE | 12/12 | 21.92 | [16.58, 27.83] | 5.62 | 0.6774 | [0.5829, 0.7811] | 0.0991 |
