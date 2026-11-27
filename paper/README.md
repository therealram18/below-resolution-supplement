# The history-length experiment

`code/history_length/effects.py` reads the model runs in `results/runs/` and gives the paper's table:

```bash
uv run python paper/code/history_length/effects.py paper/results/runs/history_length paper/results/runs/eligible_configurations.json
```

```text
MASE at 512 steps of history over MASE at the documented length, 70 eligible configurations
variant              units sources  effect  95% interval              p     Holm p
chronos_bolt_base       70      14   +2.7%   -0.9% to +6.2%      0.1146     0.1146
moirai2_small           70      14   +6.2%   +1.6% to +10.4%   0.005127    0.01025
timesfm25               70      14  +15.3%   +4.1% to +32.6%  0.0002441  0.0009766
toto__A__seed42         70      14  +15.3%   +4.0% to +27.1%  0.0002441  0.0009766
```

A configuration is eligible when its median series length is at least 1,024 steps. For each model, d = log(MASE at 512 steps / MASE at the documented length) on each configuration. The effect is exp(mean d) - 1, and the p-value is exact over all sign flips of the sources' totals of d.

## Disclosure checklist

Two honest evaluations of the same pretrained model can differ in five settings that a leaderboard's files do not record. An entry fills in this form (Table 10 of the paper) as `disclosure.yaml` beside its results:

```yaml
history_length:
  argument:             # the length the model was asked to use, in steps
  median_received:      # steps the model received per series, median across evaluation windows
  minimum_received:     # the smallest number of steps it received
aggregation_rule:       # for example "MASE per unit; the board ranks before it averages"
evaluation_windows:
  per_unit:             # rolling windows per unit
  stride:               # steps between window starts
normalisation:          # for example "none before the model" or "inputs divided by the mean of the history"
sampling:
  paths:                # sample paths drawn; 0 for a model that gives quantiles directly
  scored:               # for example "the median path for MASE, nine quantiles for CRPS"
```

The evaluation harness sets three of the five fields, so a board can collect them without asking the submitter. Each run's `run_meta.json` records the history length received (`context_length_used`), the history available (`maximal_context_length`) and the sample paths (`num_samples`).
