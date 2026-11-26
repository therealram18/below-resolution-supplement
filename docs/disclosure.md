# Disclosure checklist for a leaderboard entry

Two honest evaluations of the same pretrained model can differ in five settings. None of the five can be
recovered from the files that forecasting leaderboards publish today. This checklist is Table 10 of the paper
as a form: an entry fills it in, and a board stores it beside the entry's results.

| Field | What the entry reports |
|---|---|
| History length | The number of past steps the model received per series, not the argument it was passed. Report the median across evaluation windows and the minimum. |
| Aggregation rule | The statistic that turns per-unit scores into the displayed number, including whether ranks are averaged before or after scaling. |
| Evaluation windows | The number of rolling windows per unit and their stride. |
| Normalisation | Whether inputs were scaled before the model saw them, and by which statistic. |
| Sampling | The number of sample paths drawn and the summary that is scored. |

Each field is a property of the evaluation run, not of the model.

## The form

Copy this block into a file named `disclosure.yaml` in the entry's results folder and fill in each value.

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

## Which fields a harness can fill in

The evaluation harness sets three of the five: history length, evaluation windows and sampling. A board can
therefore collect them without asking the submitter. The harness that ran the models (`experiments/history_length/` at commit `3ab82ad`) records two
of them for every run in a `run_meta.json` file. It does not record the evaluation windows, which GIFT-Eval's
own data loader sets.

| Field | Key in `run_meta.json` | In `paper/results/runs/history_length/toto__A__seed42/ctx_default/per_config/ett1__H__medium.run_meta.json` |
|---|---|---|
| History length received | `context_length_used` | 4096 |
| History length available | `maximal_context_length` | 15020 |
| Sample paths | `num_samples` | 256 |

Why history length matters: on 70 configurations, 512 steps of history in place of the documented length raised
the error of three of four open models, two of them by 15%. This command prints the result:

```bash
uv run python paper/code/history_length/effects.py paper/results/runs/history_length paper/results/runs/eligible_configurations.json
```
