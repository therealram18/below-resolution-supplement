# Vernier

Vernier computes a report from a leaderboard's own table of scores. It gives the board's resolution, which is the smallest difference between two entries that the board's data can detect (a test at the 5% level finds it eight times in ten). It lists which pairs among the top ten entries the data can separate, and the tier of entries among the first 40 places that are not separated from the leader. It also replays the board's git history and lists the top-ten pairs whose order was reversed by other entries arriving or changing while their own results stayed the same. The repository also holds the history-length analysis: how much a model's error (MASE) changes when it is given 512 steps of history instead of its documented length.

In the board's table, a unit is one dataset configuration (dataset, frequency and horizon) and its source is the dataset it was cut from. Units of one source are correlated, so every test treats them as one cluster.

## Install

Python 3.9 or later.

    pip install -r requirements.txt

## Run

Run the commands from the repository's root folder. The report on the GIFT-Eval leaderboard as pinned on 2026-09-07, with entries ordered by mean rank on MASE as the board displays them (output shortened):

    python -m vernier report snapshot/snapshot_2026-09-07/results

    mean-rank, metric MASE, ties first: 128 entries shown with all 97 units from 28 sources
    place  entry                                  mean  gap to next         95% interval       p
        1  STRIDE w/ Synapse                   14.6186       4.2474   -2.3998 to 10.8947  0.1889
        2  EXAONE-Forecast-Agent               18.8660       0.4433   -6.6844 to 7.5710   0.8952
    ...
        9  RegLab-AH-1                         24.3093       1.3299    0.0945 to 2.5653   0.0364
       10  RacineCast-1-NC                     25.6392
    Resolution, the gap a 5% test detects 8 times in 10 (median over neighbours): 6.8010
    Median gap between neighbours: 0.9072
    Neighbouring pairs separated at 5%: 1 of 9, 0 after Holm
    All pairs separated at 5%: 12 of 45, 0 after Holm
    Leading tier, not separated from the leader at 5% among the first 40 places: 3 (STRIDE w/ Synapse, EXAONE-Forecast-Agent, LS-MoE)

Each row compares a place with the next one: the gap in mean rank, its 95% interval and the p-value of a t test whose standard error is clustered by source (CR2). `--rule geometric-mean` orders entries by the geometric mean of their error divided by the error of the entry `Seasonal_Naive`. `--metric CRPS` uses the board's CRPS column instead of MASE. `--ties average` gives tied entries the average of their ranks instead of breaking ties by entry name.

The replay of the board's history (output shortened):

    python -m vernier replay <clone of the board's repository>

    mean rank, ties first: 141 updates examined, 14 reversed 20 pairs of top-ten entries whose own results did not change
    date        commit   placed higher before          placed lower before
    2024-12-05  16367d7  Moirai_small                  TFT
    2025-05-27  71fbbd0  YingLong_110m                 chronos_bolt_small
    ...
    2026-09-07  b61c48a  CastStar                      Falcon-Agent

The board's history is not in this repository. Get it with `git clone https://huggingface.co/spaces/Salesforce/GIFT-Eval <clone of the board's repository>`, the GIFT-Eval leaderboard's Hugging Face Space. The paper's numbers use the history up to commit b61c48aa, so run `git checkout b61c48aa` inside the clone first; the replay reads the history up to the commit that is checked out. An update is a commit that changed `results/` and kept the same units as the commit before it.

The effect of 512 steps of history on each model (full output):

    python -m vernier history-length runs/history_length --eligible runs/eligible_configurations.json

    MASE at 512 steps of history over MASE at the documented length, 70 eligible configurations
    variant              units sources  effect  95% interval              p     Holm p
    chronos_bolt_base       70      14   +2.7%   -0.9% to +6.2%      0.1146     0.1146
    moirai2_small           70      14   +6.2%   +1.6% to +10.4%   0.005127    0.01025
    timesfm25               70      14  +15.3%   +4.1% to +32.6%  0.0002441  0.0009766
    toto__A__seed42         70      14  +15.3%   +4.0% to +27.1%  0.0002441  0.0009766

A configuration is eligible when its median series length is at least 1,024 steps. For each model, d = log(MASE at 512 steps / MASE at the documented length) on each configuration, and the effect is exp(mean d) - 1, so +15.3% means MASE is 15.3% higher with 512 steps. The p-value comes from an exact test over all sign flips of the sources' totals of d, and the Holm p-value adjusts it over the models shown. Only models with results on every eligible configuration are shown.

## Use on another board

Write the board's table as a CSV file with the columns `entry`, `unit`, `source` and `score`, one row per entry and unit. Lower scores are better. A source groups the units that share data, such as the frequencies and horizons cut from one dataset. Only entries with a score on every unit are compared, and at least ten are needed.

    entry,unit,source,score
    Auto_Arima,bitbrains_fast_storage/5T/long,bitbrains_fast_storage,1.14
    Auto_Arima,bitbrains_fast_storage/5T/medium,bitbrains_fast_storage,1.22

    python -m vernier report scores.csv

`--metric` applies only to GIFT-Eval folders, and `--rule geometric-mean` needs an entry named `Seasonal_Naive`.

## Use from Python

    from vernier import gift_eval_order, load_gift_eval, ranks, report

    board = load_gift_eval("snapshot/snapshot_2026-09-07/results")
    values = ranks(board.scores, ties="first")
    result = report(values, board.sources, board.entries, gift_eval_order(board, values))
    print(result.resolution, result.tier)

## The paper's numbers

Each command below starts with `python -m vernier`, and each test is in `tests/test_paper_numbers.py`.

| Number in the paper | Command | Test |
|---|---|---|
| Resolution 6.80 and median gap 0.91 between neighbours (mean rank on MASE) | `report snapshot/snapshot_2026-09-07/results` | `test_mean_rank_on_mase` |
| 1 of 9 neighbouring pairs and 12 of 45 pairs separated at 5%, none after Holm | `report snapshot/snapshot_2026-09-07/results` | `test_mean_rank_on_mase` |
| Leading tier of 3 (mean rank on MASE) | `report snapshot/snapshot_2026-09-07/results` | `test_mean_rank_on_mase` |
| Leading tier of 17 (geometric mean of scaled error) | `report snapshot/snapshot_2026-09-07/results --rule geometric-mean` | `test_geometric_mean_of_scaled_error` |
| 20 reversed pairs in 14 of 141 updates | `replay <clone of the board's repository>` | `test_replay_of_the_leaderboard_history` |
| 18 reversed pairs with tied ranks averaged | `replay <clone of the board's repository> --ties average` | `test_replay_of_the_leaderboard_history` |
| History-length effects +2.7%, +6.2%, +15.3%, +15.3% on 70 configurations | `history-length runs/history_length --eligible runs/eligible_configurations.json` | `test_history_length_on_70_eligible_configurations` |

The tests also check the report on CRPS and the history-length effects on the 12 configurations of the submitted paper. Run them with:

    python -m pytest tests

The replay test needs the board's history: it runs only when the environment variable `GIFT_EVAL_CLONE` names a clone of the board's repository, and is skipped otherwise. It reads the history up to commit b61c48aa whatever is checked out.

    GIFT_EVAL_CLONE=<clone of the board's repository> python -m pytest tests

## Folders

- `vernier/`: the tool.
- `tests/`: the paper's key numbers, recomputed from the files in this repository.
- `snapshot/`: the GIFT-Eval leaderboard's results and app files as of 2026-09-07 (commit b61c48aa), unchanged; `results/` holds one folder per entry.
- `runs/`: the model runs' results: `history_length/` for the history-length analysis, `submitted/` for the submitted paper's runs on 12 configurations, and the list of eligible configurations.
- `experiments/history_length/`: the code that ran the models, with its logic unchanged; only its input paths, the options of make_jobs.py that name them (`--eligible`, `--selected`) and some comments were updated for this layout.
- `paper/`: the camera-ready source, bibliographies, style file and figures.
- `record/`: stored outputs (JSON, CSV and text files) and Markdown write-ups of the analyses the paper reports.

## What is not rerunnable here

The analyses of the other boards (fev-bench, BOOM, TIME and the GIFT-Eval sub-boards), the calibration simulations and the checks against reference software are stored as results in `record/`. The scripts that produced them are in this repository's history before this change. The model runs need a GPU, the GIFT-Eval data and one Python environment per model, whose requirements are in `experiments/history_length/requirements/`.

## Licence

Apache License 2.0, see `LICENSE`. The files in `snapshot/` are copied from the GIFT-Eval leaderboard's repository.
