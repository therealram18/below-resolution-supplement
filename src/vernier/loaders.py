"""How each leaderboard's own files become a Board.

Every loader repeats the arithmetic of the board's own code, so the order Vernier lists is the order the board
displays. A loader takes a tree of the board's files (sources.Folder or sources.Commit), the metric and the
definition of a source, and returns a Board.
"""
import ast
import csv
import io
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from .board import CRPS, MASE, Board, make_board
from .sources import BoardError


def _rows(content) -> list:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def _number(text) -> float:
    return float(text) if text and text.strip() else np.nan


def _one(parsed, default=None):
    """Return the parsed content of a file that a tree holds at most once."""
    return next(iter(parsed.values()), default)


# GIFT-Eval: results/<entry>/all_results.csv and config.json. app.py sorts by declared leakage, then mean rank.

def read_gift_results(content) -> dict:
    """Return {entry: {unit: {metric: score}}} of an all_results.csv file."""
    found = {}
    for row in _rows(content):
        found.setdefault(row["model"], {})[row["dataset"]] = {"MASE": _number(row.get(MASE)),
                                                              "CRPS": _number(row.get(CRPS))}
    return found


def read_gift_config(content) -> dict:
    """Return {entry: listing band} of a config.json: entries that declare no test-data leakage come first."""
    try:
        config = json.loads(content)
        return {config["model"]: {"No": 0, "Yes": 1}.get(config.get("testdata_leakage"), 2)}
    except (json.JSONDecodeError, KeyError, TypeError):  # GIFT-Eval's history holds a config.json that is not valid
        return {}  # JSON; an entry whose config names no model is not listed


def gift_eval(tree, metric="MASE", sources="source") -> Board:
    metric = {MASE: "MASE", CRPS: "CRPS"}.get(metric, metric)
    result_files, config_files = tree.files("*/all_results.csv"), tree.files("*/config.json")
    results, own_results, band, own_config = {}, {}, {}, {}
    for path, found in tree.parsed("*/all_results.csv", read_gift_results).items():
        for entry, scores in found.items():
            results[entry] = {unit: by_metric[metric] for unit, by_metric in scores.items()}
            own_results[entry] = result_files[path]
    for path, found in tree.parsed("*/config.json", read_gift_config).items():
        band.update(found)
        own_config.update(dict.fromkeys(found, config_files[path]))
    entries = sorted(results)  # the board ranks on a table in this order
    return make_board({entry: results[entry] for entry in entries}, lambda unit: unit.split("/")[0],
                      {entry: band.get(entry) for entry in entries},  # no config.json: the board does not list it
                      {entry: (own_results[entry], own_config.get(entry)) for entry in entries})


# fev-bench and Chronos Benchmark II: one results/<name>.csv per entry in the fev repository.

FEV_BASELINE, FEV_LEAKAGE = "Seasonal Naive", "Chronos-Bolt"  # as scripts/generate_fev_bench_figures.py sets them


def read_fev_results(content) -> dict:
    """Return {entry: {task: ({metric: error}, trained on the task's data)}} of a results file."""
    found = {}
    for row in _rows(content):
        errors = {metric: _number(row.get(metric)) for metric in ("test_error", "SQL", "MASE", "WQL", "WAPE")}
        task = row.get("task_name") or row["dataset_config"]
        found.setdefault(row["model_name"], {})[task] = (errors, row.get("trained_on_this_dataset") == "True")
    return found


def read_fev_tasks(content) -> list:
    return [task.get("task_name") or task["dataset_config"] for task in yaml.safe_load(content)["tasks"]]


def read_fev_hidden(content) -> set:
    """Return the names of the result files that models.yaml hides from the board."""
    return {name for name, about in yaml.safe_load(content).items() if isinstance(about, dict) and about.get("hidden")}


def _fev_tables(tree, column, tasks_file):
    """Return the entries, the tasks, errors and leakage flags (entries x tasks) and each entry's file."""
    files, hidden = tree.files("results/*.csv"), _one(tree.parsed("models.yaml", read_fev_hidden), set())
    rows = {entry: (by_task, files[path])
            for path, found in tree.parsed("results/*.csv", read_fev_results).items()
            if Path(path).stem not in hidden for entry, by_task in found.items()}
    tasks = _one(tree.parsed(tasks_file, read_fev_tasks))
    if tasks is None and tasks_file != "tasks.yaml":
        raise BoardError(f"the board's files hold no {tasks_file}")
    tasks = sorted(tasks or {task for by_task, _ in rows.values() for task in by_task})
    entries, missing = sorted(rows), ({column: np.nan}, False)
    cells = [[rows[entry][0].get(task, missing) for task in tasks] for entry in entries]
    error = np.array([[errors[column] for errors, _ in row] for row in cells], float).reshape(len(entries), len(tasks))
    leak = np.array([[flag for _, flag in row] for row in cells], bool).reshape(len(entries), len(tasks))
    return entries, tasks, error, leak, {entry: rows[entry][1] for entry in entries}


def fev_bench(tree, metric="SQL", sources="source", tasks_file="tasks.yaml") -> Board:
    """Return fev-bench's table of relative error, as fev.analysis.leaderboard builds it.

    Each error is divided by the baseline's. Where an entry was trained on the task's data, the leakage model's
    value replaces its own. Values are clipped to 0.01 to 100, and a missing value becomes the baseline's, 1.
    """
    entries, tasks, error, leak, files = _fev_tables(tree, "test_error" if metric == "SQL" else metric, tasks_file)
    absent = [name for name in (FEV_BASELINE, FEV_LEAKAGE) if name not in entries]
    if absent or np.isnan(error[entries.index(FEV_BASELINE)]).any():
        raise BoardError(f"fev-bench needs complete results of {FEV_BASELINE} and the entry {FEV_LEAKAGE}")
    relative = error / error[entries.index(FEV_BASELINE)]
    relative = np.where(leak, relative[entries.index(FEV_LEAKAGE)], relative)
    relative = np.nan_to_num(np.clip(relative, 0.01, 100), nan=1.0)
    family = [task.rsplit("_", 1)[0] for task in tasks]  # a task's name without its last part
    return Board(entries, tasks, family, relative, [0] * len(entries), files)


def chronos_source(task: str) -> str:
    """Return the collection a Chronos Benchmark II task was cut from; one collection at several frequencies."""
    for prefix, source in (("monash_m1_", "m1"), ("monash_m3_", "m3"), ("monash_tourism_", "tourism"), ("m4_", "m4")):
        if task.startswith(prefix):
            return source
    return "ETT" if task in ("ETTh", "ETTm") else task


def chronos_bench_ii(tree, metric="MASE", sources="source") -> Board:
    """Return Chronos Benchmark II's table of errors, with the entries that have a result on every task."""
    entries, tasks, error, _, files = _fev_tables(tree, "test_error" if metric == "MASE" else metric, "tasks.yaml")
    complete = [i for i, row in enumerate(error) if not np.isnan(row).any()]
    kept = [entries[i] for i in complete]
    return Board(kept, tasks, [chronos_source(task) for task in tasks], error[complete], [0] * len(kept),
                 {entry: files[entry] for entry in kept})


# TIME: results/<entry>/<dataset>/<frequency>/<horizon>/metrics.npz, arrays over series, windows and variates.

def read_time_metrics(content) -> dict:
    """Return {metric: mean over all series, windows and variates} of a metrics.npz file, as the board reduces it."""
    arrays = np.load(io.BytesIO(content))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an array of only NaN has the mean NaN
        return {metric: float(np.nanmean(arrays[metric])) for metric in ("MASE", "CRPS") if metric in arrays.files}


def time_board(tree, metric="MASE", sources="source") -> Board:
    pattern = "*/*/*/*/metrics.npz"
    ids, results, files = tree.files(pattern), {}, {}
    for path, scores in tree.parsed(pattern, read_time_metrics).items():
        entry, dataset, frequency, horizon, _ = path.split("/")
        if horizon in ("short", "medium", "long"):
            results.setdefault(entry, {})[f"{dataset}/{frequency}/{horizon}"] = scores.get(metric, np.nan)
            files.setdefault(entry, []).append(ids[path])
    scaled = results.get("seasonal_naive")
    if scaled:  # the board skips a unit on which its baseline has no result
        results = {entry: {unit: score for unit, score in scores.items() if unit in scaled}
                   for entry, scores in results.items()}
    entries = sorted(results)
    return make_board({entry: results[entry] for entry in entries}, lambda unit: unit.split("/")[0],
                      files={entry: tuple(sorted(files[entry])) for entry in entries})


# BOOM: boom/results/<entry>/all_results.csv in the toto repository, one row per series and horizon.

# boom/utils/leaderboard.py breaks tied ranks by the order in which its machine lists the result folders, and
# the files do not record that order. This is the order that reproduces the board's published table of
# 2026-10-26 in all six columns. Later entries follow in alphabetical order. The baseline is always last.
BOOM_LISTING = ("moirai_1.1_base", "autoarima", "toto_2.0_4m", "moirai_1.1_large", "timer", "toto", "autoets",
                "moirai_1.1_small", "toto_2.0_2.5b", "moirai_2_small", "chronos_2", "time_moe_50M", "autotheta",
                "toto_2.0_313m", "timesfm_2_5_200m", "chronos_bolt_small", "toto_2.0_1b", "visionts",
                "chronos_bolt_base", "time_moe_200M", "toto_2.0_22m", "timesfm_2_0_500m")
BOOM_BASELINE = "seasonalnaive"


def read_boom_results(content) -> dict:
    """Return {unit: {metric: score}} of one entry's all_results.csv."""
    return {row["dataset"]: {"MASE": _number(row[MASE]), "CRPS": _number(row[CRPS])} for row in _rows(content)}


def read_boom_low_variance(content) -> set:
    """Return the series that the board's own code lists as low-variance and ranks apart."""
    return next(set(ast.literal_eval(node.value)) for node in ast.parse(content.decode()).body
                if isinstance(node, ast.Assign) and node.targets[0].id == "LOW_VARIANCE_DATASETS")


def boom_source(properties, definition) -> Callable:
    """Return the function from a unit to its source: the series, or its domain with or without its type."""
    def source(unit):
        series = unit.split("/")[0]
        about = properties.get(series)
        if definition == "source" or about is None:
            return series if definition == "source" else "no properties"
        domains = "|".join(sorted(about["domain"]))
        if definition == "domain" and len(about["domain"]) == 1:
            return domains
        return f"{domains}#{about['type']}"  # a series with several domains keeps its full combination
    return source


def boom(tree, metric="CRPS", sources="source") -> Board:
    """Return BOOM's table of scaled scores on the series the board ranks together, as its leaderboard.py does.

    The board ranks the low-variance series and those where the baseline's MASE is zero apart. On the others
    it replaces an infinite or empty score by the entry's mean, and divides by the baseline's score.
    """
    files = tree.files("results/*/all_results.csv")
    scores = {path.split("/")[1]: by_unit
              for path, by_unit in tree.parsed("results/*/all_results.csv", read_boom_results).items()}
    apart, properties = (_one(tree.parsed(path, read)) for path, read in (
        ("utils/leaderboard.py", read_boom_low_variance), ("boom_properties.json", json.loads)))
    if BOOM_BASELINE not in scores or apart is None or properties is None:
        raise BoardError(f"BOOM needs results/{BOOM_BASELINE}/all_results.csv, utils/leaderboard.py, which lists "
                         "the low-variance series, and boom_properties.json")
    later = sorted(set(scores) - set(BOOM_LISTING) - {BOOM_BASELINE})
    entries = [entry for entry in BOOM_LISTING if entry in scores] + later + [BOOM_BASELINE]
    apart = apart | {unit.split("/")[0] for unit, score in scores[BOOM_BASELINE].items() if score["MASE"] == 0}
    units = sorted(unit for unit in scores[BOOM_BASELINE] if unit.split("/")[0] not in apart)
    table = np.array([[scores[entry].get(unit, {}).get(metric, np.nan) for unit in units] for entry in entries])
    held = np.array([[unit in scores[entry] for unit in units] for entry in entries])
    table[np.isinf(table)] = np.nan
    with np.errstate(invalid="ignore"):  # an entry with no finite score keeps NaN and is not listed
        mean = np.nansum(table, axis=1, keepdims=True) / (~np.isnan(table)).sum(1, keepdims=True)
    table = np.where(np.isnan(table) & held, mean, table)  # an entry with no row on a unit is not listed either
    return Board(entries, units, [boom_source(properties, sources)(unit) for unit in units], table / table[-1],
                 [0] * len(entries), {entry: files[f"results/{entry}/all_results.csv"] for entry in entries})
