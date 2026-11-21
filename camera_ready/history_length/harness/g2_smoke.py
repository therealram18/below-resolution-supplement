#!/usr/bin/env python3
"""
Small driver for the two G2 models, used for the local CPU smoke before run_grid.py had the
g2_models hook. It does what run_grid.py's run() does for one model: same Dataset construction
(except that to_univariate comes from G2_TO_UNIVARIATE, the hook's intended rule), same builder
contract, same metrics, same evaluate_model call, same CSV columns. Helpers (storage-name map,
config naming, CSV columns) are imported from a run_grid.py file so they cannot drift.

Usage (local, from camera_ready/gpu_round):
  envs_local/timestral/bin/python harness/g2_smoke.py --model timestral209m \
      --configs harness/configs/smoke_config.json --context 512 --out envs_local/smoke_out \
      --gift-eval-root ../../harness/gift-eval --data ../../harness/data
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_run_grid(path: Path):
    spec = importlib.util.spec_from_file_location("run_grid_for_g2_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["timesfm3", "timestral209m"])
    p.add_argument("--configs", required=True)
    p.add_argument("--context", required=True, choices=["default", "512"])
    p.add_argument("--out", required=True)
    p.add_argument("--gift-eval-root", default=str(HERE / "gift-eval"))
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--run-grid", default=str(HERE / "run_grid.py"),
                   help="run_grid.py to take the naming helpers from")
    args = p.parse_args()

    gift_root = Path(args.gift_eval_root).resolve()
    os.environ["GIFT_EVAL"] = str(Path(args.data).resolve())
    sys.path.insert(0, str(gift_root / "src"))
    sys.path.insert(0, str(HERE))

    rg = load_run_grid(Path(args.run_grid))
    import g2_models
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
                                    MeanWeightedSumQuantileLoss)
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    # Same list as run_grid.py run().
    metrics = [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(),
        MSIS(), RMSE(), NRMSE(), ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]
    properties = json.loads((gift_root / "notebooks" / "dataset_properties.json").read_text())
    configs_list = json.loads(Path(args.configs).read_text())

    out_dir = Path(args.out) / args.model / f"ctx_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"
    new_file = not csv_path.exists()
    meta = {"model": args.model, "context_mode": args.context, "run_grid_helpers_from": str(args.run_grid),
            "package_versions": rg.package_versions(g2_models.G2_PACKAGES[args.model]), "configs": []}
    had_fail = False
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(rg.CSV_COLUMNS)
        for entry in configs_list:
            dataset_name, term = entry["dataset"], entry["term"]
            t0 = time.time()
            record = {"dataset": dataset_name, "term": term}
            try:
                storage_name = rg.resolve_storage_name(dataset_name)
                probe = Dataset(name=storage_name, term=term, to_univariate=False)
                to_univariate = g2_models.G2_TO_UNIVARIATE[args.model] and probe.target_dim != 1
                dataset = Dataset(name=storage_name, term=term, to_univariate=to_univariate)
                ds_config = rg.dataset_config_name(dataset_name, term, properties)
                predictor, context_used, revision, batch_size = g2_models.G2_BUILDERS[args.model](
                    dataset, args.context
                )
                res = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(dataset.freq),
                )
                writer.writerow([
                    ds_config, args.model,
                    res["MSE[mean]"][0], res["MSE[0.5]"][0], res["MAE[0.5]"][0], res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0], res["MSIS"][0], res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0], res["ND[0.5]"][0], res["mean_weighted_sum_quantile_loss"][0],
                    properties[rg.pretty_key(dataset_name)]["domain"],
                    properties[rg.pretty_key(dataset_name)]["num_variates"],
                ])
                fh.flush()
                record.update({"ds_config": ds_config, "status": "ok", "to_univariate": to_univariate,
                               "context_length_used": context_used, "model_revision": revision,
                               "MASE[0.5]": float(res["MASE[0.5]"][0]),
                               "wall_time_seconds": time.time() - t0})
                print(f"[ok] {ds_config} ctx={args.context} MASE[0.5]={float(res['MASE[0.5]'][0]):.6f} "
                      f"wall={time.time() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                had_fail = True
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                               "wall_time_seconds": time.time() - t0})
                print(f"[FAILED] {dataset_name}/{term}: {type(exc).__name__}: {exc}", file=sys.stderr)
            meta["configs"].append(record)
            (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return 1 if had_fail else 0


if __name__ == "__main__":
    sys.exit(main())
