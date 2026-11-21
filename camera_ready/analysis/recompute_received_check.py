"""N12 cross-check: recompute the 48 default-setting received lengths from the on-disk datasets, WITHOUT writing
effective_context.csv (imports helper functions of effective_context/compute_effective_context.py, does not call main()).
Writes received_runs.json (records: model, configuration, requested_length, received_length, source_file + extras)."""
import sys, json, csv, os
sys.dont_write_bytecode = True
ROOT = "fmts"
sys.path.insert(0, ROOT + "/effective_context")
import compute_effective_context as c
import numpy as np
from gift_eval.data import Dataset
CSV = ROOT + "/effective_context/effective_context.csv"
saved = {(r["model"], r["config"]): r for r in csv.DictReader(open(CSV)) if r["setting"] == "default"}
sel = json.loads(c.SELECTED_CONFIGS_PATH.read_text())["selected"]
recs = []; mism = []
for e in sel:
    ds = Dataset(name=c.run_grid.resolve_storage_name(f"{e['base_dataset']}/{e['frequency']}"), term=e["term"], to_univariate=False)
    P, W, mn = ds.prediction_length, ds.windows, ds._min_series_length
    avail = c.available_history(c.series_lengths(ds), P, W).ravel()
    for m in c.MODELS:
        nominal, eff = c.effective_for_model(m, "default", avail, ds.target_dim, mn, W, P)
        s = saved[(m, e["config"])]
        if float(s["eff_median"]) != float(np.median(eff)) or int(s["nominal"]) != nominal: mism.append((m, e["config"]))
        recs.append(dict(model=m, configuration=e["config"], requested_length=int(nominal), received_length=float(np.median(eff)),
                         source_file="effective_context/effective_context.csv (column eff_median; setting=default)",
                         received_min=int(eff.min()), received_max=int(eff.max()),
                         prediction_length=int(P), windows=int(W), shortest_series=int(mn),
                         toto_cap_logic="min(4096, shortest-(W+1)*P) = %d" % c.toto_context_length(4096, mn, W, P) if m == "toto" else None))
json.dump(recs, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "received_runs.json"), "w"), indent=1)
print(len(recs), "records; mismatches vs csv:", mism)
print([ (r["model"], r["received_length"]) for r in recs if r["configuration"] == "us_births/W/short"])
print([r for r in recs if r["configuration"]=="us_births/W/short" and r["model"]=="toto"])
