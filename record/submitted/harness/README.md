# GIFT-Eval inference harness

Evaluates 4 open time-series foundation models on GIFT-Eval dataset-configs
using GIFT-Eval's own `Dataset` class and gluonts `evaluate_model()`, so
results land in the same `all_results.csv` shape as the leaderboard at
huggingface.co/spaces/Salesforce/GIFT-Eval.

**Status: the grid has run.** All 96 cells (4 models x 2 context lengths x 12
dataset-configs) completed across three passes on one GPU; see
`results/RUNS.md` for the full story, including two failure passes that were
diagnosed and rerun. See "What's been verified" below for what was checked
before the real runs.

## Layout

```
harness/
  gift-eval/            GIFT-Eval repo, cloned from SalesforceAIResearch/gift-eval
  vendor/toto/           DataDog/toto repo, cloned (needed for Toto-Open-Base-1.0)
  data/                   downloaded GIFT-Eval train/test dataset (~1.5 GB)
  envs/                   one venv per purpose (see below)
    data/                 data download + Dataset-class verification
    chronos_bolt_base/
    moirai2_small/
    timesfm25/
    toto/
  requirements/           pinned top-level deps per env, with source comments
  configs/
    selected_configs.json example config list (3 configs used for data verification)
  run_grid.py             the evaluation script
  verify_data.py          the data-loading verification script (step 4)
  results/                run_grid.py output lands here by default
```

## Why 5 separate venvs

Tried to keep this to one shared env; gave up on purpose once the conflicts
turned out to be real and already documented upstream:

- `uni2ts` (Moirai 2.0) pins `gluonts~=0.14.3`; gift-eval needs
  `gluonts~=0.15.1`. uni2ts's own example notebook documents installing
  uni2ts first, then reinstalling `gluonts==0.15.1` on top. Verified this
  actually resolves and imports cleanly.
- DataDog/toto's `requirements.txt` pins `gluonts[torch]==0.16.2`, same shape
  of conflict. Same fix (install toto's requirements, then gift-eval on top,
  which pulls gluonts back to 0.15.1).
- `timesfm` documents needing a python 3.11.x environment specifically.

All 5 venvs use python 3.11 (via `uv venv --python 3.11`) so this constraint
is satisfied everywhere, not just for the timesfm env.

## Data

GIFT-Eval's evaluation dataset (Salesforce/GiftEval on Hugging Face) is
public and small: **1.5 GB total** on disk (the HF API's file listing reports
~1.59 GB across 58 arrow files, 28 dataset directories) -- well under the
10 GB "only download what's needed" threshold, so the full dataset was
downloaded, no subsetting needed. No Hugging Face token was needed or used
(public dataset; downloaded with an unauthenticated `hf download` call).

```bash
envs/data/bin/hf download Salesforce/GiftEval --repo-type=dataset --local-dir data
```

`run_grid.py` and `verify_data.py` both set `GIFT_EVAL=harness/data` via
`os.environ` directly (not a `.env` file) before touching
`gift_eval.data.Dataset`, so there's nothing to configure by hand.

## Env setup

```bash
# data env (download + verification only, no torch)
uv venv envs/data --python 3.11
uv pip install --python envs/data/bin/python -e ./gift-eval "huggingface_hub[cli]"

# chronos_bolt_base
uv venv envs/chronos_bolt_base --python 3.11
uv pip install --python envs/chronos_bolt_base/bin/python \
    -e ./gift-eval "chronos-forecasting==2.3.2" "torch>=2.1"

# moirai2_small (order matters: uni2ts first, gift-eval second)
uv venv envs/moirai2_small --python 3.11
uv pip install --python envs/moirai2_small/bin/python "uni2ts==2.0.0" "torch>=2.1,<2.5"
uv pip install --python envs/moirai2_small/bin/python -e ./gift-eval

# timesfm25 (PyPI timesfm==3.0.2 already ships the 2.5-torch backend,
# no need to git clone google-research/timesfm as gift-eval's notebook says)
uv venv envs/timesfm25 --python 3.11
uv pip install --python envs/timesfm25/bin/python -e ./gift-eval "timesfm[torch]==3.0.2"

# toto (order matters: toto's requirements.txt first, gift-eval second,
# then the vendored toto repo itself with --no-deps, then re-pin setuptools)
git clone https://github.com/DataDog/toto vendor/toto
uv venv envs/toto --python 3.11
uv pip install --python envs/toto/bin/python -r vendor/toto/requirements.txt
uv pip install --python envs/toto/bin/python -e ./gift-eval
uv pip install --python envs/toto/bin/python -e ./vendor/toto --no-deps
uv pip install --python envs/toto/bin/python "setuptools<81" --reinstall
```

Each `requirements/<env>.txt` has the same commands plus the reasoning inline.

### The pkg_resources / setuptools footgun

`setuptools>=81` removed the `pkg_resources` module. `gluonts.torch` imports
`lightning.pytorch`, and `lightning.fabric` still does
`__import__("pkg_resources").declare_namespace(...)` at import time. Only bit
this in the `toto` env (its requirements.txt doesn't pin setuptools and `uv`
installed 84.0.0 there); `chronos_bolt_base` and `timesfm25` never import
`gluonts.torch` at all (their predictor wrappers are plain classes, not
`gluonts.torch.PyTorchPredictor` subclasses) so they never hit this; `uni2ts`
happened to end up with an env where it didn't trigger either. Fixed by
pinning `setuptools<81` in the toto env. If this resurfaces on the Linux CUDA
box with a different resolution order, same fix.

## Data-loading verification (done, no model)

`verify_data.py` loads 3 dataset-configs via `gift_eval.data.Dataset` --
one short univariate, one medium-term multivariate, one long-term
multivariate -- and prints frequency, prediction length, target
dimensionality, window count and test-series count. No model involved.

```bash
envs/data/bin/python verify_data.py
```

Output (this run):

```
--- m4_weekly / short ---
  freq:              W-SUN
  prediction_length: 13
  target_dim:        1
  windows:           1
  test series count: 359
--- bizitobs_l2c/H / medium ---
  freq:              H
  prediction_length: 480
  target_dim:        7
  windows:           1
  test series count: 1
--- ett1/H / long ---
  freq:              H
  prediction_length: 720
  target_dim:        7
  windows:           3
  test series count: 3
```

(`test series count` for the multivariate configs is the number of
multivariate item-series, i.e. before any per-model univariate splitting --
`bizitobs_l2c/H` is 1 series with 7 variates, `ett1/H` has 3 rolling windows
of the same 7-variate series.)

## Context length per model

`--context default` and `--context 512` are implemented differently per
model because each package exposes context control differently. All of this
was checked against primary sources (checkpoint `config.json`, package
source, GIFT-Eval's own reference notebooks) -- see the source comments
directly above each `..._DEFAULT_CONTEXT` constant in `run_grid.py` for the
exact citations.

| Model | `default` | `512` | Source / caveat |
|---|---|---|---|
| Chronos-Bolt-base | 2048 | forced by truncating the input series to the last 512 points ourselves before calling `pipeline.predict()` | `context_length` is **baked into the checkpoint** (`amazon/chronos-bolt-base/config.json` -> `chronos_config.context_length: 2048`) and is not an API parameter -- `ChronosBoltPipeline.predict()`/`.embed()` always internally truncate to the rightmost `self.model_context_length` values, verified from `chronos_bolt.py` source. Feeding it fewer than 2048 points is the only lever. |
| Moirai-2.0-R-small | 4000 | 512, passed straight through | **Real discrepancy, not resolved:** GIFT-Eval's own `notebooks/moirai2.ipynb` hardcodes `context_length=4000` when constructing `Moirai2Forecast` -- that's what produced the leaderboard's published numbers, so `default` replicates it. But the checkpoint's own `Salesforce/moirai-2.0-R-small/config.json` reports `"max_seq_len": 512`, and uni2ts's own getting-started notebook uses yet a third value (`context_length=1680`) in a different example. `context_length` is a legitimate `Moirai2Forecast` constructor kwarg either way (the model apparently doesn't hard-reject values above its configured `max_seq_len`), so both 4000 and 512 are mechanically valid to pass -- but which one is "correct" relative to the model's actual training/positional-embedding limit could not be determined from public docs without running it. Flagging this for the pre-registration step rather than guessing. |
| TimesFM-2.5-200M | dynamic per batch: full available history, rounded up to a multiple of the model's patch size, capped at 15360 | 512, rounded up to a patch multiple, i.e. each series' context truncated to its last 512 points before the batch is compiled | GIFT-Eval's `notebooks/timesfm2p5.ipynb` recompiles the model per batch with `max_context = min(15360, ceil(longest_series/patch)*patch)` -- there is no single fixed "default context", it's whatever the data has. The model card / GitHub README advertise "up to 16k context, up from 2048" (16384); gift-eval's actual recipe caps at 15360, which is what we replicate. |
| Toto-Open-Base-1.0 | 4096, further capped by `min(4096, available_history)` per GIFT-Eval's own `get_maximal_context_length()` | 512, same `min(512, available_history)` capping | GIFT-Eval's `notebooks/toto.ipynb`: `DEFAULT_CONTEXT_LENGTH = 4096`. Toto's checkpoint `config.json` has no `max_seq_len`-style field and the model card advertises "variable context lengths" -- no architectural cap to report, unlike Chronos-Bolt and Moirai 2.0. |

## Running the full grid

Once the pre-registration gate clears, for each of the 4 models, in that
model's own venv:

```bash
envs/chronos_bolt_base/bin/python run_grid.py \
    --model chronos_bolt_base --configs configs/selected_configs.json \
    --context default --out results
envs/chronos_bolt_base/bin/python run_grid.py \
    --model chronos_bolt_base --configs configs/selected_configs.json \
    --context 512 --out results

envs/moirai2_small/bin/python run_grid.py \
    --model moirai2_small --configs configs/selected_configs.json \
    --context default --out results
envs/moirai2_small/bin/python run_grid.py \
    --model moirai2_small --configs configs/selected_configs.json \
    --context 512 --out results

envs/timesfm25/bin/python run_grid.py \
    --model timesfm25 --configs configs/selected_configs.json \
    --context default --out results
envs/timesfm25/bin/python run_grid.py \
    --model timesfm25 --configs configs/selected_configs.json \
    --context 512 --out results

envs/toto/bin/python run_grid.py \
    --model toto --configs configs/selected_configs.json \
    --context default --out results
envs/toto/bin/python run_grid.py \
    --model toto --configs configs/selected_configs.json \
    --context 512 --out results
```

That's 8 runs total (4 models x 2 context settings). `configs/selected_configs.json`
currently holds the 3 configs used for data-loading verification -- swap in
the pre-registered config list before running for real; the file is just a
JSON list of `{"dataset": ..., "term": ...}` objects, matching
`gift_eval.data.Dataset(name=..., term=...)` argument names directly. The
full GIFT-Eval grid is 97-98 dataset-configs (per dataset x applicable terms);
running all of them for all 4 models x 2 contexts is 8x that many
individual `evaluate_model()` calls.

Each run appends to `results/<model>/ctx_<context>/all_results.csv` (same
columns as GIFT-Eval's sample `results/naive/all_results.csv`) and writes/
updates `results/<model>/ctx_<context>/run_meta.json` after every config,
with per-config wall time, the context length actually used, the model's HF
revision (commit sha, looked up via `huggingface_hub.HfApi().model_info()` --
no token needed, it's metadata on a public repo), and installed package
versions for that env. A failed config is recorded in `run_meta.json` with
its exception and the loop continues rather than aborting the whole grid.

### Runtime

Measured: all 96 cells ran across three passes on one GPU, on 2026-09-12,
within roughly 70 minutes total (pass 1 started around 13:50 UTC, pass 3
finished before about 15:00 UTC) -- see `results/RUNS.md` for the full
three-pass story (two failure-and-rerun rounds, both disclosed) and
`results/grid_run_log.txt` / `results/logs_per_run/` for per-cell wall
times. Before the runs happened, the expected shape was: Chronos-Bolt-base
and TimesFM-2.5-200M are small (<250M params) and should
be the fastest per series; Moirai-2.0-R-small is comparably small; Toto-
Open-Base-1.0's own notebook defaults to `num_samples=256` (sampling-based,
not direct quantile heads like the other three), which will make it the
slowest of the four per series, and its `context_length=512` runs should be
meaningfully faster than its `default` (4096-capped) runs since attention
cost scales with context length. `--context 512` runs should generally be
faster than `default` for all 4 models given the above table (default context
is >=512 in every case except Moirai, where it's identical). Long-term
configs (prediction_length x15 the short-term one, e.g. `ett1/H/long` at
720 steps) will dominate wall time over short-term configs.

## What's been verified vs. not

This section is the pre-registration checkpoint: what was checked BEFORE
the real runs, as the admission gate for running any model at all. The
real runs have since happened -- see `results/RUNS.md` for what they found
(96/96 cells, two failure-and-rerun passes, all disclosed).

Verified (this session, on this Mac, no CUDA):
- All 5 venvs build and their direct dependencies resolve without conflict
  (including the two documented gluonts version fights).
- `gift_eval.data.Dataset` loads real data for 3 configs and reports correct
  freq/prediction_length/target_dim/windows (see above).
- Every `import` statement inside each model's `build_*()` function in
  `run_grid.py` succeeds in that model's venv, against the actually-installed
  package (not just syntax-checked).
- Every constructor/method signature `run_grid.py` calls
  (`Moirai2Forecast.__init__`, `Moirai2Forecast.predict`,
  `configs.ForecastConfig.__init__`, `TotoPredictor.create_for_eval`) was
  checked with `inspect.signature()` against the real installed package and
  matches what the code passes.
- `huggingface_hub.HfApi().model_info(...)` (the revision-lookup helper)
  works unauthenticated against a public model repo.
- All pure helper functions in `run_grid.py` (`pretty_key`,
  `dataset_config_name`, `to_univariate_for`, `package_versions`) were
  exercised directly with assertions.
- `run_grid.py --help` runs cleanly (argparse only touches stdlib at import
  time; all model/gift-eval imports are inside functions, so no env needs
  every package installed just to parse arguments).

At that pre-registration checkpoint, none of the following had been done
yet, per the hard constraint against running any model before the gate
cleared -- all of it has since happened for real; see `results/RUNS.md`
for the full three-pass record:
- Every model's `.from_pretrained()` / checkpoint weights was downloaded.
- `predictor.predict()` was called for all 4 models, across all 96 grid
  cells (4 models x 2 context lengths x 12 dataset-configs).
- `evaluate_model()` ran end to end for the full grid, not a stand-in --
  including the two failure-and-rerun passes `results/RUNS.md` discloses.
