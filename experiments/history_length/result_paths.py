"""Names and paths of result files, shared by run_grid.py, dispatch.py,
merge_results.py and make_jobs.py. Standard library only, so the dispatcher
can import it from any python.

One run = one (model variant, context setting, configuration). Its files:
  <out>/<variant>/ctx_<default|512>/per_config/<dataset with "/" -> "__">__<term>.csv
  <out>/<variant>/ctx_<default|512>/per_config/<dataset with "/" -> "__">__<term>.run_meta.json
A run is done when its csv exists (both files are written to a temporary
name and renamed, the meta file first).
"""
import os
from pathlib import Path

TOTO_SAMPLING_SETTINGS = ("A", "A_tf32off", "A_nocache", "A_bs16")
TOTO_SETTINGS = ("point",) + TOTO_SAMPLING_SETTINGS
SMOKE_ENV = "TOTO_SMOKE_SAMPLES"
SMOKE_SUFFIX = "__SMOKE"


def smoke_samples():
    """The TOTO_SMOKE_SAMPLES override as an int, or None when unset/empty."""
    raw = os.environ.get(SMOKE_ENV, "").strip()
    return int(raw) if raw else None


def base_variant(model: str, toto_setting: str = "point", seed: int = 42) -> str:
    """Variant folder name without the smoke suffix. Toto's sampling
    settings get toto__<setting>__seed<k>; everything else keeps its model
    name (plain `toto` is the first round's point-forecast behaviour)."""
    if model == "toto" and toto_setting != "point":
        if toto_setting not in TOTO_SAMPLING_SETTINGS:
            raise ValueError(f"unknown toto setting {toto_setting!r}")
        return f"toto__{toto_setting}__seed{int(seed)}"
    return model


def variant_name(model: str, toto_setting: str = "point", seed: int = 42) -> str:
    """Variant folder name actually written. When TOTO_SMOKE_SAMPLES is set
    in the environment, every variant gets the __SMOKE suffix, so a run made
    with the smoke override can never sit in a real run's folder."""
    name = base_variant(model, toto_setting, seed)
    if smoke_samples() is not None:
        name += SMOKE_SUFFIX
    return name


def config_stem(dataset: str, term: str) -> str:
    """us_births/W + short -> us_births__W__short"""
    return f"{dataset.replace('/', '__')}__{term}"


def ctx_dir(out, variant: str, context: str) -> Path:
    return Path(out) / variant / f"ctx_{context}"


def per_config_dir(out, variant: str, context: str) -> Path:
    return ctx_dir(out, variant, context) / "per_config"


def result_csv(out, variant: str, context: str, dataset: str, term: str) -> Path:
    return per_config_dir(out, variant, context) / f"{config_stem(dataset, term)}.csv"


def result_meta(out, variant: str, context: str, dataset: str, term: str) -> Path:
    return per_config_dir(out, variant, context) / f"{config_stem(dataset, term)}.run_meta.json"


def job_result_csv(out, job: dict) -> Path:
    """Result csv of a job dict as written by make_jobs.py."""
    variant = variant_name(job["model"], job.get("toto_setting") or "point", job.get("seed", 42))
    return result_csv(out, variant, job["context"], job["dataset"], job["term"])


def atomic_write_text(path, text: str) -> None:
    """Write to a temporary name in the same folder, then rename, so a file
    that exists under its final name is always complete. The temporary name
    does not end in .csv or .json, so globs for finished files never see it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
