#!/usr/bin/env python3
"""Write uv constraint files that make runpod_setup.sh resolve every package
to the version the first round installed on the H100.

Source: grid_results_h100/provenance_h100/setup_logs/setup.log (the first
round's h100_setup.sh run under `set -x`). Every
  + uv pip install --python envs/<env>/bin/python ...
line starts one install step; the " + <package>==<version>" lines under it
are what that step installed. Editable local packages (gift-eval, toto-ts,
lines with "(from file://") are left out. One file per step:
  requirements/first_round_constraints/<env>__step<k>.txt
runpod_setup.sh passes step k's file with -c to the k-th install command of
that environment. The requested packages (the pins) are unchanged; the
constraints only fix what an open requirement such as "torch>=2.1" resolves
to (on 2026-11-19 a fresh resolve gave torch 2.15.0 and pyarrow 26.0.1; the
first round had torch 2.14.0 and pyarrow 25.0.1).

  python3 make_constraints.py [--log <setup.log>]
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMTS = HERE.parents[2]
DEFAULT_LOG = FMTS / "grid_results_h100" / "provenance_h100" / "setup_logs" / "setup.log"
STEP_RE = re.compile(r"^\+ uv pip install --python envs/(\w+)/bin/python (.*)$")
PKG_RE = re.compile(r"^ \+ ([A-Za-z0-9_.\-\[\]]+)==(\S+)$")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--out-dir", default=str(HERE / "requirements" / "first_round_constraints"))
    a = p.parse_args()

    steps = []  # (env, command, [pkg==ver])
    current = None  # the step whose output is being read, or None
    for line in Path(a.log).read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if m:
            current = (m.group(1), m.group(2), [])
            steps.append(current)
            continue
        if line.startswith("+ "):  # any other traced shell command ends the step
            current = None
            continue
        if current is not None:
            m = PKG_RE.match(line)
            if m and "(from " not in line:
                current[2].append(f"{m.group(1)}=={m.group(2)}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = {}
    for env, cmd, pkgs in steps:
        counter[env] = counter.get(env, 0) + 1
        k = counter[env]
        body = (
            f"# first round, envs/{env}, install step {k}: uv pip install {cmd}\n"
            f"# from {Path(a.log).relative_to(FMTS) if Path(a.log).is_relative_to(FMTS) else a.log}\n"
            + "".join(x + "\n" for x in sorted(pkgs, key=str.lower))
        )
        (out / f"{env}__step{k}.txt").write_text(body)
        print(f"{env}__step{k}.txt  {len(pkgs):4d} packages  <- {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
