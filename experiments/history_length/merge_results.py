#!/usr/bin/env python3
"""Build <out>/<variant>/ctx_<c>/all_results.csv from the per-run files in
<out>/<variant>/ctx_<c>/per_config/*.csv (one header + one row each, written
by run_grid.py), rows sorted by the `dataset` column. Standard library only.

Safe to rerun: the merged file is rebuilt from scratch each time and written
to a temporary name, then renamed. Files that are not finished results
(temporary names, *.run_meta.json) are never read.

Usage:
  python3 merge_results.py --out results                # every variant and context
  python3 merge_results.py --out results --variant toto__A__seed42 --context default
"""
import argparse
import csv
import io
import sys
from pathlib import Path

from result_paths import atomic_write_text


def merge_one(ctx_dir: Path):
    """Merge one <variant>/ctx_<c> folder. Returns (rows written, problems)."""
    per_config = ctx_dir / "per_config"
    files = sorted(per_config.glob("*.csv"))
    header = None
    rows = []
    problems = []
    for f in files:
        with f.open(newline="") as fh:
            lines = list(csv.reader(fh))
        if len(lines) != 2:
            problems.append(f"{f}: expected header + 1 row, found {len(lines)} lines")
            continue
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            problems.append(f"{f}: header differs from {files[0].name}")
            continue
        rows.append(lines[1])
    if header is None:
        return 0, problems
    idx = header.index("dataset")
    rows.sort(key=lambda r: r[idx])
    seen = {}
    for r in rows:
        if r[idx] in seen:
            problems.append(f"{ctx_dir}: dataset {r[idx]!r} appears twice")
        seen[r[idx]] = True
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    atomic_write_text(ctx_dir / "all_results.csv", buf.getvalue())
    return len(rows), problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results", help="results root (default: results)")
    p.add_argument("--variant", help="only this variant folder")
    p.add_argument("--context", choices=["default", "512"], help="only this context")
    args = p.parse_args()

    root = Path(args.out)
    ctx_dirs = sorted(d.parent for d in root.glob("*/ctx_*/per_config") if d.is_dir())
    if args.variant:
        ctx_dirs = [d for d in ctx_dirs if d.parent.name == args.variant]
    if args.context:
        ctx_dirs = [d for d in ctx_dirs if d.name == f"ctx_{args.context}"]
    all_problems = []
    for d in ctx_dirs:
        n, problems = merge_one(d)
        all_problems += problems
        print(f"MERGED {d.relative_to(root)} rows={n}")
    for line in all_problems:
        print(f"MERGE_PROBLEM {line}", file=sys.stderr)
    print(f"MERGE_DONE folders={len(ctx_dirs)} problems={len(all_problems)}")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
