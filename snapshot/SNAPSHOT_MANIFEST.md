# Pinned GIFT-Eval snapshot — 2026-09-07 (the paper's data)

All paper numbers come from this snapshot, not the live board (which grew 125 → 130 models between Sep 6 and Sep 7).

- Path: `papers/fmts/snapshot_2026-09-07/` (git clone of huggingface.co/spaces/Salesforce/GIFT-Eval)
- Git commit: `b61c48aa3197a1a111e405c0e6f600128a1efee3` (space's own commit, 2026-09-07 10:51:31 +0800)
- Contents verified: 130 model dirs, each with `all_results.csv` (98 lines = 97 configs + header) and `config.json`; 5 aggregate CSVs at `results/` top level; `app.py` sort key confirmed unchanged at line 189: `sort_val=['Test Leak.', 'MASE_Rank']`
- Combined md5 of all 130 CSVs (null-safe, sorted): `0eb39c134bfcd390c9e4b0e12aac115d`
- Earlier pilot snapshot (2026-09-06, 125 models, NOT leakage-filtered, kept for provenance): `papers/fmts/audit_pilot/`
