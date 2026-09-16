SUPPLEMENTARY MATERIAL — README
================================

This zip has three top-level directories: code/, results/, and snapshot/.

code/audit/
  Scripts that check the public GIFT-Eval leaderboard snapshot for data
  leakage and compute the two audit results reported in the paper.
    run_audit.py         — the leakage audit itself
    result2_bands.py     — computes Result 2 as score bands
    result2_grid.py      — computes Result 2 as a full grid
  Each script's JSON output is included next to it (audit_results.json,
  result2_bands.json, result2_grid.json), along with a short write-up of
  each result (AUDIT_DAY1.md, RESULT2_BANDS.md, RESULT2_GRID.md).
    table1_200k.py      — Table 1 refreshed at 200,000 bootstrap draws,
                           plus a CRPS_Rank Holm check at 100,000 draws.
                           Output: table1_200k.json, write-up:
                           TABLE1_200K.md.

code/selection/
  Scripts that pick which GIFT-Eval dataset-configs the paper's own model
  grid was run on, and why.
    select_configs.py, select_configs_v1.py   — the selection scripts
                                                 (v1 was an earlier pass;
                                                 the final rule is
                                                 SELECTION_RULE.md)
    compute_median_lengths.py                 — computes median series
                                                 length per config, used
                                                 as a selection input
    SELECTION_RULE.md, SELECTION_RULE_v1.md   — the selection rule,
                                                 explained
    selected_configs.json, selected_configs_v1.json,
    config_median_lengths.json                — their outputs

code/harness/
  The evaluation harness used to run the paper's 4 models against the
  selected dataset-configs. Only the harness code is included — not the
  per-model Python environments, not the downloaded GIFT-Eval dataset, and
  not the gift-eval / toto library clones themselves (all three are public
  and documented below so they can be reproduced exactly).
    run_grid.py     — runs one model against a list of configs at a given
                       context length, writes all_results.csv + run_meta.json
    verify_data.py  — a data-loading sanity check (no model involved)
    README.md       — full harness documentation: per-model environment
                       setup, why each model needs its own environment,
                       how context length is handled per model, and what
                       was and wasn't verified before the real runs
    requirements/   — pinned dependencies per model environment

code/poolcheck/
  Extra checks that sanity-check the audit and Result-2 claims against a
  second, independent dataset (fev-bench) and against the GIFT-Eval
  leaderboard's own git history, not just a single static snapshot.
    resolution_budget.py, pool_perturbations.py         — GIFT-Eval snapshot
                                                            checks (CPU only,
                                                            read snapshot/)
    pool_dependence_replay.py,
    pool_dependence_replay_method_first_fast.py          — replay the
                                                            GIFT-Eval
                                                            leaderboard's
                                                            public git
                                                            history commit by
                                                            commit; needs a
                                                            local clone (not
                                                            included here):
                                                              git clone https://huggingface.co/spaces/Salesforce/GIFT-Eval
    fev_resolution.py, fev_dropone.py                    — replicate the
                                                            fev-bench
                                                            leaderboard and
                                                            check its
                                                            resolution and
                                                            drop-one
                                                            stability; read
                                                            pivot_SQL.csv and
                                                            leaderboard_SQL.csv
                                                            (both included —
                                                            fev-bench's own
                                                            published tables,
                                                            pinned space
                                                            commit a7553dc)
  Each script documents its exact required inputs at the top of its own
  file.

code/effective_context/
  compute_effective_context.py — computes, for every (model, config,
  setting) cell of the 4-model x 12-config x 2-setting grid, the effective
  context length: how much past history the model actually saw after its
  own truncation rule. CPU only, no model runs; needs the downloaded
  GIFT-Eval dataset (see code/harness/README.md for the exact download
  command). Output: effective_context.csv, write-up: EFFECTIVE_CONTEXT.md.
  Required inputs are documented at the top of the script.

results/
  The raw output of the paper's model grid: 4 models x 2 context lengths
  (default and 512) = 8 runs, each an all_results.csv in the same
  directory shape run_grid.py writes (results/<model>/ctx_<context>/).
    chronos_bolt_base/ctx_default/all_results.csv
    chronos_bolt_base/ctx_512/all_results.csv
    moirai2_small/ctx_default/all_results.csv
    moirai2_small/ctx_512/all_results.csv
    timesfm25/ctx_default/all_results.csv
    timesfm25/ctx_512/all_results.csv
    toto/ctx_default/all_results.csv
    toto/ctx_512/all_results.csv
    grid_run_log.txt — the run log for all 8 runs (progress lines per
                        config/model/context, and any failed configs)

snapshot/
  The pinned public GIFT-Eval leaderboard snapshot that all paper numbers
  are computed from (not the live leaderboard, which changed model count
  between when the snapshot was taken and the day after).
    SNAPSHOT_MANIFEST.md         — what the snapshot is, when it was taken,
                                    and a combined md5 of its 130
                                    all_results.csv files (one per model
                                    dir), computed as:
                                      find snapshot_2026-09-07/results \
                                        -name 'all_results.csv' -print0 \
                                        | sort -z | xargs -0 md5 | md5
                                    (null-safe find, sorted, so the combined
                                    hash doesn't depend on directory listing
                                    order)
    snapshot_2026-09-07/results/ — all 130 leaderboard model directories
                                    (each an all_results.csv + config.json)
                                    plus 5 leaderboard-wide summary CSVs
    snapshot_2026-09-07/app.py,
    snapshot_2026-09-07/src/     — the leaderboard app code itself, kept
                                    because code/audit/run_audit.py checks
                                    it for the sort key the audit depends on

Provenance
----------
The shipped snapshot_2026-09-07/results/ is byte-identical to the public
GIFT-Eval Hugging Face space at commit b61c48aa3197a1a111e405c0e6f600128a1efee3
(verified 2026-09-15: all 265 files under results/, zero mismatches).
code/poolcheck/'s replay and resolution scripts originated in an external
adversarial review of this work, and were verified and extended by the
authors from there -- if you're asking "who wrote this," that's the answer.
A clone of the GIFT-Eval space made after 2026-09-14 will have more commits
than this snapshot (the board keeps adding models); the shipped poolcheck
scripts pin their replay to the snapshot commit above by default specifically
so they keep reproducing regardless, with an --unpinned flag to replay to
the clone's current HEAD instead.

Reproduction, in three commands
------------------------------
The scripts were originally written to live inside one project folder where
audit/, selection/, harness/, and the snapshot were all sibling directories,
each script finding its inputs using paths relative to its own location. This
zip reorganizes those same files into code/audit/, code/selection/,
code/harness/, results/, and snapshot/ for clarity. The three scripts below
have already been patched (path constants only -- no change to seeds, the
bootstrap replicate count B, or any selection/ranking logic) to resolve their
inputs relative to their own file location for THIS zip's layout, so they run
as-extracted, no editing required. Run them from anywhere; the relative paths
are computed from each script's own location via Path/__file__, not from your
current directory.

1. Run the leakage audit against the pinned snapshot:
     python3 code/audit/run_audit.py snapshot/snapshot_2026-09-07
   It writes audit_results.json and AUDIT_DAY1.md next to itself, i.e. into
   code/audit/, overwriting the copies already in this zip. Reproduces
   audit_results.json byte-identical under CPython 3.11 (last-digit float
   formatting may differ under other versions); a small tolerance in
   comparisons is appropriate.

2. Recompute the Result-2 sampling-noise reference bands:
     python3 code/audit/result2_bands.py snapshot/snapshot_2026-09-07
   Reads snapshot/snapshot_2026-09-07 and code/selection/selected_configs.json,
   writes result2_bands.json and RESULT2_BANDS.md next to itself. Reproduces
   every number in result2_bands.json exactly; the only field that differs
   from the shipped copy is "snapshot_root" itself (a path string, redacted
   to "<REDACTED_PATH>/..." in the shipped copy for anonymization -- it will
   show your actual snapshot path instead when you run this).

3. Recompute Result 2 from the grid's raw CSVs:
     python3 code/audit/result2_grid.py snapshot/snapshot_2026-09-07
   Reads snapshot/snapshot_2026-09-07, code/selection/selected_configs.json,
   code/audit/result2_bands.json, and results/ (this zip's flattened stand-in
   for the original grid_results_h100/results), writes result2_grid.json and
   RESULT2_GRID.md next to itself. Reproduces every computed number exactly;
   the only fields that differ from the shipped copy are "snapshot_root" (see
   above) and "result2_bands_json_md5" (an md5 of result2_bands.json itself,
   which also differs only because of the same redacted snapshot_root field
   above -- it is a hash of a path-bearing file, not a computed result).

For how the model grid in results/ was actually run (three passes, two
failure-and-rerun rounds, all disclosed), see results/RUNS.md. For the
checks backing the paper's CRPS_Rank, Holm, and high-B numbers, see
code/audit/critic_checks.py (output: critic_checks.json, write-up:
CRITIC_CHECKS.md).

select_configs.py (code/selection/) is included as the pre-registration
record, not as a script you can run from this zip: it needs a
"domain" (Energy, Transport, etc.) label for every one of the 97 configs in
the snapshot, and that comes from dataset_properties.json inside the
GIFT-Eval repo clone, which is not included here (see "External code and
data" below). The two domain-shaped files that ARE shipped alongside it
don't substitute: config_median_lengths.json has no domain field at all, and
selected_configs.json's "domain" field only covers the 12 configs the script
already picked -- using either to backfill the other 97 would mean either
inventing data or a circular read of the script's own output, so this script
is left unpatched and documentation-only. Its actual output,
selected_configs.json, ships next to it and is exactly what
result2_bands.py and result2_grid.py read; that output is unaffected by this
limitation. select_configs_v1.py has the same limitation for the same reason.

The .md file next to each script (AUDIT_DAY1.md, RESULT2_BANDS.md,
RESULT2_GRID.md, SELECTION_RULE.md) explains what the script computes and
why, in more depth than this file does.

External code and data this depends on
---------------------------------------
- GIFT-Eval harness code: https://github.com/SalesforceAIResearch/gift-eval
  pinned at commit 9e88bb6
- Toto model code: https://github.com/DataDog/toto
  pinned at commit c079fd7
- GIFT-Eval evaluation dataset: the public "Salesforce/GiftEval" dataset on
  Hugging Face (see code/harness/README.md for the exact download command)
- Model weights: all 4 models evaluated in the paper are loaded from their
  public Hugging Face model repos, named in the paper itself. No weights
  are included in this zip.
