# TIME: replay of the whole history (45 states, 44 updates)

Spec: DESIGN_ROUND2.md S6. Scripts: time_prerefresh_plan.py, time_prerefresh_fetch.py, time_prerefresh_replay.py, time_prerefresh_report.py (all in this folder). Numbers: boards/time_full_history.json. Downloaded files: boards/time_output_pre_refresh/objects/ (727 files, 47,604,992 bytes, each checked against its LFS sha256).

## 1. Why the first round could not score the first 16 states (facts)

- The history clone (boards/time_output_history) holds Git LFS pointer files only (its .git is 3.3 MB); the actual files were never fetched for old commits. The first round scored every state from the current (head) files on disk and matched them to a commit's files by name and content hash. A model-state counted as scoreable only if its folder name exists at head and its 98 metrics.npz files were byte-identical to the head files.
- The old per-entry files are NOT absent and the layout did NOT change. At every one of the 45 commits that touch results/ (first with results: ca2d156c, 2026-02-09; last 3bf103cb, 2026-10-31) each model folder has results/<model>/<dataset>/<freq>/<horizon>/{config.json, metrics.npz, predictions.npz} for the same 98 units (294 files per model; 13 models at ca2d156c, 22 at 297ae66a and c43961bb, 43 at 3bf103cb). The board's loader needs metrics.npz and predictions.npz; both are listed for every model at every commit (checked in the replay).
- The baseline file exists at every commit (results/seasonal_naive, 98 units). It is identical to the head file in 96 units and differs in exactly two units, Crypto/D/short and Global_Influenza/W/short, in every commit from ca2d156c to 297ae66a (16 states). It was replaced at c43961bb (2026-06-02 05:50:07 UTC, "Update Crypto/D and Global_Influenza/W results (#34)"); from c43961bb on it equals the head file.
- So the first-round loader did not "use the current baseline for all states"; it threw away every model-state that did not match head. For the first 16 states that discarded the baseline itself (differs from head in those two units), so the geometric-mean rule had no baseline and the state was skipped. The first round's sentence that the rule "cannot be computed" for those states described its own method, not the repository.
- Why later states had fewer entries than the board showed: at c43961bb 22 entries were present, 13 scored. The 9 dropped had files that differ from head: 8 (Eidos, PatchTST-FM-R1, Timer-S1, Toto-2.0-1B, Toto-2.0-2.5B, Toto-2.0-22m, Toto-2.0-313m, Toto-2.0-4m) because they were later re-uploaded (Eidos and the five Toto-2.0 folders: the two refreshed units, in 4d7793e9 and d086573e; Timer-S1: all 98 units, a2d8c250; PatchTST-FM-R1: 35 units, 3963cf80), and 1 (PatchTST-FM-TsMix10M) because that folder name no longer exists at head. Later gaps: Tabby (bb420f38 to 03ab980c, before the 2026-09-30 refresh), and the early renamed folders (chronos2_base, chronos_bolt_base, kairos, moirai2, moirai_base, ToTo) whose files are identical under their new names but were counted as missing.
- Files that were refreshed after upload, besides the two units: TiRex, chronos2_base, chronos_bolt_base (all 98 units re-uploaded, 2026-02-22), kairos (66 units, changed together with its rename to Kairos in 985df4da, so it counts as a new entry there), Timer-S1 (98 units, a2d8c250), Eidos (98 units at 25e9cf50, two units at d086573e), PatchTST-FM-R1 (35 units, 3963cf80), Tabby (98 units, 66eea9da).

## 2. What was fetched and how

From the same repository (huggingface.co/datasets/Real-TSF/TIME-Output), for each of the 45 commits the LFS identifiers of all metrics.npz were read from the pointer files in the existing clone (4,941 distinct files, 327.7 MB in all). 4,214 of them are the head files already on disk; the other 727 files were downloaded (47.6 MB, 3 min 25 s, via the commit-specific resolve URL; all 727 matched their sha256). predictions.npz and config.json were not downloaded (the scores use metrics.npz only, as in the first round; their presence is read from the tree listing).

## 3. How each state was scored

- Pool displayed at a commit: every model folder in results/ at that commit (the board's code, src/about.py and utils.get_all_datasets_results, lists every folder and keeps a unit only if metrics.npz and predictions.npz exist; all folders qualify at all 45 commits, every one with all 98 units). Seasonal_naive is an entry (it sits at 1.000 in the displayed table) and counts in the ranks. Pool sizes: 13 at the first state, 22 at c43961bb, 43 at the last.
- Per state: the board's own reduction (nanmean of the MASE and CRPS arrays per file); the board's own compute_ranks (rank "first" within each (dataset/freq, horizon)), normalize_by_seasonal_naive and get_overall_leaderboard (from boards/time_lb/src, with the same stand-ins for gradio, timebench and the data download as in audit_time.py). The baseline is the seasonal_naive file at that commit. My unrounded geometric means and mean ranks give the same displayed order as the board's get_overall_leaderboard in 45 of 45 states (largest difference to the board's rounded numbers 5.0e-4).
- Exact ties: none in 44 of 45 states. In e14243f8 (2026-06-02 05:50:31 to 05:53:35 UTC) the folders PatchTST-FM-TsMix10M and PatchTST-FM-Extended were both present and identical in 96 of 98 files, so 193 unit-metric cells tie. The board breaks such ties by folder listing order, which depends on the machine; I broke them by name and re-ran with the reverse order (section 5).
- An entry is "untouched" in an update when it is present before and after (a folder renamed with identical content counts as the same entry) and all 98 of its files are identical. A reversal is a pair of untouched entries, both in the previous state's top ten under that rule, whose order of aggregate value flips. Top ten and order are per rule, as in the first round.
- Data refresh (c43961bb): the baseline changed in two units and 13 of 22 entries changed in those same two units (Chronos-2, Chronos-bolt, Kairos, Moirai2, Moirai_base, TiRex, TimesFM-1.0/2.0/2.5, Toto, seasonal_naive, sundial_base, visiontspp_base); 9 entries kept their old data there and were corrected in the next commits (4d7793e9, a2d8c250, d086573e, 3963cf80). The baseline change moves the geometric-mean score of every entry, so by the paper's definition no pair is untouched in that update; I count it separately ("baseline refresh") and still list its result: no pair of own-file-untouched top-ten entries flips under any of the four rules (the top-ten flips in that update all involve a changed entry: Chronos-2 or TimesFM-2.5 against Toto-2.0-22m, PatchTST-FM-R1, Timer-S1). Between c43961bb and 3963cf80 the board displayed a mixed pool (old and new data on those two units); the replay scores it as displayed.

## 4. Result

Columns: updates replayed; updates with at least one swapped untouched top-ten pair; number of such pairs; then the same three numbers excluding the baseline refresh; for the 28 updates the first round replayed (2026-06-02 c43961bb state to 2026-10-31); and the first round's pool reproduced by this script (check). Each cell in the last three columns is updates / updates with swap / pairs.

| Rule | Updates | With a swap | Pairs | Ordinary updates (no baseline change) | The first round's 28 updates, full pool | First round's pool, rerun here |
|---|---|---|---|---|---|---|
| MASE (norm.), geometric mean (board's displayed sort) | 44 | 0 | 0 | 43 / 0 / 0 | 28 / 0 / 0 | 28 / 0 / 0 |
| CRPS (norm.), geometric mean | 44 | 0 | 0 | 43 / 0 / 0 | 28 / 0 / 0 | 28 / 0 / 0 |
| MASE_rank | 44 | 12 | 13 | 43 / 12 / 13 | 28 / 10 / 11 | 28 / 5 / 6 |
| CRPS_rank | 44 | 5 | 6 | 43 / 5 / 6 | 28 / 4 / 4 | 28 / 4 / 4 |

The baseline-refresh update (297ae66a -> c43961bb, 2026-06-02): 0 swaps under all four rules. First-round counts to compare: board's rule 0 in 28 (here 0 in 44); MASE_rank 6 pairs in 5 of 28 (here 13 pairs in 12 of 44; 11 pairs in 10 of the same 28); CRPS_rank 4 pairs in 4 of 28 (here 6 pairs in 5 of 44; 4 pairs in 4 of the same 28). The CRPS columns of the geometric-mean rule: 0 in 44, as in the first round.

### Events, MASE_rank (pair order is the order before)
- 2026-03-13 6eb02691 -> 2026-04-03 5b7dab51 (Submit evaluation results for PatchTST-FM-TsMix10M (#15)); arrived/changed: PatchTST-FM-TsMix10M; pair Toto vs Eidos: places 5,6 before -> 7,6 after
- 2026-04-03 5b7dab51 -> 2026-04-03 d6d1916a (Submit evaluation results for PatchTST-FM-R1 (#14)); arrived/changed: PatchTST-FM-R1; pair Chronos-2 vs PatchTST-FM-TsMix10M: places 1,2 before -> 2,1 after
- 2026-06-02 3963cf80 -> 2026-06-02 e14243f8 (Submit evaluation results for PatchTST-FM-Extended (#26)); arrived/changed: PatchTST-FM-Extended; pair PatchTST-FM-TsMix10M vs Chronos-2: places 4,5 before -> 6,5 after
- 2026-06-02 8ec9db10 -> 2026-07-11 2ffa8721 (Add OmniScient results (fine-tuned Chronos-2 derivative, 360); arrived/changed: OmniScient; pair TimesFM-2.5 vs PatchTST-FM-R1: places 6,7 before -> 7,6 after
- 2026-07-11 2ffa8721 -> 2026-07-14 42495a47 (whenxuan: update the benchmark results for litespecformer (#); arrived/changed: litespecformer; pair PatchTST-FM-R1 vs TimesFM-2.5: places 6,7 before -> 7,6 after
- 2026-07-14 42495a47 -> 2026-08-05 73a177cb (Submit evaluation results for TS-ICL (#38)); arrived/changed: TS-ICL; pair TimesFM-2.5 vs PatchTST-FM-R1: places 6,7 before -> 7,6 after
- 2026-08-28 e308bbc7 -> 2026-09-16 bb420f38 (Submit evaluation results for Tabby (#40)); arrived/changed: Tabby; pair PatchTST-FM-Extended vs Chronos-2: places 5,6 before -> 6,5 after
- 2026-09-16 bb420f38 -> 2026-09-23 82e43a1e (Submit evaluation results for t0_beta (#41)); arrived/changed: t0_beta; pair Chronos-2 vs PatchTST-FM-Extended: places 5,6 before -> 6,5 after
- 2026-09-26 65119f56 -> 2026-09-28 03ab980c (Submit evaluation results for QiYao-M (#43)); arrived/changed: QiYao-M; pair PatchTST-FM-Extended vs Chronos-2: places 5,6 before -> 7,6 after
- 2026-09-28 03ab980c -> 2026-09-30 66eea9da (Refresh Tabby TIME results: seed 42, max(H, 128) (#44)); arrived/changed: Tabby; pair Chronos-2 vs PatchTST-FM-Extended: places 6,7 before -> 7,6 after
- 2026-10-05 246dc3f1 -> 2026-10-09 4b07d9e5 (Submit evaluation results for TimeRSI (#48)); arrived/changed: TimeRSI; pair ChakraTS vs TimesFM-3: places 2,3 before -> 4,3 after
- 2026-10-05 246dc3f1 -> 2026-10-09 4b07d9e5 (Submit evaluation results for TimeRSI (#48)); arrived/changed: TimeRSI; pair PatchTST-FM-Extended vs Chronos-2: places 7,8 before -> 9,8 after
- 2026-10-16 3d63e2a1 -> 2026-10-16 9a5e670d (Submit evaluation results for Timestral-28M (#52)); arrived/changed: Timestral-28M; pair Timestral-5M vs Toto-2.0-1B: places 8,9 before -> 10,9 after

### Events, CRPS_rank
- 2026-05-15 7a1b5500 -> 2026-05-15 3d7c681e (Submit evaluation results for Toto-2.0-1B (#22)); arrived/changed: Toto-2.0-1B; pair Toto-2.0-22m vs PatchTST-FM-R1: places 4,5 before -> 6,5 after
- 2026-05-15 7a1b5500 -> 2026-05-15 3d7c681e (Submit evaluation results for Toto-2.0-1B (#22)); arrived/changed: Toto-2.0-1B; pair Toto-2.0-4m vs Eidos: places 9,10 before -> 11,10 after
- 2026-06-02 e14243f8 -> 2026-06-02 8ec9db10 (Rename to PatchTST-FM-Extended (#35)); arrived/changed: none (removal); pair PatchTST-FM-Extended vs Chronos-2: places 4,5 before -> 5,4 after
- 2026-06-02 8ec9db10 -> 2026-07-11 2ffa8721 (Add OmniScient results (fine-tuned Chronos-2 derivative, 360); arrived/changed: OmniScient; pair TiRex vs Timer-S1: places 9,10 before -> 11,10 after
- 2026-09-30 66eea9da -> 2026-09-30 5787f898 (Refresh Tabby-Pretrain TIME results: max(H, 128) (#45)); arrived/changed: Tabby-Pretrain; pair Toto-2.0-1B vs Tabby: places 5,6 before -> 6,5 after
- 2026-09-30 5787f898 -> 2026-10-05 246dc3f1 (Submit evaluation results for ChakraTS (#46)); arrived/changed: ChakraTS; pair Tabby vs Toto-2.0-1B: places 5,6 before -> 7,6 after

### Geometric-mean rules (MASE and CRPS)
No reversal in any of the 44 updates. Among the updates, 10 changed an entry's own files (c01dea39, 9d12c44d, 0ffbc360, 25e9cf50, c43961bb, 4d7793e9, a2d8c250, d086573e, 3963cf80, 66eea9da) (re-uploads, refreshes); there top-ten pairs including the changed entry do flip, as they must, and those are not reversals of untouched pairs.

## 5. Differences from the first round, explained

The first round's result is reproduced exactly by this code once its pool is imposed (last column: same 5 updates and 6 pairs for MASE_rank, 4 and 4 for CRPS_rank, same pairs and dates). So every difference is the pool:
- 16 earlier updates (before c43961bb) that were not scored before: MASE_rank 2 swaps (6eb02691 -> 5b7dab51: Toto vs Eidos; 5b7dab51 -> d6d1916a: Chronos-2 vs PatchTST-FM-TsMix10M), CRPS_rank 2 pairs in 1 update (7a1b5500 -> 3d7c681e: Toto-2.0-22m vs PatchTST-FM-R1 and Toto-2.0-4m vs Eidos), none for the geometric mean.
- Inside the first round's 28 updates, MASE_rank gains 5 updates / 5 pairs because the full pool contains entries the first round dropped: 3963cf80 -> e14243f8 (the duplicate-name pair, see ties), e308bbc7 -> bb420f38, bb420f38 -> 82e43a1e and 65119f56 -> 03ab980c (Tabby, absent from the first round's pool in those states, was present on the board), and 03ab980c -> 66eea9da (Tabby's refresh moves its ranks; Chronos-2 and PatchTST-FM-Extended swap). The first round's 5 events all remain.
- CRPS_rank inside the 28: the first round's event d086573e -> 3963cf80 (Timer-S1 vs TiRex) does not occur in the full pool (it came from the first round's smaller pool); instead e14243f8 -> 8ec9db10 (removing the duplicate folder) flips PatchTST-FM-Extended and Chronos-2. The total of 4 pairs matches by coincidence. The first round's other 3 events remain.
- Tie sensitivity (e14243f8): with ties broken in reverse name order, MASE_rank is 13 pairs in 12 updates (the e14243f8 event is replaced by Chronos-2 vs PatchTST-FM-Extended at 8ec9db10) and CRPS_rank is 5 pairs in 4 updates (the 8ec9db10 event disappears). The three events around the duplicate folder therefore depend on listing order, which I cannot know; the other 12 MASE_rank pairs and 5 CRPS_rank pairs do not.
- Geometric mean: 0 in 28 before and 0 in 44 now; nothing differs.

## 6. Still not covered

- predictions.npz were not downloaded; their presence is taken from the commit's file listing.
- The board's displayed state at a commit is rebuilt with the current board code (boards/time_lb at its present version), not the code that ran at the time. The code at old commits of the leaderboard space was not examined.
- Whether the board's listing order (tie-breaking) at e14243f8 matches either of my two orders is unknown.
