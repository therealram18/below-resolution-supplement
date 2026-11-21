"""Tests for analyze_gpu_round.py (DESIGN_ROUND3.md with Revision 1), fixed before any GPU-round result exists.

Run: <python> -B test_analyze_gpu_round.py   (unittest; uses only the first round's files, the board's rows and
synthetic values). Set GPU_ROUND_KEEP_SYNTHETIC=<folder> to keep the report of the full synthetic layout.
"""
import sys

sys.dont_write_bytecode = True  # camera_ready/ is read only

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gpu_round as A  # noqa: E402
from exact_signflip import TOL, exact_sum_p  # noqa: E402  (camera_ready/ is on sys.path via A)
import vernier_stats as vs  # noqa: E402

FIRST = A.FIRST_ROUND
PCT_CHANGE = A.R / "audit" / "pct_change.json"
KEEP = os.environ.get("GPU_ROUND_KEEP_SYNTHETIC")
QUICK = ["--level-draws", "200"]          # plumbing runs; the level check itself is tested at 20,000 draws
REF = A.load_reference(A.Reader())


def make_first_round_layout(root, seeds=("42",)):
    """The first round's files in the new layout; first-round Toto stands in for setting A."""
    root = Path(root)
    for m in A.DETERMINISTIC:
        for L in A.LENGTHS:
            shutil.copytree(FIRST / m / f"ctx_{L}", root / m / f"ctx_{L}")
    for s in seeds:
        for L in A.LENGTHS:
            shutil.copytree(FIRST / "toto" / f"ctx_{L}", root / f"toto__A__seed{s}" / f"ctx_{L}")


def run(results, out, *extra):
    return A.main(["--results", str(results), "--out", str(out), *QUICK, *extra])


def brute_force(x):
    """Two-sided and one-sided sign-flip p by listing every sign pattern."""
    x = np.asarray(x, float)
    G = len(x)
    W = 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)[None, :]) & 1) - 1
    sums = W @ x
    obs = x.sum()
    two = np.mean(np.abs(sums) >= abs(obs) * (1 - TOL))
    one = np.mean(sums >= obs - TOL * abs(obs))
    return float(two), float(one)


def rewrite(path, change):
    """Apply change(rows) to a results csv in place."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    rows = change(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_results(root, variant, L, vals):
    d = Path(root) / variant / f"ctx_{L}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "all_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", A.MASE_COL])
        for c, v in vals.items():
            w.writerow([c, variant, v if isinstance(v, str) else repr(v)])


def family(d_by_model, cfg, fam):
    blocks = {m: A.effect_block(cfg, np.asarray(x, float), REF["top10"], REF["sn"]) for m, x in d_by_model.items()}
    for g in A.GROUPINGS:
        A.family_tests(blocks, fam, g)
    return blocks


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)


class FirstRoundReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_first_round_layout(cls.tmp / "res", seeds=("42", "43", "44"))
        cls.out = run(cls.tmp / "res", cls.tmp / "out")
        cls.md = (cls.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        with open(PCT_CHANGE) as f:
            cls.pct = json.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_effects_equal_pct_change_json(self):
        expect = {"chronos_bolt_base": "chronos_bolt_base", "moirai2_small": "moirai2_small",
                  "timesfm25": "timesfm25", "toto__A__seed42": "toto"}
        got = {}
        for subset in ("first_12", "eligible_70"):   # only the 12 are present, so the 70 reduce to them
            for m, key in expect.items():
                b = self.out["G1"][subset]["models"][m]
                self.assertEqual(b["n_configurations"], 12)
                self.assertEqual(b["n_sources"], 9)
                self.assertEqual(round(b["effect_pct"], 1), self.pct[key]["pct_change"], (subset, m))
                got[m] = round(b["effect_pct"], 1)
        self.assertEqual(sorted(got.values()), [-2.0, 6.8, 20.6, 22.2])

    def test_S_on_12_is_9_34(self):
        self.assertEqual(round(self.out["reference"]["S"]["first_12"]["pct"], 2), 9.34)
        for m in A.G1_MODELS:
            self.assertEqual(round(self.out["G1"]["first_12"]["models"][m]["S"]["pct"], 2), 9.34)

    def test_board_and_first_round_toto_gmeans(self):
        g3 = self.out["G3"]["reference"]
        self.assertEqual(round(g3["board_gmean_scaled_on_12"], 4), 0.7699)
        self.assertTrue(g3["board_value_equals_paper_0.7699_to_4_decimals"])
        self.assertEqual(round(g3["first_round_point_setting"]["gmean_scaled_ours"], 4), 0.8126)
        self.assertTrue(g3["first_round_point_setting"]["equals_paper_0.8126_to_4_decimals"])
        a = self.out["G3"]["reading_a_within_1pct"]["toto__A__seed42"]
        self.assertEqual(round(a["gmean_scaled_ours"], 4), 0.8126)
        self.assertEqual(a["verdict"], "FAIL")

    def test_machine_check_passes_on_identical_files(self):
        mc = self.out["machine_check"]
        self.assertEqual(mc["verdict"], "PASS")
        self.assertEqual(mc["values_compared"], 72)
        self.assertEqual(mc["largest_relative_difference"], 0.0)
        self.assertEqual(mc["d_c_sign_changes"], [])

    def test_g4_identical_seeds_have_zero_range(self):
        g4 = self.out["G4"]
        self.assertEqual(g4["range_over_seeds"]["effect_pct"]["range"], 0.0)
        self.assertEqual(round(g4["seeds"]["42"]["effect_pct"], 1), 20.6)

    def test_m4_rule_on_first_round_toto(self):
        # the first round's Toto gave the same MASE at both lengths there; the rule replaces the 512 row
        self.assertEqual(self.out["m4_daily_rule"]["toto__A__seed42"], "applied; a 512 row was present and was replaced")
        self.assertEqual(self.out["G1"]["first_12"]["models"]["toto__A__seed42"]["d_c"][A.M4_RULE_CONFIG], 0.0)

    def test_parts_without_files_not_run_and_freeze_printed(self):
        self.assertEqual(self.out["G2"]["status"], "NOT RUN")
        self.assertEqual(self.out["G1"]["new_58"]["status"], "NOT RUN")
        self.assertEqual(self.out["run_records"]["status"], "NOT RUN")
        self.assertEqual(self.out["design"]["freeze_utc"], "2026-11-20T06:30:00Z")
        first_lines = self.md.splitlines()[:4]
        self.assertTrue(any("nothing that finishes after 2026-11-20 12:00 IST enters the camera-ready" in x
                            for x in first_lines))


class Reference(unittest.TestCase):
    def test_S_values_and_range(self):
        S = {k: round(v["pct"], 2) for k, v in REF["S_ref"].items()}
        self.assertEqual(S, {"eligible_70": 11.76, "new_58": 12.71, "first_12": 9.34})
        s70 = REF["S_ref"]["eligible_70"]
        self.assertEqual((round(s70["loo_smallest"]["pct"], 2), round(s70["loo_largest"]["pct"], 2)), (8.51, 13.22))
        self.assertEqual((s70["loo_smallest"]["left_out"], s70["loo_largest"]["left_out"]), ("ett1", "jena_weather"))

    def test_top10_recomputed_equals_N11_and_retest_gift(self):
        with open(A.CR / "retest_gift.json") as f:
            j = json.load(f)
        self.assertEqual(list(REF["top10_names"]), j["gmean_pairs_order_top10"])
        self.assertEqual(list(REF["top10_names"]), list(A.TOP10_DOCUMENTED))

    def test_sources_and_selection(self):
        self.assertEqual(sorted(REF["source_sizes_70"], reverse=True), [12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1])
        self.assertEqual(len(REF["source_sizes_12"]), 9)
        self.assertEqual(sorted(REF["source_sizes_12"], reverse=True), [3, 2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(REF["sources_70"]["bizitobs"], 12)
        self.assertEqual(REF["sources_70"]["ett"], 12)
        self.assertEqual(REF["sources_70"]["bitbrains"], 6)
        self.assertEqual(len(REF["eligible70"]), 70)
        self.assertEqual(len({A.base_of(c) for c in REF["eligible70"]}), 18)
        self.assertEqual(len(REF["new58"]), 58)
        o = REF["overlap_58"]
        self.assertEqual((o["from_a_base_dataset_among_the_12"], o["same_series_as_one_of_the_12"]), (43, 20))

    def test_freeze_parsed_from_design(self):
        sentence, when = A.design_freeze(A.DESIGN_FILE)
        self.assertTrue(sentence.startswith("nothing that finishes after 2026-11-20 12:00 IST"))
        self.assertEqual(when.strftime("%Y-%m-%dT%H:%M"), "2026-11-20T06:30")


class SignFlip(unittest.TestCase):
    def test_one_and_two_sided_equal_brute_force(self):
        rng = np.random.default_rng(20261119)
        cases = [rng.normal(size=10) for _ in range(20)]
        cases += [rng.normal(0.4, 1, size=10) for _ in range(10)]
        cases += [rng.integers(-3, 4, size=10).astype(float) for _ in range(20)]   # many tied pattern sums
        cases += [rng.normal(size=7), rng.normal(size=11), np.zeros(10)]
        for x in cases:
            two, one = brute_force(x)
            self.assertEqual(A.two_sided_sum_p(x), two)
            self.assertEqual(exact_sum_p(x), two)
            self.assertEqual(A.one_sided_sum_p(x), one)

    def _check_interval(self, configs, d, grouping):
        names, s, n = A.group_sums(configs, d, grouping)
        lo, hi, k = A.signflip_interval(s, n)
        est = float(np.mean(d))
        self.assertTrue(lo < est < hi)
        # at the estimate the observed sum is zero up to rounding, so p is 1 in exact arithmetic; in floating
        # point a few patterns whose sums also cancel can fall either side of the rounding-level threshold
        # (and exact_sum_p alone can exceed 1 there by double counting; two_sided_sum_p caps it)
        self.assertGreater(A.two_sided_sum_p(s - n * est), 0.99)
        self.assertLessEqual(A.two_sided_sum_p(s - n * est), 1.0)
        for end, inward in ((lo, 1), (hi, -1)):
            eps = 1e-6 * max(1.0, abs(end))
            self.assertGreaterEqual(exact_sum_p(s - n * (end + inward * eps)), 0.05)
            self.assertLess(exact_sum_p(s - n * (end - inward * eps)), 0.05)
            self.assertGreaterEqual(exact_sum_p(s - n * end), 0.05)

    def test_inverted_interval_first_round(self):
        for m in ("chronos_bolt_base", "moirai2_small", "timesfm25", "toto"):
            used, d, _ = A.paired_logs(REF["first"][m]["default"], REF["first"][m]["512"], REF["sel12"])
            for g in A.GROUPINGS:
                self._check_interval(used, d, g)

    def test_inverted_interval_unequal_groups(self):
        rng = np.random.default_rng(7)
        cfg = REF["eligible70"]
        bases = sorted({A.base_of(c) for c in cfg})
        gi = np.array([bases.index(A.base_of(c)) for c in cfg])
        for mu in (0.0, 0.15, -0.1):
            d = mu + rng.normal(0, 0.2, size=len(cfg)) + rng.normal(0, 0.1, size=18)[gi]
            for g in A.GROUPINGS:
                self._check_interval(cfg, d, g)

    def test_interval_unbounded_with_five_groups(self):
        lo, hi, k = A.signflip_interval([0.1, 0.2, -0.05, 0.3, 0.1], [1, 1, 1, 1, 1])
        self.assertEqual(k, 2)
        self.assertTrue(math.isinf(lo) and math.isinf(hi))

    def test_signed_rank_zero_keeps_rank_with_sign_zero(self):
        p, sr = A.signed_rank_p([0.0, 1.0, -2.0, 3.0])
        self.assertEqual(list(sr), [0.0, 2.0, -3.0, 4.0])
        self.assertEqual(p, exact_sum_p([0.0, 2.0, -3.0, 4.0]))

    def test_sources_and_base_datasets_give_different_verdicts(self):
        cfg = REF["eligible70"]
        bases = sorted({A.base_of(c) for c in cfg})
        bi = np.array([bases.index(A.base_of(c)) for c in cfg])
        rng = np.random.default_rng(57)
        d = 0.04 + rng.normal(0, 0.1, 18)[bi] + rng.normal(0, 0.02, 70)
        b = family({"m": d}, cfg, ["m"])["m"]          # one model: Holm leaves p unchanged
        self.assertEqual(b["by_source"]["T1"]["n_sign_patterns"], 2 ** 14)
        self.assertEqual(b["by_base_dataset"]["T1"]["n_sign_patterns"], 2 ** 18)
        self.assertFalse(b["by_source"]["T1"]["rejects_at_5pct_after_holm"])
        self.assertTrue(b["by_base_dataset"]["T1"]["rejects_at_5pct_after_holm"])
        self.assertTrue(b["by_source"]["used_for_claims"])
        self.assertFalse(b["by_base_dataset"]["used_for_claims"])
        # source totals are sums of the base-dataset totals
        s_src = b["by_source"]["T1"]["group_sums"]
        s_bd = b["by_base_dataset"]["T1"]["group_sums"]
        self.assertAlmostEqual(s_src["ett"], s_bd["ett1"] + s_bd["ett2"], places=12)
        self.assertAlmostEqual(s_src["bizitobs"], sum(s_bd[k] for k in s_bd if k.startswith("bizitobs")), places=12)


class HolmAndT2(unittest.TestCase):
    def test_holm_hand_worked(self):
        # sorted: 0.005*4 = 0.02 (reject at 5%), 0.01*3 = 0.03 (reject), 0.03*2 = 0.06 (stop), 0.04*1 -> 0.06
        p = [0.01, 0.04, 0.03, 0.005]
        adj = A.holm_adjusted(p)
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(list(adj <= 0.05), [True, False, False, True])
        self.assertEqual(list(vs.holm(p, 0.05)), [True, False, False, True])
        np.testing.assert_allclose(A.holm_adjusted([0.01, 0.02]), [0.02, 0.02])
        np.testing.assert_allclose(A.holm_adjusted([0.3, 0.5]), [0.6, 0.6])
        np.testing.assert_allclose(A.holm_adjusted([0.9, 0.8, 0.01]), [1.0, 1.0, 0.03])
        np.testing.assert_allclose(A.holm_adjusted([0.001, 1.0, 1.0, 1.0]), [0.004, 1.0, 1.0, 1.0])

    def test_t2_gating_and_holm_over_all_four(self):
        cfg = REF["eligible70"]
        bases = sorted({A.base_of(c) for c in cfg})
        gi = np.array([bases.index(A.base_of(c)) for c in cfg])
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 0.01, size=len(cfg))
        weak = np.where(gi % 2 == 0, 0.5, -0.5) + noise
        d = {
            "pos_strong": 0.4 + noise,                                   # rejects, positive -> T2
            "neg_strong": -0.4 + noise,                                  # rejects, negative -> no T2
            "pos_weak": weak - weak.mean() + 0.01,                       # positive mean, T1 does not reject
        }
        blocks = family(d, cfg, ["pos_strong", "neg_strong", "pos_weak", "absent_model"])
        b = blocks["pos_strong"]["by_source"]
        self.assertEqual(b["T1"]["p_two_sided"], 2 / 2 ** 14)
        self.assertTrue(b["T1"]["rejects_at_5pct_after_holm"])
        self.assertEqual(b["T1"]["holm_family_missing_entered_as_p_1"], ["absent_model"])
        self.assertTrue(b["T2"]["computed"])
        self.assertEqual(b["T2"]["p_one_sided"], 1 / 2 ** 14)
        self.assertEqual(b["T2"]["holm_family"], ["pos_strong", "neg_strong", "pos_weak", "absent_model"])
        self.assertAlmostEqual(b["T2"]["p_holm"], 4 / 2 ** 14)       # the three others enter with p = 1
        self.assertTrue(b["T2"]["rejects_at_2_5pct_after_holm"])
        n = blocks["neg_strong"]["by_source"]
        self.assertTrue(n["T1"]["rejects_at_5pct_after_holm"])
        self.assertFalse(n["T2"]["computed"])
        self.assertEqual(n["T2"]["reason"], "the effect is not positive")
        self.assertEqual(n["T2"]["p_holm"], 1.0)
        w = blocks["pos_weak"]["by_source"]
        self.assertFalse(w["T1"]["rejects_at_5pct_after_holm"])
        self.assertFalse(w["T2"]["computed"])
        self.assertEqual(w["T2"]["reason"], "T1 does not reject after Holm")
        self.assertFalse(w["T2"]["rejects_at_2_5pct_after_holm"])

    def test_t2_holm_two_reach_one_gated_out(self):
        cfg = REF["eligible70"]
        rng = np.random.default_rng(9)
        d = {"a": 0.20 + rng.normal(0, 0.05, 70), "b": 0.14 + rng.normal(0, 0.08, 70),
             "gated": -0.3 + rng.normal(0, 0.01, 70), "c": 0.25 + rng.normal(0, 0.02, 70)}
        blocks = family(d, cfg, ["a", "b", "gated", "c"])
        t2 = {m: blocks[m]["by_source"]["T2"] for m in d}
        self.assertFalse(t2["gated"]["computed"])
        p = [t2[m]["p_one_sided"] if t2[m]["computed"] else 1.0 for m in ("a", "b", "gated", "c")]
        self.assertEqual(sum(x < 1.0 for x in p), 3)
        # hand-worked Holm on the four p-values, the gated model at p = 1
        order = sorted(range(4), key=lambda i: p[i])
        expect, run_ = [0.0] * 4, 0.0
        for k, i in enumerate(order):
            run_ = max(run_, min(1.0, (4 - k) * p[i]))
            expect[i] = run_
        for m, e in zip(("a", "b", "gated", "c"), expect):
            self.assertAlmostEqual(t2[m]["p_holm"], e, places=15)

    def test_t2_one_sided_against_brute_force_and_interval(self):
        cfg = REF["sel12"]
        rng = np.random.default_rng(11)
        seen = set()
        for mu in (0.12, 0.15, 0.2, 0.3, 0.5):
            d = mu + rng.normal(0, 0.1, size=12)
            blocks = family({"m": d}, cfg, ["m"])
            for g in A.GROUPINGS:
                t2 = blocks["m"][f"by_{g}"]["T2"]
                if not t2["computed"]:
                    continue
                names, s, n = A.group_sums(cfg, d, g)
                x = s - n * math.log1p(blocks["m"]["S"]["value"])
                np.testing.assert_allclose(list(t2["shifted_group_totals"].values()), x, rtol=0, atol=1e-12)
                self.assertEqual(t2["p_one_sided"], brute_force(x)[1])
                # T2 rejecting at 2.5% (before Holm) is the same as the interval's lower end above ln(1 + S)
                self.assertEqual(t2["p_one_sided"] <= 0.025, t2["interval_low_above_ln_1_plus_S"])
                seen.add(t2["p_one_sided"] <= 0.025)
        self.assertEqual(seen, {True, False})


class LevelCheck(unittest.TestCase):
    def test_port_reproduces_review_numbers(self):
        """Same draws as review_scratch/level_sim_t1t2.py (18 base datasets, first-round Toto, its seeds
        20261119 + 7 and + 8) give the review's rates."""
        with open(HERE / "review_scratch" / "level_sim_t1t2.json") as f:
            rv = {(r["scenario"], r["mode"]): r for r in json.load(f)["results"]}
        bases = sorted({A.base_of(c) for c in REF["eligible70"]})
        sizes = [sum(A.base_of(c) == b for c in REF["eligible70"]) for b in bases]
        used, d, _ = A.paired_logs(REF["first"]["toto"]["default"], REF["first"]["toto"]["512"], REF["sel12"])
        for mode, rmode, k in (("independent", "indep", 7), ("shared", "shared", 8)):
            lc = A.level_check(d, sizes, direction=1, draws=20000, seed=A.LEVEL_SEED + k)
            r = rv[("centred_toto", rmode)]
            self.assertAlmostEqual(lc[mode]["two_sided_pct"], r["T1_two_sided"], places=6)
            self.assertAlmostEqual(lc[mode]["positive_pct"], r["T1_reject_positive"], places=6)
            self.assertAlmostEqual(lc[mode]["negative_pct"], r["T1_reject_negative"], places=6)
            self.assertEqual(lc[mode]["in_direction_of_observed_effect_pct"], lc[mode]["positive_pct"])

    def test_direction_and_degenerate_pool(self):
        lc = A.level_check([0.1, -0.2, 0.3, 0.0], [2, 1, 1], direction=-1, draws=300)
        self.assertEqual(lc["shared"]["in_direction_of_observed_effect_pct"], lc["shared"]["negative_pct"])
        self.assertEqual(lc["observed_direction"], "negative")
        z = A.level_check([0.2, 0.2, 0.2], [3, 2], direction=1, draws=100)   # centred: all zero, no claim possible
        self.assertEqual(z["shared"]["two_sided_pct"], 0.0)
        self.assertEqual(z["independent"]["two_sided_pct"], 0.0)

    def test_level_check_in_report_uses_14_and_9_sources(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            make_first_round_layout(tmp / "res")
            for m in A.G2_MODELS:
                for L, fac in (("default", 1.0), ("512", 1.1)):
                    write_results(tmp / "res", m, L, {c: REF["pub"][m][c] * fac for c in REF["sel12"]})
            out = run(tmp / "res", tmp / "out")
        finally:
            shutil.rmtree(tmp)
        lc = out["G1"]["eligible_70"]["models"]["timesfm25"]["level_check"]
        self.assertEqual(sorted(lc["group_sizes"], reverse=True), [12, 12, 7, 6, 6, 6, 6, 3, 3, 3, 2, 2, 1, 1])
        self.assertEqual(lc["draws"], 200)
        self.assertEqual(lc["seed"], 20261119)
        lc2 = out["G2"]["tests_on_12"]["models"]["timesfm3"]["level_check"]
        self.assertEqual(len(lc2["group_sizes"]), 9)
        self.assertNotIn("level_check", out["G1"]["first_12"]["models"]["timesfm25"])


class MachineCheck(TempDir):
    def _bump(self, rel):
        """Scale one Chronos-Bolt documented-length MASE (a configuration whose d_c is far from zero)."""
        make_first_round_layout(self.tmp / "res")
        used, d, _ = A.paired_logs(REF["first"]["chronos_bolt_base"]["default"],
                                   REF["first"]["chronos_bolt_base"]["512"], REF["sel12"])
        c = used[int(np.argmax(np.abs(d)))]
        self.assertGreater(abs(d[used.index(c)]), 0.01)

        def change(rows):
            for r in rows:
                if r["dataset"] == c:
                    r[A.MASE_COL] = repr(float(r[A.MASE_COL]) * (1 + rel))
            return rows
        rewrite(self.tmp / "res" / "chronos_bolt_base" / "ctx_default" / "all_results.csv", change)
        out = run(self.tmp / "res", self.tmp / "out")
        return out, (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()

    def test_pass_below_1e_4(self):
        out, _ = self._bump(5e-5)
        self.assertEqual(out["machine_check"]["verdict"], "PASS")
        self.assertEqual(out["G1"]["status"], "RUN")

    def test_continue_between(self):
        out, md = self._bump(5e-4)
        self.assertEqual(out["machine_check"]["verdict"], "CONTINUE")
        self.assertEqual(out["G1"]["status"], "RUN")                 # CONTINUE does not hold the rest
        self.assertIn("run one configuration twice on one card", md)

    def test_stop_above_1e_3_holds_the_rest(self):
        out, md = self._bump(2e-3)
        self.assertEqual(out["machine_check"]["verdict"], "STOP")
        for part in ("G1", "G2", "G3", "G4"):
            self.assertEqual(out[part]["status"], "NOT RUN")
        out = run(self.tmp / "res", self.tmp / "out2", "--analyse-despite-machine-check")
        self.assertEqual(out["G1"]["status"], "RUN")

    def test_stop_on_sign_change_alone(self):
        sel = REF["sel12"]
        first = {m: {"default": {c: 1.0 for c in sel}, "512": {c: 1.00004 for c in sel}} for m in A.DETERMINISTIC}
        res = {m: {"default": dict(first[m]["default"]), "512": dict(first[m]["512"])} for m in A.DETERMINISTIC}
        res["moirai2_small"]["default"][sel[0]] = 1.00004
        res["moirai2_small"]["512"][sel[0]] = 1.0                   # d_c flips sign; each value moves 4e-5
        mc = A.machine_check(res, first, sel)
        self.assertLess(mc["largest_relative_difference"], 1e-4)
        self.assertEqual(mc["verdict"], "STOP")
        self.assertEqual([x["config"] for x in mc["d_c_sign_changes"]], [sel[0]])
        mc = A.machine_check(first, first, sel)
        self.assertEqual(mc["verdict"], "PASS")


class G3Readings(TempDir):
    def _layout(self, factors, absent_seed=None):
        make_first_round_layout(self.tmp / "res")
        shutil.rmtree(self.tmp / "res" / "toto__A__seed42")
        for (seed, v), f in zip(A.TOTO_SEEDS, factors):
            if seed == absent_seed:
                continue
            for L in A.LENGTHS:
                write_results(self.tmp / "res", v, L, {c: REF["pub"]["toto"][c] * f * (1.15 if L == "512" else 1.0)
                                                       for c in REF["sel12"]})
        out = run(self.tmp / "res", self.tmp / "out")
        return out, (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()

    def test_both_readings_pass(self):
        out, md = self._layout((1.005, 1.02, 1.03))
        a = out["G3"]["reading_a_within_1pct"]["toto__A__seed42"]
        b = out["G3"]["reading_b_within_noise"]
        self.assertAlmostEqual(a["relative_gap"], 0.005, places=12)
        self.assertEqual(a["verdict"], "PASS")
        board = out["G3"]["reference"]["board_gmean_scaled_on_12"]
        np.testing.assert_allclose(b["range"], [1.005 * board, 1.03 * board], rtol=1e-12)
        np.testing.assert_allclose(b["widened_range"], [0.9925 * board, 1.0425 * board], rtol=1e-12)
        self.assertTrue(b["inside"])
        self.assertEqual(b["verdict"], "PASS")
        self.assertIn("the notebook's setting reproduces the board", md)

    def test_noise_reading_passes_when_1pct_fails(self):
        out, _ = self._layout((1.02, 0.97, 1.04))      # gap 2%; range 0.97..1.04 widened to 0.935..1.075
        self.assertEqual(out["G3"]["reading_a_within_1pct"]["toto__A__seed42"]["verdict"], "FAIL")
        self.assertEqual(out["G3"]["reading_b_within_noise"]["verdict"], "PASS")

    def test_both_readings_fail(self):
        out, md = self._layout((1.03, 1.035, 1.04))    # range 1.03..1.04 widened to 1.025..1.045: board outside
        self.assertEqual(out["G3"]["reading_a_within_1pct"]["toto__A__seed42"]["verdict"], "FAIL")
        self.assertEqual(out["G3"]["reading_b_within_noise"]["verdict"], "FAIL")
        self.assertIn("Setting A misses both readings. DESIGN_ROUND3.md then runs one variant on the 12 "
                      "configurations: setting A with the faster matrix products off.", md)
        self.assertEqual(out["G3"]["reading_a_within_1pct"][A.TOTO_FALLBACK]["status"], "NOT RUN")
        self.assertEqual(set(out["G3"]["reading_a_within_1pct"]), {"toto__A__seed42", "toto__A_tf32off__seed42"})

    def test_noise_reading_not_run_without_a_seed(self):
        out, _ = self._layout((1.0, 1.0, 1.0), absent_seed="44")
        self.assertEqual(out["G3"]["reading_b_within_noise"]["status"], "NOT RUN")
        self.assertIn("44", out["G3"]["reading_b_within_noise"]["reason"])
        self.assertEqual(out["G3"]["reading_a_within_1pct"]["toto__A__seed42"]["verdict"], "PASS")


class MissingAndOddValues(TempDir):
    def test_missing_at_one_length_dropped_for_that_model_only(self):
        make_first_round_layout(self.tmp / "res")
        rewrite(self.tmp / "res" / "timesfm25" / "ctx_512" / "all_results.csv",
                lambda rows: [r for r in rows if r["dataset"] != "ett1/H/medium"])
        out = run(self.tmp / "res", self.tmp / "out")
        fam = out["G1"]["first_12"]["models"]
        self.assertEqual(fam["timesfm25"]["n_configurations"], 11)
        self.assertEqual(fam["timesfm25"]["dropped"], [{"config": "ett1/H/medium", "missing_at": ["512"]}])
        self.assertNotIn("ett1/H/medium", fam["timesfm25"]["d_c"])
        self.assertEqual(fam["timesfm25"]["S"]["configurations"], 11)
        for m in ("chronos_bolt_base", "moirai2_small", "toto__A__seed42"):
            self.assertEqual(fam[m]["n_configurations"], 12)
            self.assertEqual(fam[m]["dropped"], [])
        d70 = out["G1"]["eligible_70"]["models"]["timesfm25"]["dropped"]
        self.assertIn({"config": "ett1/H/medium", "missing_at": ["512"]}, d70)
        cc = out["G1"]["first_12"]["common_configurations"]
        self.assertEqual(cc["n_configurations"], 11)
        self.assertFalse(cc["same_as_every_model_used"])
        self.assertEqual(cc["models"]["chronos_bolt_base"]["n_configurations"], 11)
        self.assertEqual(out["machine_check"]["verdict"], "PASS")    # on the values present
        self.assertFalse(out["machine_check"]["complete"])
        md = (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        self.assertIn("ett1/H/medium (missing at 512)", md)
        self.assertIn("Every model on the 11 configurations all of them finished", md)

    def test_non_finite_rows_are_missing_and_named(self):
        make_first_round_layout(self.tmp / "res")
        bad = {}

        def change(rows):
            for r, v in zip(rows, ("nan", "inf", "", "-inf")):
                bad[r["dataset"]] = v
                r[A.MASE_COL] = v
            return rows
        p = self.tmp / "res" / "moirai2_small" / "ctx_512" / "all_results.csv"
        rewrite(p, change)
        out = run(self.tmp / "res", self.tmp / "out")
        b = out["G1"]["first_12"]["models"]["moirai2_small"]
        self.assertEqual(sorted(x["config"] for x in b["dropped"]), sorted(bad))
        self.assertTrue(all(x["missing_at"] == ["512"] for x in b["dropped"]))
        probs = out["files_read"][str(p.resolve())]["problems"]
        self.assertEqual(len(probs), 4)
        md = (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        self.assertIn("Problems found in the result files", md)

    def test_m4_daily_rule(self):
        make_first_round_layout(self.tmp / "res", seeds=("42", "43"))
        drop_m4 = lambda rows: [r for r in rows if r["dataset"] != A.M4_RULE_CONFIG]  # noqa: E731
        # seed 42: no 512 row (the run is skipped by rule) -> d_c = 0, configuration kept
        rewrite(self.tmp / "res" / "toto__A__seed42" / "ctx_512" / "all_results.csv", drop_m4)
        # seed 43: a 512 row with a different value -> replaced by the documented value
        rewrite(self.tmp / "res" / "toto__A__seed43" / "ctx_512" / "all_results.csv",
                lambda rows: [dict(r, **{A.MASE_COL: "9.0"}) if r["dataset"] == A.M4_RULE_CONFIG else r for r in rows])
        # Chronos-Bolt: the rule does not apply -> configuration dropped
        rewrite(self.tmp / "res" / "chronos_bolt_base" / "ctx_512" / "all_results.csv", drop_m4)
        out = run(self.tmp / "res", self.tmp / "out")
        toto = out["G1"]["first_12"]["models"]["toto__A__seed42"]
        self.assertEqual(toto["n_configurations"], 12)
        self.assertEqual(toto["d_c"][A.M4_RULE_CONFIG], 0.0)
        self.assertEqual(out["m4_daily_rule"]["toto__A__seed42"], "applied")
        self.assertEqual(out["m4_daily_rule"]["toto__A__seed43"], "applied; a 512 row was present and was replaced")
        self.assertEqual(out["G4"]["seeds"]["43"]["configurations_both_lengths"], 12)
        self.assertAlmostEqual(out["G4"]["seeds"]["43"]["effect_pct"], out["G4"]["seeds"]["42"]["effect_pct"], places=10)
        self.assertEqual(out["m4_daily_rule"]["toto__A__seed44"], "not applied: the 512 results file is absent")
        cb = out["G1"]["first_12"]["models"]["chronos_bolt_base"]
        self.assertEqual(cb["dropped"], [{"config": A.M4_RULE_CONFIG, "missing_at": ["512"]}])
        # Toto with the documented value missing: the configuration is missing
        rewrite(self.tmp / "res" / "toto__A__seed42" / "ctx_default" / "all_results.csv", drop_m4)
        out = run(self.tmp / "res", self.tmp / "out2")
        toto = out["G1"]["first_12"]["models"]["toto__A__seed42"]
        self.assertEqual(toto["dropped"], [{"config": A.M4_RULE_CONFIG, "missing_at": ["documented", "512"]}])


class RunRecords(TempDir):
    def test_run_records_summary(self):
        make_first_round_layout(self.tmp / "res")
        rev_toto = sorted(REF["first_round_revisions"]["toto"])[0]
        pc = self.tmp / "res" / "toto__A__seed42" / "ctx_default" / "per_config"
        pc.mkdir()
        base = {"status": "ok", "gpu_name": "NVIDIA A100-SXM4-80GB", "torch_version": "2.7.0",
                "model_revision": rev_toto, "float32_matmul_precision": "high", "deterministic_algorithms": True,
                "samples_per_batch_final": 256, "toto_batch_size": 16, "wall_time_seconds": 1800.0,
                "peak_gpu_memory_gb": 30.5, "torch_cuda_version": "12.6", "run_grid_sha256": "abc",
                "finished_at": "2026-11-20T01:00:00Z"}
        recs = [base, dict(base, xformers_installed=True, samples_per_batch_final=128, peak_gpu_memory_gb=42.0),
                dict(base, model_revision="0000000", finished_at="2026-11-20T07:00:00Z", wall_time_seconds=1200.0)]
        for i, r in enumerate(recs):
            (pc / f"cfg{i}__short.run_meta.json").write_text(json.dumps(r))
        (pc / "cfg9__short.failed.run_meta.json").write_text(json.dumps({"status": "failed", "error": "x"}))
        (pc / "cfg8__short.run_meta.json").write_text("{not json")
        pc2 = self.tmp / "res" / "chronos_bolt_base" / "ctx_512" / "per_config"
        pc2.mkdir()
        (pc2 / "a__short.run_meta.json").write_text(json.dumps({"status": "ok"}))   # every field missing
        out = run(self.tmp / "res", self.tmp / "out")
        rr = out["run_records"]
        self.assertEqual(rr["status"], "RUN")
        t = rr["variants"]["toto__A__seed42"]
        self.assertEqual((t["files"], t["runs_ok"], t["runs_failed"], len(t["unreadable"])), (5, 3, 1, 1))
        self.assertTrue(t["xformers_installed_flag"])
        self.assertTrue(t["hf_revision_differs_from_first_round"])
        self.assertEqual(t["samples_per_batch_final"], {"256": 2, "128": 1, "not recorded": 1})
        self.assertEqual(t["peak_memory"]["peak_gpu_memory_gb"]["max"], 42.0)
        self.assertEqual(t["cuda_version"], {"12.6": 3, "not recorded": 1})
        self.assertEqual(t["run_grid_checksum"], {"abc": 3, "not recorded": 1})
        self.assertEqual(t["wall_time_seconds_total"], 4800.0)
        self.assertEqual([x["finished_at"] for x in t["runs_finished_after_freeze"]], ["2026-11-20T07:00:00Z"])
        c = rr["variants"]["chronos_bolt_base"]
        self.assertEqual(c["card"], {"not recorded": 1})
        self.assertEqual(c["wall_time_seconds_total"], "not recorded")
        self.assertEqual(c["peak_memory"], "not recorded")
        self.assertFalse(c["hf_revision_differs_from_first_round"])
        md = (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        self.assertIn("FLAG: must not be installed", md)
        self.assertIn("DIFFERS FROM THE FIRST ROUND", md)
        self.assertIn("finished after the freeze (1)", md)


class EndToEnd(TempDir):
    def test_full_synthetic_layout_runs_every_part(self):
        """Made-up values for every variant on all 70 configurations (the 12 of the three deterministic
        models copied from the first round so the machine check passes). Checks plumbing only."""
        rng = np.random.default_rng(5)
        root = self.tmp / "res"
        for v in A.ALL_VARIANTS:
            src = {"timesfm3": REF["pub"]["timesfm3"], "timestral209m": REF["pub"]["timestral209m"]}.get(v)
            cfgs = REF["sel12"] if v in A.G2_MODELS or v == A.TOTO_FALLBACK else REF["eligible70"]
            for L in A.LENGTHS:
                if v == A.TOTO_FALLBACK and L == "512":
                    continue
                fac = 1.0 if L == "default" else 1.12
                vals = {}
                for c in cfgs:
                    if v.startswith("toto") and L == "512" and c == A.M4_RULE_CONFIG:
                        continue                                          # skipped by rule
                    if v in A.DETERMINISTIC and c in REF["sel12"]:
                        vals[c] = REF["first"][v][L][c]
                    elif src is not None:
                        vals[c] = src[c] * fac * math.exp(rng.normal(0, 0.003))
                    else:
                        vals[c] = REF["pub"]["toto"][c] * fac * math.exp(rng.normal(0, 0.1))
                write_results(root, v, L, vals)
        out = run(root, self.tmp / "out")
        self.assertEqual(out["machine_check"]["verdict"], "PASS")
        for part in ("G3", "G4", "G1", "G2"):
            self.assertEqual(out[part]["status"], "RUN", part)
        for key, n in (("eligible_70", 70), ("new_58", 58), ("first_12", 12)):
            for m in A.G1_MODELS:
                self.assertEqual(out["G1"][key]["models"][m]["n_configurations"], n)
        t = out["G1"]["eligible_70"]["models"]["toto__A__seed42"]
        self.assertEqual((t["n_sources"], t["n_base_datasets"]), (14, 18))
        self.assertEqual(t["by_source"]["T1"]["n_sign_patterns"], 2 ** 14)
        self.assertEqual(round(t["S"]["pct"], 2), 11.76)
        self.assertEqual(round(out["G1"]["new_58"]["models"]["toto__A__seed42"]["S"]["pct"], 2), 12.71)
        self.assertEqual(out["G2"]["reproduction"]["timesfm3"]["verdict"], "PASS")
        self.assertEqual(out["G3"]["reading_b_within_noise"]["status"], "RUN")
        self.assertIsNotNone(out["G4"]["range_over_seeds"]["effect_pct"]["range"])
        self.assertEqual(out["G1"]["confirmation_on_58"]["overlap"]["from_a_base_dataset_among_the_12"], 43)
        md = (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        self.assertIn("Of the 58, 43 come from a base dataset that is already among the 12 and 20 use the same "
                      "series as one of the 12 at another forecast horizon.", md)
        self.assertIn("on the 58 after Holm over the four models", md)
        self.assertIn("The same tests on base datasets, not used for any claim", md)
        if KEEP:
            shutil.copytree(self.tmp / "out", Path(KEEP), dirs_exist_ok=True)

    def test_empty_root_runs_to_the_end(self):
        (self.tmp / "empty").mkdir()
        r = subprocess.run([sys.executable, "-B", str(HERE / "analyze_gpu_round.py"), "--results",
                            str(self.tmp / "empty"), "--out", str(self.tmp / "out")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.tmp / "out" / "GPU_ROUND_RESULTS.json") as f:
            out = json.load(f)
        for part in ("machine_check", "G3", "G4", "G1", "G2", "run_records"):
            self.assertEqual(out[part]["status"], "NOT RUN", part)
            self.assertTrue(out[part]["reason"])
        md = (self.tmp / "out" / "GPU_ROUND_RESULTS.md").read_text()
        self.assertGreaterEqual(md.count("Not run"), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
