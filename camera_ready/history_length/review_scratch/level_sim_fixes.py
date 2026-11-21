"""Review scratch (round 3), follow-up to level_sim_t1t2.py: two candidate repairs
for T1's level under skewed nulls, in the same 18-group layout and the same null
shapes (first-round d_c per model, centred; and the mirror image).

 1. studentized exact sign-flip test (CR1 t statistic over all sign patterns;
    camera_ready/exact_signflip.exact_student_p, imported, not edited);
 2. the sum test with a stricter nominal level, chosen now so that the worst
    one-directional false rejection rate over these shapes is at most 2.5%.

Reports, per scenario, the two-sided rejection rate at 5% and the rate in each
direction, for both statistics; then the calibrated nominal level for the sum
test. Run: harness/envs/chronos_bolt_base/bin/python review_scratch/level_sim_fixes.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = Path("fmts")
sys.path.insert(0, str(R / "camera_ready"))
sys.path.insert(0, str(HERE))
from exact_signflip import exact_student_p  # noqa: E402
from level_sim_t1t2 import D, GID, MODELS, SIZES, draw, sign_flip_counts  # noqa: E402

REPS = 10000
SEED = 20261120


def run(args):
    name, pool, mode, seed = args
    rng = np.random.default_rng(seed)
    out = np.zeros((REPS, 4))  # p_sum_two, p_sum_upper, p_stud_two, sign
    for r in range(REPS):
        d = draw(rng, pool, mode)
        s = np.bincount(GID, weights=d, minlength=18)
        p2, pu = sign_flip_counts(s)
        out[r] = (p2, pu, exact_student_p(s, SIZES), np.sign(s.sum()))
    return name, mode, out


def main():
    jobs = []
    k = 0
    rng0 = np.random.default_rng(SEED)
    sym = rng0.standard_t(3, size=200000) * 0.1
    sym = np.concatenate([sym, -sym])
    jobs.append(("symmetric_t3", sym, "shared", SEED + k)); k += 1
    for name in MODELS + ["pooled48"]:
        pool = D[name] - D[name].mean()
        for mode in ("shared", "half", "indep"):
            jobs.append((f"centred_{name}", pool, mode, SEED + k)); k += 1
            jobs.append((f"mirrored_{name}", -pool, mode, SEED + k)); k += 1
    with Pool(min(10, os.cpu_count() or 2)) as p:
        res = p.map(run, jobs)
    rows = []
    print(f"{'scenario':28s} {'mode':6s} | sum: {'2s%':>5s} {'+%':>5s} {'-%':>5s} | stud: {'2s%':>5s} {'+%':>5s} {'-%':>5s}")
    for name, mode, o in res:
        p2, pu, ps, sg = o.T
        row = dict(scenario=name, mode=mode,
                   sum_two=100 * np.mean(p2 <= .05), sum_pos=100 * np.mean((p2 <= .05) & (sg > 0)),
                   sum_neg=100 * np.mean((p2 <= .05) & (sg < 0)),
                   stud_two=100 * np.mean(ps <= .05), stud_pos=100 * np.mean((ps <= .05) & (sg > 0)),
                   stud_neg=100 * np.mean((ps <= .05) & (sg < 0)))
        rows.append(row)
        print(f"{name:28s} {mode:6s} | {row['sum_two']:9.2f} {row['sum_pos']:5.2f} {row['sum_neg']:5.2f} | "
              f"{row['stud_two']:10.2f} {row['stud_pos']:5.2f} {row['stud_neg']:5.2f}")
    # calibrated nominal two-sided level for the sum test: largest a on a grid with
    # max over scenarios of the per-direction rate <= 2.5%
    grid = np.round(np.arange(0.002, 0.0501, 0.001), 3)
    cal = {}
    for a in grid:
        worst = 0.0
        for name, mode, o in res:
            p2, pu, ps, sg = o.T
            worst = max(worst, np.mean((p2 <= a) & (sg > 0)), np.mean((p2 <= a) & (sg < 0)))
        cal[float(a)] = 100 * worst
    ok = [a for a in grid if cal[float(a)] <= 2.5]
    a_star = float(max(ok)) if ok else None
    print("worst one-direction rate of the sum test by nominal two-sided level:",
          {a: round(v, 2) for a, v in cal.items() if a in (0.01, 0.02, 0.025, 0.03, 0.04, 0.05)})
    print("largest nominal two-sided level with worst one-direction rate <= 2.5%:", a_star)
    Path(__file__).with_suffix(".json").write_text(json.dumps(
        {"reps": REPS, "seed": SEED, "rows": rows, "calibration_worst_one_direction_pct": cal,
         "calibrated_two_sided_level": a_star}, indent=1))


if __name__ == "__main__":
    main()
