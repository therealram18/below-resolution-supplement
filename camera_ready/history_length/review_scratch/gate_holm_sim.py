"""Review scratch (round 3): error rate of the T1 -> T2 gate with Holm over the
models that reach T2 (DESIGN_ROUND3.md lines 85-95), against Holm over all four
models with p = 1 for a model that does not reach T2.

Synthetic, symmetric data (so every single test is exact and only the multiplicity
rule is under study). Real layout: 18 base datasets, sizes from the 70.
Four models; dataset effects correlated across models (rho) because all models are
scored on the same series. delta0 = ln(1 + S) with S = 11.76% (S on the 70,
review_scratch/spread_S.json).

Scenarios (true mean effect per model):
  A: (d0, d0, 0, 0)  two models exactly at the T2 boundary, two with no effect
  B: (d0, d0, d0, d0)
  C: (d0, 0.5 d0, 0, 0)
Reported: share of replicates with at least one false T2 claim (a model whose true
effect is <= d0 declared above S), and with at least one false claim of either kind.
Run: harness/envs/chronos_bolt_base/bin/python review_scratch/gate_holm_sim.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from level_sim_t1t2 import SIZES, sign_flip_counts  # noqa: E402

D0 = float(np.log1p(0.1176))
REPS = 20000
SEED = 20261121
N = SIZES.sum()


def holm(p, alpha):
    p = np.asarray(p, float)
    order = np.argsort(p)
    rej = np.zeros(len(p), bool)
    k = len(p)
    for i, j in enumerate(order):
        if p[j] <= alpha / (k - i):
            rej[j] = True
        else:
            break
    return rej


def one(mu, sigma, rho, rng):
    z = rng.standard_normal(18)
    s = np.empty((4, 18))
    for m in range(4):
        u = rng.standard_normal(18)
        s[m] = SIZES * mu[m] + sigma * np.sqrt(SIZES) * (np.sqrt(rho) * z + np.sqrt(1 - rho) * u)
    p1 = np.empty(4); sgn = np.empty(4); p2 = np.empty(4)
    for m in range(4):
        p1[m], _ = sign_flip_counts(s[m])
        sgn[m] = np.sign(s[m].sum())
        _, p2[m] = sign_flip_counts(s[m] - SIZES * D0)
    r1 = holm(p1, 0.05)
    gate = r1 & (sgn > 0)
    # plan: Holm over the models that pass the gate
    idx = np.where(gate)[0]
    r2_plan = np.zeros(4, bool)
    if len(idx):
        r2_plan[idx] = holm(p2[idx], 0.025)
    # alternative: Holm over all four, p = 1 for models that do not pass the gate
    r2_fixed = holm(np.where(gate, p2, 1.0), 0.025)
    return r1, sgn, r2_plan, r2_fixed


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for label, mu in (("A", [D0, D0, 0, 0]), ("B", [D0] * 4), ("C", [D0, D0 / 2, 0, 0])):
        for sigma in (0.25, 0.37, 0.5):
            rho = 0.5
            f2p = f2f = fany = 0
            pw1 = 0
            for _ in range(REPS):
                r1, sgn, r2p, r2f = one(mu, sigma, rho, rng)
                true_h2 = np.array(mu) <= D0 + 1e-12
                true_h1 = np.array(mu) == 0
                e2p = (r2p & true_h2).any(); e2f = (r2f & true_h2).any()
                e1 = (r1 & true_h1).any()
                f2p += e2p; f2f += e2f; fany += (e1 or e2p)
                pw1 += r1[0]
            key = f"{label}_sigma{sigma}"
            out[key] = dict(mu=mu, sigma=sigma, rho=rho,
                            T1_power_model1_pct=100 * pw1 / REPS,
                            false_T2_plan_pct=100 * f2p / REPS,
                            false_T2_fixed4_pct=100 * f2f / REPS,
                            any_false_claim_plan_pct=100 * fany / REPS)
            print(key, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[key].items() if k != "mu"})
    Path(__file__).with_suffix(".json").write_text(json.dumps({"reps": REPS, "seed": SEED, "delta0": D0, "results": out}, indent=1))


if __name__ == "__main__":
    main()
