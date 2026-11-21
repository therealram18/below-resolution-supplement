"""S7 check A helpers: Chronos Benchmark II without trained-on cells. Imports first-round code, changes nothing there."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_fev_family as af
import vernier_board as vb

LEAKREF = "chronos_bolt_base"  # zero-shot Chronos-Bolt, as the fev-bench script uses Chronos-Bolt as replacement model


def data():
    cb = af.board_c()
    srcm = af.write_c_sources()
    return cb, srcm


def variant(cb, v):
    """Returns models, tasks, mase, wql (models x tasks) for 'lib' (replace flagged with LEAKREF), 'drop' (drop tasks with any flag), 'orig'."""
    M, W, L = cb["mase"].copy(), cb["wql"].copy(), cb["leak"]
    tasks = np.array(cb["tasks"])
    if v == "lib":
        r = cb["models"].index(LEAKREF)
        assert not L[r].any()
        M = np.where(L, M[r][None, :], M)
        W = np.where(L, W[r][None, :], W)
    elif v == "drop":
        keep = ~L.any(0)
        M, W, tasks = M[:, keep], W[:, keep], tasks[keep]
    return cb["models"], list(tasks), M, W


def rules(models, M, W):
    bi = models.index(af.BASE_C)
    relM = af.clip_rel(M / M[bi]); relW = af.clip_rel(W / W[bi])
    mr = vb.mean_rank_rule("average")
    return {"wr": -af.winrate_tasks(relM), "rank": np.stack([mr(M[:, [j]]) for j in range(M.shape[1])], 1), "gm": np.log(relM)}
