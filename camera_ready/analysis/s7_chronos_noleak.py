"""Usage: s7_chronos_noleak.py report <lib|drop> <wr|rank|gm> | replay | check | assemble.  Seed 20261111 (report_from_matrix default).
Parts cached in boards/s7_parts/."""
import sys, os, json, time
import numpy as np
from s7_common import *

PARTS = os.path.join(HERE, "boards", "s7_parts"); os.makedirs(PARTS, exist_ok=True)
SEED = 20261111


def order(X):
    return af.order_of(X.mean(1))


def do_report(v, rule):
    cb, srcm = data()
    models, tasks, M, W = variant(cb, v)
    X = rules(models, M, W)[rule]
    cl = np.unique([srcm[t] for t in tasks], return_inverse=True)[1]
    r = vb.report_from_matrix(X, cl, models, order(X), seed=SEED)
    r["tasks"] = tasks
    r["known_shift"] and r["known_shift"].pop("per_pair", None)
    return r


def do_check():
    """Library's own leaderboard with leakage_imputation_model vs my replacement (win rate order)."""
    cb, srcm = data()
    df = af.load_dir(af.C_DIR)
    lb = af.fev.analysis.leaderboard(df, metric_column="test_error", missing_strategy="error", baseline_model=af.BASE_C,
                                     task_columns="task_name", leakage_imputation_model=LEAKREF).sort_values("win_rate", ascending=False)
    models, tasks, M, W = variant(cb, "lib")
    wr = pd_series(models, rules(models, M, W)["wr"])
    own = wr.sort_values(kind="stable")
    d = float(np.abs(lb["win_rate"].reindex(own.index).to_numpy() - (-own.to_numpy())).max())
    return dict(order_equal=list(lb.index) == list(own.index), max_abs_diff_win_rate=d, lib_top10=list(lb.index[:10]))


def pd_series(models, X):
    import pandas as pd
    return pd.Series(X.mean(1), index=models)


def replay_states(only_tasks=None):
    H = af.History(af.C_DIR, "C")
    out = []
    for c, date in af.commits_of(af.C_DIR):
        files, units, _, _ = H.snapshot(c)
        dfs = {m: H.csv(o) for m, o in files.items()}
        import pandas as pd
        allrows = pd.concat(dfs.values(), ignore_index=True)
        if units is None:
            units = sorted(allrows.task_name.unique())
        if only_tasks is not None:
            units = [t for t in units if t in only_tasks]
        ix = {t: i for i, t in enumerate(units)}
        T = len(units); ents = {}
        for stem, oid in files.items():
            df = dfs[stem]
            for m in df.model_name.unique():
                sub = df[(df.model_name == m) & df.task_name.isin(units)]
                v, lk = np.full(T, np.nan), np.zeros(T)
                for t, x, l in zip(sub.task_name, sub.test_error, sub.trained_on_this_dataset):
                    v[ix[t]] = x; lk[ix[t]] = float(bool(l))
                if np.isnan(v).any():
                    continue
                ents[m] = (np.concatenate([v, lk]), oid, 0)
        out.append(dict(date=date, commit=c[:7], entries=ents, units=tuple(units)))
    return out


def make_rule(kind, mode):
    mr = vb.mean_rank_rule("average")
    def rule(S, names):
        S = np.asarray(S, float); T = S.shape[1] // 2
        M, L = S[:, :T], S[:, T:] > 0.5
        if af.BASE_C not in names:
            return None
        if mode == "lib":
            if LEAKREF not in names:
                return None
            M = np.where(L, M[names.index(LEAKREF)][None, :], M)
        elif mode == "drop":
            M = M[:, ~L.any(0)]
            if M.shape[1] == 0:
                return None
        if kind == "rank":
            return mr(M)
        E = af.clip_rel(M / M[names.index(af.BASE_C)][None, :])
        return af.wr_rule(E) if kind == "wr" else af.gm_rule(E)
    return rule


def do_replay():
    states = replay_states()
    cb, _ = data()
    keep = set(np.array(cb["tasks"])[~cb["leak"].any(0)])  # tasks with no flagged cell at the current state, fixed for every state
    states_drop = replay_states(keep)
    res = dict(drop_tasks_kept=sorted(keep))
    for mode in ("orig", "lib", "drop"):
        for kind in ("wr", "rank", "gm"):
            h = af.replay(states_drop if mode == "drop" else states, make_rule(kind, "orig" if mode == "drop" else mode))
            res[f"{mode}_{kind}"] = dict(transitions=h["transitions"], with_reversal=h["transitions_with_reversal"], pairs=h["reversed_pairs"],
                                         events=h["events"], skipped=h["transition_accounting"]["skipped"],
                                         flagged_cells_by_state=[int(sum(v[0][len(st["units"]):].sum() for v in st["entries"].values())) for st in (states_drop if mode == "drop" else states)], n_entries_by_state=[len(st["entries"]) for st in (states_drop if mode == "drop" else states)], dates=[st["date"] for st in states])
            print(mode, kind, h["transitions"], h["transitions_with_reversal"], h["reversed_pairs"], flush=True)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.time()
    if cmd == "report":
        key = f"report_{sys.argv[2]}_{sys.argv[3]}"; fn = lambda: do_report(sys.argv[2], sys.argv[3])
    elif cmd == "replay":
        key, fn = "replay", do_replay
    elif cmd == "check":
        key, fn = "check", do_check
    p = os.path.join(PARTS, key + ".json")
    if not os.path.exists(p):
        r = fn(); r["runtime_s"] = round(time.time() - t0, 1); af.jdump(r, p)
    print(key, "done", round(time.time() - t0), "s")
