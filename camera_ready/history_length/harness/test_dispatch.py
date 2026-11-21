#!/usr/bin/env python3
"""Tests for dispatch.py with a fake job command (DISPATCH_FAKE_CMD), no GPU,
no model. Standard library only:  python3 test_dispatch.py

The fake job is this file run as `test_dispatch.py --fake-run <run_grid args>`.
It reads the job from DISPATCH_JOB_JSON, whose extra test fields say how long
to sleep (fake_sleep), on which attempts to fail (fake_fail_attempts), and
whether to leave a child process behind (fake_child). It records start and
end, card, attempt, environment and arguments in the file TEST_TRACE, and on
success writes the result csv where run_grid.py would (DISPATCH_RESULT_CSV).

Covered: an exclusive job never shares a card; at most --slots-per-gpu jobs
per card; a failing job is retried once, alone on a card; a job over the
timeout is killed with its process group and marked; an existing result is
skipped; a second dispatcher on other cards does not clash (and a third one
asking for a card in use is refused); STOP file and low-disk guard; dry run
on the real job files. Revision 1: per-job timeout_min, --no-exclusive,
CUBLAS_WORKSPACE_CONFIG for Toto sampling jobs only.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
TMP_DIRS = []  # removed at the end when every test passed


def mkdtemp(prefix):
    d = Path(tempfile.mkdtemp(prefix=prefix))
    TMP_DIRS.append(d)
    return d


# ---------------------------------------------------------------- fake job
def fake_run(argv):
    job = json.loads(os.environ["DISPATCH_JOB_JSON"])
    attempt = int(os.environ["DISPATCH_ATTEMPT"])
    trace = os.environ["TEST_TRACE"]
    rec = {
        "id": job["id"], "attempt": attempt, "card": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(), "pgid": os.getpgid(0), "argv": argv,
        "env": {k: os.environ.get(k) for k in ("CHRONOS_BATCH_SIZE", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "CUBLAS_WORKSPACE_CONFIG")},
    }

    def write(ev):
        line = json.dumps(dict(rec, ev=ev, t=time.time())) + "\n"
        fd = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.write(fd, line.encode())
        os.close(fd)

    write("start")
    if job.get("fake_child"):
        child = subprocess.Popen(["sleep", "120"])
        Path(job["fake_child"]).write_text(str(child.pid))
    time.sleep(float(job.get("fake_sleep", 0.3)))
    if attempt in job.get("fake_fail_attempts", []):
        write("end_fail")
        print("fake failure", file=sys.stderr)
        return 3
    out = Path(os.environ["DISPATCH_RESULT_CSV"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text("dataset,model\nx,y\n")
    os.replace(tmp, out)
    write("end_ok")
    return 0


# ---------------------------------------------------------------- helpers
def job(jid, exclusive=False, **extra):
    """A job dict shaped like make_jobs.py's, with a unique dataset name."""
    j = {"id": jid, "model": "toto" if exclusive else "chronos_bolt_base", "env": "fake",
         "toto_setting": "A" if exclusive else None, "seed": 42 if exclusive else None,
         "context": "default", "dataset": f"fake_{jid}/H", "term": "short", "weight": extra.pop("weight", 1.0),
         "exclusive": exclusive}
    j.update(extra)
    return j


def write_jobs(path, jobs):
    Path(path).write_text("".join(json.dumps(j) + "\n" for j in jobs))


def dispatcher_cmd(tmp, files, gpus, extra=()):
    return [PY, str(HERE / "dispatch.py"), "--jobs", *map(str, files), "--gpus", gpus, "--slots-per-gpu", "2",
            "--out", str(tmp / "results"), "--logs", str(tmp / "logs"), "--status", str(tmp / "status.jsonl"),
            "--state-dir", str(tmp / "state"), "--stop-file", str(tmp / "STOP"), "--poll-sec", "0.05",
            "--kill-grace-sec", "1", "--min-free-gb", "0", *extra]


def env_for(tmp):
    env = os.environ.copy()
    env.pop("TOTO_SMOKE_SAMPLES", None)
    env["DISPATCH_FAKE_CMD"] = f"{PY} {HERE / 'test_dispatch.py'} --fake-run"
    env["TEST_TRACE"] = str(tmp / "trace.jsonl")
    env["HF_HOME"] = "/fake/hf_home"
    return env


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def intervals(trace):
    """(id, attempt) -> {card, start, end}"""
    iv = {}
    for r in trace:
        key = (r["id"], r["attempt"])
        d = iv.setdefault(key, {"card": r["card"], "start": None, "end": None})
        if r["ev"] == "start":
            d["start"] = r["t"]
        else:
            d["end"] = r["t"]
    return iv


def overlapping(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def done_line(out):
    lines = [l for l in out.splitlines() if l.startswith("DISPATCH_DONE")]
    return lines[-1] if lines else ""


def check(cond, msg, fails):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---------------------------------------------------------------- tests
def test_main(fails):
    print("test_main: 2 fake cards x 2 slots; exclusive, retry, timeout, skip")
    tmp = mkdtemp("disp_main_")
    pidfile = tmp / "child.pid"
    jobs = [
        job("excl_a", exclusive=True, fake_sleep=1.0, weight=50),
        job("excl_b", exclusive=True, fake_sleep=1.0, weight=40),
        *[job(f"plain_{i}", fake_sleep=0.6, weight=10 + i) for i in range(5)],
        job("fail_once", fake_sleep=0.3, fake_fail_attempts=[1], weight=30),
        job("fail_always", fake_sleep=0.3, fake_fail_attempts=[1, 2], weight=20),
        job("too_long", fake_sleep=60, fake_child=str(pidfile), weight=60, timeout_min=0.05),
        job("already_done", weight=100),
    ]
    write_jobs(tmp / "a.jsonl", jobs)
    # pre-existing result for already_done (toto/chronos variant path as run_grid writes it)
    sys.path.insert(0, str(HERE))
    from result_paths import job_result_csv
    env = env_for(tmp)
    os.environ.pop("TOTO_SMOKE_SAMPLES", None)
    pre = job_result_csv(tmp / "results", jobs[-1])
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_text("dataset,model\nx,y\n")

    # default timeout 5 min; only too_long carries its own timeout_min (0.05 min = 3 s)
    p = subprocess.run(dispatcher_cmd(tmp, [tmp / "a.jsonl"], "0,1", ["--timeout-min", "5"]),
                       env=env, capture_output=True, text=True, timeout=120)
    out = p.stdout
    (tmp / "dispatch.out").write_text(out + p.stderr)
    trace = read_jsonl(tmp / "trace.jsonl")
    status = read_jsonl(tmp / "status.jsonl")
    iv = intervals(trace)
    print(f"  ({done_line(out)}; rc={p.returncode}; folder {tmp})")

    check(done_line(out) == "DISPATCH_DONE ok=8 failed=1 timeout=1 skipped=1", "final counts ok=8 failed=1 timeout=1 skipped=1", fails)
    check(p.returncode == 1, "dispatcher exits 1 when something failed", fails)
    check(not any(r["id"] == "already_done" for r in trace), "existing result is skipped (never started)", fails)
    check({r["card"] for r in trace} <= {"0", "1"}, "only cards 0 and 1 used", fails)
    # exclusive jobs and retries never share a card
    alone = [k for k in iv if k[0].startswith("excl_") or k[1] == 2]
    shared = [(k, o) for k in alone for o in iv if o != k and iv[o]["card"] == iv[k]["card"] and iv[o]["end"] and iv[k]["end"] and overlapping(iv[k], iv[o])]
    check(len([k for k in alone if k[0].startswith("excl_")]) == 2 and not shared,
          f"exclusive jobs and retries ran alone on their card (overlaps: {shared})", fails)
    # at most 2 at a time per card
    worst = 0
    for k in iv:
        if iv[k]["start"] is None:
            continue
        t = iv[k]["start"] + 1e-6
        n = sum(1 for o in iv if iv[o]["card"] == iv[k]["card"] and iv[o]["start"] is not None
                and iv[o]["start"] <= t and (iv[o]["end"] is None or iv[o]["end"] > t))
        worst = max(worst, n)
    check(worst <= 2, f"never more than 2 jobs on one card (max seen {worst})", fails)
    att = lambda jid: sorted(r["attempt"] for r in status if r["id"] == jid)
    check(att("fail_once") == [1, 2] and [r["outcome"] for r in status if r["id"] == "fail_once"] == ["failed", "ok"],
          "a job failing once is retried once and then ok", fails)
    check(att("fail_always") == [1, 2], "a job failing twice is retried exactly once", fails)
    tl = [r for r in status if r["id"] == "too_long"]
    check(len(tl) == 1 and tl[0]["timeout"] is True and tl[0]["outcome"] == "timeout" and tl[0]["timeout_min"] == 0.05
          and tl[0]["wall_seconds"] < 30, "job over its own timeout_min (0.05) is killed, marked timeout, not retried", fails)
    others = [r for r in status if r["id"] != "too_long"]
    check(all(r["timeout_min"] == 5 for r in others), "jobs without timeout_min use --timeout-min", fails)
    alive = None
    if pidfile.exists():
        cpid = int(pidfile.read_text())
        time.sleep(0.3)
        try:
            os.kill(cpid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        if alive:
            os.kill(cpid, 9)
    check(alive is False, "the timed-out job's child process was killed with its group", fails)
    base_env = {"CHRONOS_BATCH_SIZE": "64", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/fake/hf_home"}
    starts = [r for r in trace if r["ev"] == "start"]
    check(all({k: r["env"][k] for k in base_env} == base_env for r in starts),
          "children get CHRONOS_BATCH_SIZE=64, expandable_segments, HF_HOME passed through", fails)
    check(all(r["env"]["CUBLAS_WORKSPACE_CONFIG"] == (":4096:8" if r["id"].startswith("excl_") else None) for r in starts),
          "CUBLAS_WORKSPACE_CONFIG=:4096:8 set for Toto sampling jobs only", fails)
    ex = [r for r in trace if r["id"] == "excl_a" and r["ev"] == "start"][0]["argv"]
    check("--toto-setting" in ex and ex[ex.index("--toto-setting") + 1] == "A" and ex[ex.index("--seed") + 1] == "42"
          and ex[ex.index("--context") + 1] == "default", "toto job gets --toto-setting A --seed 42 --context default", fails)
    keys = {"id", "returncode", "wall_seconds", "card", "start_utc", "end_utc", "timeout"}
    check(all(keys <= set(r) for r in status) and len(status) == 12, f"status.jsonl has 12 complete lines, one per attempt (has {len(status)})", fails)
    check(sum(1 for l in out.splitlines() if l.startswith("PROGRESS")) == 12, "one PROGRESS line per finished attempt (12)", fails)
    check((tmp / "logs" / "fail_once.retry.log").exists() and (tmp / "logs" / "fail_once.log").exists(), "retry has its own log", fails)
    # rerun: everything done or final -> only failed/timeout jobs run again, ok ones skipped
    p2 = subprocess.run(dispatcher_cmd(tmp, [tmp / "a.jsonl"], "0,1", ["--timeout-min", "5"]),
                        env=env, capture_output=True, text=True, timeout=120)
    check("skipped=9" in done_line(p2.stdout), f"rerun skips the 9 finished results ({done_line(p2.stdout)})", fails)


def test_two_dispatchers(fails):
    print("test_two_dispatchers: cards 0,1 and 2,3 at once, overlapping job files; third on card 1 refused")
    tmp = mkdtemp("disp_two_")
    x = [job(f"x{i}", fake_sleep=0.8, weight=10) for i in range(6)] + [job("xe", exclusive=True, fake_sleep=0.8, weight=5)]
    y = [job(f"y{i}", fake_sleep=0.5, weight=10) for i in range(4)] + [job("ye", exclusive=True, fake_sleep=0.5, weight=5)]
    write_jobs(tmp / "x.jsonl", x)
    write_jobs(tmp / "y.jsonl", y)
    env = env_for(tmp)
    pa = subprocess.Popen(dispatcher_cmd(tmp, [tmp / "x.jsonl"], "0,1"), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.6)
    pb = subprocess.Popen(dispatcher_cmd(tmp, [tmp / "y.jsonl", tmp / "x.jsonl"], "2,3"), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.2)
    pc = subprocess.run(dispatcher_cmd(tmp, [tmp / "y.jsonl"], "1"), env=env, capture_output=True, text=True, timeout=60)
    oa, _ = pa.communicate(timeout=120)
    ob, _ = pb.communicate(timeout=120)
    (tmp / "a.out").write_text(oa)
    (tmp / "b.out").write_text(ob)
    trace = read_jsonl(tmp / "trace.jsonl")
    status = read_jsonl(tmp / "status.jsonl")
    print(f"  (A: {done_line(oa)} | B: {done_line(ob)}; folder {tmp})")
    starts = {}
    for r in trace:
        if r["ev"] == "start":
            starts.setdefault(r["id"], []).append(r["card"])
    check(pc.returncode == 2 and "CARD_BUSY" in pc.stdout, "a dispatcher asking for a card in use refuses to start", fails)
    check(sorted(starts) == sorted(j["id"] for j in x + y) and all(len(v) == 1 for v in starts.values()),
          "every job of both files ran exactly once", fails)
    check(all(starts[j["id"]][0] in ("2", "3") for j in y), "dispatcher B ran its own jobs on cards 2,3 only", fails)
    a_ids = {l.split()[1] for l in oa.splitlines() if l.startswith("START ")}
    check(all(starts[i][0] in ("0", "1") for i in a_ids), "dispatcher A used cards 0,1 only", fails)
    check(len(status) == len(x) + len(y) and len({r["id"] for r in status}) == len(status), "status.jsonl: one intact line per job", fails)
    check(pa.returncode == 0 and pb.returncode == 0, f"both dispatchers exit 0 (A={pa.returncode}, B={pb.returncode})", fails)
    check(not any((tmp / "state" / "cards").glob("*.claim")) and not any((tmp / "state" / "claims").glob("*.claim")),
          "card and job claims released at the end", fails)


def test_no_exclusive(fails):
    print("test_no_exclusive: --no-exclusive lets two exclusive jobs share one card")
    tmp = mkdtemp("disp_noex_")
    write_jobs(tmp / "e.jsonl", [job("ex1", exclusive=True, fake_sleep=1.0), job("ex2", exclusive=True, fake_sleep=1.0)])
    p = subprocess.run(dispatcher_cmd(tmp, [tmp / "e.jsonl"], "0", ["--no-exclusive"]), env=env_for(tmp), capture_output=True, text=True, timeout=60)
    iv = intervals(read_jsonl(tmp / "trace.jsonl"))
    a, b = iv.get(("ex1", 1)), iv.get(("ex2", 1))
    check(p.returncode == 0 and a and b and overlapping(a, b), "with --no-exclusive both ran at once on card 0", fails)
    tmp2 = mkdtemp("disp_noex_")
    write_jobs(tmp2 / "e.jsonl", [job("ex1", exclusive=True, fake_sleep=1.0), job("ex2", exclusive=True, fake_sleep=1.0)])
    p = subprocess.run(dispatcher_cmd(tmp2, [tmp2 / "e.jsonl"], "0"), env=env_for(tmp2), capture_output=True, text=True, timeout=60)
    iv = intervals(read_jsonl(tmp2 / "trace.jsonl"))
    a, b = iv.get(("ex1", 1)), iv.get(("ex2", 1))
    check(p.returncode == 0 and a and b and not overlapping(a, b), "without it they ran one after the other", fails)


def test_stop_and_disk(fails):
    print("test_stop_and_disk: STOP file and low free disk stop new jobs")
    tmp = mkdtemp("disp_stop_")
    write_jobs(tmp / "s.jsonl", [job(f"s{i}", fake_sleep=0.2) for i in range(3)])
    env = env_for(tmp)
    (tmp / "STOP").write_text("")
    p = subprocess.run(dispatcher_cmd(tmp, [tmp / "s.jsonl"], "0"), env=env, capture_output=True, text=True, timeout=60)
    check("DISPATCH_STOPPED" in p.stdout and not (tmp / "trace.jsonl").exists() and "not_started=3" in p.stdout,
          "STOP file: no job starts, 3 reported not started", fails)
    (tmp / "STOP").unlink()
    p = subprocess.run(dispatcher_cmd(tmp, [tmp / "s.jsonl"], "0", ["--min-free-gb", "1e12"]), env=env, capture_output=True, text=True, timeout=60)
    check("free disk" in p.stdout and not (tmp / "trace.jsonl").exists(), "free disk below the limit: no job starts", fails)


def test_dry_run_real(fails):
    print("test_dry_run_real: --dry-run on the real job files")
    files = [HERE / "jobs" / f for f in ("gate.jsonl", "g3_seed42.jsonl", "g4_seeds.jsonl", "g1_det.jsonl",
                                         "g1_toto.jsonl", "g3_fallback.jsonl", "g2.jsonl")]
    tmp = mkdtemp("disp_dry_")
    env = os.environ.copy()
    env.pop("DISPATCH_FAKE_CMD", None)
    p = subprocess.run([PY, str(HERE / "dispatch.py"), "--jobs", *map(str, files), "--gpus", "0,1,2,3", "--dry-run",
                        "--out", str(tmp / "results"), "--logs", str(tmp / "logs"), "--state-dir", str(tmp / "state")],
                       env=env, capture_output=True, text=True, timeout=120)
    print("\n".join("    " + l for l in p.stdout.splitlines() if l.startswith(("DISPATCH_START", "DRYRUN", "DISPATCH_DONE"))))
    check(p.returncode == 0 and "jobs=665" in p.stdout and "DISPATCH_DONE" in p.stdout, "dry run reads 665 jobs and exits 0", fails)
    check(not (tmp / "state").exists() and not (tmp / "logs").exists(), "dry run starts nothing and claims nothing", fails)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-run":
        sys.exit(fake_run(sys.argv[2:]))
    fails = []
    for t in (test_main, test_two_dispatchers, test_no_exclusive, test_stop_and_disk, test_dry_run_real):
        t(fails)
    if not fails:
        for d in TMP_DIRS:
            shutil.rmtree(d, ignore_errors=True)
    print(f"TEST_DISPATCH {'PASS' if not fails else 'FAIL'} failures={len(fails)}")
    sys.exit(1 if fails else 0)
