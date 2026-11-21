#!/usr/bin/env python3
"""Run job files (JSON lines from make_jobs.py) on several GPU cards.
Standard library only; run it from any python 3.9+ in the harness folder.

  python3 dispatch.py --jobs jobs/gate.jsonl jobs/g3g4.jsonl --gpus 0,1,2,3

Order: earlier job files first; within a file the heaviest job first. A job
whose result csv already exists is skipped. Each job is one subprocess
  envs/<env>/bin/python run_grid.py --model .. --configs <one-config json>
      --context .. --out <out> [--toto-setting .. --seed ..]
with CUDA_VISIBLE_DEVICES=<card>, CHRONOS_BATCH_SIZE=64,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, for Toto's sampling
settings also CUBLAS_WORKSPACE_CONFIG=:4096:8 (DESIGN_ROUND3.md Revision 1),
and the rest of this process's environment (HF_HOME included), in its own
process group, output to <logs>/<job id>.log.

Cards and slots: every card has --slots-per-gpu slots. An "exclusive" job
(every Toto sampling job) takes all slots of one card, unless --no-exclusive
is given, in which case it takes one slot like any other job. When the first
job that cannot start is exclusive, one card is held for it (no new jobs go
to that card) until it is free, so exclusive jobs are not starved.

Each finished job appends one line to status.jsonl (id, returncode,
wall_seconds, card, start_utc, end_utc, timeout, attempt, ...). A job that
runs longer than its own "timeout_min" (from the job file; --timeout-min
for a job without one) is killed with its whole process group and marked
"timeout": true; it is not retried. When no first-attempt job is left
in the queue, each failed job is retried once, alone on a card
(logs/<job id>.retry.log). No new job starts while free disk under --out is
below --min-free-gb or the file --stop-file (harness/STOP) exists.

A job counts as ok only when it exits 0 and its result csv exists.

Several dispatchers can run at once on different cards: each claims its cards
(<state-dir>/cards/gpu<k>.claim) and refuses to start if one is held by a
live dispatcher; each job is claimed (<state-dir>/claims/<id>.claim) before
it starts, so the same job never runs twice at once.

Prints PROGRESS per finished job and finally
  DISPATCH_DONE ok=<n> failed=<n> timeout=<n> skipped=<n>

Testing: DISPATCH_FAKE_CMD="<command>" replaces `envs/<env>/bin/python
run_grid.py` by that command (same arguments follow). The child also gets
DISPATCH_JOB_JSON, DISPATCH_RESULT_CSV and DISPATCH_ATTEMPT.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from result_paths import TOTO_SAMPLING_SETTINGS, atomic_write_text, job_result_csv  # noqa: E402

HOST = socket.gethostname()
CUBLAS_REQUIRED = ":4096:8"
REQUIRED_FIELDS = ("id", "model", "env", "context", "dataset", "term", "weight", "exclusive")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(msg):
    print(msg, flush=True)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Claim:
    """A file created with O_EXCL holding {pid, host}. A claim left by a dead
    process on this host is taken over."""

    def __init__(self, path):
        self.path = Path(path)
        self.held = False

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"pid": os.getpid(), "host": HOST, "since": utc_now()})
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = self.owner()
                if owner and owner.get("host") == HOST and not pid_alive(int(owner.get("pid", -1))):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                return False
            with os.fdopen(fd, "w") as fh:
                fh.write(body)
            self.held = True
            return True
        return False

    def owner(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def release(self):
        if self.held:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.held = False


def load_jobs(files, no_exclusive=False):
    jobs, seen = [], set()
    for fi, f in enumerate(files):
        file_jobs = []
        for li, line in enumerate(Path(f).read_text().splitlines()):
            if not line.strip():
                continue
            j = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in j]
            if missing:
                raise ValueError(f"{f}:{li + 1}: job without {missing}")
            if j["id"] in seen:
                log(f"DUPLICATE_JOB {j['id']} in {f}: kept the earlier one")
                continue
            seen.add(j["id"])
            exclusive = bool(j["exclusive"]) and not no_exclusive
            file_jobs.append({"spec": j, "file": str(f), "fi": fi, "li": li, "attempt": 1, "exclusive": exclusive})
        file_jobs.sort(key=lambda x: (-float(x["spec"]["weight"]), x["li"]))
        jobs += file_jobs
    return jobs


def is_toto_sampling(spec):
    return spec["model"] == "toto" and spec.get("toto_setting") in TOTO_SAMPLING_SETTINGS


def build_command(spec, cfg_path, out_dir):
    args = [
        "--model", spec["model"],
        "--configs", str(cfg_path),
        "--context", str(spec["context"]),
        "--out", str(out_dir),
    ]
    if is_toto_sampling(spec):
        args += ["--toto-setting", spec["toto_setting"], "--seed", str(spec.get("seed", 42))]
    fake = os.environ.get("DISPATCH_FAKE_CMD")
    if fake:
        return shlex.split(fake) + args
    return [str(HERE / "envs" / spec["env"] / "bin" / "python"), str(HERE / "run_grid.py")] + args


class Dispatcher:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out).resolve()
        self.logs = Path(a.logs).resolve()
        self.status = Path(a.status).resolve()
        self.state = Path(a.state_dir).resolve()
        self.stop_file = Path(a.stop_file).resolve()
        self.cards = [c.strip() for c in a.gpus.split(",") if c.strip()]
        if len(set(self.cards)) != len(self.cards):
            raise SystemExit(f"--gpus lists a card twice: {a.gpus}")
        self.slots = a.slots_per_gpu
        self.default_timeout_min = a.timeout_min
        self.used = {c: 0 for c in self.cards}
        self.excl = {c: False for c in self.cards}
        self.reserved = {}  # card -> job id held for an exclusive job
        self.running = []
        self.pending = []
        self.to_retry = []
        self.counts = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0, "interrupted": 0}
        self.finished = 0
        self.total = 0
        self.stopping = None
        self.interrupted = False
        self.card_claims = []

    # ---------------------------------------------------------------- placement
    def place(self, job):
        """Card for this job now, or None."""
        if job["exclusive"]:
            free = [c for c in self.cards if self.used[c] == 0 and self.reserved.get(c) in (None, job["spec"]["id"])]
            mine = [c for c in free if self.reserved.get(c) == job["spec"]["id"]]
            return (mine or free or [None])[0]
        ok = [c for c in self.cards if c not in self.reserved and not self.excl[c] and self.used[c] < self.slots]
        if not ok:
            return None
        return min(ok, key=lambda c: (self.used[c], self.cards.index(c)))

    def reserve_for(self, job):
        jid = job["spec"]["id"]
        if jid in self.reserved.values():
            return True
        cand = [c for c in self.cards if c not in self.reserved]
        if not cand:
            return False
        c = min(cand, key=lambda c: (self.used[c], self.cards.index(c)))
        self.reserved[c] = jid
        return True

    def release_reservation(self, jid):
        for c in [c for c, j in self.reserved.items() if j == jid]:
            del self.reserved[c]

    def take(self, job, card):
        self.used[card] = self.slots if job["exclusive"] else self.used[card] + 1
        self.excl[card] = job["exclusive"]
        self.release_reservation(job["spec"]["id"])

    def give_back(self, job, card):
        self.used[card] = 0 if job["exclusive"] else self.used[card] - 1
        if job["exclusive"]:
            self.excl[card] = False

    # ------------------------------------------------------------------ checks
    def result_exists(self, job):
        return job_result_csv(self.out, job["spec"]).exists()

    def stop_reason(self):
        if self.interrupted:
            return "signal"
        if self.stop_file.exists():
            return f"stop file {self.stop_file}"
        free_gb = shutil.disk_usage(self.out).free / 1e9
        if free_gb < self.a.min_free_gb:
            return f"free disk {free_gb:.1f} GB < {self.a.min_free_gb} GB under {self.out}"
        return None

    # ------------------------------------------------------------------ launch
    def launch(self, job, card):
        spec = job["spec"]
        jid = spec["id"]
        claim = Claim(self.state / "claims" / f"{jid}.claim")
        if not claim.acquire():
            owner = claim.owner() or {}
            log(f"SKIP_CLAIMED {jid} (running under dispatcher pid {owner.get('pid')} on {owner.get('host')})")
            self.counts["skipped"] += 1
            self.release_reservation(jid)
            return False
        cfg = self.logs / "job_configs" / f"{jid}.json"
        atomic_write_text(cfg, json.dumps([{"dataset": spec["dataset"], "term": spec["term"]}]))
        cmd = build_command(spec, cfg, self.out)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(card),
                "CHRONOS_BATCH_SIZE": "64",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "DISPATCH_JOB_JSON": json.dumps(spec),
                "DISPATCH_RESULT_CSV": str(job_result_csv(self.out, spec)),
                "DISPATCH_ATTEMPT": str(job["attempt"]),
            }
        )
        if is_toto_sampling(spec):
            env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_REQUIRED  # before torch is imported in the child
        log_path = self.logs / (f"{jid}.log" if job["attempt"] == 1 else f"{jid}.retry.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "a")  # a rerun appends after the earlier attempt's output
        start_utc = utc_now()
        fh.write(f"# dispatch start={start_utc} card={card} attempt={job['attempt']} cmd={shlex.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.take(job, card)
        timeout_min = float(spec.get("timeout_min") or self.default_timeout_min)
        self.running.append(
            {"job": job, "card": card, "proc": proc, "fh": fh, "t0": time.monotonic(), "start_utc": start_utc,
             "timeout_s": timeout_min * 60.0,
             "claim": claim, "kill_stage": 0, "kill_t": None, "timeout": False, "log": str(log_path)}
        )
        log(f"START {jid} card={card} attempt={job['attempt']} weight={spec['weight']} exclusive={job['exclusive']} "
            f"timeout_min={timeout_min:g}")
        return True

    # --------------------------------------------------------------- schedule
    def schedule(self):
        blocked_earlier = False
        for job in list(self.pending):
            jid = job["spec"]["id"]
            if self.result_exists(job):
                self.pending.remove(job)
                self.release_reservation(jid)
                self.counts["skipped"] += 1
                log(f"SKIP_EXISTS {jid}")
                continue
            card = self.place(job)
            if card is not None:
                self.pending.remove(job)
                self.launch(job, card)
                continue
            if job["exclusive"] and not blocked_earlier and self.reserve_for(job):
                continue  # waits for its held card; later jobs may still use other cards
            blocked_earlier = True

    # ------------------------------------------------------------------- reap
    def reap(self):
        now = time.monotonic()
        for r in list(self.running):
            proc = r["proc"]
            if r["kill_stage"] == 0 and now - r["t0"] > r["timeout_s"]:
                r["timeout"] = True
                self.kill_group(r, signal.SIGTERM)
                r["kill_stage"], r["kill_t"] = 1, now
                log(f"TIMEOUT {r['job']['spec']['id']} after {now - r['t0']:.0f}s: process group terminated")
            if r["kill_stage"] == 1:
                if now - r["kill_t"] < self.a.kill_grace_sec:
                    continue
                self.kill_group(r, signal.SIGKILL)
                r["kill_stage"] = 2
            if proc.poll() is None:
                continue
            self.finish(r)

    def kill_group(self, r, sig):
        try:
            os.killpg(r["proc"].pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def finish(self, r, interrupted=False):
        self.running.remove(r)
        job, spec = r["job"], r["job"]["spec"]
        r["fh"].close()
        self.give_back(job, r["card"])
        rc = r["proc"].returncode
        wall = time.monotonic() - r["t0"]
        csv_ok = self.result_exists(job)
        if interrupted:
            outcome = "interrupted"
        elif r["timeout"]:
            outcome = "timeout"
        elif rc == 0 and csv_ok:
            outcome = "ok"
        else:
            outcome = "failed"
        line = {
            "id": spec["id"],
            "returncode": rc,
            "wall_seconds": round(wall, 1),
            "card": r["card"],
            "start_utc": r["start_utc"],
            "end_utc": utc_now(),
            "timeout": bool(r["timeout"]),
            "timeout_min": r["timeout_s"] / 60.0,
            "attempt": job["attempt"],
            "outcome": outcome,
            "result_csv_exists": csv_ok,
            "job_file": job["file"],
            "log": r["log"],
            "dispatcher_pid": os.getpid(),
        }
        if interrupted:
            line["interrupted"] = True
        self.append_status(line)
        r["claim"].release()
        self.finished += 1
        if outcome == "failed" and job["attempt"] == 1 and not interrupted:
            self.to_retry.append(job)
            log(f"PROGRESS {self.finished} done | FAILED_WILL_RETRY {spec['id']} rc={rc} wall={wall:.0f}s card={r['card']} {line['end_utc']}")
        else:
            self.counts[outcome] += 1
            log(
                f"PROGRESS {self.finished} done | {outcome.upper()} {spec['id']} rc={rc} wall={wall:.0f}s card={r['card']} "
                f"attempt={job['attempt']} | ok={self.counts['ok']} failed={self.counts['failed']} "
                f"timeout={self.counts['timeout']} skipped={self.counts['skipped']} running={len(self.running)} "
                f"queued={len(self.pending)} {line['end_utc']}"
            )

    def append_status(self, line):
        data = (json.dumps(line) + "\n").encode()
        self.status.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.status, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)  # one write per line: lines from two dispatchers never interleave
        finally:
            os.close(fd)

    def queue_retries(self):
        if self.to_retry and not any(j["attempt"] == 1 for j in self.pending):
            for job in self.to_retry:
                retry = dict(job, attempt=2, exclusive=True)
                self.pending.append(retry)
                log(f"RETRY_QUEUED {job['spec']['id']} (alone on a card)")
            self.to_retry = []

    # -------------------------------------------------------------------- run
    def claim_cards(self):
        for c in self.cards:
            cl = Claim(self.state / "cards" / f"gpu{c}.claim")
            if not cl.acquire():
                owner = cl.owner() or {}
                for held in self.card_claims:
                    held.release()
                log(f"CARD_BUSY card {c} is held by dispatcher pid {owner.get('pid')} on {owner.get('host')}; not starting")
                return False
            self.card_claims.append(cl)
        return True

    def on_signal(self, signum, frame):
        self.interrupted = True

    def run(self):
        jobs = load_jobs(self.a.jobs, no_exclusive=self.a.no_exclusive)
        self.total = len(jobs)
        for job in jobs:
            if self.result_exists(job):
                self.counts["skipped"] += 1
            else:
                self.pending.append(job)
        log(f"DISPATCH_START {utc_now()} jobs={self.total} already_done={self.counts['skipped']} "
            f"queued={len(self.pending)} cards={','.join(self.cards)} slots_per_card={self.slots} "
            f"default_timeout_min={self.a.timeout_min} no_exclusive={self.a.no_exclusive} out={self.out}")
        if self.a.dry_run:
            return self.dry_run()
        if not self.claim_cards():
            return 2
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        try:
            while True:
                self.reap()
                if self.stopping is None:
                    reason = self.stop_reason()
                    if reason:
                        self.stopping = reason
                        log(f"DISPATCH_STOPPING no new jobs: {reason}")
                if self.interrupted:
                    self.kill_all()
                    break
                if self.stopping is None:
                    self.queue_retries()
                    self.schedule()
                idle = not self.running
                if idle and (self.stopping is not None or (not self.pending and not self.to_retry)):
                    break
                time.sleep(self.a.poll_sec)
        finally:
            for r in list(self.running):
                self.kill_group(r, signal.SIGKILL)
            for cl in self.card_claims:
                cl.release()
        not_started = len(self.pending) + len(self.to_retry)
        self.counts["failed"] += len(self.to_retry)  # failed, retry never started (stop or signal)
        if self.stopping or self.interrupted:
            log(f"DISPATCH_STOPPED reason={self.stopping or 'signal'} not_started={len(self.pending)} "
                f"failed_not_retried={len(self.to_retry)} interrupted={self.counts['interrupted']}")
        c = self.counts
        log(f"DISPATCH_DONE ok={c['ok']} failed={c['failed']} timeout={c['timeout']} skipped={c['skipped']}")
        if self.interrupted:
            return 130
        return 0 if (c["failed"] == 0 and c["timeout"] == 0 and not_started == 0 and c["interrupted"] == 0) else 1

    def kill_all(self):
        log(f"DISPATCH_INTERRUPTED killing {len(self.running)} running job(s)")
        for r in self.running:
            self.kill_group(r, signal.SIGTERM)
        deadline = time.monotonic() + self.a.kill_grace_sec
        while time.monotonic() < deadline and any(r["proc"].poll() is None for r in self.running):
            time.sleep(0.2)
        for r in list(self.running):
            self.kill_group(r, signal.SIGKILL)
            r["proc"].wait()
            self.finish(r, interrupted=True)

    # ---------------------------------------------------------------- dry run
    def dry_run(self):
        by_file = {}
        for j in self.pending:
            by_file.setdefault(j["file"], []).append(j)
        for f, js in by_file.items():
            log(f"DRYRUN file={f} queued={len(js)} exclusive={sum(j['exclusive'] for j in js)} "
                f"weight={sum(float(j['spec']['weight']) for j in js) / 3600:.2f} h")
        log("DRYRUN first 15 in queue order:")
        for j in self.pending[:15]:
            log(f"  {j['spec']['weight']:9.1f}s excl={int(j['exclusive'])} {j['spec']['id']}")
        if self.pending:
            log(f"  example command: {shlex.join(build_command(self.pending[0]['spec'], self.logs / 'job_configs' / (self.pending[0]['spec']['id'] + '.json'), self.out))}")
        envs = sorted({j["spec"]["env"] for j in self.pending})
        missing = [e for e in envs if not (HERE / "envs" / e / "bin" / "python").exists()]
        log(f"DRYRUN envs needed={envs} missing_here={missing}")
        # Simulate the schedule with weights as run times (ignores slow-down
        # from two jobs sharing a card, retries and timeouts).
        t, events, order = 0.0, [], list(self.pending)
        self.pending = order
        while self.pending or events:
            for job in list(self.pending):
                card = self.place(job)
                if card is not None:
                    self.pending.remove(job)
                    self.take(job, card)
                    events.append((t + float(job["spec"]["weight"]), job, card))
                elif job["exclusive"] and self.reserve_for(job):
                    continue
                else:
                    break
            if not events:
                break
            events.sort(key=lambda e: e[0])
            t, job, card = events.pop(0)
            self.give_back(job, card)
        log(f"DRYRUN simulated wall time {t / 3600:.2f} h on {len(self.cards)} card(s) x {self.slots} slot(s)")
        log(f"DISPATCH_DONE ok=0 failed=0 timeout=0 skipped={self.counts['skipped']} (dry run)")
        return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", nargs="+", required=True, help="job files; earlier files have priority")
    p.add_argument("--gpus", default="0,1,2,3", help="cards this dispatcher may use, e.g. 0,1")
    p.add_argument("--slots-per-gpu", type=int, default=2)
    p.add_argument("--timeout-min", type=float, default=90.0, help="for jobs without their own timeout_min")
    p.add_argument("--no-exclusive", action="store_true",
                   help="exclusive jobs (Toto sampling) take one slot and share a card like any other job")
    p.add_argument("--out", default="results")
    p.add_argument("--logs", default="logs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", default=str(HERE / "status.jsonl"), help="status file (default harness/status.jsonl)")
    p.add_argument("--stop-file", default=str(HERE / "STOP"), help="no new jobs while this file exists")
    p.add_argument("--state-dir", default=str(HERE / ".dispatch_state"), help="card and job claim files")
    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--poll-sec", type=float, default=2.0)
    p.add_argument("--kill-grace-sec", type=float, default=15.0, help="SIGTERM to SIGKILL delay on timeout")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sys.exit(Dispatcher(a).run())
