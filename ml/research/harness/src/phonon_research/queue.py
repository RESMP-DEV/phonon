"""`phonon-research queue <jobs.yaml>` / `queue status`: a two-GPU job runner.

One command launches it. The scheduler lives in its own tmux window, every job gets its own
tmux window and log under /data/phonon_queue/<run>/, and the whole state is a single
state.json that is rewritten after every transition, so the Mac's ssh dropping changes
nothing. Re-invoking the same jobs.yaml is safe: a live scheduler is left alone, a dead one is
restarted, done jobs are never relaunched, and a failed job is retried once.

A job is {name, kind: corpus|train|eval|shell, gpu: 0|1|any, needs: [names], config|command}.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .paths import HARNESS, QUEUE_ROOT, REPO, hf_env
from .util import log, load_yaml, python_exe, read_json, resolve_config, stamp, table, write_json

TERMINAL = ("done", "failed", "skipped")
POLL = 3.0
HEARTBEAT_EVERY = 20 * 60


def detect_gpus() -> list[int]:
    """The pool a jobs.yaml gets when it does not name one. gpubox grew from two cards to four;
    hard-coding [0, 1] silently halved the queue, so ask the driver."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        gs = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        if gs:
            return gs
    except Exception:
        pass
    return [0, 1]


def run_dir(run: str) -> Path:
    return QUEUE_ROOT / run


def state_path(run: str) -> Path:
    return run_dir(run) / "state.json"


def session(run: str) -> str:
    return f"pq-{run}"


def _tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, check=check)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------- state
def load_state(run: str) -> dict:
    return read_json(state_path(run), default={}) or {}


def save_state(st: dict) -> None:
    write_json(state_path(st["run"]), st)


def job_command(job: dict, st: dict) -> str:
    kind = job["kind"]
    d = run_dir(st["run"])
    res = d / f"{job['name']}.result.json"
    py = python_exe()
    cli = [py, "-m", "phonon_research.cli"]
    if kind == "train":
        argv = cli + ["train", str(job["config"]), "--gpu", str(job.get("gpu", "any")),
                      "--result", str(res)]
    elif kind == "eval":
        if job.get("config"):
            argv = cli + ["eval-config", str(job["config"]), "--gpu", str(job.get("gpu", "any")),
                          "--result", str(res)]
        else:
            argv = cli + ["eval", str(job["adapter"]), "--gpu", str(job.get("gpu", "any")),
                          "--result", str(res)]
            if job.get("sets"):
                argv += ["--sets", ",".join(job["sets"])]
            if job.get("backend"):
                argv += ["--backend", job["backend"]]
    elif kind == "corpus":
        argv = cli + ["corpus", str(job["config"])]
        if job.get("only"):
            argv += ["--only", ",".join(job["only"])]
    elif kind == "shell":
        return job["command"]
    else:
        raise SystemExit(f"unknown job kind {kind!r}")
    return " ".join(f"'{a}'" if " " in str(a) else str(a) for a in argv)


def timeout_prefix(job: dict) -> str:
    """A job may cap its own wall clock; a hung dependency then fails instead of blocking."""
    t = job.get("timeout_s")
    return f"timeout -k 30 {int(t)} " if t else ""


def write_job_script(job: dict, st: dict, gpu) -> Path:
    d = run_dir(st["run"])
    n = job["name"]
    body = f"""#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/data/hf HF_HUB_CACHE=/data/hf/hub HUGGINGFACE_HUB_CACHE=/data/hf/hub
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_TORCH_BACKEND=cu130 PYTHONPATH={HARNESS}/src
export PHONON_PYTHON={python_exe()}
export CUDA_VISIBLE_DEVICES={'' if str(gpu) == 'cpu' else gpu}
cd {REPO}
date +%s.%N > {d}/{n}.start
{{ {timeout_prefix(job)}{job_command(job, st)} ; }} >> {d}/{n}.log 2>&1
rc=$?
date +%s.%N > {d}/{n}.end
echo $rc > {d}/{n}.rc
"""
    p = d / f"{n}.sh"
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


def _stamp_file(p: Path) -> float | None:
    try:
        return float(p.read_text().strip())
    except Exception:
        return None


# --------------------------------------------------------------------- build
def build_state(jobs_yaml: str | Path, run: str | None = None) -> dict:
    jobs_yaml = resolve_config(jobs_yaml)
    cfg = load_yaml(jobs_yaml)
    run = run or cfg.get("run") or Path(jobs_yaml).stem
    d = run_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    st = load_state(run) or {"run": run, "jobs": {}, "created": stamp()}
    st["jobs_yaml"] = str(Path(jobs_yaml).resolve())
    st["gpus"] = cfg.get("gpus") or detect_gpus()
    st["slots_per_gpu"] = int(cfg.get("slots_per_gpu", 1))
    st["cpu_slots"] = int(cfg.get("cpu_slots", 4))
    st["agent"] = cfg.get("agent")
    st["max_attempts"] = int(cfg.get("max_attempts", 2))
    for j in cfg["jobs"]:
        name = j["name"]
        prev = st["jobs"].get(name, {})
        rec = {"name": name, "kind": j["kind"], "gpu_req": j.get("gpu", "any"),
               "needs": j.get("needs", []), "config": j.get("config"),
               "command": j.get("command"), "adapter": j.get("adapter"),
               "sets": j.get("sets"), "backend": j.get("backend"), "only": j.get("only"),
               "timeout_s": j.get("timeout_s"),
               "log": str(d / f"{name}.log"),
               "result": str(d / f"{name}.result.json")}
        # resume: a done job is never relaunched; a failed or skipped one is queued again
        # (the fix that follows a failure is usually in the config or the code it calls).
        if prev.get("status") in ("done", "running"):
            rec.update({k: v for k, v in prev.items()
                        if k in ("status", "attempts", "gpu", "queued_at", "launched_at",
                                 "started_at", "ended_at", "exit_code", "wall_s", "inner_s",
                                 "dispatch_s", "overhead_s", "window")})
        else:
            rec.update({"status": "pending", "attempts": 0, "queued_at": stamp()})
        st["jobs"][name] = rec
    save_state(st)
    return st


# --------------------------------------------------------------------- scheduler
def _free_gpu(st: dict, want) -> int | None:
    used: dict[str, int] = {}
    for j in st["jobs"].values():
        if j["status"] == "running" and j.get("gpu") is not None:
            used[str(j["gpu"])] = used.get(str(j["gpu"]), 0) + 1
    if str(want) == "cpu":
        return "cpu" if used.get("cpu", 0) < int(st.get("cpu_slots", 4)) else None
    cands = st["gpus"] if str(want) == "any" else [int(want)]
    for g in cands:
        if used.get(str(g), 0) < st["slots_per_gpu"]:
            return g
    return None


def _deps_ok(st: dict, job: dict) -> bool | None:
    """True = runnable, False = wait, None = dependency failed (skip)."""
    for n in job.get("needs") or []:
        dep = st["jobs"].get(n)
        if dep is None:
            return None
        if dep["status"] in ("failed", "skipped"):
            return None
        if dep["status"] != "done":
            return False
    return True


def _launch(st: dict, job: dict, gpu) -> None:
    d = run_dir(st["run"])
    n = job["name"]
    for suf in (".rc", ".start", ".end", ".result.json"):
        (d / f"{n}{suf}").unlink(missing_ok=True)
    job["gpu"] = gpu                      # before the script is written: the CLI is told which card
    write_job_script(job, st, gpu)
    sess = session(st["run"])
    win = f"{n}" if job["attempts"] == 0 else f"{n}#{job['attempts']}"
    _tmux("new-window", "-d", "-t", f"{sess}:", "-n", win, f"bash {d}/{n}.sh")
    job.update({"status": "running", "attempts": job["attempts"] + 1,
                "launched_at": time.time(), "window": win})
    log(f"launch {n} gpu={gpu} attempt={job['attempts']}")


def _finish(st: dict, job: dict, rc: int) -> None:
    d = run_dir(st["run"])
    n = job["name"]
    t_start = _stamp_file(d / f"{n}.start")
    t_end = _stamp_file(d / f"{n}.end") or time.time()
    res = read_json(d / f"{n}.result.json", default={}) or {}
    job["started_at"] = t_start
    job["ended_at"] = t_end
    job["exit_code"] = rc
    job["wall_s"] = round(t_end - t_start, 2) if t_start else None
    job["dispatch_s"] = round(t_start - job["launched_at"], 2) if t_start else None
    job["inner_s"] = res.get("inner_seconds")
    if job["wall_s"] is not None and job["inner_s"] is not None:
        job["overhead_s"] = round(job["wall_s"] - job["inner_s"] + (job["dispatch_s"] or 0), 2)
    job["status"] = "done" if rc == 0 else "failed"
    if rc != 0 and job["attempts"] < st["max_attempts"]:
        job["status"] = "pending"
        log(f"{n} failed rc={rc}, retrying ({job['attempts']}/{st['max_attempts']})")
    else:
        log(f"{n} {job['status']} rc={rc} wall={job['wall_s']}s inner={job['inner_s']}s")


def scheduler(jobs_yaml: str | Path, run: str | None = None) -> int:
    st = build_state(jobs_yaml, run)
    run = st["run"]
    d = run_dir(run)
    st["scheduler"] = {"pid": os.getpid(), "heartbeat": time.time(), "started": stamp()}
    save_state(st)
    last_hb = 0.0
    log(f"scheduler run={run} gpus={st['gpus']} jobs={len(st['jobs'])}")
    while True:
        changed = False
        # 1. reap running jobs
        for job in st["jobs"].values():
            if job["status"] != "running":
                continue
            rc_file = d / f"{job['name']}.rc"
            if rc_file.exists():
                _finish(st, job, int(rc_file.read_text().strip() or 1))
                changed = True
            elif not _window_alive(run, job.get("window", job["name"])) and (
                    time.time() - job["launched_at"] > 20):
                log(f"{job['name']} window vanished without an exit code")
                _finish(st, job, 254)
                changed = True
        # 2. start what is runnable
        for job in st["jobs"].values():
            if job["status"] != "pending":
                continue
            ok = _deps_ok(st, job)
            if ok is None:
                job["status"] = "skipped"
                job["exit_code"] = None
                log(f"skip {job['name']}: dependency failed")
                changed = True
                continue
            if not ok:
                continue
            gpu = _free_gpu(st, job["gpu_req"])
            if gpu is None:
                continue
            _launch(st, job, gpu)
            changed = True
        st["scheduler"]["heartbeat"] = time.time()
        if changed:
            save_state(st)
        else:
            save_state(st)
        if all(j["status"] in TERMINAL for j in st["jobs"].values()):
            break
        if st.get("agent") and time.time() - last_hb > HEARTBEAT_EVERY:
            subprocess.run(["overnight-compute", "heartbeat", "--agent", st["agent"],
                            "--ttl", "30m"], capture_output=True)
            last_hb = time.time()
        time.sleep(POLL)
    st["finished"] = stamp()
    st["scheduler"]["pid"] = None
    save_state(st)
    print(status_table(st))
    n_fail = sum(1 for j in st["jobs"].values() if j["status"] != "done")
    log(f"queue {run} finished, {n_fail} job(s) not done")
    return 1 if n_fail else 0


def _window_alive(run: str, win: str) -> bool:
    out = _tmux("list-windows", "-t", session(run), "-F", "#{window_name}")
    if out.returncode != 0:
        return False
    return win in out.stdout.split()


# --------------------------------------------------------------------- submit
def submit(jobs_yaml: str | Path, run: str | None = None, foreground: bool = False) -> int:
    st = build_state(jobs_yaml, run)
    run = st["run"]
    sched = st.get("scheduler") or {}
    if _alive(sched.get("pid")) and time.time() - (sched.get("heartbeat") or 0) < 120:
        print(f"scheduler already running (pid {sched['pid']}) for run {run}")
        print(status_table(load_state(run)))
        return 0
    if all(j["status"] in TERMINAL for j in st["jobs"].values()) and st["jobs"]:
        print(f"nothing to do: every job in run {run} is terminal")
        print(status_table(st))
        return 0
    if foreground:
        return scheduler(jobs_yaml, run)
    sess = session(run)
    if _tmux("has-session", "-t", sess).returncode != 0:
        _tmux("new-session", "-d", "-s", sess, "-n", "idle", "sleep infinity")
    cmd = (f"PYTHONPATH={HARNESS}/src PHONON_PYTHON={python_exe()} "
           f"{python_exe()} -m phonon_research.cli queue {resolve_config(jobs_yaml).resolve()} "
           f"--run {run} --scheduler 2>&1 | tee -a {run_dir(run)}/scheduler.log")
    _tmux("new-window", "-d", "-t", f"{sess}:", "-n", "sched", f"bash -lc \"{cmd}\"")
    log(f"scheduler started in tmux session {sess} window sched")
    for _ in range(20):
        time.sleep(0.5)
        s = load_state(run)
        if (s.get("scheduler") or {}).get("pid"):
            break
    print(status_table(load_state(run)))
    print(f"\nlogs: {run_dir(run)}/<job>.log   state: {state_path(run)}")
    print(f"follow: tmux attach -t {sess}   status: phonon-research queue status --run {run}")
    return 0


def status_table(st: dict) -> str:
    rows = []
    for j in st.get("jobs", {}).values():
        rows.append([j["name"], j["kind"], j.get("gpu", j["gpu_req"]), j["status"],
                     j.get("attempts", 0),
                     "" if j.get("wall_s") is None else f"{j['wall_s']:.1f}",
                     "" if j.get("inner_s") is None else f"{j['inner_s']:.1f}",
                     "" if j.get("overhead_s") is None else f"{j['overhead_s']:.1f}",
                     "" if j.get("exit_code") is None else j["exit_code"]])
    head = ["job", "kind", "gpu", "status", "try", "wall_s", "inner_s", "overhead_s", "rc"]
    out = [f"run {st.get('run')}  ({st.get('jobs_yaml')})", table(rows, head)]
    return "\n".join(out)


def status(run: str | None = None, jobs_yaml: str | Path | None = None) -> int:
    if run is None:
        if jobs_yaml is None:
            runs = sorted(QUEUE_ROOT.glob("*/state.json"), key=lambda p: p.stat().st_mtime)
            if not runs:
                print(f"no runs under {QUEUE_ROOT}")
                return 1
            run = runs[-1].parent.name
        else:
            run = load_yaml(jobs_yaml).get("run") or resolve_config(jobs_yaml).stem
    st = load_state(run)
    if not st:
        print(f"no state for run {run}")
        return 1
    print(status_table(st))
    sched = st.get("scheduler") or {}
    live = _alive(sched.get("pid")) and time.time() - (sched.get("heartbeat") or 0) < 120
    print(f"scheduler: {'alive pid ' + str(sched.get('pid')) if live else 'not running'}"
          f"   finished: {st.get('finished', '-')}")
    return 0
