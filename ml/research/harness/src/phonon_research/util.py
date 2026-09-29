"""Small shared helpers: jsonl io, timing, subprocess, git rev, yaml."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from .paths import REPO


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def resolve_config(path: str | Path) -> Path:
    """Config paths work from anywhere: as given, then relative to the repo root."""
    p = Path(path)
    if p.exists():
        return p
    alt = REPO / p
    if alt.exists():
        return alt
    raise SystemExit(f"config not found: {path}")


def load_yaml(path: str | Path) -> dict:
    with resolve_config(path).open(encoding="utf-8") as h:
        return yaml.safe_load(h) or {}


def dump_yaml(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        yaml.safe_dump(obj, h, sort_keys=False, default_flow_style=False, width=100)
    tmp.replace(p)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def git_rev(repo: Path = REPO) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def now() -> float:
    return time.time()


def stamp(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timed(label: str) -> Iterator[dict]:
    d: dict = {"label": label, "t0": time.time()}
    t0 = time.perf_counter()
    log(f"START {label}")
    try:
        yield d
    finally:
        d["seconds"] = time.perf_counter() - t0
        log(f"DONE  {label} [{d['seconds']:.1f}s]")


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None,
        log_path: Path | None = None) -> tuple[int, float]:
    """Run a command, stream to log_path (or inherit stdout). Returns (rc, wall seconds)."""
    log("exec: " + " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as h:
            rc = subprocess.call([str(c) for c in cmd], env=env, cwd=str(cwd or REPO),
                                 stdout=h, stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call([str(c) for c in cmd], env=env, cwd=str(cwd or REPO))
    return rc, time.perf_counter() - t0


def python_exe() -> str:
    """The interpreter of the resolved harness environment (no uv resolve per call)."""
    return os.environ.get("PHONON_PYTHON", sys.executable)


def table(rows: list[list[Any]], head: list[str]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(head[i]), *(len(r[i]) for r in cells)) if cells else len(head[i])
              for i in range(len(head))]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(head)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in cells:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)
