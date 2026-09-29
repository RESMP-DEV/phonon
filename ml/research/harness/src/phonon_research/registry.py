"""The two registries: adapters (written by `train`) and eval sets (read by `eval`)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .paths import ADAPTERS_YAML, SETS_YAML
from .util import dump_yaml, load_yaml


def _lock(path: Path, timeout: float = 30.0):
    """Crude cross-process lock so two queue jobs never interleave a registry write."""
    lock = path.with_suffix(path.suffix + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock
        except FileExistsError:
            if time.time() - t0 > timeout:
                lock.unlink(missing_ok=True)
            time.sleep(0.2)


def adapters() -> dict:
    return load_yaml(ADAPTERS_YAML) if ADAPTERS_YAML.exists() else {}


def record_adapter(name: str, record: dict) -> None:
    lock = _lock(ADAPTERS_YAML)
    try:
        reg = adapters()
        history = (reg.get(name) or {}).get("history", [])
        prev = {k: v for k, v in (reg.get(name) or {}).items() if k != "history"}
        if prev:
            history = (history + [prev])[-5:]
        record["history"] = history
        reg[name] = record
        dump_yaml(ADAPTERS_YAML, reg)
    finally:
        lock.unlink(missing_ok=True)


def resolve_adapter(ref: str) -> tuple[str, str | None]:
    """name in the registry, or a path. Returns (path, registry name or None)."""
    reg = adapters()
    if ref in reg:
        return reg[ref]["adapter"], ref
    p = Path(ref)
    if p.exists():
        for n, r in reg.items():
            if r.get("adapter") == str(p):
                return str(p), n
        return str(p), None
    raise SystemExit(f"adapter {ref!r} is neither a registry name nor an existing path")


def sets() -> dict:
    return load_yaml(SETS_YAML) if SETS_YAML.exists() else {}


def resolve_sets(names: list[str] | None) -> list[dict]:
    reg = sets()
    known = reg.get("sets") or {}
    if not names:
        names = list(reg.get("default") or known)
    out = []
    for n in names:
        if n not in known:
            raise SystemExit(f"unknown eval set {n!r}; registry has {sorted(known)}")
        out.append({"name": n, **known[n]})
    return out
