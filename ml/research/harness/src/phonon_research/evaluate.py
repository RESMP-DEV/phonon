"""`phonon-research eval <adapter> [--sets ...]`: hand an adapter to phonon-bench.

The harness owns the registry of eval sets (registry/sets.yaml) and the adapter lookup; the
numbers come from the sibling benchmark package research/bench_v0 (`phonon-bench`). When that
package is not on disk yet the call is a recorded stub: the exact command line is printed and
written to the result file, and the run is marked `stub` so no table ever shows a number the
benchmark did not produce.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from . import registry
from .paths import BENCH_PROJECT, REPO, hf_env
from .util import git_rev, load_yaml, log, read_json, resolve_config, stamp, write_json


def bench_available() -> bool:
    return (BENCH_PROJECT / "pyproject.toml").exists()


def base_model_of(adapter: str) -> str | None:
    """phonon-bench defaults to LFM2.5-1.2B; an adapter knows its own base, so pass it."""
    cfg = read_json(Path(adapter) / "adapter_config.json", default={}) or {}
    return cfg.get("base_model_name_or_path")


def bench_argv(adapter: str | None, set_names: list[str], backend: str, label: str,
               extra: list[str] | None = None, base: str | None = None,
               exe: str | None = None) -> list[str]:
    """`exe` is a phonon-bench executable in another venv: vLLM pins its own torch and cannot
    share the bench venv, so the queue could not reach the fastest backend without this."""
    argv = ([str(exe), "run"] if exe
            else ["uv", "run", "--project", str(BENCH_PROJECT), "phonon-bench", "run"])
    argv += ["--backend", backend, "--label", label]
    if adapter:
        argv += ["--adapter", adapter]
    base = base or base_model_of(adapter)
    if base:
        argv += ["--base", base]
    if set_names:
        argv += ["--sets", ",".join(set_names)]
    return argv + [str(e) for e in (extra or [])]


def run_eval(adapter_ref: str, set_names: list[str] | None = None, backend: str = "hf",
             label: str | None = None, gpu: int | str | None = None,
             extra: list[str] | None = None, result_path: str | Path | None = None,
             base: str | None = None, exe: str | None = None) -> dict:
    if adapter_ref in (None, "", "none"):
        adapter, reg_name = None, None       # a merged / converted model, named by --base
    else:
        adapter, reg_name = registry.resolve_adapter(adapter_ref)
    sets = registry.resolve_sets(set_names)
    names = [s["name"] for s in sets]
    label = label or (reg_name or (Path(adapter).name if adapter else backend))
    argv = bench_argv(adapter, names, backend, label, extra, base, exe)
    res = {"kind": "eval", "adapter": adapter, "adapter_name": reg_name, "sets": names,
           "backend": backend, "label": label, "git_rev": git_rev(), "finished": stamp(),
           "command": " ".join(argv)}
    if not bench_available():
        res.update({"status": "stub", "rc": 0, "inner_seconds": 0.0,
                    "reason": f"{BENCH_PROJECT} does not exist yet (sibling agent opus-bench)",
                    "sets_resolved": {s["name"]: s.get("rows_file") or s.get("path")
                                      for s in sets}})
        log("phonon-bench is not installed yet; recording a STUB eval")
        log("would run: " + res["command"])
    else:
        log("exec: " + res["command"])
        t0 = time.perf_counter()
        rc = subprocess.call(argv, env=hf_env(gpu), cwd=str(REPO))
        res.update({"status": "ok" if rc == 0 else "failed", "rc": rc,
                    "inner_seconds": round(time.perf_counter() - t0, 2)})
    if result_path:
        write_json(result_path, res)
    return res


def run_eval_config(config_path: str | Path, gpu: int | str | None = None,
                    result_path: str | Path | None = None) -> dict:
    cfg = load_yaml(resolve_config(config_path))
    return run_eval(cfg.get("adapter"), cfg.get("sets"), cfg.get("backend", "hf"),
                    cfg.get("label"), cfg.get("gpu", gpu), cfg.get("extra"), result_path,
                    cfg.get("base"), cfg.get("exe"))
