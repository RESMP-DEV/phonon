"""`phonon-research train <config.yaml>`: named LoRA runs through research/corrector_v0/train_lora.py.

train_lora.py is called, never forked. The harness only builds the argv, runs it with the
already-resolved harness interpreter (so a run pays no uv resolve), times it, and writes the
registry entry: adapter path, train_meta.json, corpus id, git rev, wall time.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from . import registry
from .paths import ADAPTER_ROOT, REPO, TRAIN_LORA, hf_env
from .util import git_rev, load_yaml, log, python_exe, read_json, resolve_config, stamp, write_json

# config key -> train_lora.py flag
KEYS = {"model": "--model", "train": "--train", "dev": "--dev", "epochs": "--epochs",
        "lr": "--lr", "max_seq_len": "--max-seq-len", "lora_r": "--lora-r",
        "lora_dropout": "--lora-dropout", "batch_size": "--batch-size",
        "grad_accum": "--grad-accum", "limit": "--limit", "max_seconds": "--max-seconds",
        "init_adapter": "--init-adapter", "keep_weight": "--keep-weight",
        "grad_checkpointing": "--grad-checkpointing", "optim": "--optim",
        "attn_impl": "--attn-impl", "sdpa_backend": "--sdpa-backend",
        "group_by_length": "--group-by-length", "fused_ce": "--fused-ce",
        "packing": "--packing", "padding_free": "--padding-free", "compile": "--compile",
        "dataloader_workers": "--dataloader-workers"}


def build_argv(cfg: dict) -> list[str]:
    argv = [python_exe(), str(TRAIN_LORA), "--name", str(cfg["name"])]
    for k, flag in KEYS.items():
        if cfg.get(k) is not None:
            argv += [flag, str(cfg[k])]
    if cfg.get("force", True):
        argv.append("--force")
    for k, v in (cfg.get("args") or {}).items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    argv += list(cfg.get("flags") or [])
    return argv


def run_train(config_path: str | Path, gpu: int | str | None = None,
              result_path: str | Path | None = None) -> dict:
    cfg = load_yaml(config_path)
    name = cfg["name"]
    gpu = cfg.get("gpu", 1) if gpu is None else gpu
    out = Path(cfg.get("adapter_dir") or ADAPTER_ROOT / name)
    if cfg.get("force", True) and out.exists():
        shutil.rmtree(out, ignore_errors=True)
    argv = build_argv(cfg)
    log(f"train {name} gpu={gpu} rows_file={cfg.get('train')}")
    log("exec: " + " ".join(argv))
    t0 = time.perf_counter()
    rc = subprocess.call(argv, env=hf_env(gpu), cwd=str(REPO))
    inner = time.perf_counter() - t0
    log(f"TRAIN_EXIT {name} rc={rc} [{inner:.1f}s]")
    meta = read_json(out / "train_meta.json", default={})
    if rc == 0 and cfg.get("cleanup", True):
        for p in list(out.glob("checkpoint-*")) + [out / "trainer"]:
            shutil.rmtree(p, ignore_errors=True)
    record = {"adapter": str(out), "name": name, "rc": rc,
              "wall_seconds": round(inner, 2),
              "train_seconds": meta.get("seconds"),
              "corpus": cfg.get("corpus"), "train_file": cfg.get("train"),
              "model": cfg.get("model") or meta.get("model"),
              "gpu": gpu, "git_rev": git_rev(), "finished": stamp(),
              "config": str(resolve_config(config_path).resolve()), "train_meta": meta}
    if rc == 0:
        registry.record_adapter(name, record)
    res = {"kind": "train", "name": name, "rc": rc, "inner_seconds": round(inner, 2),
           "adapter": str(out), "train_meta_seconds": meta.get("seconds"),
           "steps": meta.get("steps"), "train_rows": meta.get("train_rows")}
    if result_path:
        write_json(result_path, res)
    return res
