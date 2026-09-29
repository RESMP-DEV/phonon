"""`phonon-research corpus <config.yaml>`: run the stages a corpus config asks for."""
from __future__ import annotations

import time
from pathlib import Path

from ..paths import CORPUS_ROOT
from ..util import git_rev, load_yaml, log, resolve_config, stamp, table, write_json
from . import external, lists, mix, pool

STAGES = ("sentences", "jobs", "tts", "asr", "pool", "lists", "mix", "verify")


def run_corpus(config_path: str | Path, only: list[str] | None = None) -> dict:
    cfg = load_yaml(config_path)
    name = cfg.get("name") or resolve_config(config_path).stem
    out_dir = Path(cfg.get("out_dir") or CORPUS_ROOT / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [s for s in (only or cfg.get("stages") or []) if s in STAGES]
    if not stages:
        raise SystemExit(f"no stages to run (config stages={cfg.get('stages')})")
    held = cfg.get("heldout") or []
    lexicon = cfg.get("lexicon")
    meta = {"name": name, "config": str(resolve_config(config_path).resolve()), "out_dir": str(out_dir),
            "git_rev": git_rev(), "started": stamp(), "stages": {}}
    log(f"corpus {name} stages={stages} out_dir={out_dir}")
    for s in stages:
        t0 = time.perf_counter()
        if s == "sentences":
            r = external.build_sentences(cfg["sentences"], out_dir)
        elif s == "jobs":
            r = external.build_jobs(cfg["jobs"], out_dir)
        elif s == "tts":
            r = external.run_tts(cfg["tts"], out_dir)
        elif s == "asr":
            r = external.run_asr(cfg["asr"], out_dir)
        elif s == "pool":
            r = pool.build_pool(cfg["pool"], out_dir)
        elif s == "lists":
            r = lists.build_lists(cfg["lists"], out_dir)
        elif s == "mix":
            r = mix.build_mix(cfg["mix"], out_dir, held, lexicon)
        elif s == "verify":
            r = mix.verify_mix(cfg["mix"], out_dir, held, lexicon)
        meta["stages"][s] = {"seconds": round(time.perf_counter() - t0, 2), "result": r}
        log(f"STAGE {s} done [{time.perf_counter() - t0:.1f}s]")
    meta["finished"] = stamp()
    meta["seconds"] = sum(v["seconds"] for v in meta["stages"].values())
    write_json(out_dir / "corpus_meta.json", meta)
    rows = [[s, f"{v['seconds']:.1f}"] for s, v in meta["stages"].items()]
    print(table(rows, ["stage", "seconds"]))
    if "verify" in meta["stages"]:
        v = meta["stages"]["verify"]["result"]
        print(table([[v["rows_checked"], v["identical"], v["different"],
                      v["missing_source_row"], f"{v['identical_fraction']:.4f}"]],
                    ["checked", "identical", "different", "missing", "fraction"]))
    return meta
