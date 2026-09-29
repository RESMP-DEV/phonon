"""Stages that drive the existing generator / TTS / ASR scripts.

Nothing is reimplemented here: sentences come from research/scaling_v0/gen_more.py (the
rotating-opener prefill generator), TTS from research/bigrun_v0/tts_jobs.py (Kokoro, the
stable-hash degradation), ASR from research/term_eval_v0/asr_clips.py (Parakeet TDT via NeMo).
The first two run in the harness environment (already resolved, no per-call uv resolve); the
ASR script needs nemo_toolkit, which lives in the repo project environment, so it is the one
stage that shells out through `uv run --project <repo>`.

`jobs` is the one piece of numerics that lived in a script (make_jobs.py) and is now a module:
for term index t, stage `mid` uses voices (t+0, t+1, t+2) mod 9 and `big` (t+3, t+4, t+5) mod 9,
halves split by t % 2.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..paths import ASR_CLIPS, GEN_SENTENCES, REPO, TTS_JOBS, hf_env
from ..util import iter_jsonl, log, python_exe, run, write_json

TRAIN_VOICES = ["af_heart", "am_adam", "bf_emma", "af_bella", "am_michael", "bm_george",
                "af_nicole", "am_puck", "bm_lewis"]


def build_sentences(cfg: dict, out_dir: Path) -> dict:
    out = Path(cfg.get("out") or out_dir / "sentences.jsonl")
    cmd = [python_exe(), str(GEN_SENTENCES),
           "--terms", str(cfg["terms"]), "--out", str(out),
           "--per-term", str(cfg.get("per_term", 8)),
           "--batch-size", str(cfg.get("batch_size", 128)),
           "--rounds", str(cfg.get("rounds", 10)),
           "--seed", str(cfg.get("seed", 918)),
           "--agent", str(cfg.get("agent", "opus-harness"))]
    if cfg.get("existing"):
        cmd += ["--existing", str(cfg["existing"])]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    rc, wall = run(cmd, env=hf_env(cfg.get("gpu", 1)))
    if rc != 0:
        raise RuntimeError(f"sentence generation failed rc={rc}")
    return {"rc": rc, "seconds": wall, "out": str(out)}


def build_jobs(cfg: dict, out_dir: Path) -> dict:
    """sentences -> TTS jobs (voice round-robin per term, split into halves)."""
    voices = cfg.get("voices") or TRAIN_VOICES
    nv = len(voices)
    halves = int(cfg.get("halves", 2))
    stages = cfg.get("stages") or {"mid": {"sidx": [0, 4], "voice_offset": 0, "n_voices": 3},
                                   "big": {"sidx": [4, 8], "voice_offset": 3, "n_voices": 3}}
    order = {}
    for i, r in enumerate(iter_jsonl(cfg["terms"])):
        order[r["term"]] = i
    by_term: dict[str, list[dict]] = {}
    n_sent_rows = 0
    for r in iter_jsonl(cfg["sentences"]):
        by_term.setdefault(r["term"], []).append(r)
        n_sent_rows += 1
    for v in by_term.values():
        v.sort(key=lambda r: int(r["id"].rsplit("_s", 1)[1]))

    out: dict[tuple, list] = {(s, h): [] for s in stages for h in range(halves)}
    vc, n_sent = Counter(), Counter()
    for term, sents in by_term.items():
        t = order.get(term)
        if t is None:
            continue
        half = t % halves
        for stage, sc in stages.items():
            lo, hi = sc.get("sidx", [0, 4])
            voff = int(sc.get("voice_offset", 0))
            k_voices = int(sc.get("n_voices", 3))
            vs = [voices[(t + voff + k) % nv] for k in range(k_voices)]
            for s in sents:
                sidx = int(s["id"].rsplit("_s", 1)[1])
                if not (lo <= sidx < hi):
                    continue
                n_sent[stage] += 1
                for v in vs:
                    vc[v] += 1
                    out[(stage, half)].append({
                        "id": f"{s['id']}__{v}", "sentence_id": s["id"], "term": term,
                        "kind": s.get("kind"), "voice": v, "sentence": s["sentence"]})
    tot = 0
    written = []
    for (stage, half), jobs in out.items():
        p = out_dir / f"jobs_{stage}_half{half}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as h:
            for j in jobs:
                h.write(json.dumps(j, ensure_ascii=False) + "\n")
        tot += len(jobs)
        written.append({"path": str(p), "clips": len(jobs)})
        log(f"{p.name}: {len(jobs)} clips")
    stats = {"terms": len(by_term), "sentences": n_sent_rows, "clips_total": tot,
             "files": written, "sentences_per_stage": dict(n_sent),
             "voice_counts": dict(sorted(vc.items()))}
    write_json(out_dir / "jobs_stats.json", stats)
    return stats


def run_tts(cfg: dict, out_dir: Path) -> dict:
    res = []
    for sh in cfg["shards"]:
        stage, half = sh["stage"], int(sh["half"])
        gpu = sh.get("gpu", 1)
        manifest = out_dir / f"tts_manifest_{stage}_half{half}.jsonl"
        cmd = [python_exe(), str(TTS_JOBS),
               "--jobs", str(out_dir / f"jobs_{stage}_half{half}.jsonl"),
               "--clips-dir", str(cfg.get("clips_dir") or out_dir / "clips"),
               "--manifest", str(manifest),
               "--shard-dir", str(cfg.get("shard_dir") or out_dir / "tts_shards"),
               "--workers", str(cfg.get("workers", 7))]
        rc, wall = run(cmd, env=hf_env(gpu))
        res.append({"stage": stage, "half": half, "gpu": gpu, "rc": rc, "seconds": wall,
                    "manifest": str(manifest)})
        log(f"TTS {stage} half{half} rc={rc} [{wall:.0f}s]")
        if rc != 0:
            raise RuntimeError(f"TTS {stage} half{half} failed rc={rc}")
    return {"shards": res}


def run_asr(cfg: dict, out_dir: Path) -> dict:
    res = []
    for sh in cfg["shards"]:
        stage, half = sh["stage"], int(sh["half"])
        gpu = sh.get("gpu", 1)
        out = out_dir / f"asr_{stage}_half{half}.jsonl"
        cmd = ["uv", "run", "--project", str(REPO), "python", str(ASR_CLIPS),
               "--manifest", str(out_dir / f"tts_manifest_{stage}_half{half}.jsonl"),
               "--out", str(out), "--batch-size", str(cfg.get("batch_size", 96))]
        rc, wall = run(cmd, env=hf_env(gpu))
        res.append({"stage": stage, "half": half, "gpu": gpu, "rc": rc, "seconds": wall,
                    "out": str(out)})
        log(f"ASR {stage} half{half} rc={rc} [{wall:.0f}s]")
        if rc != 0:
            raise RuntimeError(f"ASR {stage} half{half} failed rc={rc}")
    return {"shards": res}
