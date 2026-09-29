"""ASR dumps -> acoustic pair pool.

Same numerics as research/bigrun_v0/step5a_pool.py and research/pool3_v0/step4a_pool3.py:
zero-edit rows are capped at `zero_edit_cap` of each (sentence_idx, voice) bucket, the kept
zero-edit rows are drawn from an id-sorted list shuffled by random.Random(seed), and the pool
is written id-sorted.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..util import iter_jsonl, log, write_json

SIDX = re.compile(r"_s(\d+)$")


def _stage_of(sidx: int, stages: dict | str | None) -> str:
    """stages: a constant name, or {name: [lo, hi)} on sentence_idx (bigrun mid/big)."""
    if stages is None:
        return "pool"
    if isinstance(stages, str):
        return stages
    for name, span in stages.items():
        lo, hi = span
        if lo <= sidx < hi:
            return name
    return "other"


def build_pool(cfg: dict, out_dir: Path) -> dict:
    src = []
    for pat in cfg["asr"]:
        p = Path(pat)
        src += sorted(p.parent.glob(p.name)) if any(c in p.name for c in "*?[") else [p]
    cap = float(cfg.get("zero_edit_cap", 0.30))
    seed = int(cfg.get("seed", 7))
    prefix = cfg.get("id_prefix", "")
    stages = cfg.get("stages")
    source_tag = cfg.get("source", "tts_parakeet")
    out_path = Path(cfg.get("out") or (out_dir / "acoustic_pool.jsonl"))

    from whisper_normalizer.english import EnglishTextNormalizer
    N = EnglishTextNormalizer()

    rows = []
    for p in src:
        if not p.exists():
            log(f"missing {p}")
            continue
        rows += list(iter_jsonl(p))
    log(f"asr rows={len(rows)} from {[p.name for p in src]}")

    buckets: dict[tuple, list] = defaultdict(list)
    zeros: dict[tuple, list] = defaultdict(list)
    dropped = 0
    for r in rows:
        hyp = (r.get("hyp") or "").strip()
        tgt = (r.get("reference") or "").strip()
        m = SIDX.search(r.get("sentence_id") or "")
        if not hyp or not tgt or not m:
            dropped += 1
            continue
        sidx = int(m.group(1))
        rec = {
            "id": f"{prefix}{r['id']}",
            "term": r["term"], "kind": r.get("kind"), "voice": r["voice"],
            "sentence_idx": sidx, "stage": _stage_of(sidx, stages),
            "degraded": r.get("degraded"),
            "input": hyp, "target": tgt, "source": source_tag,
        }
        key = (sidx, r["voice"])
        (zeros if N(hyp) == N(tgt) else buckets)[key].append(rec)

    rng = random.Random(seed)
    out_rows, n_zero_kept, n_zero_avail = [], 0, 0
    for key in sorted(set(buckets) | set(zeros)):
        kept, ze = buckets.get(key, []), zeros.get(key, [])
        n_zero_avail += len(ze)
        n_keep = min(len(ze), int(cap / (1 - cap) * len(kept)))
        ze = sorted(ze, key=lambda r: r["id"])
        rng.shuffle(ze)
        z = ze[:n_keep]
        n_zero_kept += len(z)
        out_rows += kept + z
    out_rows.sort(key=lambda r: r["id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "asr_rows": len(rows), "dropped_empty": dropped, "pool": str(out_path),
        "pool_rows": len(out_rows), "pool_terms": len({r["term"] for r in out_rows}),
        "zero_edit_available": n_zero_avail, "zero_edit_kept": n_zero_kept,
        "zero_edit_fraction_pool": n_zero_kept / max(len(out_rows), 1),
        "voices": dict(Counter(r["voice"] for r in out_rows)),
        "stages": dict(Counter(r["stage"] for r in out_rows)),
        "sentence_idx": dict(sorted(Counter(r["sentence_idx"] for r in out_rows).items())),
        "kinds": dict(Counter(r["kind"] for r in out_rows).most_common()),
    }
    write_json(out_dir / "pool_stats.json", stats)
    return stats
