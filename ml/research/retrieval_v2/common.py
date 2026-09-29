"""Shared pool loaders and eval-set loaders for retrieval v2."""
from __future__ import annotations
import json, re
from pathlib import Path

LEX_FULL = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
LEX_V1 = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl")
ASR_NEW = Path("/data/phonon_bigrun_v0/heldout_new_asr.jsonl")
ASR_OLD_UNSEEN = Path("/data/phonon_term_eval_v1/asr.jsonl")
ASR_OLD_SEEN = Path("/data/phonon_term_eval_v0/asr.jsonl")
REAL_HOLDOUT = Path("/data/phonon_corrector_v0/datesplit/holdout_date_audio.jsonl")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _pool(srcs, lower=True):
    seen, terms, rank, kind = set(), [], {}, {}
    for src in srcs:
        for r in read_jsonl(src):
            t = (r.get("term") or "").strip()
            key = t.lower() if lower else t
            if not t or key in seen:
                continue
            if not re.sub(r"[^a-z0-9]+", "", t.lower()):
                continue
            seen.add(key)
            terms.append(t)
            rank[t] = float(r.get("rank_score") or 0.0)
            kind[t] = r.get("kind") or ""
    return terms, rank, kind


def pool_big():
    """15.6k pool: full lexicon + both held-out sets (step5b_lists.build_retriever)."""
    return _pool([LEX_FULL, HELD_OLD, HELD_NEW])


def pool_small():
    """4.7k pool used by the heldout_old conditions (retrieve.load_pool)."""
    return _pool([LEX_V1, HELD_OLD], lower=False)


LEX_POOL2 = Path("/data/phonon_pool2_v0/terms_pool2.jsonl")
HELD_P2 = Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl")
ASR_POOL2 = [Path("/data/phonon_pool2_v0/asr_heldout_half0.jsonl"),
             Path("/data/phonon_pool2_v0/asr_heldout_half1.jsonl")]


def pool_union():
    """38.2k union pool: bigrun lexicon + pool2 + every held-out set (pool2_v0/step4b_lists)."""
    return _pool([LEX_FULL, HELD_OLD, HELD_NEW, LEX_POOL2, HELD_P2])


def load_pool2_rows():
    rows = []
    for p in ASR_POOL2:
        rows += read_jsonl(p)
    rows.sort(key=lambda r: r["id"])
    return rows
