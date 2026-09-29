"""Retrieval top-k lists for a row file.

`retriever: v1` is research/vocab_v1/retrieve.Retriever, the one bigrun/pool2/pool3 trained on
(0.5 char + 0.5 double-metaphone over whole terms).  `retriever: v2` is
research/retrieval_v2/retrieve2.RetrieverV2 (sub-word realisations, phonemes, two-stage rerank).
Both are imported, never copied, so list numerics cannot drift from the shipped code.

`budget` turns the fixed top-k into the index_v0 adaptive budget
min(cap, max(30, mult x hypothesis words)) with a score threshold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..paths import research_syspath
from ..util import iter_jsonl, log, write_json


def term_sources(sources: list[str]) -> tuple[list[str], dict[str, float], dict[str, str]]:
    seen: set[str] = set()
    terms: list[str] = []
    rank: dict[str, float] = {}
    kind: dict[str, str] = {}
    for src in sources:
        for r in iter_jsonl(src):
            t = (r.get("term") or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            terms.append(t)
            rank[t] = float(r.get("rank_score") or 0.0)
            kind[t] = r.get("kind") or "other"
    return terms, rank, kind


def build_retriever(cfg: dict):
    research_syspath()
    terms, rank, kind = term_sources(cfg["sources"])
    version = str(cfg.get("retriever", "v1"))
    log(f"retriever={version} terms={len(terms)}")
    if version == "v1":
        from retrieve import Retriever  # research/vocab_v1

        return Retriever(terms, rank), len(terms)
    from retrieve2 import Index, Phoneme, RetrieverV2  # research/retrieval_v2

    v2 = cfg.get("v2") or {}
    ph = None
    if v2.get("phonemes", True):
        ph = Phoneme()
        ix_tmp = Index(terms, rank, kind, use_phonemes=False, nreal=int(v2.get("nreal", 5)))
        n_new = ph.warm(ix_tmp.words())
        if n_new:
            ph.save()
        log(f"g2p warmed {n_new} new words")
    ix = Index(terms, rank, kind, use_phonemes=ph is not None, ph=ph,
               nreal=int(v2.get("nreal", 5)))
    R = RetrieverV2(ix, nmax=int(v2.get("nmax", 6)),
                    w_char=float(v2.get("w_char", 0.5)), w_meta=float(v2.get("w_meta", 0.5)),
                    w_ph=float(v2.get("w_ph", 0.0)),
                    stage1=tuple(v2["stage1"]) if v2.get("stage1") else None,
                    rerank=int(v2.get("rerank", 0)))
    return R, len(terms)


def budget_len(nwords: int, mult: int = 3, cap: int = 80) -> int:
    return min(cap, max(30, mult * nwords))


def dump_lists(R, rows_path: str | Path, key: str, k: int, out: Path,
               block: int = 12000, budget: dict | None = None) -> dict:
    t0 = time.perf_counter()
    rows = list(iter_jsonl(rows_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    n_terms = 0
    with out.open("w", encoding="utf-8") as h:
        for s in range(0, len(rows), block):
            chunk = rows[s:s + block]
            lists = R.topk([r[key] for r in chunk], k=k)
            for r, L in zip(chunk, lists):
                terms = [t for t, _ in L]
                if budget:
                    thr = float(budget.get("threshold", 0))
                    b = budget_len(len((r[key] or "").split()),
                                   int(budget.get("mult", 3)), int(budget.get("cap", 80)))
                    terms = [t for t, sc in L[:b] if sc >= thr]
                n_terms += len(terms)
                h.write(json.dumps({"id": r["id"], "terms": terms}, ensure_ascii=False) + "\n")
            log(f"  {out.name} {min(s + block, len(rows))}/{len(rows)} "
                f"[{time.perf_counter() - t0:.0f}s]")
    return {"rows": len(rows), "out": str(out), "mean_list_len": n_terms / max(len(rows), 1),
            "seconds": time.perf_counter() - t0}


def build_lists(cfg: dict, out_dir: Path) -> dict:
    R, n_terms = build_retriever(cfg)
    stats = {"retriever": cfg.get("retriever", "v1"), "pool_terms": n_terms, "dumps": {}}
    for d in cfg["dumps"]:
        out = Path(d["out"]) if "/" in str(d["out"]) else out_dir / d["out"]
        stats["dumps"][out.name] = dump_lists(
            R, d["rows"], d.get("key", "input"), int(d["k"]), out,
            block=int(cfg.get("block", 12000)), budget=cfg.get("budget"))
    write_json(out_dir / "lists_stats.json", stats)
    return stats
