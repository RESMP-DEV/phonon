"""Frozen benchmark fixtures.

`phonon-bench fixtures build` copies the research-tree eval sets into
/data/phonon_bench_v0/sets/ as flat jsonl with one schema, and writes manifest.json with a
sha256 and a row count per file.  Nothing else in the package reads the research tree, so a run
depends only on these files.

Row schema (every set):
    id, input, reference, term, kind, voice, retrieved_has_term,
    vocab_retrieved, vocab_oracle, true_terms
`term` is "" on the real-dictation sets; `vocab_oracle` is [] where the source has none.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .metrics import read_jsonl, squash, write_jsonl

BENCH = Path("/data/phonon_bench_v0")
SETS_DIR = BENCH / "sets"

# ---- sources in the research tree (read once, at build time) ----------------
SRC = {
    "real_580": {
        "kind": "real",
        "rows": "/data/phonon_corrector_v0/datesplit/holdout_date_audio.jsonl",
        "lists": "/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl",
        "note": "580-row date-split real-audio holdout; v1 retrieved lists (30 slots)",
    },
    "real_holdout120": {
        "kind": "real",
        "rows": "pairs:wispr_holdout120",
        "lists": None,
        "note": "120 Wispr dictations, Parakeet v2 raw in, kept text out; no vocabulary line",
    },
    "term_old_seen": {
        "kind": "term",
        "rows": "/data/phonon_vocab_v1/eval_conditions.jsonl",
        "note": "term_eval_v0 asr.jsonl clips (300 repo-walk terms, seen voices), v1 lists",
    },
    "term_old_unseen": {
        "kind": "term",
        "rows": "/data/phonon_retrieval_v2/cond_unseen_v2.jsonl",
        "note": "term_eval_v1 asr.jsonl clips (same terms, unseen voices), retrieval v2 lists",
    },
    "term_new": {
        "kind": "term",
        "rows": "/data/phonon_retrieval_v2/cond_new_v2.jsonl",
        "note": "bigrun heldout_new (300 new terms, new voices), retrieval v2 lists",
    },
    "term_pool2": {
        "kind": "term",
        "rows": "/data/phonon_retrieval_v2/cond_pool2_v2.jsonl",
        "note": "pool2 held-out asr (300 terms outside the repo walk), retrieval v2 lists",
    },
    "term_pool3": {
        "kind": "term",
        "rows": "/data/phonon_pool3_v0/eval_conditions_pool3.jsonl",
        "note": "pool3 held-out asr (300 terms from 150 public repos), v2 lists + budget lists",
    },
}
NUMERICS_SOURCES = ("real_580", "term_new")
NUMERICS_N = 500
NUMERICS_SEED = 20260921

LEXICONS = {
    "lexicon_big.jsonl": ["/data/phonon_synth_v0/lexicon_ranked.jsonl",
                          "/data/phonon_term_eval_v0/terms_heldout.jsonl",
                          "/data/phonon_bigrun_v0/terms_heldout_new.jsonl"],
    "lexicon_small.jsonl": ["/data/phonon_synth_v1/lexicon_ranked.jsonl",
                            "/data/phonon_term_eval_v0/terms_heldout.jsonl"],
    "lexicon_union.jsonl": ["/data/phonon_synth_v0/lexicon_ranked.jsonl",
                            "/data/phonon_term_eval_v0/terms_heldout.jsonl",
                            "/data/phonon_bigrun_v0/terms_heldout_new.jsonl",
                            "/data/phonon_pool2_v0/terms_pool2.jsonl",
                            "/data/phonon_pool2_v0/terms_heldout_pool2.jsonl",
                            "/data/phonon_pool3_v0/terms_pool3.jsonl",
                            "/data/phonon_pool3_v0/terms_heldout_pool3.jsonl"],
}
AUX = {
    "english_words.txt": "/data/phonon_asr_errors_v0/english_words.txt",
    "g2p_cache.json": "/data/phonon_retrieval_v2/g2p_cache.json",
    "kind_prior.json": "/data/phonon_retrieval_v2/prior.json",
}

SET_ORDER = ["real_580", "real_holdout120", "term_old_seen", "term_old_unseen", "term_new",
             "term_pool2", "term_pool3", "numerics_500"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wispr_holdout120() -> list[dict]:
    pairs = read_jsonl("/data/phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl")
    pk = {r["id"]: r for r in read_jsonl("/data/phonon_corrector_v0/parakeet_v2_wispr.jsonl")}
    out = []
    for r in pairs:
        if r.get("split") != "wispr_holdout120":
            continue
        out.append({"id": r["id"], "input": pk.get(r["id"], {}).get("parakeet_raw") or "",
                    "reference": r.get("target") or ""})
    return out


def _norm_row(r: dict, kind: str, lists: dict | None) -> dict:
    rid = r["id"]
    extra = (lists or {}).get(rid, {})
    vr = r.get("vocab_retrieved", extra.get("vocab_retrieved", []))
    return {
        "id": rid,
        "input": r.get("input") if kind == "real" else r.get("hyp"),
        "reference": r.get("reference", ""),
        "term": r.get("term", ""),
        "kind": r.get("kind", ""),
        "voice": r.get("voice", ""),
        "retrieved_has_term": bool(r.get("retrieved_has_term", False)),
        "vocab_retrieved": list(vr or []),
        "vocab_oracle": list(r.get("vocab_oracle", []) or []),
        "true_terms": list(r.get("true_terms", extra.get("true_terms", [])) or []),
    }


def build(out_dir: Path = SETS_DIR, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sets": {}, "lexicons": {}, "aux": {}, "numerics": {
        "sources": list(NUMERICS_SOURCES), "n": NUMERICS_N, "seed": NUMERICS_SEED}}
    built: dict[str, list[dict]] = {}
    for name, spec in SRC.items():
        dst = out_dir / f"{name}.jsonl"
        if spec["rows"] == "pairs:wispr_holdout120":
            src_rows = _wispr_holdout120()
        else:
            src_rows = read_jsonl(spec["rows"])
        lists = None
        if spec.get("lists"):
            lists = {r["id"]: r for r in read_jsonl(spec["lists"])}
        rows = [_norm_row(r, spec["kind"], lists) for r in src_rows]
        built[name] = rows
        if force or not dst.exists():
            write_jsonl(dst, rows)
        manifest["sets"][name] = {"rows": len(rows), "kind": spec["kind"],
                                  "note": spec["note"], "source": spec["rows"],
                                  "sha256": sha256(dst)}

    # ---- numerics_500: fixed-seed sample of the two generation-heavy sets ----
    pool = []
    for s in NUMERICS_SOURCES:
        for r in built[s]:
            pool.append(dict(r, id=f"{s}:{r['id']}", set=s))
    rng = random.Random(NUMERICS_SEED)
    rng.shuffle(pool)
    num = sorted(pool[:NUMERICS_N], key=lambda r: r["id"])
    dst = out_dir / "numerics_500.jsonl"
    if force or not dst.exists():
        write_jsonl(dst, num)
    manifest["sets"]["numerics_500"] = {
        "rows": len(num), "kind": "numerics",
        "note": f"{NUMERICS_N} rows sampled seed={NUMERICS_SEED} from "
                f"{'+'.join(NUMERICS_SOURCES)}, for teacher-forced numerics",
        "source": "+".join(NUMERICS_SOURCES), "sha256": sha256(dst)}

    # ---- lexicons -----------------------------------------------------------
    for fname, srcs in LEXICONS.items():
        dst = out_dir / fname
        if force or not dst.exists():
            seen, rows = set(), []
            for src in srcs:
                p = Path(src)
                if not p.exists():
                    continue
                for r in read_jsonl(p):
                    t = (r.get("term") or "").strip()
                    k = t.lower()
                    if not t or k in seen or not squash(t):
                        continue
                    seen.add(k)
                    rows.append({"term": t, "kind": r.get("kind") or "",
                                 "rank_score": float(r.get("rank_score") or 0.0)})
            write_jsonl(dst, rows)
        manifest["lexicons"][fname] = {"terms": sum(1 for _ in dst.open()),
                                       "sources": srcs, "sha256": sha256(dst)}

    # ---- aux files ----------------------------------------------------------
    for fname, src in AUX.items():
        dst = out_dir / fname
        p = Path(src)
        if not p.exists():
            manifest["aux"][fname] = {"error": f"missing source {src}"}
            continue
        if force or not dst.exists():
            dst.write_bytes(p.read_bytes())
        manifest["aux"][fname] = {"bytes": dst.stat().st_size, "source": src,
                                  "sha256": sha256(dst)}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_manifest(sets_dir: Path = SETS_DIR) -> dict:
    return json.loads((sets_dir / "manifest.json").read_text())


def load_set(name: str, sets_dir: Path = SETS_DIR, limit: int = 0) -> list[dict]:
    rows = read_jsonl(sets_dir / f"{name}.jsonl")
    return rows[:limit] if limit else rows


def verify(sets_dir: Path = SETS_DIR) -> list[str]:
    """Returns the list of files whose sha256 no longer matches the manifest."""
    man = load_manifest(sets_dir)
    bad = []
    for group, key in (("sets", ".jsonl"), ("lexicons", ""), ("aux", "")):
        for name, meta in man.get(group, {}).items():
            if "sha256" not in meta:
                continue
            p = sets_dir / (f"{name}{key}" if group == "sets" else name)
            if not p.exists() or sha256(p) != meta["sha256"]:
                bad.append(str(p))
    return bad
