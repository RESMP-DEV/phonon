"""STEP 3: term pool 3 from the symbol index of 150 public repos + tree-sitter keyword lists.

Filters follow research/pool2_v0/build_pool2.py (3+ chars, speakable, not plain English,
dedupe against pools 1-2 and every held-out set).  300 stratified held-out terms.
"""
from __future__ import annotations
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

PR = Path("/data/phonon_pool3_v0/per_repo")
OUT = Path("/data/phonon_pool3_v0")
ENGLISH = Path("/data/phonon_asr_errors_v0/english_words.txt")
EXCLUDE = [
    Path("/data/phonon_synth_v0/lexicon_ranked.jsonl"),
    Path("/data/phonon_bigrun_v0/terms_pool.jsonl"),
    Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl"),
    Path("/data/phonon_term_eval_v0/terms_heldout.jsonl"),
    Path("/data/phonon_pool2_v0/terms_pool2.jsonl"),
    Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl"),
]
N_HELD = 300
SEED = 20260918
KIND_BONUS = {"cli_tool": 3, "flag": 2, "library": 2, "product": 2, "module": 1,
              "acronym": 1, "file": 0, "identifier": 0}

RUN_ID = re.compile(r"^[0-9a-f]{8,}$", re.I)
HEXISH = re.compile(r"^(0x)?[0-9a-f]{6,}$", re.I)
HEX_ADDR = re.compile(r"0x[0-9a-f]{4,}", re.I)


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def speakable(term: str) -> bool:
    if not term or len(term) < 2 or len(term) > 40:
        return False
    if RUN_ID.match(term) or HEXISH.match(term) or HEX_ADDR.search(term):
        return False
    if term.count("_") >= 4:
        return False
    if len(re.sub(r"[^A-Za-z]", "", term)) < 2:
        return False
    return True


def classify(term: str, hint: str) -> str:
    if hint:
        return hint
    if term.startswith("--") or (term.startswith("-") and len(term) <= 3):
        return "flag"
    if re.fullmatch(r"[A-Z]{2,8}", term):
        return "acronym"
    if "/" in term or term.endswith((".py", ".rs", ".cu", ".cuh", ".go", ".ts", ".md",
                                     ".toml", ".json", ".h", ".cpp")):
        return "file"
    return "identifier"


def mangle_score(term: str, english: set) -> float:
    s = 0.0
    low = term.lower().strip("-")
    alpha = re.sub(r"[^a-z]", "", low)
    s += -3.0 if (low in english or alpha in english) else 2.0
    if re.search(r"\d", term):
        s += 2.0
    if "_" in term:
        s += 2.0
    if re.search(r"[a-z][A-Z]", term):
        s += 3.0
    if re.fullmatch(r"[A-Z]{2,8}", term):
        s += 2.0
    if "-" in term and not term.startswith("-"):
        s += 1.0
    if term.startswith("--"):
        s += 1.5
    if len(alpha) <= 1:
        s -= 4.0
    return s


KIND_MAP = {"module": "module", "file": "file", "flag": "flag"}


def grammar_keywords():
    """Tier: the tree-sitter grammars' own keyword lists (anonymous node kinds)."""
    out = []
    try:
        from tree_sitter_language_pack import get_language
    except Exception as exc:
        print(f"grammar keywords unavailable: {exc}", flush=True)
        return out
    for lg in ["python", "rust", "c", "cpp", "cuda", "go", "javascript", "typescript",
               "tsx", "markdown", "bash", "toml", "yaml", "json", "make", "cmake"]:
        try:
            L = get_language(lg)
        except Exception:
            continue
        n = 0
        for i in range(L.node_kind_count):
            if L.node_kind_is_named(i):
                continue
            k = L.node_kind_for_id(i) or ""
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,40}", k):
                out.append((k, f"grammar:{lg}", "", 4.0, 1))
                n += 1
        print(f"  grammar {lg}: {n}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=6000)
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    english = {w.strip().lower() for w in ENGLISH.read_text(encoding="utf-8").splitlines()
               if w.strip()}
    banned = set()
    for p in EXCLUDE:
        banned |= {(r.get("term") or "").lower() for r in read_jsonl(p)}
    print(f"english={len(english)} banned={len(banned)}", flush=True)

    files = sorted(PR.glob("*.jsonl"))
    print(f"per-repo files={len(files)}", flush=True)
    raw_by_lang, repos_by_lang = Counter(), defaultdict(set)
    # best candidate per term: (term -> [lang, source, kind_hint, score_w, count])
    cand: dict[str, list] = {}
    for f in files:
        lang, rest = f.stem.split("__", 1)
        repo = rest.replace("__", "/")
        repos_by_lang[lang].add(repo)
        for r in read_jsonl(f):
            t = r["term"]
            raw_by_lang[lang] += 1
            if r["count"] < a.min_count:
                continue
            hint = KIND_MAP.get(r["kind"], "")
            cur = cand.get(t.lower())
            if cur is None or r["count"] > cur[4]:
                cand[t.lower()] = [lang, f"gh:{lang}:{repo}", hint, 5.0, r["count"], t]
    print(f"repo candidates (count>={a.min_count}) = {len(cand)}", flush=True)
    gk = grammar_keywords()
    n_gk = 0
    for k, src, hint, w, c in gk:
        if k.lower() not in cand:
            cand[k.lower()] = ["grammar", src, hint, w, c, k]
            n_gk += 1
    print(f"grammar keywords added={n_gk}", flush=True)

    drop = Counter()
    kept = []
    FLAG_OK = re.compile(r"--[A-Za-z][A-Za-z0-9-]{1,40}")
    for tl, (lang, source, hint, w, count, term) in cand.items():
        if term.startswith("--"):
            m = FLAG_OK.match(term)
            if not m:
                drop["bad_flag"] += 1; continue
            term = m.group(0); tl = term.lower()
        if tl in banned:
            drop["already_in_pool12"] += 1; continue
        if len(term) < 3:
            drop["short"] += 1; continue
        if term.isalpha() and term.islower() and tl in english:
            drop["english"] += 1; continue
        if not speakable(term):
            drop["unspeakable"] += 1; continue
        kind = classify(term, hint)
        ms = mangle_score(term, english)
        rank = ms + KIND_BONUS.get(kind, 0) + w + min(3.0, 0.5 * (count ** 0.5))
        kept.append({"term": term, "kind": kind, "source": source, "lang": lang,
                     "count": count, "mangle_score": round(ms, 3),
                     "rank_score": round(rank, 4)})
    print(f"after_filter={len(kept)} drops={dict(drop)}", flush=True)

    # ---- balance: top N per language ------------------------------------
    by_lang = defaultdict(list)
    for p in kept:
        by_lang[p["lang"]].append(p)
    pool = []
    per_lang_kept = {}
    for lang, rows in by_lang.items():
        rows.sort(key=lambda p: (-p["rank_score"], -p["count"], p["term"]))
        cap = a.per_lang if lang != "grammar" else 10 ** 6
        pool += rows[:cap]
        per_lang_kept[lang] = min(len(rows), cap)
    print(f"pool after per-language cap={len(pool)} {per_lang_kept}", flush=True)

    # ---- stratified held-out draw by (kind, lang) -----------------------
    strata = defaultdict(list)
    for p in pool:
        strata[(p["kind"], p["lang"])].append(p)
    keys = sorted(strata)
    quota = {k: N_HELD * len(strata[k]) / len(pool) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rem = sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))
    i = 0
    while sum(alloc.values()) < N_HELD and i < 10000:
        k = rem[i % len(rem)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
        i += 1
    rng = random.Random(SEED)
    picked = []
    for k in keys:
        ck = sorted(strata[k], key=lambda p: p["term"])
        rng.shuffle(ck)
        picked += ck[: alloc[k]]
    picked.sort(key=lambda p: (-p["rank_score"], p["term"]))
    held = {p["term"] for p in picked}

    with (OUT / "terms_heldout_pool3.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(picked):
            rec = dict(p); rec["term_id"] = f"p3h{j:04d}"
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    train = [p for p in pool if p["term"] not in held]
    train.sort(key=lambda p: (-p["rank_score"], p["term"]))
    with (OUT / "terms_pool3.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(train):
            rec = dict(p); rec["term_id"] = f"p3{j:05d}"
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "per_repo_files": len(files),
        "repos_by_lang": {k: len(v) for k, v in sorted(repos_by_lang.items())},
        "raw_terms_by_lang": dict(raw_by_lang.most_common()),
        "min_count": a.min_count, "per_lang_cap": a.per_lang,
        "candidates_after_mincount": len(cand), "grammar_keywords": n_gk,
        "drops": dict(drop), "after_filter": len(kept),
        "pool_terms": len(train), "heldout_terms": len(picked),
        "kept_by_lang": per_lang_kept,
        "pool_kinds": dict(Counter(p["kind"] for p in train).most_common()),
        "pool_by_lang": dict(Counter(p["lang"] for p in train).most_common()),
        "heldout_kinds": dict(Counter(p["kind"] for p in picked).most_common()),
        "heldout_by_lang": dict(Counter(p["lang"] for p in picked).most_common()),
        "heldout_examples": [p["term"] for p in picked[:20]],
    }
    (OUT / "pool3_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)
    print("STEP 3 pool done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
