"""pool2 step 1c: merge every non-repo-walk source into terms_pool2 + terms_heldout_pool2.

Filters and kind heuristics follow research/bigrun_v0/step1_pool.py and the lexicon builder
(research/synth_v0/walk_repos.py classify_term/mangle_score, filter_lexicon.py speakable/score).
"""
from __future__ import annotations
import json, random, re
from collections import Counter, defaultdict
from pathlib import Path

R = Path("/data/phonon_pool2_v0/raw")
OUT = Path("/data/phonon_pool2_v0")
ENGLISH = Path("/data/phonon_asr_errors_v0/english_words.txt")
EXCLUDE = [
    Path("/data/phonon_bigrun_v0/terms_pool.jsonl"),
    Path("/data/phonon_bigrun_v0/terms_heldout_new.jsonl"),
    Path("/data/phonon_term_eval_v0/terms_heldout.jsonl"),
]
N_HELD = 300
SEED = 20260918
CAP_IDENTIFIER = 12000
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


def mangle_score(term: str, english: set[str]) -> float:
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    english = {w.strip().lower() for w in ENGLISH.read_text(encoding="utf-8").splitlines()
               if w.strip()}
    banned: set[str] = set()
    for p in EXCLUDE:
        banned |= {(r.get("term") or "").lower() for r in read_jsonl(p)}
    print(f"english={len(english)} banned={len(banned)}", flush=True)

    # ---- gather (term, source, kind_hint, weight) ------------------------
    cand: list[tuple[str, str, str, float]] = []

    # A. installed-package symbols
    sp = R / "symbols.jsonl"
    if sp.exists():
        for r in read_jsonl(sp):
            hint = "module" if r["sym_kind"] == "module" else ""
            cand.append((r["term"], r["source"], hint, 4.0))

    # A2. CUDA / cuBLAS / cuDNN API names
    f = R / "cuda_api.txt"
    if f.exists():
        for t in f.read_text(encoding="utf-8").split():
            cand.append((t, "cuda_headers", "identifier", 5.0))

    # B. package registries
    f = R / "pypi_top.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = d.get("rows") or d.get("data") or []
        for i, r in enumerate(rows[:4000]):
            nm = r.get("project") or r.get("name") or ""
            if nm:
                cand.append((nm, "pypi_top4000", "library", 6.0 - 3.0 * i / 4000))
    for p in range(1, 11):
        f = R / f"crates_p{p}.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for i, c in enumerate(d.get("crates") or []):
            nm = c.get("name") or ""
            if nm:
                cand.append((nm, "crates_top1000", "library", 6.0 - 3.0 * ((p - 1) * 100 + i) / 1000))
    f = R / "npm_names.txt"
    if f.exists():
        names = [t for t in f.read_text(encoding="utf-8").split() if t]
        for i, nm in enumerate(names):
            cand.append((nm, "npm_registry_search", "library", 5.0))

    # C. CLI tools and long flags
    f = R / "exes.txt"
    if f.exists():
        for t in f.read_text(encoding="utf-8").split():
            cand.append((t, "path_executables", "cli_tool", 5.0))
    f = R / "help_flags.txt"
    if f.exists():
        for t in f.read_text(encoding="utf-8").split():
            cand.append((t, "cli_long_flags", "flag", 5.0))

    # D. local HF model / dataset names
    f = R / "hf_hub.txt"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "--" not in line:
                continue
            parts = line.split("--")
            name = parts[-1]
            cand.append((name, "hf_hub", "product", 6.0))
            if len(parts) > 2 and parts[1] not in ("", "None"):
                cand.append((parts[1], "hf_hub", "product", 5.0))

    print(f"raw candidates={len(cand)}", flush=True)

    # ---- filter + dedupe -------------------------------------------------
    drop = Counter()
    seen: set[str] = set()
    pool: list[dict] = []
    src_raw = Counter(s for _, s, _, _ in cand)
    for term, source, hint, w in cand:
        term = term.strip()
        tl = term.lower()
        if not term:
            drop["empty"] += 1; continue
        if tl in seen:
            drop["dup_internal"] += 1; continue
        if tl in banned:
            drop["already_in_pool1"] += 1; seen.add(tl); continue
        if len(term) < 3:
            drop["short"] += 1; seen.add(tl); continue
        if term.isalpha() and term.islower() and tl in english:
            drop["english"] += 1; seen.add(tl); continue
        if not speakable(term):
            drop["unspeakable"] += 1; seen.add(tl); continue
        seen.add(tl)
        kind = classify(term, hint)
        rank = mangle_score(term, english) + KIND_BONUS.get(kind, 0) + w
        pool.append({"term": term, "kind": kind, "source": source, "count": None,
                     "mangle_score": round(mangle_score(term, english), 3),
                     "rank_score": round(rank, 4)})
    print(f"after_filter={len(pool)} drops={dict(drop)}", flush=True)

    # ---- cap identifiers -------------------------------------------------
    ident = [p for p in pool if p["kind"] == "identifier"]
    rest = [p for p in pool if p["kind"] != "identifier"]
    n_ident_before = len(ident)
    if len(ident) > CAP_IDENTIFIER:
        ident.sort(key=lambda p: (-p["rank_score"], p["term"]))
        ident = ident[:CAP_IDENTIFIER]
    pool = rest + ident
    print(f"identifiers {n_ident_before} -> {len(ident)}; pool={len(pool)}", flush=True)

    # ---- stratified held-out draw by (kind, source) ----------------------
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for p in pool:
        strata[(p["kind"], p["source"])].append(p)
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
    picked: list[dict] = []
    for k in keys:
        cand_k = sorted(strata[k], key=lambda p: p["term"])
        rng.shuffle(cand_k)
        picked += cand_k[: alloc[k]]
    picked.sort(key=lambda p: (-p["rank_score"], p["term"]))
    held = {p["term"] for p in picked}

    with (OUT / "terms_heldout_pool2.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(picked):
            rec = dict(p); rec["term_id"] = f"p2h{j:04d}"
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train = [p for p in pool if p["term"] not in held]
    train.sort(key=lambda p: (-p["rank_score"], p["term"]))
    with (OUT / "terms_pool2.jsonl").open("w", encoding="utf-8") as h:
        for j, p in enumerate(train):
            rec = dict(p); rec["term_id"] = f"p2{j:05d}"
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "raw_candidates": len(cand), "raw_by_source": dict(src_raw.most_common()),
        "drops": dict(drop), "identifiers_before_cap": n_ident_before,
        "identifier_cap": CAP_IDENTIFIER,
        "pool_terms": len(train), "heldout_terms": len(picked),
        "pool_kinds": dict(Counter(p["kind"] for p in train).most_common()),
        "pool_sources": dict(Counter(p["source"] for p in train).most_common()),
        "heldout_kinds": dict(Counter(p["kind"] for p in picked).most_common()),
        "heldout_sources": dict(Counter(p["source"] for p in picked).most_common()),
        "kind_by_source": {s: dict(Counter(p["kind"] for p in train if p["source"] == s))
                           for s in sorted({p["source"] for p in train})},
    }
    (OUT / "pool2_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "kind_by_source"}, indent=2),
          flush=True)
    print("STEP 1 done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
