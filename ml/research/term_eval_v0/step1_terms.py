"""Step 1: held-out term set of 300, stratified over kind x rank_score decile."""
from __future__ import annotations
import json, random, re, sys
from pathlib import Path
from collections import Counter, defaultdict

LEX = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
SYNTH1 = Path("/data/phonon_synth_v1/synth_pairs_v1.jsonl")
SYNTH2 = Path("/data/phonon_synth_v2/synth_pairs_v2.jsonl")
TRAIN = Path("/data/phonon_corrector_v0/train.jsonl")
OUT = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
N_PICK = 300
SEED = 1717

KEEP_SHORT = {"gpu","cpu","wer","tts","asr","api","llm","lora","moe","cu","sm","nv","ptx","sass","itn","pnc","sft","rl"}
JUNK = {"software","implied","limited","copyright","license","warranty","the","and"}

def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def usable(row):
    t = row["term"]
    if len(t) <= 2 and t.lower() not in KEEP_SHORT:
        return False
    if t.lower() in JUNK:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    return True

def corpus_blob(path, field="target"):
    parts = []
    with path.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            v = row.get(field) or ""
            if v:
                parts.append(v.lower())
    return "\n".join(parts)

def main():
    rng = random.Random(SEED)
    lex = [r for r in load_jsonl(LEX) if usable(r)]
    print(f"lexicon rows={len(lex)} (after filter)", flush=True)

    print("loading corpora blobs...", flush=True)
    blob_s1 = corpus_blob(SYNTH1)
    blob_s2 = corpus_blob(SYNTH2)
    blob_tr = corpus_blob(TRAIN)
    print(f"blob sizes MB: s1={len(blob_s1)/1e6:.1f} s2={len(blob_s2)/1e6:.1f} train={len(blob_tr)/1e6:.1f}", flush=True)

    # membership flags for the whole usable lexicon (needed for step 5 too)
    def bmatch(t, b):
        return re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", b) is not None
    for r in lex:
        t = r["term"].lower()
        r["in_synth_v1"] = t in blob_s1
        r["in_synth_v2"] = t in blob_s2
        r["in_train"] = t in blob_tr
        r["in_synth_v1_wb"] = bmatch(t, blob_s1)
        r["in_synth_v2_wb"] = bmatch(t, blob_s2)
        r["in_train_wb"] = bmatch(t, blob_tr)

    # rank_score deciles over the usable pool
    scores = sorted(r["rank_score"] for r in lex)
    cuts = [scores[int(len(scores) * k / 10)] for k in range(1, 10)]
    def decile(s):
        d = 0
        for c in cuts:
            if s >= c:
                d += 1
        return min(d, 9)
    for r in lex:
        r["decile"] = decile(r["rank_score"])

    strata = defaultdict(list)
    for r in lex:
        strata[(r["kind"], r["decile"])].append(r)

    total = len(lex)
    # largest-remainder allocation
    raw = {k: N_PICK * len(v) / total for k, v in strata.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    # every non-empty stratum gets at least 1 if it can
    for k in strata:
        if alloc[k] == 0:
            alloc[k] = 1
    while sum(alloc.values()) > N_PICK:
        k = max(alloc, key=lambda k: (alloc[k] - raw[k], alloc[k]))
        if alloc[k] > 1:
            alloc[k] -= 1
        else:
            # drop the least deserving singleton
            ks = [k2 for k2 in alloc if alloc[k2] == 1]
            k2 = min(ks, key=lambda k2: raw[k2])
            alloc[k2] = 0
    while sum(alloc.values()) < N_PICK:
        k = max(strata, key=lambda k: (raw[k] - alloc[k], len(strata[k])))
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
        else:
            raw[k] = -1e9

    picked = []
    for k, n in sorted(alloc.items()):
        cands = list(strata[k])
        if n <= 0 or not cands:
            continue
        rng.shuffle(cands)
        cands.sort(key=lambda r: (
            0 if (r.get("mangle_score") or 0) >= 1 else 1,
            0 if not r["in_train"] else 1,
            -float(r["rank_score"]),
        ))
        take = cands[: min(n, len(cands))]
        picked.extend(take)

    # top up if some strata were short
    if len(picked) < N_PICK:
        have = {r["term"] for r in picked}
        rest = [r for r in lex if r["term"] not in have]
        rng.shuffle(rest)
        rest.sort(key=lambda r: (
            0 if (r.get("mangle_score") or 0) >= 1 else 1,
            0 if not r["in_train"] else 1,
            -float(r["rank_score"]),
        ))
        picked.extend(rest[: N_PICK - len(picked)])
    picked = picked[:N_PICK]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as h:
        for i, r in enumerate(picked):
            rec = {
                "term_id": f"t{i:04d}",
                "term": r["term"],
                "kind": r["kind"],
                "count": r.get("count"),
                "mangle_score": r.get("mangle_score"),
                "rank_score": r.get("rank_score"),
                "decile": r["decile"],
                "in_synth_v1": r["in_synth_v1"],
                "in_synth_v2": r["in_synth_v2"],
                "in_train": r["in_train"],
                "in_synth_v1_wb": r["in_synth_v1_wb"],
                "in_synth_v2_wb": r["in_synth_v2_wb"],
                "in_train_wb": r["in_train_wb"],
            }
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "n_picked": len(picked),
        "kinds": dict(Counter(r["kind"] for r in picked)),
        "deciles": dict(sorted(Counter(r["decile"] for r in picked).items())),
        "mangle_ge_1": sum(1 for r in picked if (r.get("mangle_score") or 0) >= 1),
        "in_synth_v1": sum(1 for r in picked if r["in_synth_v1"]),
        "in_synth_v2": sum(1 for r in picked if r["in_synth_v2"]),
        "in_synth_either": sum(1 for r in picked if r["in_synth_v1"] or r["in_synth_v2"]),
        "in_train": sum(1 for r in picked if r["in_train"]),
        "in_synth_either_wb": sum(1 for r in picked if r["in_synth_v1_wb"] or r["in_synth_v2_wb"]),
        "in_train_wb": sum(1 for r in picked if r["in_train_wb"]),
        "in_neither": sum(1 for r in picked if not (r["in_synth_v1"] or r["in_synth_v2"] or r["in_train"])),
        "lexicon_usable": len(lex),
        "lexicon_in_train": sum(1 for r in lex if r["in_train"]),
        "lexicon_in_synth_either": sum(1 for r in lex if r["in_synth_v1"] or r["in_synth_v2"]),
        "rank_score_median_usable": sorted(r["rank_score"] for r in lex)[len(lex)//2],
    }
    Path("/data/phonon_term_eval_v0/terms_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    # full annotated lexicon for step 5
    with Path("/data/phonon_term_eval_v0/lexicon_annotated.jsonl").open("w", encoding="utf-8") as h:
        for r in lex:
            h.write(json.dumps({k: r[k] for k in ("term","kind","count","mangle_score","rank_score","decile","in_synth_v1","in_synth_v2","in_train","in_synth_v1_wb","in_synth_v2_wb","in_train_wb")}, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print("STEP 1 done", flush=True)

if __name__ == "__main__":
    main()
