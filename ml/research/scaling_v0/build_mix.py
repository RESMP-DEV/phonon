"""scaling_v0: one vocab_v1-recipe training mix per scaling point.

Real half: every row of train_date.jsonl (the honest date split), 20 percent with no vocabulary
line. Acoustic half: the selected rows, 20 percent no-line, 30 percent restraint (true term
deleted from the list, target keeps the mangled span as heard), rest a retrieved list with the
true term forced in. Identical to research/vocab_v1/build_vocab1_date.py except that the
acoustic rows are a named selection and the retrieval lists come from the cache.

Selections
  size:<N>   nested prefix of the 8-sentence x 3-seen-voice pool, terms-first
  axis:<n>   a fixed sub-grid of the full 8-sentence x 6-voice pool at ~9,000 rows
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from build_data_v1 import keep_as_heard  # noqa: E402
from vocab_common import Lexicon, chat_messages, heldout_terms, read_jsonl, seeded  # noqa: E402

D = Path("/data/phonon_scaling_v0")
OUT = D / "mixes"
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
POOL = D / "acoustic_pool.jsonl"
BIG = D / "acoustic_36k.jsonl"
SEEN_VOICES = ("af_heart", "am_adam", "bf_emma")
ALL_VOICES = ("af_heart", "am_adam", "bf_emma", "af_bella", "am_michael", "bm_george")
TOPK = 30
P_NOLINE = 0.20
P_RESTRAINT = 0.30
AXIS_N = 8900


def nested_order(rows: list[dict], salt: str) -> list[dict]:
    """Terms-first nested ordering: a fixed term permutation, then a fixed order inside a term.
    Any prefix is therefore a superset of every shorter prefix."""
    by_term: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_term.setdefault(r["term"], []).append(r)
    terms = sorted(by_term)
    random.Random(f"{salt}:terms").shuffle(terms)
    out = []
    for t in terms:
        rs = by_term[t]
        random.Random(f"{salt}:{t}").shuffle(rs)
        out += rs
    return out


def select(spec: str) -> tuple[list[dict], dict]:
    kind, val = spec.split(":", 1)
    if kind == "size":
        n = int(val)
        rows = read_jsonl(BIG) if n else []
        chosen = nested_order(rows, "size")[:n] if n else []
    elif kind == "axis":
        rows = read_jsonl(POOL)
        if val == "terms":       # many terms, 2 sentences, 3 voices
            sel = [r for r in rows if r["sentence_idx"] < 2 and r["voice"] in SEEN_VOICES]
        elif val == "sentences":  # few terms, all 8 sentences, 3 voices
            terms = sorted({r["term"] for r in rows})
            random.Random("axis:sentences").shuffle(terms)
            keep = set(terms[:540])
            sel = [r for r in rows if r["term"] in keep and r["voice"] in SEEN_VOICES]
        elif val == "voices":     # few terms, 2 sentences, all 6 voices
            terms = sorted({r["term"] for r in rows})
            random.Random("axis:voices").shuffle(terms)
            keep = set(terms[:1080])
            sel = [r for r in rows if r["term"] in keep and r["sentence_idx"] < 2]
        else:
            raise SystemExit(f"unknown axis {val}")
        chosen = nested_order(sel, f"axis:{val}")[:AXIS_N]
    else:
        raise SystemExit(f"unknown spec {spec}")
    meta = {
        "spec": spec, "rows": len(chosen),
        "terms": len({r["term"] for r in chosen}),
        "sentences_per_term": (len({(r["term"], r["sentence_idx"]) for r in chosen})
                               / max(len({r["term"] for r in chosen}), 1)),
        "voices": sorted({r["voice"] for r in chosen}),
        "sentence_idx": sorted({r["sentence_idx"] for r in chosen}),
    }
    return chosen, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)

    held = set(heldout_terms())
    lex = Lexicon(exclude=held)
    ac, meta = select(args.spec)
    ac = [r for r in ac if (r.get("term") or "") not in held]
    log(json.dumps(meta))

    real = read_jsonl(TRAIN_REAL)
    L_real = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_real.jsonl")}
    L_ac = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_acoustic.jsonl")}

    rows_out = []
    n = {"real_noline": 0, "real_list": 0, "ac_noline": 0, "ac_restraint": 0, "ac_normal": 0,
         "ac_forced_term": 0, "ac_restraint_failed": 0}
    for row in real:
        rng = seeded("v1real", row["id"])
        terms = L_real[row["id"]]
        true_terms = [t for t in lex.find_terms(row["target"]) if t not in held]
        if rng.random() < P_NOLINE:
            vocab = []
            n["real_noline"] += 1
        else:
            vocab = list(terms)
            rng.shuffle(vocab)
            n["real_list"] += 1
        rows_out.append({"id": row["id"], "source": row.get("source", "real"),
                         "branch": "noline" if not vocab else "list",
                         "input": row["input"], "target": row["target"],
                         "true_terms": true_terms, "vocab": vocab,
                         "messages": chat_messages(row["input"], vocab, row["target"])})

    for row in ac:
        rng = seeded("v1ac", row["id"])
        term = row.get("term") or ""
        terms = L_ac[row["id"]]
        u = rng.random()
        if u < P_NOLINE:
            vocab, target, branch = [], row["target"], "noline"
            n["ac_noline"] += 1
        elif u < P_NOLINE + P_RESTRAINT:
            kept = keep_as_heard(row["input"], row["target"], term)
            if kept is None:
                kept = row["input"]
                n["ac_restraint_failed"] += 1
            vocab = [t for t in terms if t != term][:TOPK]
            rng.shuffle(vocab)
            target, branch = kept, "restraint"
            n["ac_restraint"] += 1
        else:
            vocab = list(terms[:TOPK])
            if term and term not in vocab:
                vocab = vocab[:TOPK - 1] + [term]
                n["ac_forced_term"] += 1
            rng.shuffle(vocab)
            target, branch = row["target"], "list"
            n["ac_normal"] += 1
        rows_out.append({"id": row["id"], "source": row.get("source", "tts_parakeet"),
                         "branch": branch, "input": row["input"], "target": target,
                         "true_terms": [term] if term else [], "vocab": vocab,
                         "messages": chat_messages(row["input"], vocab, target)})

    p = OUT / f"train_{args.name}.jsonl"
    with p.open("w", encoding="utf-8") as h:
        for r in rows_out:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"name": args.name, "train_rows": len(rows_out), "real_rows": len(real),
             "acoustic_rows": len(ac), "selection": meta, "branches": n}
    (OUT / f"stats_{args.name}.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(json.dumps({k: v for k, v in stats.items() if k != "selection"}))
    log(f"MIX {args.name} DONE [{time.perf_counter()-t0:.0f}s] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
