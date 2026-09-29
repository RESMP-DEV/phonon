"""bigrun_v0 step 5c: v1-recipe rows -> train_mid.jsonl and train_big.jsonl."""
from __future__ import annotations
import json, random, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
import vocab_common  # noqa: E402
vocab_common.LEXICON_PATH = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
from build_data_v1 import keep_as_heard  # noqa: E402
from vocab_common import Lexicon, chat_messages, read_jsonl, seeded  # noqa: E402

D = Path("/data/phonon_bigrun_v0")
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = D / "terms_heldout_new.jsonl"
TOPK = 30
P_NOLINE = 0.20
P_RESTRAINT = 0.30
REPEAT_MID = 4
REPEAT_BIG = 8
SEED = 918


def main() -> int:
    t0 = time.perf_counter()
    held = {r["term"] for r in read_jsonl(HELD_OLD)} | {r["term"] for r in read_jsonl(HELD_NEW)}
    lex = Lexicon(exclude=held)
    print(f"lexicon_rows={len(lex.rows)} held={len(held)}", flush=True)

    L_real = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_real.jsonl")}
    L_ac = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_acoustic.jsonl")}
    real = read_jsonl(TRAIN_REAL)
    ac = read_jsonl(D / "acoustic_pool.jsonl")
    print(f"real={len(real)} acoustic={len(ac)} [{time.perf_counter()-t0:.0f}s]", flush=True)

    n = Counter()

    def real_row(row: dict, rep: int) -> dict:
        rid = f"{row['id']}#r{rep}"
        rng = seeded("v1real", rid)
        terms = L_real.get(row["id"], [])
        true_terms = [t for t in lex.find_terms(row["target"]) if t not in held]
        if rng.random() < P_NOLINE:
            vocab = []
            n["real_noline"] += 1
        else:
            vocab = list(terms)
            rng.shuffle(vocab)
            n["real_list"] += 1
        return {"id": rid, "source": row.get("source", "real"),
                "branch": "noline" if not vocab else "list",
                "input": row["input"], "target": row["target"],
                "true_terms": true_terms, "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, row["target"])}

    def ac_row(row: dict) -> dict:
        rng = seeded("v1ac", row["id"])
        term = row.get("term") or ""
        terms = L_ac.get(row["id"], [])
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
        return {"id": row["id"], "source": "tts_parakeet", "branch": branch,
                "stage": row["stage"], "term": term, "kind": row.get("kind"),
                "voice": row["voice"], "sentence_idx": row["sentence_idx"],
                "input": row["input"], "target": target,
                "true_terms": [term] if term else [], "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, target)}

    ac_rows = [ac_row(r) for r in ac]
    print(f"acoustic rows built [{time.perf_counter()-t0:.0f}s]", flush=True)
    real_mid = [real_row(r, k) for k in range(REPEAT_MID) for r in real]
    real_big = [real_row(r, k) for k in range(REPEAT_BIG) for r in real]
    print(f"real rows built [{time.perf_counter()-t0:.0f}s]", flush=True)

    ac_mid = [r for r in ac_rows if r["stage"] == "mid"]
    stats = {}
    for name, acr, rr in (("mid", ac_mid, real_mid), ("big", ac_rows, real_big)):
        rows = acr + rr
        random.Random(SEED).shuffle(rows)
        p = D / f"train_{name}.jsonl"
        with p.open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {
            "path": str(p), "rows": len(rows), "acoustic_rows": len(acr), "real_rows": len(rr),
            "real_unique": len(real), "repeat": REPEAT_MID if name == "mid" else REPEAT_BIG,
            "acoustic_terms": len({r["term"] for r in acr}),
            "real_to_acoustic": round(len(rr) / max(len(acr), 1), 4),
            "branches": dict(Counter(r["branch"] for r in rows)),
        }
        print(json.dumps(stats[name], indent=2), flush=True)
    stats["branch_counters"] = dict(n)
    (D / "mix_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"STEP 5c done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
