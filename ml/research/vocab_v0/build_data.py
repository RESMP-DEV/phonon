"""Step 1: build vocabulary-in-prompt training sets and the eval vocabulary conditions."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab_common import (  # noqa: E402
    Lexicon, chat_messages, heldout_terms, read_jsonl, seeded,
)

OUT = Path("/data/phonon_vocab_v0")
TRAIN_REAL = Path("/data/phonon_corrector_v0/train.jsonl")
ACOUSTIC = Path("/data/phonon_synth_v3/acoustic_pairs.jsonl")
ASR = Path("/data/phonon_term_eval_v0/asr.jsonl")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    t0 = time.perf_counter()
    held = heldout_terms()
    lex = Lexicon(exclude=set(held))  # training lists never contain held-out terms
    lex_eval = Lexicon()              # eval lists may (neighbours of the true term)
    print(f"lexicon rows={len(lex.rows)} train-allowed={len(lex.allowed)} heldout={len(held)}",
          flush=True)

    stats = {}

    # ---------- real rows -------------------------------------------------
    real = read_jsonl(TRAIN_REAL)
    real_vocab, real_novocab = [], []
    n_with_true = 0
    for row in real:
        rng = seeded("real", row["id"])
        true_terms = [t for t in lex.find_terms(row["target"]) if t not in held]
        if true_terms:
            n_with_true += 1
            k = rng.randint(max(5, len(true_terms) + 1), 30)
            kinds = [lex.kind_of(t) for t in true_terms]
            dis = lex.sample_distractors(rng, k - len(true_terms), kinds, set(true_terms))
            vocab = true_terms + dis
            rng.shuffle(vocab)
        elif rng.random() < 0.5:
            vocab = lex.sample_distractors(rng, rng.randint(5, 30), [], set())
        else:
            vocab = []
        real_vocab.append({
            "id": row["id"], "source": row["source"], "input": row["input"],
            "target": row["target"], "true_terms": true_terms, "vocab": vocab,
            "messages": chat_messages(row["input"], vocab, row["target"]),
        })
        real_novocab.append({
            "id": row["id"], "source": row["source"], "input": row["input"],
            "target": row["target"], "true_terms": true_terms, "vocab": [],
            "messages": chat_messages(row["input"], None, row["target"]),
        })
    stats["real_rows"] = len(real_vocab)
    stats["real_rows_with_true_term"] = n_with_true
    print(f"real rows={len(real_vocab)} with_true_term={n_with_true} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---------- acoustic rows --------------------------------------------
    ac = read_jsonl(ACOUSTIC)
    ac_vocab, ac_novocab = [], []
    dropped = 0
    for row in ac:
        term = row.get("term") or ""
        if term in held:
            dropped += 1
            continue
        rng = seeded("ac", row["id"])
        kind = row.get("kind") or lex.kind_of(term)
        k = rng.randint(5, 30)
        dis = lex.sample_distractors(rng, k - 1, [kind], {term})
        vocab = [term] + dis
        rng.shuffle(vocab)
        ac_vocab.append({
            "id": row["id"], "source": row.get("source", "tts_parakeet"), "input": row["input"],
            "target": row["target"], "true_terms": [term], "vocab": vocab,
            "messages": chat_messages(row["input"], vocab, row["target"]),
        })
        ac_novocab.append({
            "id": row["id"], "source": row.get("source", "tts_parakeet"), "input": row["input"],
            "target": row["target"], "true_terms": [term], "vocab": [],
            "messages": chat_messages(row["input"], None, row["target"]),
        })
    stats["acoustic_rows"] = len(ac_vocab)
    stats["acoustic_rows_dropped_heldout"] = dropped
    print(f"acoustic rows={len(ac_vocab)} dropped_heldout={dropped}", flush=True)

    write_jsonl(OUT / "train_vocab_real.jsonl", real_vocab)
    write_jsonl(OUT / "train_vocab_real_acoustic.jsonl", real_vocab + ac_vocab)
    write_jsonl(OUT / "train_novocab_real_acoustic.jsonl", real_novocab + ac_novocab)

    # ---------- eval conditions on the 2700 term clips --------------------
    asr = read_jsonl(ASR)
    cond = []
    n_missing = 0
    for row in asr:
        rng = seeded("eval", row["id"])
        term, kind = row["term"], row.get("kind") or "other"
        rank = float(held.get(term, {}).get("rank_score") or 0.0)
        oracle_d = lex_eval.nearest(kind, rank, 9, {term})
        if len(oracle_d) < 9:
            oracle_d += lex_eval.sample_distractors(rng, 9 - len(oracle_d), [kind],
                                                    {term, *oracle_d})
        oracle = [term] + oracle_d
        rng.shuffle(oracle)
        neigh = lex_eval.nearest(kind, rank, 30, {term})
        present = rng.random() < 0.8
        realistic = ([term] + neigh[:29]) if present else neigh[:30]
        if not present:
            n_missing += 1
        rng.shuffle(realistic)
        cond.append({
            "id": row["id"], "term": term, "kind": kind, "voice": row["voice"],
            "reference": row["reference"], "hyp": row["hyp"],
            "vocab_oracle": oracle, "vocab_realistic": realistic,
            "realistic_has_term": present,
        })
    stats["eval_rows"] = len(cond)
    stats["eval_realistic_missing_term"] = n_missing
    write_jsonl(OUT / "eval_conditions.jsonl", cond)
    print(f"eval rows={len(cond)} realistic_missing_term={n_missing}", flush=True)

    # ---------- real-dictation holdouts: vocab from reference -------------
    sys.path.insert(0, "/home/user/phonon/research/corrector_v0")
    from eval_corrector import eval_rows  # noqa: E402

    sets = eval_rows()
    real_eval = {}
    for name in ("wispr_holdout120", "wispr_text_holdout"):
        rows = sets[name]
        out_rows = []
        for row in rows:
            rng = seeded("realeval", name, row["id"])
            true_terms = lex_eval.find_terms(row["reference"])
            kinds = [lex_eval.kind_of(t) for t in true_terms] or []
            dis = lex_eval.sample_distractors(rng, max(0, 30 - len(true_terms)), kinds,
                                              set(true_terms))
            vocab = true_terms + dis
            rng.shuffle(vocab)
            out_rows.append({
                "id": row["id"], "input": row["input"], "reference": row["reference"],
                "true_terms": true_terms, "vocab_ref": vocab,
            })
        real_eval[name] = out_rows
        write_jsonl(OUT / f"eval_{name}.jsonl", out_rows)
        with_terms = sum(1 for r in out_rows if r["true_terms"])
        stats[f"{name}_rows"] = len(out_rows)
        stats[f"{name}_rows_with_lexicon_term"] = with_terms
        print(f"{name} rows={len(out_rows)} with_lexicon_term={with_terms}", flush=True)

    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)
    print(f"STEP 1 done ({time.perf_counter()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
