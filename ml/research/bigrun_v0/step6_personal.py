"""bigrun_v0 step 6: the stage-2 personal training set.

The user's own real pairs only (`train_date.jsonl`, the honest date split), one repetition, with
the same v1 row recipe the big run used for its real rows: 20 percent no-line, otherwise the
retrieved top-30 list from `lists_real.jsonl`, shuffled. Row ids and seeded draws match repetition
0 of `train_big.jsonl` exactly.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
import vocab_common  # noqa: E402

vocab_common.LEXICON_PATH = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
from vocab_common import Lexicon, chat_messages, read_jsonl, seeded  # noqa: E402

D = Path("/data/phonon_bigrun_v0")
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = D / "terms_heldout_new.jsonl"
OUT = D / "train_personal.jsonl"
P_NOLINE = 0.20
SEED = 918


def main() -> int:
    t0 = time.perf_counter()
    held = {r["term"] for r in read_jsonl(HELD_OLD)} | {r["term"] for r in read_jsonl(HELD_NEW)}
    lex = Lexicon(exclude=held)
    L_real = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_real.jsonl")}
    real = read_jsonl(TRAIN_REAL)
    print(f"real={len(real)} lists={len(L_real)} held={len(held)}", flush=True)
    n = Counter()
    rows = []
    for row in real:
        rid = f"{row['id']}#r0"
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
        rows.append({"id": rid, "source": row.get("source", "real"),
                     "branch": "noline" if not vocab else "list",
                     "input": row["input"], "target": row["target"],
                     "true_terms": true_terms, "vocab": vocab,
                     "messages": chat_messages(row["input"], vocab, row["target"])})
    random.Random(SEED).shuffle(rows)
    with OUT.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"path": str(OUT), "rows": len(rows), "branches": dict(n)}
    (D / "personal_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)
    print(f"STEP 6 done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
