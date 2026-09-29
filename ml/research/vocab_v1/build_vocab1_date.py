"""Training mix for `vocab1_date`: the v1 restraint recipe with train_date.jsonl as the real
source, so no row of the real-audio date holdout is in training."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")

from build_data_v1 import keep_as_heard  # noqa: E402
from retrieve import load_pool  # noqa: E402
from vocab_common import Lexicon, chat_messages, heldout_terms, read_jsonl, seeded  # noqa: E402

OUT = Path("/data/phonon_vocab_v1")
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
ACOUSTIC = Path("/data/phonon_synth_v3/acoustic_pairs.jsonl")
TOPK = 30
P_NOLINE = 0.20
P_RESTRAINT = 0.30


def main() -> int:
    t0 = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731
    held = heldout_terms()
    lex = Lexicon(exclude=set(held))
    R_train = load_pool(exclude=set(held))
    log(f"pool_train={len(R_train.terms)}")

    real = read_jsonl(TRAIN_REAL)
    real_lists = R_train.topk([r["input"] for r in real], k=TOPK, log=log)
    ac_all = read_jsonl(ACOUSTIC)
    ac = [r for r in ac_all if (r.get("term") or "") not in held]
    ac_lists = R_train.topk([r["input"] for r in ac], k=TOPK + 1, log=log)

    rows_out = []
    n = {"real_noline": 0, "real_list": 0, "ac_noline": 0, "ac_restraint": 0, "ac_normal": 0,
         "ac_forced_term": 0, "ac_restraint_failed": 0}
    for row, L in zip(real, real_lists):
        rng = seeded("v1real", row["id"])
        terms = [t for t, _ in L]
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

    for row, L in zip(ac, ac_lists):
        rng = seeded("v1ac", row["id"])
        term = row.get("term") or ""
        terms = [t for t, _ in L]
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

    p = OUT / "train_vocab1_date.jsonl"
    with p.open("w", encoding="utf-8") as h:
        for r in rows_out:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"train_rows": len(rows_out), "real_rows": len(real), "acoustic_rows": len(ac),
             "branches": n}
    (OUT / "build_stats_date.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(json.dumps(stats))
    log(f"BUILD DATE DONE [{time.perf_counter()-t0:.0f}s] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
