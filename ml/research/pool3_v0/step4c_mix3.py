"""pool3 step 4c: v1-recipe rows for pool 3 + 25 percent replay of the pool-2 acoustic rows
+ 15 percent replay of the pool-1 (train_big) acoustic rows + the train_date real rows x4."""
from __future__ import annotations
import json, random, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
import vocab_common  # noqa: E402
vocab_common.LEXICON_PATH = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
from build_data_v1 import keep_as_heard  # noqa: E402
from vocab_common import chat_messages, read_jsonl, seeded  # noqa: E402

D = Path("/data/phonon_pool3_v0")
BIG = Path("/data/phonon_bigrun_v0/train_big.jsonl")
P2 = Path("/data/phonon_pool2_v0/train_pool2.jsonl")
TOPK = 30
P_NOLINE = 0.20
P_RESTRAINT = 0.30
REPLAY_P2 = 0.25
REPLAY_P1 = 0.15
REAL_REPEATS = 4
SEED = 3918


def main() -> int:
    t0 = time.perf_counter()
    L_ac = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_acoustic3.jsonl")}
    ac = read_jsonl(D / "acoustic_pool3.jsonl")
    print(f"acoustic pool3={len(ac)} lists={len(L_ac)} [{time.perf_counter()-t0:.0f}s]",
          flush=True)

    n = Counter()

    def ac_row(row: dict) -> dict:
        rng = seeded("v1ac3", row["id"])
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
        return {"id": row["id"], "source": "tts_parakeet_pool3", "branch": branch,
                "stage": "pool3", "term": term, "kind": row.get("kind"),
                "voice": row["voice"], "sentence_idx": row["sentence_idx"],
                "input": row["input"], "target": target,
                "true_terms": [term] if term else [], "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, target)}

    new_rows = [ac_row(r) for r in ac]
    print(f"pool3 rows built={len(new_rows)} [{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---- replay of the pool-2 acoustic rows ------------------------------
    rng2 = random.Random(SEED)
    replay2, n_avail2 = [], 0
    with P2.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") == "tts_parakeet_pool2":
                n_avail2 += 1
                if rng2.random() < REPLAY_P2:
                    r["source"] = "tts_parakeet_replay_p2"
                    replay2.append(r)
    print(f"replay pool2 {len(replay2)}/{n_avail2} [{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---- replay of the pool-1 acoustic rows + the real rows --------------
    rng1 = random.Random(SEED + 1)
    replay1, real_all, n_avail1 = [], [], 0
    with BIG.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") == "tts_parakeet":
                n_avail1 += 1
                if rng1.random() < REPLAY_P1:
                    r["source"] = "tts_parakeet_replay_p1"
                    replay1.append(r)
            else:
                rep = r["id"].rsplit("#r", 1)[-1]
                if rep.isdigit() and int(rep) < REAL_REPEATS:
                    real_all.append(r)
    print(f"replay pool1 {len(replay1)}/{n_avail1} real {len(real_all)} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    rows = new_rows + replay2 + replay1 + real_all
    random.Random(SEED).shuffle(rows)
    out = D / "train_pool3.jsonl"
    with out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    terms_new = {r["term"] for r in new_rows if r.get("term")}
    terms_r2 = {r.get("term") for r in replay2 if r.get("term")}
    terms_r1 = {r.get("term") for r in replay1 if r.get("term")}
    stats = {
        "path": str(out), "rows": len(rows),
        "pool3_rows": len(new_rows), "pool3_terms": len(terms_new),
        "replay_p2_rows": len(replay2), "replay_p2_available": n_avail2,
        "replay_p2_fraction": REPLAY_P2, "replay_p2_terms": len(terms_r2),
        "replay_p1_rows": len(replay1), "replay_p1_available": n_avail1,
        "replay_p1_fraction": REPLAY_P1, "replay_p1_terms": len(terms_r1),
        "real_rows": len(real_all), "real_unique": len(real_all) // REAL_REPEATS,
        "real_repeat": REAL_REPEATS,
        "distinct_terms_total": len(terms_new | terms_r2 | terms_r1),
        "distinct_terms_cumulative_note":
            "terms seen by bigrun_xxl_r16 across its whole history = pool1 + pool2 + pool3",
        "branches": dict(Counter(r.get("branch") for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "branch_counters": dict(n),
    }
    (D / "mix3_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"STEP 4c done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
