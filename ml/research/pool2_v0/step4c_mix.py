"""pool2 step 4c: v1-recipe rows for pool 2 + 25 percent replay of train_big acoustic rows
+ the train_date real rows repeated 4x (lists as built in bigrun_v0)."""
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

D = Path("/data/phonon_pool2_v0")
BIG = Path("/data/phonon_bigrun_v0/train_big.jsonl")
TOPK = 30
P_NOLINE = 0.20
P_RESTRAINT = 0.30
REPLAY_FRAC = 0.25
REAL_REPEATS = 4
SEED = 2918


def main() -> int:
    t0 = time.perf_counter()
    L_ac = {r["id"]: r["terms"] for r in read_jsonl(D / "lists_acoustic2.jsonl")}
    ac = read_jsonl(D / "acoustic_pool2.jsonl")
    print(f"acoustic pool2={len(ac)} lists={len(L_ac)} [{time.perf_counter()-t0:.0f}s]",
          flush=True)

    n = Counter()

    def ac_row(row: dict) -> dict:
        rng = seeded("v1ac2", row["id"])
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
        return {"id": row["id"], "source": "tts_parakeet_pool2", "branch": branch,
                "stage": "pool2", "term": term, "kind": row.get("kind"),
                "voice": row["voice"], "sentence_idx": row["sentence_idx"],
                "input": row["input"], "target": target,
                "true_terms": [term] if term else [], "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, target)}

    new_rows = [ac_row(r) for r in ac]
    print(f"pool2 rows built={len(new_rows)} [{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---- replay + real rows straight out of train_big.jsonl ---------------
    replay, real_all = [], []
    n_replay_avail = 0
    rng = random.Random(SEED)
    with BIG.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") == "tts_parakeet":
                n_replay_avail += 1
                if rng.random() < REPLAY_FRAC:
                    r["source"] = "tts_parakeet_replay"
                    replay.append(r)
            else:
                rep = r["id"].rsplit("#r", 1)[-1]
                if rep.isdigit() and int(rep) < REAL_REPEATS:
                    real_all.append(r)
    print(f"replay {len(replay)}/{n_replay_avail} real {len(real_all)} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    rows = new_rows + replay + real_all
    random.Random(SEED).shuffle(rows)
    out = D / "train_pool2.jsonl"
    with out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "path": str(out), "rows": len(rows),
        "pool2_rows": len(new_rows), "pool2_terms": len({r["term"] for r in new_rows}),
        "replay_rows": len(replay), "replay_available": n_replay_avail,
        "replay_fraction": REPLAY_FRAC,
        "replay_fraction_actual": len(replay) / max(n_replay_avail, 1),
        "replay_terms": len({r.get("term") for r in replay if r.get("term")}),
        "real_rows": len(real_all), "real_unique": len(real_all) // REAL_REPEATS,
        "real_repeat": REAL_REPEATS,
        "distinct_terms_total": len({r["term"] for r in new_rows} |
                                    {r.get("term") for r in replay if r.get("term")}),
        "branches": dict(Counter(r["branch"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "branch_counters": dict(n),
    }
    (D / "mix2_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"STEP 4c done [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
