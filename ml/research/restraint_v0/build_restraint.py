"""restraint_v0: variant training files, same recipe as bigrun_v0/step5c_mix.py with knobs moved.

Variants written to /data/phonon_restraint_v0/:
  train_a.jsonl   restraint50_zero45 : P_RESTRAINT 0.50, zero-edit (target==input) cap 0.45
  train_b.jsonl   real12             : real rows x12, everything else as mid
  train_d2.jsonl  stage-2 mix        : real rows x4 + a seeded 25% replay of the acoustic rows

Seeds, branch draws and row ids are identical to step5c_mix.py so anything the knob does not
touch is byte-comparable with train_mid.jsonl.
"""
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

BIG = Path("/data/phonon_bigrun_v0")
D = Path("/data/phonon_restraint_v0")
TRAIN_REAL = Path("/data/phonon_corrector_v0/datesplit/train_date.jsonl")
HELD_OLD = Path("/data/phonon_term_eval_v0/terms_heldout.jsonl")
HELD_NEW = BIG / "terms_heldout_new.jsonl"
TOPK = 30
P_NOLINE = 0.20
SEED = 918
REPLAY_FRAC = 0.25


def main() -> int:
    t0 = time.perf_counter()
    D.mkdir(parents=True, exist_ok=True)
    held = {r["term"] for r in read_jsonl(HELD_OLD)} | {r["term"] for r in read_jsonl(HELD_NEW)}
    lex = Lexicon(exclude=held)
    L_real = {r["id"]: r["terms"] for r in read_jsonl(BIG / "lists_real.jsonl")}
    L_ac = {r["id"]: r["terms"] for r in read_jsonl(BIG / "lists_acoustic.jsonl")}
    real = read_jsonl(TRAIN_REAL)
    ac_all = read_jsonl(BIG / "acoustic_pool.jsonl")
    ac = [r for r in ac_all if r["stage"] == "mid"]
    print(f"real={len(real)} acoustic_mid={len(ac)} held={len(held)} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    n = Counter()

    def real_row(row: dict, rep: int) -> dict:
        rid = f"{row['id']}#r{rep}"
        rng = seeded("v1real", rid)
        terms = L_real.get(row["id"], [])
        true_terms = [t for t in lex.find_terms(row["target"]) if t not in held]
        if rng.random() < P_NOLINE:
            vocab = []
        else:
            vocab = list(terms)
            rng.shuffle(vocab)
        return {"id": rid, "source": row.get("source", "real"),
                "branch": "noline" if not vocab else "list",
                "input": row["input"], "target": row["target"],
                "true_terms": true_terms, "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, row["target"])}

    def ac_row(row: dict, p_restraint: float) -> dict:
        rng = seeded("v1ac", row["id"])
        term = row.get("term") or ""
        terms = L_ac.get(row["id"], [])
        u = rng.random()
        if u < P_NOLINE:
            vocab, target, branch = [], row["target"], "noline"
        elif u < P_NOLINE + p_restraint:
            kept = keep_as_heard(row["input"], row["target"], term)
            if kept is None:
                kept = row["input"]
                n[f"ac_restraint_failed_p{p_restraint}"] += 1
            vocab = [t for t in terms if t != term][:TOPK]
            rng.shuffle(vocab)
            target, branch = kept, "restraint"
        else:
            vocab = list(terms[:TOPK])
            if term and term not in vocab:
                vocab = vocab[:TOPK - 1] + [term]
            rng.shuffle(vocab)
            target, branch = row["target"], "list"
        return {"id": row["id"], "source": "tts_parakeet", "branch": branch,
                "stage": row["stage"], "term": term, "kind": row.get("kind"),
                "voice": row["voice"], "sentence_idx": row["sentence_idx"],
                "input": row["input"], "target": target,
                "true_terms": [term] if term else [], "vocab": vocab,
                "messages": chat_messages(row["input"], vocab, target)}

    def zero_edit(r: dict) -> bool:
        return r["input"].strip() == r["target"].strip()

    stats: dict = {}

    # ---- reference: the mid recipe, to report its own zero-edit rate --------
    mid_rows = [ac_row(r, 0.30) for r in ac]
    mid_zero = sum(1 for r in mid_rows if zero_edit(r))
    stats["mid_reference"] = {
        "acoustic_rows": len(mid_rows),
        "branches": dict(Counter(r["branch"] for r in mid_rows)),
        "zero_edit_rows": mid_zero,
        "zero_edit_frac": round(mid_zero / len(mid_rows), 4),
    }
    print(json.dumps(stats["mid_reference"], indent=1), flush=True)

    # ---- a: restraint 0.50, zero-edit cap 0.45 -----------------------------
    a_rows = [ac_row(r, 0.50) for r in ac]
    z_idx = [i for i, r in enumerate(a_rows) if zero_edit(r)]
    z_before = len(z_idx)
    cap_n = int(0.45 * len(a_rows))
    demoted = 0
    if z_before > cap_n:
        # demote the excess back to the corrected-target branch, seeded order. A demoted row
        # can still be zero-edit (the ASR already had it right), so keep going until the
        # count is under the cap or the candidate list runs out.
        rng = random.Random(SEED + 45)
        order = list(z_idx)
        rng.shuffle(order)
        live = z_before
        for i in order:
            if live <= cap_n:
                break
            a_rows[i] = ac_row(ac[i], 0.0)   # p_restraint 0 -> list/noline branch
            demoted += 1
            if not zero_edit(a_rows[i]):
                live -= 1
    z_after = sum(1 for r in a_rows if zero_edit(r))
    stats["a_acoustic"] = {
        "p_restraint": 0.50, "zero_edit_cap": 0.45, "acoustic_rows": len(a_rows),
        "zero_edit_before_cap": z_before, "zero_edit_frac_before": round(z_before/len(a_rows), 4),
        "cap_rows": cap_n, "demoted": demoted,
        "zero_edit_after_cap": z_after, "zero_edit_frac_after": round(z_after/len(a_rows), 4),
        "branches": dict(Counter(r["branch"] for r in a_rows)),
    }
    print(json.dumps(stats["a_acoustic"], indent=1), flush=True)

    real4 = [real_row(r, k) for k in range(4) for r in real]
    real12 = [real_row(r, k) for k in range(12) for r in real]

    def write(name: str, rows: list[dict]) -> dict:
        random.Random(SEED).shuffle(rows)
        p = D / f"train_{name}.jsonl"
        with p.open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = {"path": str(p), "rows": len(rows),
             "branches": dict(Counter(r["branch"] for r in rows)),
             "zero_edit_rows": sum(1 for r in rows if zero_edit(r)),
             "real_rows": sum(1 for r in rows if r["source"] != "tts_parakeet")}
        s["zero_edit_frac"] = round(s["zero_edit_rows"] / len(rows), 4)
        print(f"{name}: {json.dumps(s)}", flush=True)
        return s

    stats["a"] = write("a", a_rows + real4)
    stats["b"] = write("b", mid_rows + real12)

    # ---- d stage 2: real x4 + 25 percent replay of the mid acoustic rows ----
    rng = random.Random(SEED + 25)
    replay_idx = rng.sample(range(len(mid_rows)), int(REPLAY_FRAC * len(mid_rows)))
    replay = [mid_rows[i] for i in replay_idx]
    stats["d2"] = write("d2", replay + [dict(r) for r in real4])
    stats["d2"]["replay_frac"] = REPLAY_FRAC
    stats["d2"]["replay_rows"] = len(replay)

    (D / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"BUILD DONE [{time.perf_counter()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
