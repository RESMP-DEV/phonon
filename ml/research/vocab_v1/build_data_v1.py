"""vocab_v1 step 1+2: retrieval lists, retrieval recall, and the restraint training mix."""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import load_pool, squash  # noqa: E402
from vocab_common import Lexicon, chat_messages, heldout_terms, read_jsonl, seeded  # noqa: E402

OUT = Path("/data/phonon_vocab_v1")
V0 = Path("/data/phonon_vocab_v0")
TRAIN_REAL = Path("/data/phonon_corrector_v0/train.jsonl")
ACOUSTIC = Path("/data/phonon_synth_v3/acoustic_pairs.jsonl")
TOPK = 30

P_NOLINE = 0.20
P_RESTRAINT = 0.30      # of acoustic rows (drawn after the no-line branch fails)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


def keep_as_heard(inp: str, tgt: str, term: str) -> str | None:
    """Target with the span that carries `term` reverted to what the ASR heard."""
    iw, tw = inp.split(), tgt.split()
    lo_t = None
    st = squash(term)
    for j, w in enumerate(tw):
        if st and st in squash(w):
            lo_t = j
            break
    if lo_t is None:                      # term spans several target words
        for j in range(len(tw)):
            for n in (2, 3, 4):
                if j + n <= len(tw) and st and st in squash(" ".join(tw[j:j + n])):
                    lo_t = j
                    break
            if lo_t is not None:
                break
    if lo_t is None:
        return None
    hi_t = lo_t
    while hi_t + 1 <= len(tw) and st not in squash(" ".join(tw[lo_t:hi_t + 1])):
        hi_t += 1
        if hi_t - lo_t > 5:
            return None
    hi_t += 1                              # exclusive
    # case-sensitive on purpose: `torch` -> `TORCH` is exactly the edit restraint must undo
    sm = difflib.SequenceMatcher(a=iw, b=tw, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        overlaps = not (j2 <= lo_t or j1 >= hi_t)
        if tag == "equal":
            out.extend(tw[j1:j2])
        elif overlaps:
            out.extend(iw[i1:i2])          # revert this edit: keep it as heard
        else:
            out.extend(tw[j1:j2])          # keep the corrected target elsewhere
    return " ".join(out)


def main() -> int:
    t0 = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731
    held = heldout_terms()
    lex = Lexicon(exclude=set(held))
    R_eval = load_pool()                     # held-out terms retrievable
    R_train = load_pool(exclude=set(held))   # never retrievable in training lists
    log(f"retrieval pool eval={len(R_eval.terms)} train={len(R_train.terms)}")
    stats: dict = {"topk": TOPK, "pool_eval": len(R_eval.terms), "pool_train": len(R_train.terms)}

    # ---------- step 1: eval retrieval lists + recall ---------------------
    cond = read_jsonl(V0 / "eval_conditions.jsonl")
    lists = R_eval.topk([r["hyp"] for r in cond], k=TOPK, log=log)
    rec = {}
    for k in (10, 30):
        rec[f"recall@{k}"] = sum(
            1 for r, L in zip(cond, lists) if r["term"] in [t for t, _ in L[:k]]) / len(cond)
    by_kind: dict[str, list[int]] = {}
    for r, L in zip(cond, lists):
        by_kind.setdefault(r["kind"], []).append(int(r["term"] in [t for t, _ in L]))
    rec["recall@30_by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}
    stats["retrieval"] = rec
    log(f"retrieval recall {json.dumps({k: v for k, v in rec.items() if k != 'recall@30_by_kind'})}")
    n_ret_missing = 0
    for r, L in zip(cond, lists):
        terms = [t for t, _ in L]
        rng = seeded("v1eval", r["id"])
        r["retrieved_has_term"] = r["term"] in terms
        n_ret_missing += 0 if r["retrieved_has_term"] else 1
        rng.shuffle(terms)
        r["vocab_retrieved"] = terms
    stats["eval_rows"] = len(cond)
    stats["eval_retrieved_missing_term"] = n_ret_missing
    write_jsonl(OUT / "eval_conditions.jsonl", cond)
    log(f"STEP 1 done: {len(cond)} clips, retrieved-missing={n_ret_missing} "
        f"[{time.perf_counter()-t0:.0f}s]")

    # ---------- step 2: training mix --------------------------------------
    real = read_jsonl(TRAIN_REAL)
    real_lists = R_train.topk([r["input"] for r in real], k=TOPK, log=log)
    ac_all = read_jsonl(ACOUSTIC)
    ac = [r for r in ac_all if (r.get("term") or "") not in held]
    stats["acoustic_dropped_heldout"] = len(ac_all) - len(ac)
    ac_lists = R_train.topk([r["input"] for r in ac], k=TOPK + 1, log=log)

    rows_out = []
    n = {"real_noline": 0, "real_list": 0, "ac_noline": 0, "ac_restraint": 0, "ac_normal": 0,
         "ac_forced_term": 0, "ac_restraint_noop": 0, "ac_restraint_failed": 0,
         "real_list_has_true_term": 0, "real_rows_with_true_term": 0}
    for row, L in zip(real, real_lists):
        rng = seeded("v1real", row["id"])
        terms = [t for t, _ in L]
        true_terms = [t for t in lex.find_terms(row["target"]) if t not in held]
        if true_terms:
            n["real_rows_with_true_term"] += 1
            if any(t in terms for t in true_terms):
                n["real_list_has_true_term"] += 1
        if rng.random() < P_NOLINE:
            vocab = []
            n["real_noline"] += 1
        else:
            vocab = list(terms)
            rng.shuffle(vocab)
            n["real_list"] += 1
        rows_out.append({
            "id": row["id"], "source": row["source"], "branch": "noline" if not vocab else "list",
            "input": row["input"], "target": row["target"], "true_terms": true_terms,
            "vocab": vocab, "messages": chat_messages(row["input"], vocab, row["target"]),
        })

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
            if kept.strip() == (row["target"] or "").strip():
                n["ac_restraint_noop"] += 1
            vocab = [t for t in terms if t != term][:TOPK]   # 31 retrieved -> always 30
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
        rows_out.append({
            "id": row["id"], "source": row.get("source", "tts_parakeet"), "branch": branch,
            "input": row["input"], "target": target, "true_terms": [term] if term else [],
            "vocab": vocab, "messages": chat_messages(row["input"], vocab, target),
        })

    write_jsonl(OUT / "train_vocab1_real_acoustic.jsonl", rows_out)
    stats["train_rows"] = len(rows_out)
    stats["train_branches"] = n
    log(f"train rows={len(rows_out)} {json.dumps(n)}")

    # ---------- real-dictation holdouts with retrieved lists ---------------
    sys.path.insert(0, "/home/user/phonon/research/corrector_v0")
    from eval_corrector import eval_rows  # noqa: E402

    sets = eval_rows()
    for name in ("wispr_holdout120", "wispr_text_holdout"):
        rows = sets[name]
        L = R_eval.topk([r["input"] for r in rows], k=TOPK, log=log)
        out_rows = []
        for row, lst in zip(rows, L):
            rng = seeded("v1realeval", name, row["id"])
            terms = [t for t, _ in lst]
            rng.shuffle(terms)
            out_rows.append({
                "id": row["id"], "input": row["input"], "reference": row["reference"],
                "true_terms": lex.find_terms(row["reference"]), "vocab_retrieved": terms,
            })
        write_jsonl(OUT / f"eval_{name}.jsonl", out_rows)
        cov = [sum(1 for t in r["true_terms"] if t in r["vocab_retrieved"]) for r in out_rows]
        tot = [len(r["true_terms"]) for r in out_rows]
        stats[f"{name}_rows"] = len(out_rows)
        stats[f"{name}_ref_term_recall@30"] = (sum(cov) / sum(tot)) if sum(tot) else None
        log(f"{name} rows={len(out_rows)} ref-term recall@30="
            f"{(sum(cov)/sum(tot) if sum(tot) else 0):.3f}")

    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"STEP 2 done ({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
