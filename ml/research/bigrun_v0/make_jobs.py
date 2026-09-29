"""bigrun_v0 step 4a: sentences -> TTS jobs, 3 training voices per (term, sentence) round-robin.

MID  = sentences 0-3, voices (t+0,t+1,t+2) mod 9 for term index t
BIG  = sentences 4-7, voices (t+3,t+4,t+5) mod 9  (the extra half)
Each half of the terms goes to one GPU.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

D = Path("/data/phonon_bigrun_v0")
TRAIN_VOICES = ["af_heart", "am_adam", "bf_emma", "af_bella", "am_michael", "bm_george",
                "af_nicole", "am_puck", "bm_lewis"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=Path, default=D / "sentences_8.jsonl")
    ap.add_argument("--terms", type=Path, default=D / "terms_pool.jsonl")
    args = ap.parse_args()

    order = {}
    for i, l in enumerate(args.terms.read_text(encoding="utf-8").splitlines()):
        if l.strip():
            order[json.loads(l)["term"]] = i
    rows = [json.loads(l) for l in args.sentences.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_term: dict[str, list[dict]] = {}
    for r in rows:
        by_term.setdefault(r["term"], []).append(r)
    for v in by_term.values():
        v.sort(key=lambda r: int(r["id"].rsplit("_s", 1)[1]))

    out = {("mid", 0): [], ("mid", 1): [], ("big", 0): [], ("big", 1): []}
    vc = Counter()
    n_sent = Counter()
    for term, sents in by_term.items():
        t = order.get(term)
        if t is None:
            continue
        half = t % 2
        for stage, sidx_lo, voff in (("mid", 0, 0), ("big", 4, 3)):
            voices = [TRAIN_VOICES[(t + voff + k) % 9] for k in range(3)]
            for s in sents:
                sidx = int(s["id"].rsplit("_s", 1)[1])
                if not (sidx_lo <= sidx < sidx_lo + 4):
                    continue
                n_sent[stage] += 1
                for v in voices:
                    vc[v] += 1
                    out[(stage, half)].append({
                        "id": f"{s['id']}__{v}", "sentence_id": s["id"], "term": term,
                        "kind": s.get("kind"), "voice": v, "sentence": s["sentence"]})

    tot = 0
    for (stage, half), jobs in out.items():
        p = D / f"jobs_{stage}_half{half}.jsonl"
        with p.open("w", encoding="utf-8") as h:
            for j in jobs:
                h.write(json.dumps(j, ensure_ascii=False) + "\n")
        tot += len(jobs)
        print(f"{p.name}: {len(jobs)} clips", flush=True)
    stats = {"terms": len(by_term), "sentences": len(rows), "clips_total": tot,
             "mid_sentences": n_sent["mid"], "big_sentences": n_sent["big"],
             "voice_counts": dict(sorted(vc.items()))}
    (D / "jobs_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print("JOBS done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
