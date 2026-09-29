"""pool3 step 3a: sentences -> TTS jobs.

Training terms: SENTS_POOL sentences x VOICES_POOL of the 9 training voices (round robin by
term index).  Held-out terms: 3 sentences x the 3 unseen voices.
Scaled down from pool 2 (4 x 3) to fit one GPU and the 02:30 stop; distinct terms are the axis
the scaling law runs on, so all 36,227 terms are kept and the per-term clip count is cut.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

D = Path("/data/phonon_pool3_v0")
TRAIN_VOICES = ["af_heart", "am_adam", "bf_emma", "af_bella", "am_michael", "bm_george",
                "af_nicole", "am_puck", "bm_lewis"]
UNSEEN = ["af_sarah", "am_fenrir", "bf_isabella"]
SENTS_POOL = 2
VOICES_POOL = 2


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def group(rows):
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["term"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["id"].rsplit("_s", 1)[1]))
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pool", "heldout"], required=True)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    if args.which == "pool":
        terms = load(D / "terms_pool3.jsonl")
        sents = load(D / "sentences_pool3.jsonl")
        tag, nsent = "pool", SENTS_POOL
    else:
        terms = load(D / "terms_heldout_pool3.jsonl")
        sents = load(D / "sentences_heldout_pool3.jsonl")
        tag, nsent = "heldout", 3
    order = {r["term"]: i for i, r in enumerate(terms)}
    by_term = group(sents)

    jobs: list[dict] = []
    vc = Counter()
    for term, ss in by_term.items():
        t = order.get(term)
        if t is None:
            continue
        voices = UNSEEN if args.which == "heldout" else \
            [TRAIN_VOICES[(t + k) % 9] for k in range(VOICES_POOL)]
        for s in ss[:nsent]:
            for v in voices:
                vc[v] += 1
                jobs.append({"id": f"{s['id']}__{v}", "sentence_id": s["id"], "term": term,
                             "kind": s.get("kind"), "voice": v, "sentence": s["sentence"]})
    shards = [[] for _ in range(args.shards)]
    for i, j in enumerate(jobs):
        shards[i % args.shards].append(j)
    for h, hj in enumerate(shards):
        p = D / f"jobs_{tag}_s{h}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for j in hj:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"{p.name}: {len(hj)} clips", flush=True)
    stats = {"which": tag, "terms": len(by_term), "sentences": len(sents),
             "sentences_used": min(nsent, 99), "voices_per_term":
             len(UNSEEN) if tag == "heldout" else VOICES_POOL,
             "clips": len(jobs), "shards": args.shards,
             "voice_counts": dict(sorted(vc.items()))}
    (D / f"jobs_stats_{tag}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
