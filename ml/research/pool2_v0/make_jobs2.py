"""pool2 step 3a: sentences -> TTS jobs. Training terms get 3 of the 9 training voices
(round robin by term index); held-out terms get the 3 unseen voices."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

D = Path("/data/phonon_pool2_v0")
TRAIN_VOICES = ["af_heart", "am_adam", "bf_emma", "af_bella", "am_michael", "bm_george",
                "af_nicole", "am_puck", "bm_lewis"]
UNSEEN = ["af_sarah", "am_fenrir", "bf_isabella"]


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
    args = ap.parse_args()

    if args.which == "pool":
        terms = load(D / "terms_pool2.jsonl")
        sents = load(D / "sentences_4_half0.jsonl") + load(D / "sentences_4_half1.jsonl")
        tag = "pool"
    else:
        terms = load(D / "terms_heldout_pool2.jsonl")
        sents = load(D / "sentences_heldout_pool2.jsonl")
        tag = "heldout"
    order = {r["term"]: i for i, r in enumerate(terms)}
    by_term = group(sents)

    jobs: list[dict] = []
    vc = Counter()
    for term, ss in by_term.items():
        t = order.get(term)
        if t is None:
            continue
        voices = UNSEEN if args.which == "heldout" else \
            [TRAIN_VOICES[(t + k) % 9] for k in range(3)]
        for s in ss[:4 if args.which == "pool" else 3]:
            for v in voices:
                vc[v] += 1
                jobs.append({"id": f"{s['id']}__{v}", "sentence_id": s["id"], "term": term,
                             "kind": s.get("kind"), "voice": v, "sentence": s["sentence"]})
    halves = [[], []]
    for i, j in enumerate(jobs):
        halves[i % 2].append(j)
    for h, hj in enumerate(halves):
        p = D / f"jobs_{tag}_half{h}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for j in hj:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"{p.name}: {len(hj)} clips", flush=True)
    stats = {"which": tag, "terms": len(by_term), "sentences": len(sents),
             "clips": len(jobs), "voices": dict(sorted(vc.items()))}
    (D / f"jobs_stats_{tag}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
