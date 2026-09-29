"""bigrun_v0 step 6a: heldout_new sentences -> TTS jobs on the 3 UNSEEN voices."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

D = Path("/data/phonon_bigrun_v0")
UNSEEN = ["af_sarah", "am_fenrir", "bf_isabella"]
SRC = D / "sentences_heldout_new.jsonl"


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_term: dict[str, list[dict]] = {}
    for r in rows:
        by_term.setdefault(r["term"], []).append(r)
    for v in by_term.values():
        v.sort(key=lambda r: int(r["id"].rsplit("_s", 1)[1]))
    jobs = []
    for term in sorted(by_term):
        for s in by_term[term][:3]:
            for v in UNSEEN:
                jobs.append({"id": f"{s['id']}__{v}", "sentence_id": s["id"], "term": term,
                             "kind": s.get("kind"), "voice": v, "sentence": s["sentence"]})
    halves = [[], []]
    for i, j in enumerate(jobs):
        halves[i % 2].append(j)
    for h, hj in enumerate(halves):
        p = D / f"jobs_heldout_half{h}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for j in hj:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"{p.name}: {len(hj)}", flush=True)
    print(json.dumps({"terms": len(by_term), "sentences": len(rows), "clips": len(jobs),
                      "voices": dict(Counter(j["voice"] for j in jobs))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
