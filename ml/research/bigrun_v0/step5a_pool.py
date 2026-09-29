"""bigrun_v0 step 5a: ASR dumps -> acoustic pair pool, zero-edit capped at 30 percent
inside each (sentence_idx, voice) bucket. stage is mid (sentence_idx 0-3) or big (4-7)."""
from __future__ import annotations
import json, random, re
from collections import Counter, defaultdict
from pathlib import Path
from whisper_normalizer.english import EnglishTextNormalizer

D = Path("/data/phonon_bigrun_v0")
SRC = [D / f"asr_{s}_half{h}.jsonl" for s in ("mid", "big") for h in (0, 1)]
POOL = D / "acoustic_pool.jsonl"
STATS = D / "pool_pairs_stats.json"
N = EnglishTextNormalizer()
SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
ZERO_EDIT_CAP = 0.30
SIDX = re.compile(r"_s(\d+)$")


def main() -> int:
    rows = []
    for p in SRC:
        if not p.exists():
            print(f"missing {p}", flush=True)
            continue
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"asr rows={len(rows)}", flush=True)
    buckets: dict[tuple, list] = defaultdict(list)
    zeros: dict[tuple, list] = defaultdict(list)
    dropped = 0
    for r in rows:
        hyp = (r.get("hyp") or "").strip()
        tgt = (r.get("reference") or "").strip()
        m = SIDX.search(r.get("sentence_id") or "")
        if not hyp or not tgt or not m:
            dropped += 1
            continue
        sidx = int(m.group(1))
        rec = {
            "id": f"big_{r['id']}",
            "term": r["term"], "kind": r.get("kind"), "voice": r["voice"],
            "sentence_idx": sidx, "stage": "mid" if sidx < 4 else "big",
            "degraded": r.get("degraded"),
            "input": hyp, "target": tgt, "source": "tts_parakeet",
        }
        key = (sidx, r["voice"])
        (zeros if N(hyp) == N(tgt) else buckets)[key].append(rec)

    rng = random.Random(7)
    out_rows, n_zero_kept, n_zero_avail = [], 0, 0
    for key in sorted(set(buckets) | set(zeros)):
        kept, ze = buckets.get(key, []), zeros.get(key, [])
        n_zero_avail += len(ze)
        n_keep = min(len(ze), int(ZERO_EDIT_CAP / (1 - ZERO_EDIT_CAP) * len(kept)))
        ze = sorted(ze, key=lambda r: r["id"])
        rng.shuffle(ze)
        z = ze[:n_keep]
        n_zero_kept += len(z)
        out_rows += kept + z
    out_rows.sort(key=lambda r: r["id"])
    with POOL.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    mid = [r for r in out_rows if r["stage"] == "mid"]
    stats = {
        "asr_rows": len(rows), "dropped_empty": dropped,
        "pool_rows": len(out_rows), "pool_terms": len({r["term"] for r in out_rows}),
        "mid_rows": len(mid), "mid_terms": len({r["term"] for r in mid}),
        "big_rows": len(out_rows), "big_terms": len({r["term"] for r in out_rows}),
        "zero_edit_available": n_zero_avail, "zero_edit_kept": n_zero_kept,
        "zero_edit_fraction_pool": n_zero_kept / max(len(out_rows), 1),
        "zero_edit_fraction_mid": sum(1 for r in mid if N(r["input"]) == N(r["target"])) / max(len(mid), 1),
        "voices": dict(Counter(r["voice"] for r in out_rows)),
        "sentence_idx": dict(sorted(Counter(r["sentence_idx"] for r in out_rows).items())),
        "kinds": dict(Counter(r["kind"] for r in out_rows).most_common()),
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print("STEP 5a done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
