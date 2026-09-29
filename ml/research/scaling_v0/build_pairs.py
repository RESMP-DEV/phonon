"""scaling_v0 step 1b: ASR dump -> acoustic pair corpus with sentence_idx and voice.

Zero-edit rows are capped at 30 percent inside every (sentence_idx, voice) bucket, so any
sub-grid selection along the term / sentence / voice axes keeps the same zero-edit fraction.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path
from collections import Counter, defaultdict
from whisper_normalizer.english import EnglishTextNormalizer

D = Path("/data/phonon_scaling_v0")
SRC = [D / "asr_half0.jsonl", D / "asr_half1.jsonl"]
POOL = D / "acoustic_pool.jsonl"
BIG = D / "acoustic_36k.jsonl"
STATS = D / "pairs_stats.json"
SEEN_VOICES = ("af_heart", "am_adam", "bf_emma")
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
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    buckets: dict[tuple, list] = defaultdict(list)
    zeros: dict[tuple, list] = defaultdict(list)
    dropped_empty = 0
    for r in rows:
        hyp = (r.get("hyp") or "").strip()
        tgt = (r.get("reference") or "").strip()
        m = SIDX.search(r.get("sentence_id") or "")
        if not hyp or not tgt or not m:
            dropped_empty += 1
            continue
        sidx = int(m.group(1))
        rec = {
            "id": f"scale_{r['id']}",
            "term": r["term"],
            "kind": r.get("kind"),
            "voice": r["voice"],
            "sentence_idx": sidx,
            "degraded": r.get("degraded"),
            "input": hyp,
            "target": tgt,
            "source": "tts_parakeet",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": hyp},
                {"role": "assistant", "content": tgt},
            ],
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
    big = [r for r in out_rows if r["voice"] in SEEN_VOICES]
    with BIG.open("w", encoding="utf-8") as h:
        for r in big:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "asr_rows": len(rows), "dropped_empty": dropped_empty,
        "pool_rows": len(out_rows), "pool_terms": len({r["term"] for r in out_rows}),
        "big_rows": len(big), "big_terms": len({r["term"] for r in big}),
        "zero_edit_available": n_zero_avail, "zero_edit_kept": n_zero_kept,
        "zero_edit_fraction_pool": n_zero_kept / max(len(out_rows), 1),
        "voices": dict(Counter(r["voice"] for r in out_rows)),
        "sentence_idx": dict(Counter(r["sentence_idx"] for r in out_rows)),
        "kinds": dict(Counter(r["kind"] for r in out_rows)),
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2), flush=True)
    print(json.dumps(stats["voices"]), flush=True)
    print("STEP 1b pairs done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
