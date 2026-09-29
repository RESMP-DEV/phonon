"""Step 5: turn the scale ASR dump into /data/phonon_synth_v3/acoustic_pairs.jsonl."""
from __future__ import annotations
import json, random, re
from pathlib import Path
from collections import Counter
from whisper_normalizer.english import EnglishTextNormalizer

SRC = Path("/data/phonon_synth_v3/asr_v3.jsonl")
OUT = Path("/data/phonon_synth_v3/acoustic_pairs.jsonl")
STATS = Path("/data/phonon_synth_v3/pairs_stats.json")
N = EnglishTextNormalizer()
SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
ZERO_EDIT_CAP = 0.30


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, zero_edit, dropped_empty = [], [], 0
    for r in rows:
        hyp = (r.get("hyp") or "").strip()
        tgt = (r.get("reference") or "").strip()
        if not hyp or not tgt:
            dropped_empty += 1
            continue
        rec = {
            "id": f"synth_v3_{r['id']}",
            "term": r["term"],
            "kind": r.get("kind"),
            "voice": r["voice"],
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
        if N(hyp) == N(tgt):
            zero_edit.append(rec)
        else:
            kept.append(rec)
    total = len(kept) + len(zero_edit)
    frac_zero = len(zero_edit) / max(total, 1)
    rng = random.Random(7)
    if frac_zero > ZERO_EDIT_CAP:
        n_keep = int(ZERO_EDIT_CAP / (1 - ZERO_EDIT_CAP) * len(kept))
        rng.shuffle(zero_edit)
        zero_kept = zero_edit[:n_keep]
    else:
        zero_kept = zero_edit
    out_rows = kept + zero_kept
    rng.shuffle(out_rows)
    with OUT.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "asr_rows": len(rows),
        "dropped_empty": dropped_empty,
        "edited_rows": len(kept),
        "zero_edit_rows_available": len(zero_edit),
        "zero_edit_fraction_before_cap": frac_zero,
        "zero_edit_rows_kept": len(zero_kept),
        "rows_written": len(out_rows),
        "zero_edit_fraction_after_cap": len(zero_kept) / max(len(out_rows), 1),
        "terms": len({r["term"] for r in out_rows}),
        "terms_all": len({r["term"] for r in rows}),
        "voices": dict(Counter(r["voice"] for r in out_rows)),
        "kinds": dict(Counter(r["kind"] for r in out_rows)),
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    print("STEP 5 pairs done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
