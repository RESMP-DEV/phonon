"""STEP 2: train_mid.jsonl -> train_mid_nbest.jsonl with an `ASR alternatives:` line on the
acoustic rows that have recogniser n-best alternatives. Real rows are untouched."""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/nbest_v0")
from altline import line_for, load_nbest, with_alt_line  # noqa: E402

SRC = Path("/data/phonon_bigrun_v0/train_mid.jsonl")
NB = Path("/data/phonon_nbest_v0/nbest_mid.jsonl")
OUT = Path("/data/phonon_nbest_v0/train_mid_nbest.jsonl")


def main() -> int:
    t0 = time.perf_counter()
    nb = load_nbest(NB)
    print(f"nbest clips={len(nb)} [{time.perf_counter()-t0:.0f}s]", flush=True)
    n = Counter()
    spans = 0
    with SRC.open(encoding="utf-8") as src, OUT.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            r = json.loads(line)
            n["rows"] += 1
            if r.get("source") == "tts_parakeet":
                n["acoustic"] += 1
                key = r["id"][4:] if r["id"].startswith("big_") else r["id"]
                rec = nb.get(key)
                if rec is not None:
                    n["acoustic_with_nbest"] += 1
                    ln = line_for(rec, r["input"])
                    if ln:
                        n["acoustic_with_line"] += 1
                        spans += len(ln.split(" | "))
                        msgs = r["messages"]
                        for m in msgs:
                            if m["role"] == "user":
                                m["content"] = with_alt_line(m["content"], ln)
                                break
                        r["alt_line"] = ln
            else:
                n["real"] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(n)
    stats["mean_spans_when_line"] = spans / max(n["acoustic_with_line"], 1)
    stats["line_share_of_acoustic"] = n["acoustic_with_line"] / max(n["acoustic"], 1)
    stats["seconds"] = time.perf_counter() - t0
    Path("/data/phonon_nbest_v0/train_build_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)
    print(f"STEP 2 build done -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
