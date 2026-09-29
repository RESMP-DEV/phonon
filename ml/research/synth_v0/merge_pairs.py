"""Union error-model pairs and TTS pairs into synth_pairs.jsonl (train_lora schema)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_ROOT, PAIRS_PATH, RESEARCH_ROOT  # noqa: E402


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    em = DATA_ROOT / "error_model_pairs.jsonl"
    tts = DATA_ROOT / "tts_pairs.jsonl"
    seen: set[tuple[str, str]] = set()
    n = 0
    n_em = 0
    n_tts = 0
    sample = []
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("w", encoding="utf-8") as dst:
        for src_path, tag in ((em, "em"), (tts, "tts")):
            for row in iter_jsonl(src_path) or []:
                raw = (row.get("input") or "").strip()
                target = (row.get("target") or "").strip()
                if not raw or not target:
                    continue
                key = (raw, target)
                if key in seen:
                    continue
                seen.add(key)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if tag == "em":
                    n_em += 1
                else:
                    n_tts += 1
                if tag == "em" and len([s for s in sample if s.get("route") == "error_model"]) < 150:
                    sample.append(row)
                elif tag == "tts" and len([s for s in sample if s.get("route") == "tts_asr"]) < 50:
                    sample.append(row)
    sample_path = RESEARCH_ROOT / "sample_200.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"synth_pairs n={n} em={n_em} tts={n_tts} -> {PAIRS_PATH}", flush=True)
    print(f"sample_200 -> {sample_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
