"""Batch ASR oracle for the lexicon on a Linux GPU box: Kokoro TTS (two voices) -> Parakeet TDT 0.6B v2 (NeMo).

Same cache format as profile_miner.oracle: {"term": ..., "voices": {voice: heard}} per line.
usage: python oracle_linux.py TERMS_JSON CACHE_JSONL [--batch 64]
Setup: lexicon/oracle_linux_setup.sh (uv venv with nemo_toolkit[asr], kokoro, misaki[en]).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

CARRIER = "let's use {} for this"
RE_CARRIER = re.compile(r"^\W*let'?s\s+use\s+(.*?)\s*for\s+this\W*$", re.I)
VOICES = ["af_heart", "am_michael"]
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"


def strip_carrier(text):
    m = RE_CARRIER.match(text.strip())
    return (m.group(1).strip(), True) if m else (text.strip().rstrip(".!?,"), False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terms"); ap.add_argument("cache"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    terms = list(dict.fromkeys(json.load(open(a.terms))))
    cache = {}
    cp = Path(a.cache)
    if cp.exists():
        for line in open(cp):
            try:
                d = json.loads(line); cache[d["term"]] = d["voices"]
            except Exception:
                pass
    todo = [t for t in terms if t not in cache]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[oracle] {len(terms)} terms, {len(cache)} cached, {len(todo)} to run", file=sys.stderr)
    if not todo:
        return
    import torch
    from kokoro import KPipeline
    import nemo.collections.asr as nemo_asr
    tts = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu")
    asr = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    asr.eval()
    if torch.cuda.is_available():
        asr = asr.cuda()
    t0 = time.time(); n = 0; mism = 0
    tmp = Path(tempfile.mkdtemp(prefix="oracle_"))
    with open(cp, "a") as out:
        for i in range(0, len(todo), a.batch):
            batch = todo[i:i + a.batch]
            paths, keys = [], []
            for term in batch:
                for v in VOICES:
                    try:
                        chunks = [aud for _, _, aud in tts(CARRIER.format(term), voice=v)]
                        audio = np.concatenate([c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks]) if chunks else np.zeros(2400, dtype=np.float32)
                    except Exception as e:
                        print(f"[oracle] tts failed for {term!r} {v}: {e}", file=sys.stderr)
                        audio = np.zeros(2400, dtype=np.float32)
                    # Kokoro is 24 kHz; Parakeet wants 16 kHz.
                    x = np.asarray(audio, dtype=np.float32)
                    idx = np.arange(0, len(x), 1.5)
                    y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
                    p = tmp / f"{len(paths)}.wav"
                    sf.write(p, y, 16000)
                    paths.append(str(p)); keys.append((term, v))
            with torch.inference_mode():
                hyps = asr.transcribe(paths, batch_size=len(paths), verbose=False)
            texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]
            res = {}
            for (term, v), text in zip(keys, texts):
                heard, ok = strip_carrier(text)
                if not ok:
                    mism += 1
                res.setdefault(term, {})[v] = heard
            for term in batch:
                out.write(json.dumps({"term": term, "voices": res.get(term, {})}, ensure_ascii=False) + "\n")
            out.flush()
            n += len(batch)
            if (i // a.batch) % 10 == 0:
                el = time.time() - t0
                print(f"[oracle] {n}/{len(todo)} {el:.0f}s {n / max(el, 1):.1f} terms/s mismatches {mism}", file=sys.stderr)
    print(f"[oracle] done {n} terms in {time.time() - t0:.0f}s, carrier mismatches {mism}", file=sys.stderr)
    print("ORACLE_DONE")


if __name__ == "__main__":
    main()
