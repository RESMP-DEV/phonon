"""Transcribe wispr_train wavs with option0 model identities.

NeMo models (unified, local v3) must run in the repo venv so NeMo stays 2.3.0:
    cd /home/user/phonon && uv run python research/corrector_v1/transcribe_train.py \\
        --family nemo --model nvidia/parakeet-unified-en-0.6b \\
        --out /data/phonon_corrector_v1/hyps_parakeet_unified.jsonl

Cohere uses the isolated transformers-cohere environment from
scripts/option0/run_sweep.sh (transformers==5.17.0). Decode settings are
imported from scripts/option0/transcribe_gate.py (English processor, 30 s chunks).

Parakeet v2 is not re-transcribed; --export-parakeet-v2 copies existing raw text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_ROOT,
    HYP_PATHS,
    OPTION0_DIR,
    PARAKEET_JSONL,
    ensure_data_dirs,
    heartbeat,
    load_done_ids,
    read_jsonl,
    set_hf_env,
    write_jsonl,
)

sys.path.insert(0, str(OPTION0_DIR))


def train_jobs() -> list[dict]:
    jobs = []
    for row in read_jsonl(PARAKEET_JSONL):
        if row.get("split") != "wispr_train":
            continue
        wav = row.get("wav") or ""
        if not wav:
            continue
        jobs.append(
            {
                "id": row["id"],
                "wav": wav,
                "target": row.get("target") or "",
            }
        )
    return jobs


def export_parakeet_v2(out: Path) -> int:
    import soundfile as sf

    ensure_data_dirs()
    rows = []
    for job in read_jsonl(PARAKEET_JSONL):
        if job.get("split") != "wispr_train":
            continue
        wav = Path(job["wav"])
        seconds = 0.0
        if wav.exists():
            seconds = float(sf.info(str(wav)).duration)
        rows.append(
            {
                "id": job["id"],
                "hypothesis": job.get("parakeet_raw") or "",
                "audio_seconds": seconds,
                "elapsed_seconds": float(job.get("elapsed_seconds") or 0.0),
            }
        )
    write_jsonl(out, rows)
    audio = sum(r["audio_seconds"] for r in rows)
    elapsed = sum(r["elapsed_seconds"] for r in rows)
    meta = {
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "family": "nemo",
        "source": str(PARAKEET_JSONL),
        "clips": len(rows),
        "audio_seconds": audio,
        "elapsed_seconds": elapsed,
        "realtime_factor": (elapsed / audio) if audio else None,
        "note": "copied from parakeet_v2_wispr.jsonl; not re-transcribed",
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)
    return 0


def transcribe(family: str, model_id: str, out: Path, batch_size: int, limit: int) -> int:
    from gates import chunk_audio, load_audio
    from transcribe_gate import load_model, transcribe_batch

    import soundfile as sf
    import torch

    set_hf_env()
    ensure_data_dirs()
    jobs = train_jobs()
    if limit:
        jobs = jobs[:limit]
    missing = [job for job in jobs if not Path(job["wav"]).exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} wavs, first={missing[0]['wav']}")
    done = load_done_ids(out)
    remaining = [job for job in jobs if job["id"] not in done]
    print(
        f"family={family} model={model_id} jobs={len(jobs)} done={len(done)} "
        f"remaining={len(remaining)} out={out}",
        flush=True,
    )
    if not remaining:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    cache_dir = DATA_ROOT / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model, processor = load_model(family, model_id)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    print(f"loaded in {load_seconds:.1f}s", flush=True)
    heartbeat()
    if family == "nemo":
        batch_size = max(1, min(batch_size, 4))
    elif family == "transformers-granite":
        batch_size = 1
    else:
        batch_size = max(1, min(batch_size, 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    last_beat = time.perf_counter()
    with out.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for index, job in enumerate(remaining, start=1):
            torch.cuda.synchronize()
            clip_start = time.perf_counter()
            row = {"id": job["id"], "audio_path": job["wav"]}
            wave = load_audio(row)
            seconds = len(wave) / 16000
            chunks = chunk_audio(wave)
            hypotheses = []
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                paths = []
                for chunk in batch:
                    digest = hashlib.sha256(chunk.tobytes()).hexdigest()
                    path = cache_dir / f"{digest}.wav"
                    if not path.exists():
                        sf.write(path, chunk, 16000, subtype="FLOAT")
                    paths.append(str(path))
                result = transcribe_batch(family, model, processor, batch, paths, model_id)
                if isinstance(result, str):
                    result = [result]
                if len(result) != len(batch):
                    raise RuntimeError(f"output length {len(result)} != batch {len(batch)}")
                hypotheses.extend(str(item).strip() for item in result)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - clip_start
            record = {
                "id": job["id"],
                "hypothesis": " ".join(h for h in hypotheses if h).strip(),
                "audio_seconds": seconds,
                "elapsed_seconds": elapsed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index}/{len(remaining)} {job['id']}: {seconds:.2f}s audio, "
                f"{elapsed:.2f}s elapsed",
                flush=True,
            )
            now = time.perf_counter()
            if index == 1 or index % 25 == 0 or now - last_beat >= 600:
                heartbeat()
                last_beat = now
    rows = read_jsonl(out)
    audio = sum(float(r.get("audio_seconds") or 0.0) for r in rows)
    elapsed = sum(float(r.get("elapsed_seconds") or 0.0) for r in rows)
    meta = {
        "model": model_id,
        "family": family,
        "clips": len(rows),
        "audio_seconds": audio,
        "elapsed_seconds": elapsed,
        "model_load_seconds": load_seconds,
        "realtime_factor": (elapsed / audio) if audio else None,
        "batch_size": batch_size,
        "chunk_seconds": 30,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)
    heartbeat()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=["nemo", "transformers-cohere"],
        default="nemo",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--export-parakeet-v2",
        action="store_true",
        help="Copy existing Parakeet v2 raw text; no GPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_hf_env()
    if args.export_parakeet_v2:
        out = args.out or HYP_PATHS["parakeet_v2"]
        return export_parakeet_v2(out)
    if not args.model:
        raise SystemExit("--model is required unless --export-parakeet-v2")
    if args.out is None:
        raise SystemExit("--out is required")
    return transcribe(args.family, args.model, args.out, args.batch_size, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
