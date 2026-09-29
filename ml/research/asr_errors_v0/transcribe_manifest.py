"""Transcribe a wav manifest using scripts/option0/transcribe_gate.py loaders.

Does not change the existing gate CLI. Writes /data/phonon_asr_errors_v0/hyps/<tag>.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from common import (
    CHUNK_DIR,
    CLIPS_PATH,
    OPTION0,
    REPO,
    heartbeat,
    hyp_path,
    load_done_ids,
    read_jsonl,
    set_hf_env,
)

sys.path.insert(0, str(OPTION0))
sys.path.insert(0, str(REPO / "src"))


def resolve_model(name: str) -> str:
    """Map a Hub id to a local snapshot so NeMo restore_from never hits the network."""
    path = Path(name)
    if path.exists():
        return name
    cache = Path(os.environ.get("HF_HOME", "/data/hf")) / "hub"
    folder = cache / ("models--" + name.replace("/", "--"))
    snap = None
    main = folder / "refs" / "main"
    if main.exists():
        candidate = folder / "snapshots" / main.read_text().strip()
        if candidate.exists():
            snap = candidate
    if snap is None:
        snaps = sorted((folder / "snapshots").glob("*")) if (folder / "snapshots").exists() else []
        snap = snaps[-1] if snaps else None
    if snap is None:
        return name
    nemo = list(snap.glob("*.nemo"))
    if nemo:
        return str(nemo[0])
    if (snap / "config.json").exists():
        return str(snap)
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--manifest", type=Path, default=CLIPS_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def write_chunk(wave, cache: Path) -> str:
    import soundfile as sf

    digest = hashlib.sha256(wave.tobytes()).hexdigest()
    path = cache / f"{digest}.wav"
    if not path.exists():
        alt = Path("/data/phonon_segments_root/option0_16k") / f"{digest}.wav"
        if alt.exists():
            return str(alt)
        sf.write(path, wave, 16000, subtype="FLOAT")
    return str(path)


def transcribe_waves(family, model, processor, model_id, waves, batch_size):
    from transcribe_gate import transcribe_batch

    import torch

    cache = CHUNK_DIR
    cache.mkdir(parents=True, exist_ok=True)
    paths = [write_chunk(wave, cache) for wave in waves]
    try:
        result = transcribe_batch(family, model, processor, waves, paths, model_id)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(waves) == 1:
            raise
        mid = max(1, len(waves) // 2)
        left = transcribe_waves(family, model, processor, model_id, waves[:mid], batch_size)
        right = transcribe_waves(family, model, processor, model_id, waves[mid:], batch_size)
        return left + right
    if isinstance(result, str):
        result = [result]
    if len(result) != len(waves):
        raise RuntimeError(f"output length {len(result)} != batch {len(waves)}")
    return [str(item).strip() for item in result]


def main() -> int:
    args = parse_args()
    set_hf_env()
    from gates import chunk_audio, load_audio
    from transcribe_gate import load_model

    import torch

    out = args.out or hyp_path(args.tag)
    clips = read_jsonl(args.manifest)
    if args.limit:
        clips = clips[: args.limit]
    done = load_done_ids(out)
    pending = [c for c in clips if c["id"] not in done]
    print(
        f"tag={args.tag} family={args.family} jobs={len(clips)} done={len(done)} "
        f"remaining={len(pending)} out={out}",
        flush=True,
    )
    if not pending:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"cuda_visible={visible!r} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    if visible.strip() != "0":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be 0, got {visible!r}")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model_id = resolve_model(args.model)
    print(f"resolved_model={model_id}", flush=True)
    model, processor = load_model(args.family, model_id)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    print(f"loaded in {load_seconds:.1f}s", flush=True)
    heartbeat()
    batch_size = max(1, args.batch_size)
    if args.family == "transformers-granite":
        batch_size = 1
    out.parent.mkdir(parents=True, exist_ok=True)
    last_beat = time.perf_counter()
    prepared = []
    for job in pending:
        wave = load_audio({"id": job["id"], "audio_path": job["wav"]})
        chunks = chunk_audio(wave)
        prepared.append((job, float(len(wave) / 16000), chunks))

    index = 0
    with out.open("a", encoding="utf-8") as handle, torch.inference_mode():
        i = 0
        while i < len(prepared):
            job, seconds, chunks = prepared[i]
            if len(chunks) != 1 or batch_size == 1:
                torch.cuda.synchronize()
                clip_start = time.perf_counter()
                pieces = []
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    pieces.extend(
                        transcribe_waves(
                            args.family, model, processor, model_id, batch, batch_size
                        )
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - clip_start
                record = {
                    "id": job["id"],
                    "hypothesis": " ".join(p for p in pieces if p).strip(),
                    "audio_seconds": seconds,
                    "elapsed_seconds": elapsed,
                    "source": "live",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                index += 1
                print(
                    f"{index}/{len(pending)} {job['id']}: {seconds:.2f}s audio, {elapsed:.2f}s",
                    flush=True,
                )
                i += 1
            else:
                group = []
                while i < len(prepared) and len(group) < batch_size and len(prepared[i][2]) == 1:
                    group.append(prepared[i])
                    i += 1
                torch.cuda.synchronize()
                clip_start = time.perf_counter()
                waves = [item[2][0] for item in group]
                texts = transcribe_waves(
                    args.family, model, processor, model_id, waves, batch_size
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - clip_start
                share = elapsed / max(1, len(group))
                for (job, seconds, _chunks), text in zip(group, texts, strict=True):
                    record = {
                        "id": job["id"],
                        "hypothesis": text,
                        "audio_seconds": seconds,
                        "elapsed_seconds": share,
                        "source": "live",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    index += 1
                    print(
                        f"{index}/{len(pending)} {job['id']}: {seconds:.2f}s audio, "
                        f"{share:.2f}s (batch {len(group)})",
                        flush=True,
                    )
            now = time.perf_counter()
            if index == 1 or index % 25 == 0 or now - last_beat >= 600:
                heartbeat()
                last_beat = now
    have = load_done_ids(out)
    meta = {
        "model": args.model,
        "resolved_model": model_id,
        "family": args.family,
        "tag": args.tag,
        "clips_written": len(have),
        "model_load_seconds": load_seconds,
        "batch_size": batch_size,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)
    heartbeat()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        from common import LOG_DIR, append_jsonl

        append_jsonl(
            LOG_DIR / "failures.jsonl",
            {"step": "transcribe_manifest", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
