"""TTS (Kokoro-82M, generic voices) then Parakeet v2 ASR on a 1500-row subset.

Reuse helpers from research/corrector_v0 without importing synthesize_tts.py
(that module clears CUDA_VISIBLE_DEVICES at import).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

CORRECTED = Path("/home/user/phonon/research/corrector_v0")
sys.path.insert(0, str(CORRECTED))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_ops import (  # noqa: E402
    KOKORO_SR,
    TARGET_SR,
    degrade_audio,
    peak_normalize,
    resample_to,
    save_wav,
)
from common import stable_hash  # noqa: E402
from paths import (  # noqa: E402
    CLEAN_PATH,
    DATA_ROOT,
    HF_HOME,
    SYSTEM_PROMPT,
    TTS_MANIFEST,
    TTS_PARAKEET,
    TTS_WAV_DIR,
)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

VOICES = ("af_heart", "am_adam", "bf_emma")


def lang_code(voice: str) -> str:
    if voice.startswith("b"):
        return "b"
    return "a"


def build_pipelines(voices: list[str], device: str = "cpu"):
    from kokoro import KPipeline

    codes = sorted({lang_code(v) for v in voices})
    return {code: KPipeline(lang_code=code, device=device, repo_id="hexgrad/Kokoro-82M") for code in codes}


def synthesize_text(pipeline, text: str, voice: str):
    import numpy as np

    pieces = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not pieces:
        return None
    return np.concatenate(pieces)


def chat_messages(raw: str, target: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": target},
    ]


def select_rows(n: int, seed: int) -> list[dict]:
    rows = []
    with CLEAN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            words = int(row.get("n_words") or len((row.get("text") or "").split()))
            if 18 <= words <= 70:
                rows.append(row)
    if len(rows) < n:
        extra = []
        with CLEAN_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                extra.append(json.loads(line))
        have = {r.get("id") for r in rows}
        for row in extra:
            if row.get("id") not in have:
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _prepare_jobs(rows: list[dict], done_ids: set, existing: set) -> tuple[list[dict], list[dict], int]:
    voices = list(VOICES)
    todo = []
    skipped_records = []
    skipped = 0
    for i, row in enumerate(rows):
        voice = voices[i % len(voices)]
        target = (row.get("text") or "").strip()
        uid = f"{row['id']}__{voice}"
        wav_path = TTS_WAV_DIR / f"{uid}.wav"
        degraded = stable_hash(f"{uid}:deg") % 2 == 0
        record = {
            "id": uid,
            "source_id": row["id"],
            "voice": voice,
            "degraded": degraded,
            "target": target,
            "terms": row.get("terms") or [],
            "wav": str(wav_path),
            "split": "tts_synth",
        }
        if uid in done_ids:
            skipped += 1
            continue
        if wav_path.name in existing and wav_path.exists():
            skipped_records.append(record)
            skipped += 1
            continue
        todo.append(record)
    return todo, skipped_records, skipped


def _synth_record(record: dict, pipelines) -> dict | None:
    import numpy as np

    uid = record["id"]
    voice = record["voice"]
    target = record["target"]
    wav_path = Path(record["wav"])
    try:
        wave24 = synthesize_text(pipelines[lang_code(voice)], target, voice)
    except Exception as exc:
        print(f"tts fail {uid}: {type(exc).__name__}: {exc}", flush=True)
        return None
    if wave24 is None or len(wave24) < 800:
        print(f"tts empty {uid}", flush=True)
        return None
    wave = resample_to(peak_normalize(wave24), KOKORO_SR, TARGET_SR)
    if record.get("degraded"):
        local = np.random.default_rng(stable_hash(uid) % (2**32))
        wave = degrade_audio(wave, TARGET_SR, local)
    save_wav(wav_path, wave, TARGET_SR)
    out = dict(record)
    out["wav_seconds"] = len(wave) / float(TARGET_SR)
    return out


def _tts_shard(payload: tuple) -> tuple[str, int, int]:
    shard_id, jobs, device, shard_path = payload
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(2)
    voices = sorted({j["voice"] for j in jobs})
    pipelines = build_pipelines(voices, device=device)
    written = 0
    failed = 0
    t0 = time.perf_counter()
    with open(shard_path, "w", encoding="utf-8") as handle:
        for i, record in enumerate(jobs, start=1):
            out = _synth_record(record, pipelines)
            if out is None:
                failed += 1
                continue
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"tts shard={shard_id} written={written}/{len(jobs)} failed={failed} "
                    f"elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
    print(f"tts shard={shard_id} done written={written} failed={failed}", flush=True)
    return shard_path, written, failed


def stage_tts(args: argparse.Namespace) -> int:
    import torch

    device = args.device
    if device == "auto":
        device = "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(8)
    TTS_WAV_DIR.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args.n, args.seed)
    voices = list(VOICES)
    print(f"tts selected={len(rows)} voices={voices} device={device} workers={args.workers}", flush=True)
    existing = {p.name for p in TTS_WAV_DIR.glob("*.wav")}
    t0 = time.perf_counter()
    mode = "a" if TTS_MANIFEST.exists() and args.resume else "w"
    done_ids = set()
    if mode == "a":
        with TTS_MANIFEST.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(json.loads(line).get("id"))
    todo, skipped_records, skipped = _prepare_jobs(rows, done_ids, existing)
    with TTS_MANIFEST.open(mode, encoding="utf-8") as handle:
        for record in skipped_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    written = 0
    workers = max(1, int(args.workers))
    if workers == 1 or device != "cpu" or len(todo) < 8:
        pipelines = build_pipelines(voices, device=device)
        with TTS_MANIFEST.open("a", encoding="utf-8") as handle:
            for i, record in enumerate(todo):
                out = _synth_record(record, pipelines)
                if out is None:
                    continue
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                if written % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"tts written={written} skipped={skipped} elapsed_min={elapsed / 60:.1f}",
                        flush=True,
                    )
    else:
        import multiprocessing as mp

        shards = [[] for _ in range(workers)]
        for i, record in enumerate(todo):
            shards[i % workers].append(record)
        shard_dir = DATA_ROOT / "tmp"
        shard_dir.mkdir(parents=True, exist_ok=True)
        payloads = []
        for i, jobs in enumerate(shards):
            if not jobs:
                continue
            payloads.append((i, jobs, device, str(shard_dir / f"tts_shard_{i}.jsonl")))
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(_tts_shard, payloads)
        with TTS_MANIFEST.open("a", encoding="utf-8") as handle:
            for shard_path, n_ok, _n_fail in results:
                written += n_ok
                with open(shard_path, encoding="utf-8") as src:
                    for line in src:
                        if line.strip():
                            handle.write(line if line.endswith("\n") else line + "\n")
    print(
        f"tts done written={written} skipped={skipped} wall_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return 0


def load_parakeet_local():
    """Load Parakeet v2 from the local HF snapshot. from_pretrained hits the hub even when cached."""
    import nemo.collections.asr as nemo_asr
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Parakeet transcription")
    src = Path("/home/user/phonon/src")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from phonon.eval import _trusted_local_nemo_connector

    name = "nvidia/parakeet-tdt-0.6b-v2"
    cache = Path(os.environ.get("HF_HOME", "/data/hf")) / "hub"
    folder = cache / ("models--" + name.replace("/", "--"))
    main = folder / "refs" / "main"
    snap = None
    if main.exists():
        candidate = folder / "snapshots" / main.read_text().strip()
        if candidate.exists():
            snap = candidate
    if snap is None:
        snaps = sorted((folder / "snapshots").glob("*")) if (folder / "snapshots").exists() else []
        snap = snaps[-1] if snaps else None
    if snap is None:
        raise FileNotFoundError(f"no local snapshot for {name} under {folder}")
    nemo_files = list(snap.glob("*.nemo"))
    if not nemo_files:
        raise FileNotFoundError(f"no .nemo in {snap}")
    model = nemo_asr.models.ASRModel.restore_from(
        str(nemo_files[0]),
        save_restore_connector=_trusted_local_nemo_connector(),
    )
    return model.cuda().eval()


def stage_asr(args: argparse.Namespace) -> int:
    from transcribe_parakeet import transcribe_paths, transcribe_wav

    jobs = []
    with TTS_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == "_meta":
                continue
            if not Path(row["wav"]).exists():
                continue
            jobs.append(row)
    done = set()
    if TTS_PARAKEET.exists():
        with TTS_PARAKEET.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line).get("id"))
    remaining = [j for j in jobs if j["id"] not in done]
    print(f"parakeet jobs={len(jobs)} done={len(done)} remaining={len(remaining)}", flush=True)
    if not remaining:
        return 0
    import torch

    torch.set_num_threads(4)
    model = load_parakeet_local()
    cache_dir = DATA_ROOT / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    bs = max(1, int(getattr(args, "asr_batch_size", 8)))
    with TTS_PARAKEET.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(remaining), bs):
            batch = remaining[start : start + bs]
            clip_t = time.perf_counter()
            try:
                texts = transcribe_paths(model, [j["wav"] for j in batch])
            except Exception as exc:
                print(f"asr batch fail {type(exc).__name__}: {exc}; falling back per-clip", flush=True)
                texts = []
                for job in batch:
                    texts.append(transcribe_wav(model, Path(job["wav"]), cache_dir, batch_size=4))
            elapsed = time.perf_counter() - clip_t
            per = elapsed / max(len(batch), 1)
            for job, text in zip(batch, texts, strict=True):
                rec = {
                    "id": job["id"],
                    "source_id": job.get("source_id"),
                    "voice": job.get("voice"),
                    "degraded": job.get("degraded"),
                    "parakeet_raw": text,
                    "target": job.get("target"),
                    "terms": job.get("terms") or [],
                    "wav": job.get("wav"),
                    "elapsed_seconds": per,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            n_done = start + len(batch)
            if start == 0 or n_done % 32 == 0 or n_done >= len(remaining):
                print(
                    f"asr {n_done}/{len(remaining)} batch={len(batch)} {per:.2f}s/clip "
                    f"elapsed_min={(time.perf_counter() - t0) / 60:.1f}",
                    flush=True,
                )
    print(f"asr done wall_s={time.perf_counter() - t0:.1f}", flush=True)
    return 0


def stage_pairs(args: argparse.Namespace) -> int:
    from common import fair_norm, pair_wer

    out_path = DATA_ROOT / "tts_pairs.jsonl"
    n = 0
    dropped = 0
    with TTS_PARAKEET.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = (row.get("parakeet_raw") or "").strip()
            target = (row.get("target") or "").strip()
            if not raw or not target:
                dropped += 1
                continue
            try:
                wer = pair_wer(target, raw, fair_norm)
            except Exception:
                dropped += 1
                continue
            if wer > 0.6:
                dropped += 1
                continue
            rec = {
                "id": f"synth_tts_{row['id']}",
                "source": f"synth_tts:{row.get('voice')}",
                "input": raw,
                "target": target,
                "messages": chat_messages(raw, target),
                "route": "tts_asr",
                "terms": row.get("terms") or [],
                "error_classes_applied": ["TTS_ASR"],
                "voice": row.get("voice"),
                "degraded": row.get("degraded"),
                "wer": wer,
            }
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"tts pairs kept={n} dropped={dropped} -> {out_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tts", "asr", "pairs"), required=True)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--asr-batch-size", type=int, default=8)
    args = parser.parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "tts":
        return stage_tts(args)
    if args.stage == "asr":
        return stage_asr(args)
    return stage_pairs(args)


if __name__ == "__main__":
    raise SystemExit(main())
