"""Kokoro-82M TTS on GPU for a sentences jsonl: 3 voices per sentence, 16 kHz, half degraded."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HF_HOME = "/data/hf"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HOME + "/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME + "/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/user/phonon/research/corrector_v0")

VOICES = ("af_heart", "am_adam", "bf_emma")


def lang_code(voice: str) -> str:
    return "b" if voice.startswith("b") else "a"


def _shard(payload):
    shard_id, jobs, shard_path, clips_dir = payload
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", HF_HOME + "/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME + "/hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, "/home/user/phonon/research/corrector_v0")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from kokoro import KPipeline
    from audio_ops import KOKORO_SR, TARGET_SR, degrade_audio, peak_normalize, resample_to, save_wav
    from common import stable_hash

    codes = sorted({lang_code(j["voice"]) for j in jobs})
    pipes = {c: KPipeline(lang_code=c, device="cuda", repo_id="hexgrad/Kokoro-82M") for c in codes}
    written = failed = 0
    audio_s = 0.0
    t0 = time.perf_counter()
    with open(shard_path, "w", encoding="utf-8") as h:
        for job in jobs:
            uid = job["id"]
            wav_path = Path(clips_dir) / f"{uid}.wav"
            try:
                pieces = [np.asarray(a, dtype=np.float32).reshape(-1)
                          for _g, _p, a in pipes[lang_code(job["voice"])](job["sentence"], voice=job["voice"])]
                if not pieces:
                    raise RuntimeError("empty")
                wave24 = np.concatenate(pieces)
            except Exception as exc:
                print(f"tts fail {uid}: {type(exc).__name__}: {exc}", flush=True)
                failed += 1
                continue
            if len(wave24) < 800:
                failed += 1
                continue
            wave = resample_to(peak_normalize(wave24), KOKORO_SR, TARGET_SR)
            degraded = stable_hash(f"{uid}:deg") % 2 == 0
            if degraded:
                rng = np.random.default_rng(stable_hash(uid) % (2**32))
                wave = degrade_audio(wave, TARGET_SR, rng)
            save_wav(wav_path, wave, TARGET_SR)
            audio_s += len(wave) / float(TARGET_SR)
            rec = dict(job)
            rec["wav"] = str(wav_path)
            rec["degraded"] = degraded
            rec["wav_seconds"] = len(wave) / float(TARGET_SR)
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 250 == 0:
                el = time.perf_counter() - t0
                print(f"shard={shard_id} {written}/{len(jobs)} {written/max(el,1e-9):.2f} clips/s", flush=True)
    el = time.perf_counter() - t0
    print(f"shard={shard_id} done written={written} failed={failed} wall_s={el:.1f}", flush=True)
    return shard_path, written, failed, audio_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--voices", default="", help="comma list; default af_heart,am_adam,bf_emma")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.sentences.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    args.clips_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    voices = tuple(v for v in args.voices.split(",") if v) or VOICES
    jobs = []
    for row in rows:
        for voice in voices:
            jobs.append({
                "id": f"{row['id']}__{voice}",
                "sentence_id": row["id"],
                "term": row["term"],
                "kind": row.get("kind"),
                "voice": voice,
                "sentence": row["sentence"],
            })
    print(f"sentences={len(rows)} clips={len(jobs)} voices={voices} workers={args.workers}", flush=True)

    import multiprocessing as mp
    tmp = args.manifest.parent / "tts_shards"
    tmp.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    shards = [[] for _ in range(workers)]
    for i, j in enumerate(jobs):
        shards[i % workers].append(j)
    payloads = [(i, s, str(tmp / f"{args.manifest.stem}_shard_{i}.jsonl"), str(args.clips_dir))
                for i, s in enumerate(shards) if s]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        results = pool.map(_shard, payloads)
    wall = time.perf_counter() - t0
    total = failed = 0
    audio_s = 0.0
    with args.manifest.open("w", encoding="utf-8") as h:
        for path, n_ok, n_fail, a_s in results:
            total += n_ok
            failed += n_fail
            audio_s += a_s
            with open(path, encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        h.write(line)
    print(f"TTS done clips={total} failed={failed} wall_s={wall:.1f} "
          f"{total/max(wall,1e-9):.2f} clips/s audio_s={audio_s:.0f} rtf={wall/max(audio_s,1e-9):.4f} "
          f"-> {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
