"""Synthesize wispr_text_train targets with Kokoro-82M, then lightly degrade half.

CPU-only (CUDA_VISIBLE_DEVICES is cleared). Caps total wav seconds at ~12 hours.
Voices default to af_heart, am_adam, bf_emma. If a smoke timing projects more
than 6 wall hours, drop to two voices.

Wavs are written through save_wav: unique sibling .wav tmp in the same
directory, then os.replace onto the final path (soundfile cannot infer format
from a .wav.tmp suffix).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_ops import (  # noqa: E402
    KOKORO_SR,
    TARGET_SR,
    degrade_audio,
    estimate_seconds,
    peak_normalize,
    resample_to,
    save_wav,
)
from common import (  # noqa: E402
    DATA_ROOT,
    TRAIN_SPLITS,
    VOICES,
    ensure_data_dirs,
    load_wispr_pairs,
    record_failure,
    set_hf_env,
    stable_hash,
)

MANIFEST = DATA_ROOT / "tts_manifest.jsonl"


def select_rows(
    rows: list[dict],
    n_voices: int,
    cap_hours: float,
    seed: int,
    order: str = "length-asc",
) -> tuple[list[dict], float]:
    candidates = [
        row
        for row in rows
        if row.get("split") == "wispr_text_train" and (row.get("target") or "").strip()
    ]
    if order == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        candidates.sort(
            key=lambda row: (
                min(estimate_seconds(row["target"], row.get("duration")), 180.0),
                str(row.get("id") or ""),
            )
        )
    cap = cap_hours * 3600.0
    selected: list[dict] = []
    total = 0.0
    for row in candidates:
        seconds = min(estimate_seconds(row["target"], row.get("duration")), 180.0)
        extra = seconds * n_voices
        if selected and total + extra > cap:
            continue
        selected.append(row)
        total += extra
        if total >= cap:
            break
    return selected, total


def lang_code(voice: str) -> str:
    if voice.startswith("b"):
        return "b"
    return "a"


def build_pipelines(voices: list[str], device: str = "cpu"):
    from kokoro import KPipeline

    codes = sorted({lang_code(v) for v in voices})
    return {
        code: KPipeline(lang_code=code, device=device, repo_id="hexgrad/Kokoro-82M")
        for code in codes
    }


def synthesize_text(pipeline, text: str, voice: str):
    import numpy as np

    pieces = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not pieces:
        return None
    return np.concatenate(pieces)


def wav_id(source_id: str, voice: str, degraded: bool) -> str:
    tag = "deg" if degraded else "clean"
    return f"{source_id}__{voice}__{tag}"


def should_degrade(source_id: str, voice: str) -> bool:
    return stable_hash(f"{source_id}:{voice}:deg") % 2 == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", nargs="+", default=list(VOICES))
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Cap unique texts (smoke).")
    parser.add_argument("--wall-hours", type=float, default=6.0)
    parser.add_argument("--force-voices", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=("length-asc", "shuffle"),
        default="length-asc",
        help="Pick training texts by estimated duration (default) or shuffle.",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "tts_wavs")
    return parser.parse_args()


def smoke_rate(rows: list[dict], voices: list[str], n: int = 4) -> float:
    """Seconds of wall per second of estimated audio; 0 if smoke fails."""
    import torch

    torch.set_num_threads(4)
    pipelines = build_pipelines(voices, device="cpu")
    sample = rows[:n]
    t0 = time.perf_counter()
    audio_s = 0.0
    ok = 0
    for row in sample:
        voice = voices[ok % len(voices)]
        wave = synthesize_text(pipelines[lang_code(voice)], row["target"], voice)
        if wave is None or len(wave) < 800:
            continue
        audio_s += len(wave) / float(KOKORO_SR)
        ok += 1
    elapsed = time.perf_counter() - t0
    if audio_s <= 0:
        raise RuntimeError("kokoro smoke produced no audio")
    print(
        f"kokoro smoke n={ok} audio_s={audio_s:.1f} wall_s={elapsed:.1f} rtf={elapsed / audio_s:.3f}",
        flush=True,
    )
    del pipelines
    return elapsed / audio_s


def main() -> int:
    args = parse_args()
    set_hf_env()
    ensure_data_dirs()
    import numpy as np
    import torch

    torch.set_num_threads(4)
    rows = [
        r
        for r in load_wispr_pairs()
        if r.get("split") in TRAIN_SPLITS and r.get("split") == "wispr_text_train"
    ]
    voices = list(args.voices)
    selected, planned = select_rows(
        rows, len(voices), args.max_hours, args.seed, order=args.order
    )
    if args.limit:
        selected = selected[: args.limit]
        planned = sum(
            min(estimate_seconds(r["target"], r.get("duration")), 180.0) * len(voices)
            for r in selected
        )
    print(
        f"selected texts={len(selected)} voices={len(voices)} planned_hours={planned / 3600:.2f}",
        flush=True,
    )
    if not selected:
        raise RuntimeError("no text-train rows selected")
    rtf = smoke_rate(selected, voices)
    projected_wall = planned * rtf
    if args.force_voices:
        voices = voices[: args.force_voices]
    elif projected_wall > args.wall_hours * 3600 and len(voices) > 2:
        print(
            f"projected wall {projected_wall / 3600:.2f}h > {args.wall_hours}h; dropping to 2 voices",
            flush=True,
        )
        voices = voices[:2]
        selected, planned = select_rows(
            rows, len(voices), args.max_hours, args.seed, order=args.order
        )
        if args.limit:
            selected = selected[: args.limit]
        print(
            f"revised texts={len(selected)} voices={voices} planned_hours={planned / 3600:.2f}",
            flush=True,
        )
    pipelines = build_pipelines(voices, device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in args.out_dir.glob("*.wav")}
    t0 = time.perf_counter()
    written = 0
    skipped = 0
    deadline = t0 + args.wall_hours * 3600
    with MANIFEST.open("w", encoding="utf-8") as handle:
        meta = {
            "voices": voices,
            "planned_hours": planned / 3600,
            "rtf_smoke": rtf,
            "texts": len(selected),
        }
        handle.write(json.dumps({"id": "_meta", **meta}) + "\n")
        for row in selected:
            if time.perf_counter() > deadline:
                print("hit wall-hour cap; stopping synthesis", flush=True)
                break
            for voice in voices:
                degraded = should_degrade(row["id"], voice)
                uid = wav_id(row["id"], voice, degraded)
                wav_path = args.out_dir / f"{uid}.wav"
                record = {
                    "id": uid,
                    "source_id": row["id"],
                    "voice": voice,
                    "degraded": degraded,
                    "target": row["target"],
                    "split": "tts_synth",
                    "wav": str(wav_path),
                    "wispr_asr": "",
                }
                if wav_path.name in existing and wav_path.exists():
                    try:
                        import soundfile as sf

                        info = sf.info(str(wav_path))
                        record["wav_seconds"] = info.frames / float(info.samplerate)
                    except Exception:
                        pass
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue
                try:
                    wave24 = synthesize_text(
                        pipelines[lang_code(voice)], row["target"], voice
                    )
                except Exception as exc:
                    print(f"tts fail {row['id']} {voice}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if wave24 is None or len(wave24) < 800:
                    print(f"tts empty {row['id']} {voice}", flush=True)
                    continue
                wave = resample_to(peak_normalize(wave24), KOKORO_SR, TARGET_SR)
                if degraded:
                    local = np.random.default_rng(stable_hash(uid) % (2**32))
                    wave = degrade_audio(wave, TARGET_SR, local)
                save_wav(wav_path, wave, TARGET_SR)
                record["wav_seconds"] = len(wave) / float(TARGET_SR)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                if written % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"tts written={written} skipped_existing={skipped} elapsed_min={elapsed / 60:.1f}",
                        flush=True,
                    )
    print(
        f"tts done written={written} skipped_existing={skipped} wall_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    meta = {
        "voices": voices,
        "written": written,
        "skipped_existing": skipped,
        "planned_hours": planned / 3600,
        "rtf_smoke": rtf,
        "texts": len(selected),
        "limit": args.limit,
        "complete": True,
    }
    (DATA_ROOT / "tts_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.limit == 0:
        (DATA_ROOT / "tts_complete.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        record_failure("synthesize_tts", f"{type(exc).__name__}: {exc}", 1)
        raise
