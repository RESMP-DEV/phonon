"""Transcribe wavs with nvidia/parakeet-tdt-0.6b-v2 via the repo NeMo venv.

Loading and decode match scripts/option0/transcribe_gate.py (family=nemo):
ASRModel.from_pretrained, cuda().eval(), transcribe(..., return_hypotheses=True),
then str(getattr(item, "text", item)).strip(). Audio is 16 kHz mono in nominal
30 s chunks with a sub-second tail merge. Do not run this with --no-project;
use `cd /home/user/phonon && uv run python research/corrector_v0/transcribe_parakeet.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_ops import chunk_wav_paths, load_wav  # noqa: E402
from common import (  # noqa: E402
    DATA_ROOT,
    PARAKEET_JSONL,
    PARAKEET_MODEL,
    asr_jobs,
    ensure_data_dirs,
    heartbeat,
    load_done_ids,
    record_failure,
    set_hf_env,
)


def load_parakeet(name: str = PARAKEET_MODEL):
    import nemo.collections.asr as nemo_asr
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Parakeet transcription")
    model = nemo_asr.models.ASRModel.from_pretrained(name)
    return model.cuda().eval()


def transcribe_paths(model, paths: list[str]) -> list[str]:
    outputs = model.transcribe(
        paths,
        batch_size=len(paths),
        return_hypotheses=True,
        num_workers=0,
        verbose=False,
    )
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return [str(getattr(item, "text", item)).strip() for item in outputs]


def transcribe_wav(model, wav_path: Path, cache_dir: Path, batch_size: int) -> str:
    wave = load_wav(wav_path)
    paths = chunk_wav_paths(wave, cache_dir)
    pieces: list[str] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        pieces.extend(transcribe_paths(model, batch))
    return " ".join(p for p in pieces if p).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PARAKEET_JSONL)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--id-field", default="id")
    return parser.parse_args()


def load_jobs(args: argparse.Namespace) -> list[dict]:
    if args.manifest is None:
        jobs = asr_jobs()
    else:
        jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        jobs = [job for job in jobs if job.get("id") not in (None, "_meta") and job.get("wav")]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def main() -> int:
    args = parse_args()
    set_hf_env()
    ensure_data_dirs()
    jobs = load_jobs(args)
    missing = [job for job in jobs if not Path(job["wav"]).exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} wavs, first={missing[0]['wav']}")
    done = load_done_ids(args.out, args.id_field)
    remaining = [job for job in jobs if str(job[args.id_field]) not in done]
    print(
        f"parakeet jobs={len(jobs)} done={len(done)} remaining={len(remaining)} out={args.out}",
        flush=True,
    )
    if not remaining:
        return 0
    import torch

    torch.set_num_threads(4)
    cache_dir = DATA_ROOT / "chunks"
    model = load_parakeet()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, min(args.batch_size, 4))
    start = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for index, job in enumerate(remaining, start=1):
            clip_start = time.perf_counter()
            text = transcribe_wav(model, Path(job["wav"]), cache_dir, batch_size)
            record = {
                "id": job["id"],
                "parakeet_raw": text,
                "target": job.get("target") or "",
                "split": job.get("split") or "",
                "wispr_asr": job.get("wispr_asr") or "",
                "wav": job["wav"],
                "elapsed_seconds": time.perf_counter() - clip_start,
            }
            for extra in ("voice", "degraded", "source_id"):
                if extra in job:
                    record[extra] = job[extra]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index}/{len(remaining)} {job['id']}: {len(text.split())} words "
                f"{record['elapsed_seconds']:.2f}s",
                flush=True,
            )
            if index == 1 or index % 50 == 0:
                heartbeat()
    print(f"finished remaining={len(remaining)} wall={time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        record_failure("transcribe_parakeet", f"{type(exc).__name__}: {exc}", 1)
        raise
