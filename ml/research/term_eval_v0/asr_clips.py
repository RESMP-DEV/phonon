"""Parakeet TDT 0.6b v2 batched ASR over a TTS manifest. Repo venv (NeMo). GPU 1."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
sys.path.insert(0, "/home/user/phonon/research/synth_v0")
sys.path.insert(0, "/home/user/phonon/research/corrector_v0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from tts_asr import load_parakeet_local
    from transcribe_parakeet import transcribe_paths

    jobs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = [j for j in jobs if Path(j["wav"]).exists()]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    remaining = [j for j in jobs if j["id"] not in done]
    print(f"asr jobs={len(jobs)} done={len(done)} remaining={len(remaining)} bs={args.batch_size}", flush=True)
    if not remaining:
        return 0

    import torch
    torch.set_num_threads(8)
    model = load_parakeet_local()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    with args.out.open("a", encoding="utf-8") as h, torch.inference_mode():
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start : start + args.batch_size]
            texts = transcribe_paths(model, [j["wav"] for j in batch])
            for job, text in zip(batch, texts):
                h.write(json.dumps({
                    "id": job["id"],
                    "sentence_id": job.get("sentence_id"),
                    "term": job["term"],
                    "kind": job.get("kind"),
                    "voice": job["voice"],
                    "degraded": job.get("degraded"),
                    "reference": job["sentence"],
                    "hyp": text,
                    "wav_seconds": job.get("wav_seconds"),
                }, ensure_ascii=False) + "\n")
            h.flush()
            n += len(batch)
            el = time.perf_counter() - t0
            if (start // args.batch_size) % 10 == 0 or n >= len(remaining):
                print(f"asr {n}/{len(remaining)} {n/max(el,1e-9):.2f} clips/s elapsed_min={el/60:.1f}", flush=True)
    el = time.perf_counter() - t0
    print(f"ASR done clips={n} wall_s={el:.1f} {n/max(el,1e-9):.2f} clips/s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
