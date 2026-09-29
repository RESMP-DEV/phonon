"""Parakeet TDT 0.6b v2 n-best hypotheses (NeMo beam / maes, return_best_hypothesis=False).

Reads a jsonl manifest (id + wav, or id + --clips-dir), writes one row per clip:
  {"id", "term", "best", "alts": [...], "scores": [...]}
`best` is the beam top hypothesis; `alts` are the remaining distinct hypotheses (up to --max-alts).
Resumable: existing out ids are skipped.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
sys.path.insert(0, "/home/user/phonon/research/synth_v0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--strategy", default="maes")
    ap.add_argument("--max-alts", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop cleanly after N minutes")
    args = ap.parse_args()

    from tts_asr import load_parakeet_local
    import torch
    from omegaconf import open_dict

    jobs = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") in (None, "_meta"):
            continue
        wav = r.get("wav") or (str(args.clips_dir / (r["id"] + ".wav")) if args.clips_dir else None)
        if not wav:
            continue
        jobs.append({"id": r["id"], "wav": wav, "term": r.get("term")})
    if args.shuffle_seed:
        import random
        random.Random(args.shuffle_seed).shuffle(jobs)
    if args.limit:
        jobs = jobs[: args.limit]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    remaining = [j for j in jobs if j["id"] not in done and Path(j["wav"]).exists()]
    print(f"nbest jobs={len(jobs)} done={len(done)} remaining={len(remaining)} "
          f"strategy={args.strategy} beam={args.beam} bs={args.batch_size}", flush=True)
    if not remaining:
        return 0

    torch.set_num_threads(8)
    model = load_parakeet_local()
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = args.strategy
        cfg.beam.beam_size = args.beam
        cfg.beam.return_best_hypothesis = False
    model.change_decoding_strategy(cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    stop = False
    with args.out.open("a", encoding="utf-8") as h, torch.inference_mode():
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start : start + args.batch_size]
            out = model.transcribe([j["wav"] for j in batch], batch_size=len(batch),
                                   return_hypotheses=True, num_workers=0, verbose=False)
            if isinstance(out, tuple):
                out = out[0]
            for job, item in zip(batch, out):
                hyps = getattr(item, "n_best_hypotheses", None)
                if hyps is None:
                    hyps = item if isinstance(item, list) else [item]
                texts, scores, seen = [], [], set()
                for hy in hyps:
                    t = str(getattr(hy, "text", hy)).strip()
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    texts.append(t)
                    s = getattr(hy, "score", None)
                    scores.append(float(s) if s is not None else None)
                    if len(texts) > args.max_alts:
                        break
                h.write(json.dumps({"id": job["id"], "term": job["term"],
                                    "best": texts[0] if texts else "",
                                    "alts": texts[1 : args.max_alts + 1],
                                    "scores": scores}, ensure_ascii=False) + "\n")
            h.flush()
            n += len(batch)
            el = time.perf_counter() - t0
            if (start // args.batch_size) % 5 == 0 or n >= len(remaining):
                print(f"nbest {n}/{len(remaining)} {n/max(el,1e-9):.2f} clips/s "
                      f"elapsed_min={el/60:.1f}", flush=True)
            if args.minutes and el > args.minutes * 60:
                stop = True
                break
    el = time.perf_counter() - t0
    print(f"NBEST done clips={n} wall_s={el:.1f} {n/max(el,1e-9):.2f} clips/s "
          f"stopped_early={stop} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
