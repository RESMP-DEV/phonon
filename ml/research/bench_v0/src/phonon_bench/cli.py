"""phonon-bench: one benchmark for the Phonon refiner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .paths import BENCH, HF

LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"

os.environ.setdefault("HF_HOME", str(HF))
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="phonon-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    r = sub.add_parser("run", help="one full benchmark run")
    r.add_argument("--backend", default="hf", choices=("hf", "mlx", "llama", "vllm"))
    r.add_argument("--label", default="", help="name for this run (default: adapter dir name)")
    r.add_argument("--base", default=LFM12, help="HF base model id or a merged directory")
    r.add_argument("--adapter", default=None, help="PEFT adapter directory")
    r.add_argument("--model-dir", default=None, help="mlx: quantized/bf16 model directory")
    r.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama: server URL")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--attn", default="sdpa")
    r.add_argument("--max-model-len", type=int, default=4096, help="vllm")
    r.add_argument("--gpu-frac", type=float, default=0.85, help="vllm")
    r.add_argument("--max-lora-rank", type=int, default=64, help="vllm")
    r.add_argument("--sets", default="", help="comma list; default all six")
    r.add_argument("--condition", default="retrieved", choices=("retrieved", "oracle", "none"))
    r.add_argument("--max-batch", type=int, default=96)
    r.add_argument("--real-max-batch", type=int, default=0,
                   help="batch for the real-dictation sets (0 = --max-batch)")
    r.add_argument("--limit", type=int, default=0, help="rows per set (0 = all)")
    r.add_argument("--max-new-tokens", type=int, default=160, help="hard cap, term sets")
    r.add_argument("--real-max-new-tokens", type=int, default=2048, help="hard cap, real sets")
    r.add_argument("--retrieval", action="store_true", default=True)
    r.add_argument("--no-retrieval", dest="retrieval", action="store_false")
    r.add_argument("--lexicon", default="lexicon_big.jsonl")
    r.add_argument("--retrieval-limit", type=int, default=0)
    r.add_argument("--retrieval-workers", type=int, default=-1,
                   help="rapidfuzz cdist threads in stage 1 (-1 = all)")
    r.add_argument("--retrieval-procs", type=int, default=0,
                   help="worker processes for the two-stage rerank (0 = auto, 1 = in-process)")
    r.add_argument("--two-stage", action="store_true", default=True)
    r.add_argument("--single-stage", dest="two_stage", action="store_false")
    r.add_argument("--numerics", action="store_true", default=True)
    r.add_argument("--no-numerics", dest="numerics", action="store_false")
    r.add_argument("--numerics-limit", type=int, default=0)
    r.add_argument("--numerics-set", default="numerics_500",
                   help="set the teacher-forced numerics and greedy agreement run on")
    r.add_argument("--build-reference", action="store_true",
                   help="(re)build the bf16 numerics reference from this run")
    r.add_argument("--ref-key", default="", help="override the reference cache key")
    r.add_argument("--ref-dir", default=str(BENCH / "ref"))
    r.add_argument("--hidden-rows", type=int, default=100)
    r.add_argument("--latency", action="store_true", default=True)
    r.add_argument("--no-latency", dest="latency", action="store_false")
    r.add_argument("--latency-rows", type=int, default=20)
    r.add_argument("--sets-dir", default=str(BENCH / "sets"))
    r.add_argument("--runs-dir", default=str(BENCH / "runs"))
    r.add_argument("--bench-jsonl", default=str(BENCH / "bench.jsonl"))

    # ---- fixtures ----
    f = sub.add_parser("fixtures", help="build or verify the frozen fixtures")
    f.add_argument("action", choices=("build", "verify", "list"))
    f.add_argument("--sets-dir", default=str(BENCH / "sets"))
    f.add_argument("--force", action="store_true")

    # ---- compare / show ----
    c = sub.add_parser("compare", help="deltas between two runs, against the noise floors")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--bench-jsonl", default=str(BENCH / "bench.jsonl"))

    s = sub.add_parser("show", help="leaderboard from bench.jsonl")
    s.add_argument("--bench-jsonl", default=str(BENCH / "bench.jsonl"))
    s.add_argument("--limit", type=int, default=40)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        from .runner import run
        return run(args)
    if args.cmd == "fixtures":
        import json

        from . import fixtures as FX
        d = Path(args.sets_dir)
        if args.action == "build":
            man = FX.build(d, force=args.force)
            print(json.dumps({k: {"rows": v["rows"], "sha256": v["sha256"][:16]}
                              for k, v in man["sets"].items()}, indent=1))
            print(json.dumps({k: v["terms"] for k, v in man["lexicons"].items()}, indent=1))
            return 0
        if args.action == "verify":
            bad = FX.verify(d)
            print("OK" if not bad else "CHANGED:\n" + "\n".join(bad))
            return 0 if not bad else 1
        print(json.dumps(FX.load_manifest(d), indent=1))
        return 0
    if args.cmd == "compare":
        from .report import compare
        print(compare(args.a, args.b, Path(args.bench_jsonl)))
        return 0
    if args.cmd == "show":
        from .report import show
        print(show(Path(args.bench_jsonl), args.limit))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
