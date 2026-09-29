"""phonon-research: one CLI for the flywheel (corpus, train, eval, queue)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="phonon-research", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="terms -> sentences -> TTS -> ASR -> training rows")
    c.add_argument("config", type=Path)
    c.add_argument("--only", default="", help="comma-separated subset of the config's stages")

    t = sub.add_parser("train", help="named LoRA run through research/corrector_v0/train_lora.py")
    t.add_argument("config", type=Path)
    t.add_argument("--gpu", default=None)
    t.add_argument("--result", type=Path, default=None)

    e = sub.add_parser("eval", help="hand an adapter to phonon-bench")
    e.add_argument("adapter")
    e.add_argument("--sets", default="")
    e.add_argument("--backend", default="hf")
    e.add_argument("--label", default=None)
    e.add_argument("--gpu", default=None)
    e.add_argument("--result", type=Path, default=None)

    ec = sub.add_parser("eval-config", help="eval from a yaml config")
    ec.add_argument("config", type=Path)
    ec.add_argument("--gpu", default=None)
    ec.add_argument("--result", type=Path, default=None)

    q = sub.add_parser("queue", help="run a jobs.yaml across both GPUs in tmux")
    q.add_argument("jobs", nargs="?", default=None)
    q.add_argument("--run", default=None)
    q.add_argument("--scheduler", action="store_true", help=argparse.SUPPRESS)
    q.add_argument("--foreground", action="store_true")

    qs = sub.add_parser("status", help="alias for `queue status`")
    qs.add_argument("--run", default=None)
    qs.add_argument("--jobs", default=None)

    r = sub.add_parser("registry", help="show the adapter / eval-set registries")
    r.add_argument("which", choices=("adapters", "sets"), default="adapters", nargs="?")

    args = ap.parse_args(argv)

    if args.cmd == "corpus":
        from .corpus.pipeline import run_corpus

        only = [s for s in args.only.split(",") if s]
        run_corpus(args.config, only or None)
        return 0

    if args.cmd == "train":
        from .train import run_train

        res = run_train(args.config, args.gpu, args.result)
        print(json.dumps(res, indent=2))
        return res["rc"]

    if args.cmd == "eval":
        from .evaluate import run_eval

        sets = [s for s in args.sets.split(",") if s]
        res = run_eval(args.adapter, sets or None, args.backend, args.label, args.gpu,
                       result_path=args.result)
        print(json.dumps(res, indent=2))
        return int(res.get("rc", 0))

    if args.cmd == "eval-config":
        from .evaluate import run_eval_config

        res = run_eval_config(args.config, args.gpu, args.result)
        print(json.dumps(res, indent=2))
        return int(res.get("rc", 0))

    if args.cmd == "queue":
        from . import queue as Q

        if args.jobs in (None, "status"):
            return Q.status(args.run, None)
        if args.scheduler:
            return Q.scheduler(args.jobs, args.run)
        return Q.submit(args.jobs, args.run, args.foreground)

    if args.cmd == "status":
        from . import queue as Q

        return Q.status(args.run, args.jobs)

    if args.cmd == "registry":
        from . import registry as R

        print(json.dumps(R.adapters() if args.which == "adapters" else R.sets(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
