import argparse
import sys

from . import candidates, extract, rank, seed


def main(argv=None):
    p = argparse.ArgumentParser(prog="profile_miner", description="Phonon vocabulary miner (dev tool)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="stage 2: user-authored text per source")
    e.add_argument("--source", action="append", choices=list(extract.SOURCES))
    e.add_argument("--until", help="ISO timestamp; drop records at or after it (held-out cutoff)")
    s = sub.add_parser("seed", help="stage 3a: identity seeds")
    s.add_argument("--no-github", action="store_true")
    c = sub.add_parser("candidates", help="stage 3b: rule-based candidates")
    c.add_argument("--min-count", type=int, default=candidates.MIN_COUNT)
    o = sub.add_parser("oracle", help="stage 3c: TTS -> Parakeet oracle (needs parakeet-mlx)")
    o.add_argument("--workers", type=int, default=12)
    o.add_argument("--top", type=int, default=6000, help="oracle budget: top N by count and breadth")
    sub.add_parser("rank", help="write mined/candidates.json")
    g = sub.add_parser("gemma", help="optional stage 3d: local Gemma pass (needs mlx-lm)")
    g.add_argument("--minutes", type=float, default=20)
    g.add_argument("--top", type=int, default=400)
    j = sub.add_parser("judge", help="stage 4b: keep/drop LoRA judge over the live top-N (needs mlx-lm)")
    j.add_argument("--top", type=int, default=300)
    j.add_argument("--adapter", default=None, help="mlx_lm adapter dir (default $PHONON_JUDGE_ADAPTER)")
    j.add_argument("--model", default=None, help="mlx model id (default $PHONON_JUDGE_MODEL or judge.DEFAULT_MODEL)")
    j.add_argument("--batch", type=int, default=8)
    d = sub.add_parser("dictionary", help="stage 5: lexicon present in sources + live names, two tiers")
    d.add_argument("--live-top", type=int, default=100)
    d.add_argument("--oracle-cache", default=None)
    lx = sub.add_parser("lexicon", help="tech lexicon: build from public lists")
    lx.add_argument("action", choices=["build"])
    lx.add_argument("--cache", default=None, help="dir with fetched lists (lexicon/fetch.sh)")
    g.add_argument("--model", default=None, help="mlx model id (default: Phonon's polish model)")
    g.add_argument("--out", default="gemma_pass.json", help="output file name under mined/")
    sc = sub.add_parser("score", help="dev-only scoring against the held-out dictionary and corpus")
    sc.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    if a.cmd == "extract":
        extract.run(a.source, a.until)
    elif a.cmd == "seed":
        seed.run(github=not a.no_github)
    elif a.cmd == "candidates":
        candidates.run(min_count=a.min_count)
    elif a.cmd == "judge":
        from . import judge
        judge.run(top=a.top, adapter=a.adapter, model_id=a.model, batch=a.batch)
    elif a.cmd == "dictionary":
        from . import dictionary
        from pathlib import Path
        dictionary.run(live_top=a.live_top, oracle_cache=Path(a.oracle_cache) if a.oracle_cache else None)
    elif a.cmd == "lexicon":
        from . import lexicon
        from pathlib import Path
        lexicon.build(Path(a.cache) if a.cache else lexicon.LEX_DIR / "cache")
    elif a.cmd == "oracle":
        from . import oracle
        oracle.run_candidates(workers=a.workers, top=a.top)
    elif a.cmd == "rank":
        rank.run()
    elif a.cmd == "gemma":
        from . import gemma_pass
        gemma_pass.run(minutes=a.minutes, top=a.top, out_name=a.out,
                       **({"model_id": a.model} if a.model else {}))
    elif a.cmd == "score":
        from . import score
        score.run(workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
