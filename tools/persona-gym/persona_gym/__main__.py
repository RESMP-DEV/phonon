import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="persona_gym",
                                description="Synthetic user machines for the vocabulary miner (dev tool)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="manufacture N personas with planted gold vocab")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-oracle", action="store_true",
                   help="skip TTS->Parakeet manglings (no GPU / parakeet busy)")
    b.add_argument("--lines-min", type=int, default=300)
    b.add_argument("--lines-max", type=int, default=800)

    r = sub.add_parser("rollout", help="run a teacher model agentically over each persona")
    r.add_argument("--personas", required=True)
    r.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-per", type=int, default=1)
    r.add_argument("--max-turns", type=int, default=24)
    r.add_argument("--no-think", action="store_true",
                   help="send chat_template_kwargs enable_thinking=false (SFT'd Qwen)")

    e = sub.add_parser("export", help="write SFT jsonl from graded rollouts")
    e.add_argument("--personas", required=True)
    e.add_argument("--rollouts", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--strict", help="json list of [persona, roll] to keep")

    g = sub.add_parser("grade", help="score rollouts against gold.json")
    g.add_argument("--personas", required=True)
    g.add_argument("--rollouts", required=True)

    z = sub.add_parser("noise", help="inject realistic repeated-boilerplate noise into persona logs")
    z.add_argument("--personas", required=True)
    z.add_argument("--out", help="write logs+gold copies here instead of in place (no repos)")
    z.add_argument("--seed", type=int, default=0)
    z.add_argument("--only", help="comma-separated persona names")

    a = p.parse_args(argv)
    if a.cmd == "build":
        from . import build
        build.build(a.n, Path(a.out), seed=a.seed, no_oracle=a.no_oracle,
                    lines_min=a.lines_min, lines_max=a.lines_max)
    elif a.cmd == "rollout":
        from . import rollout
        rollout.run(Path(a.personas), a.endpoint, a.model, Path(a.out),
                    n_per=a.n_per, max_turns=a.max_turns, no_think=a.no_think)
    elif a.cmd == "export":
        from . import sft
        sft.export(Path(a.personas), Path(a.rollouts), Path(a.out),
                   Path(a.strict) if a.strict else None)
    elif a.cmd == "noise":
        from . import noise
        noise.run(Path(a.personas), Path(a.out) if a.out else None, a.seed,
                  set(a.only.split(",")) if a.only else None)
    elif a.cmd == "grade":
        from . import grade
        grade.run(Path(a.personas), Path(a.rollouts))


if __name__ == "__main__":
    sys.exit(main())
