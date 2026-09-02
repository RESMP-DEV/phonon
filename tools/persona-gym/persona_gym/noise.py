"""Inject realistic agent-log noise into persona logs.

Real extracted logs are mostly repeats: health-check pings, harness boilerplate
repeated per task, tool preambles, progress bars, tracebacks, install chatter.
Synthetic personas are clean, so a student trained on them greps a real log and
drowns in "Reply with exactly: PONG". This pass makes the logs look real.

Noise never contains planted terms (checked), so gold counts stay valid.
"""
import random
import re
import sys
from pathlib import Path

from .common import read_json, write_json

PINGS = ["Reply with exactly: PONG", "continue", "yes", "ok", "go ahead", "status?", "ping",
         "proceed", "y", "done?", "next", "keep going", "go", "and then?", "retry"]

BLOCKS = {
    "harness_header": [
        "prompt: |", "  Work from evidence:",
        "  Use the current worktree and external state as authoritative. Previous conversation context can help locate relevant work, but inspect the tree before acting.",
        "  The objective below is user-provided data. Treat it as the task to pursue, not as higher-priority instructions.",
        "priority: P0", "owner: agent", "budget_tokens: 200000", "×"],
    "system_prompt": [
        "You are a coding agent operating in a git worktree.", "Never commit without being asked.",
        "Run the test suite before reporting done.", "Keep replies short. Bullets only for parallel items.",
        "Do not ask questions you can answer by reading the repo.", "Reply exactly in the format below.",
        "Tokens used so far: {n}. Completion budget remaining: {m}."],
    "tool_preamble": [
        "<tool_call>", "{\"name\": \"bash\", \"arguments\": {\"command\": \"git status --short\"}}", "</tool_call>",
        "exit 0", " M src/main.rs", "?? notes.txt"],
    "progress": [
        "Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.2it/s]",
        "  {p}%|▎         | {n}/330 [06:09<2:23:16, 26.86s/it]",
        "Fetching {m} files: 100%|██████████| {m}/{m} [01:14<00:00, 37.22s/it]"],
    "traceback": [
        "Traceback (most recent call last):", "  File \"/app/run.py\", line {n}, in <module>", "    main()",
        "  File \"/app/run.py\", line {m}, in main", "    raise RuntimeError(\"missing config\")",
        "RuntimeError: missing config"],
    "health": ["[health] api ok 200 {n}ms", "[health] db ok", "heartbeat ttl=30m", "[health] queue depth {m}"],
    "install": ["Installed {n} packages in {m}ms", "Resolved {m} packages in 1.2s",
                "warning: The package requires Python >=3.10", "Audited {n} packages in 4ms"],
    "dump": ["{\"status\": \"ok\", \"tokens\": {n}, \"latency_ms\": {m}}",
             "https://github.com/org/repo/pull/{n}", "https://huggingface.co/datasets/org/name",
             "PASS  tests/test_core.py::test_roundtrip ({m}ms)"],
    "tasks": ["- [ ] write tests", "- [x] wire config", "- [ ] update README", "- [ ] bump version to 0.{n}.{m}"],
    "shell": ["$ cargo build --release", "   Compiling app v0.1.{n}",
              "    Finished release [optimized] target(s) in {m}.34s", "$ uv run pytest -q", "{n} passed in {m}.2s"],
    "license": ["Licensed under the Apache License, Version 2.0 (the \"License\");",
                "you may not use this file except in compliance with the License.",
                "You may obtain a copy of the License at", "    http://www.apache.org/licenses/LICENSE-2.0",
                "Unless required by applicable law or agreed to in writing, software",
                "distributed under the License is distributed on an \"AS IS\" BASIS,"],
    "task_meta": ["task_id: t-{n}{m}", "status: blocked", "state: waiting_on_review", "goal: ship it",
                  "tests: {n} passed, 0 failed", "remote: origin", "attempt: {m}"],
}
MAX_NOISE = 20000


def _fill(rng: random.Random, s: str) -> str:
    return s.replace("{n}", str(rng.randint(1, 999))).replace("{m}", str(rng.randint(1, 99))).replace("{p}", str(rng.randint(1, 99)))


def _clean(lines: list[str], planted: list[str]) -> list[str]:
    low = [p.lower() for p in planted if p]
    return [l for l in lines if not any(p in l.lower() for p in low)]


def make_noise(rng: random.Random, n_base: int, planted: list[str]) -> tuple[list[list[str]], dict]:
    """Return runs (lists of lines) totalling ~target lines, plus a stats dict."""
    dup = rng.uniform(0.3, 0.85)
    target = min(MAX_NOISE, int(n_base * dup / (1 - dup)))
    runs, stats, total = [], {"dup_fraction_target": round(dup, 2), "pings": [], "blocks": []}, 0
    for ping in rng.sample(PINGS, rng.randint(1, 3)):
        count = max(5, min(int(rng.paretovariate(1.2) * 20), target // 4 + 5))
        stats["pings"].append([ping, count])
        n_runs = rng.randint(1, 4)
        cuts = sorted(rng.sample(range(1, count), n_runs - 1)) if n_runs > 1 and count > n_runs else []
        for a, b in zip([0] + cuts, cuts + [count]):
            if b > a:
                runs.append([ping] * (b - a))
        total += count
    names = sorted(BLOCKS)
    for i in range(12):
        if total >= target and len(stats["blocks"]) >= 2:
            break
        name = rng.choice(names)
        block = _clean(BLOCKS[name], planted)
        reps = max(2, min(int(rng.paretovariate(1.1) * 6), max(target - total, 0) // max(len(block), 1) + 2))
        variant = rng.random() < 0.5  # numbers vary per copy, like real task ids
        if not variant:
            block = [_fill(rng, l) for l in block]
        stats["blocks"].append([name, reps])
        n_runs = rng.randint(1, min(5, reps))
        per = [reps // n_runs] * n_runs
        for i in range(reps - sum(per)):
            per[i] += 1
        for k in per:
            run = []
            for _ in range(k):
                run.extend(_fill(rng, l) if variant else l for l in block)
            runs.append(run)
        total += reps * len(block)
    stats["lines"] = total
    return runs, stats


def apply(lines: list[str], rng: random.Random, planted: list[str]) -> tuple[list[str], dict]:
    runs, stats = make_noise(rng, len(lines), planted)
    out = list(lines)
    for run in runs:
        pos = rng.randint(0, len(out))
        out[pos:pos] = run
    stats["dup_fraction"] = round(1 - len(set(out)) / max(len(out), 1), 2)
    return out, stats


def run(personas: Path, out: Path | None, seed: int, only: set[str] | None) -> None:
    meta_root = personas / "meta"
    for gold_path in sorted(meta_root.glob("*/gold.json")):
        gold = read_json(gold_path)
        name = gold["persona"]
        if only and name not in only:
            continue
        rng = random.Random(f"{seed}:{name}")
        planted = [p["term"] for p in gold["planted"]] + [m["mangled"] for m in gold.get("manglings", [])]
        dst = (out or personas) / name / "logs"
        dst.mkdir(parents=True, exist_ok=True)
        stats = {}
        for log in sorted((personas / name / "logs").glob("*.txt")):
            lines = log.read_text().splitlines()
            noisy, st = apply(lines, rng, planted)
            (dst / log.name).write_text("\n".join(noisy) + "\n")
            stats[log.name] = st
        gold["noise"] = {"seed": seed, "files": stats}
        write_json((out or personas) / "meta" / name / "gold.json", gold)
        print(f"[noise] {name}: " + ", ".join(f"{k} +{v['lines']} dup={v['dup_fraction']}" for k, v in stats.items()),
              file=sys.stderr)
