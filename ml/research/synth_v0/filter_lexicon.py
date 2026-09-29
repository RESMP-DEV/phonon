"""Rank a speakable lexicon and rebuild markdown-heavy topic seeds."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from paths import LEXICON_PATH, REPO_ROOTS, TOPICS_PATH

RANKED = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
HEX_ADDR = re.compile(r"_80[0-9A-Fa-f]{5,}|_8[0-9A-Fa-f]{7,}|0x[0-9A-Fa-f]{4,}")
RUN_ID = re.compile(r"^\d{8}_\d{6}")
HEXISH = re.compile(r"^[0-9a-f]{8,}$", re.I)
CAMEL_OK = re.compile(r"[a-z][A-Z]|_")
QUOTAS = {
    "/home/user/phonon": 7000,
    "/home/user/cuda": 4000,
    "/home/user/benchmarks": 1500,
    "/home/user/experiments": 800,
    "/home/user/box-install": 300,
    "/home/user/dev": 3500,
}

KIND_BONUS = {
    "person": 50,
    "machine": 40,
    "cli_tool": 25,
    "product": 20,
    "library": 15,
    "acronym": 12,
    "flag": 8,
    "module": 5,
    "file": 3,
    "identifier": 0,
}


def root_of(path: str) -> str:
    for root in QUOTAS:
        if path.startswith(root):
            return root
    return "/other"


def speakable(term: str) -> bool:
    if not term or len(term) < 2 or len(term) > 40:
        return False
    if RUN_ID.match(term) or HEXISH.match(term) or HEX_ADDR.search(term):
        return False
    if term.endswith((".yaml", ".yml", ".json")) and term.count("_") >= 3:
        return False
    if term.count("_") >= 4:
        return False
    letters = re.sub(r"[^A-Za-z]", "", term)
    if len(letters) < 2:
        return False
    return True


def score_row(row: dict) -> float:
    s = float(row.get("mangle_score") or 0) + KIND_BONUS.get(row.get("kind"), 0)
    s += min(8.0, (row.get("count") or 1) ** 0.25)
    return s


SKIP_TOPIC_NAMES = {
    "license",
    "license.md",
    "license.txt",
    "copying",
    "copying.md",
    "notice",
    "notice.md",
    "code_of_conduct.md",
    "security.md",
}
SKIP_TOPIC_RE = re.compile(
    r"copyright|licensed under|permitted to copy|all rights reserved|spdx-|"
    r"\bdco\b|apache license|gnu general public|mit license|creative commons|"
    r"you hereby grant|contributor license",
    re.I,
)


def rebuild_topics() -> int:
    from walk_repos import DOC_NAMES, SENT_SPLIT, LICENSE_LINE

    sentences = []
    seen = set()
    md_names = DOC_NAMES | {"readme.md", "agents.md", "devlog.md", "spec.md", "goal.md", "product.md"}
    skip_dirs = {".git", "node_modules", ".venv", "target", "datasets", "runs", "vendor", "third_party"}
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(p.lower() in skip_dirs for p in path.parts):
                continue
            name = path.name.lower()
            if name in SKIP_TOPIC_NAMES or name.startswith("license"):
                continue
            if path.suffix.lower() not in {".md", ".rst"} and name not in md_names:
                continue
            try:
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            preferred = name in md_names or name.startswith("readme")
            for raw in SENT_SPLIT.split(text.replace("\n", " ")):
                s = " ".join(raw.split()).lstrip("#> -")
                if not (50 <= len(s) <= 320):
                    continue
                if LICENSE_LINE.search(s) or SKIP_TOPIC_RE.search(s) or s.count("`") > 6:
                    continue
                if s.count("http") >= 2:
                    continue
                words = s.split()
                if len(words) < 8:
                    continue
                key = s.lower()[:160]
                if key in seen:
                    continue
                seen.add(key)
                sentences.append(
                    {
                        "text": s,
                        "source_file": str(path),
                        "kind": "markdown",
                        "preferred": preferred,
                    }
                )
    by_root = defaultdict(list)
    for t in sentences:
        by_root[root_of(t["source_file"])].append(t)
    out = []
    for root, rows in by_root.items():
        rows.sort(key=lambda r: (0 if r.get("preferred") else 1, -len(r["text"])))
        cap = 1600 if root in {"/home/user/phonon", "/home/user/cuda", "/home/user/dev"} else 800
        out.extend(rows[:cap])
    out = out[:8000]
    for i, t in enumerate(out):
        t["id"] = f"topic_{i:05d}"
        t.pop("preferred", None)
    with TOPICS_PATH.open("w", encoding="utf-8") as handle:
        for t in out:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"topics rebuilt n={len(out)} roots={ {k: len(v) for k,v in by_root.items()} }", flush=True)
    return len(out)


def main() -> int:
    buckets: dict[str, list] = defaultdict(list)
    n_in = 0
    with LEXICON_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            term = row.get("term") or ""
            if not speakable(term):
                continue
            if (row.get("count") or 0) < 2 and row.get("kind") not in {"person", "machine", "cli_tool", "product"}:
                continue
            src = row.get("source_file") or (row.get("files") or [""])[0]
            buckets[root_of(src)].append(row)
    ranked = []
    for root, rows in buckets.items():
        rows.sort(key=score_row, reverse=True)
        cap = QUOTAS.get(root, 400)
        ranked.extend(rows[:cap])
    # always keep people/machines/cli if they survived speakable()
    seen = {r["term"] for r in ranked}
    with LEXICON_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") in {"person", "machine", "cli_tool", "product", "library"}:
                if row["term"] not in seen and speakable(row["term"]):
                    ranked.append(row)
                    seen.add(row["term"])
    ranked.sort(key=score_row, reverse=True)
    with RANKED.open("w", encoding="utf-8") as handle:
        for row in ranked:
            row["rank_score"] = round(score_row(row), 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"ranked {len(ranked)} / {n_in} -> {RANKED}", flush=True)
    print("sample", [r["term"] for r in ranked[:40]], flush=True)
    rebuild_topics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
