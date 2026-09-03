"""Stage 2: deterministic extraction of user-authored text, one line per message.

Sources: Claude Code, Codex, Grok Build, local git repos. Files are streamed
line by line; nothing is loaded whole. Phonon App Support and Wispr Flow data
are never read (held-out eval set).
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .common import HOME, PHONON_SUPPORT, out_dir, write_json

WS = re.compile(r"\s+")
DOC_NAMES = {"readme", "agents", "devlog", "spec", "claude"}
TREE_SKIP = {".git", "node_modules", "target", ".venv", "venv", "__pycache__", ".build", "dist", "build", ".cache"}
# Repos whose docs quote the held-out evaluation set (dictionary and corpus).
REPO_DOC_EXCLUDE = {"phonon"}
# Codex messages that are harness wrappers, not user text.
# Chat messages over LONG_WORDS words are pastes (logs, files); keep the authored head only.
LONG_WORDS = 300
KEEP_WORDS = 120
CODEX_SKIP_PREFIX = ("<environment_context>", "# AGENTS.md instructions", "<user_instructions>", "<permissions", "<turn_aborted")


def one_line(s: str) -> str:
    return WS.sub(" ", s).strip()


def head_of_paste(line: str, stats) -> str:
    words = line.split()
    if len(words) > LONG_WORDS:
        stats["long_truncated"] += 1
        return " ".join(words[:KEEP_WORDS])
    return line


def _ts_ok(ts, until):
    if not until or not ts:
        return True
    return str(ts)[:19] < until


def extract_claude(until, stats):
    root = HOME / ".claude" / "projects"
    for path in sorted(glob.glob(str(root / "*" / "*.jsonl"))):
        stats["files"] += 1
        with open(path, "rb") as f:
            for raw in f:
                if b'"type":"user"' not in raw and b'"type": "user"' not in raw:
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    stats["bad_json"] += 1
                    continue
                if d.get("type") != "user":
                    continue
                if not _ts_ok(d.get("timestamp"), until):
                    stats["after_until"] += 1
                    continue
                m = d.get("message") or {}
                c = m.get("content")
                if not isinstance(c, str):
                    stats["non_string"] += 1
                    continue
                line = one_line(c)
                if not line:
                    continue
                if line.startswith("<"):
                    stats["harness_lines"] += 1
                    continue
                yield head_of_paste(line, stats)


def extract_codex(until, stats):
    root = HOME / ".codex" / "sessions"
    for path in sorted(glob.glob(str(root / "**" / "*.jsonl"), recursive=True)):
        stats["files"] += 1
        with open(path, "rb") as f:
            for raw in f:
                if b'"role":"user"' not in raw and b'"role": "user"' not in raw:
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    stats["bad_json"] += 1
                    continue
                p = d.get("payload")
                if not isinstance(p, dict) or p.get("role") != "user":
                    continue
                if not _ts_ok(d.get("timestamp"), until):
                    stats["after_until"] += 1
                    continue
                for part in p.get("content") or []:
                    if not isinstance(part, dict) or part.get("type") != "input_text":
                        continue
                    text = part.get("text") or ""
                    if text.lstrip().startswith(CODEX_SKIP_PREFIX):
                        stats["harness_messages"] += 1
                        continue
                    for ln in text.split("\n"):
                        ln = one_line(ln)
                        if not ln:
                            continue
                        if ln.startswith(("#", "<", "```")) or len(ln.split()) > 80:
                            stats["harness_lines"] += 1
                            continue
                        yield ln


def extract_grok(until, stats):
    root = HOME / ".grok" / "sessions"
    for path in sorted(glob.glob(str(root / "*" / "prompt_history.jsonl"))):
        stats["files"] += 1
        with open(path, "rb") as f:
            for raw in f:
                try:
                    d = json.loads(raw)
                except Exception:
                    stats["bad_json"] += 1
                    continue
                if not _ts_ok(d.get("timestamp"), until):
                    stats["after_until"] += 1
                    continue
                p = d.get("prompt")
                if isinstance(p, str):
                    line = one_line(p)
                    if line:
                        yield head_of_paste(line, stats)


def find_repos(max_depth=3):
    """Git repos under $HOME to depth 3, excluding Library and dot dirs."""
    repos = []
    home = str(HOME)
    for dirpath, dirnames, filenames in os.walk(home):
        rel = os.path.relpath(dirpath, home)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth == 0:
            dirnames[:] = [d for d in dirnames if d != "Library" and not d.startswith(".")]
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in TREE_SKIP]
        if os.path.isdir(os.path.join(dirpath, ".git")) or os.path.isfile(os.path.join(dirpath, ".git")):
            repos.append(Path(dirpath))
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
    return repos


def extract_repos(until, stats):
    repos = find_repos()
    stats["repos"] = len(repos)
    names = []
    for repo in repos:
        if str(repo).startswith(str(PHONON_SUPPORT)):
            continue
        names.append(repo.name)
        if repo.name.lower() in REPO_DOC_EXCLUDE:
            stats["repos_docs_excluded"] += 1
            continue
        # Docs at repo root.
        for entry in sorted(os.listdir(repo)):
            stem, ext = os.path.splitext(entry)
            if ext.lower() != ".md" or stem.lower().split(".")[0] not in DOC_NAMES:
                continue
            stats["doc_files"] += 1
            try:
                with open(repo / entry, "r", errors="replace") as f:
                    for ln in f:
                        ln = one_line(ln)
                        if ln and not ln.startswith(("```", "|", "<")):
                            yield ln
            except OSError:
                pass
        # File tree to depth 2, names only.
        tree = []
        for d1 in sorted(os.listdir(repo)):
            if d1 in TREE_SKIP or d1.startswith("."):
                continue
            tree.append(d1)
            p1 = repo / d1
            if p1.is_dir():
                try:
                    for d2 in sorted(os.listdir(p1)):
                        if d2 not in TREE_SKIP and not d2.startswith("."):
                            tree.append(d2)
                except OSError:
                    pass
        stats["tree_entries"] += len(tree)
        for entry in tree[:400]:
            yield entry
    write_json(out_dir() / "extract" / "repos.json", sorted(set(names)))


# Source files whose identifiers are mined (document frequency: one count per file).
CODE_EXT = {"py", "rs", "swift", "ts", "tsx", "js", "jsx", "go", "c", "cc", "cpp", "cu", "cuh", "h", "hpp", "sh", "zsh",
            "toml", "yaml", "yml", "cmake", "mm", "kt", "java", "rb", "lua", "sql", "proto", "md", "txt"}
CODE_SKIP_SUFFIX = (".min.js", ".lock", "-lock.json", ".ipynb")
CODE_MAX_BYTES = 400_000
CODE_MAX_FILES_PER_REPO = 4000
CODE_MAX_FILES = 120_000
RE_CODE_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[\-_.][A-Za-z0-9]+)*")
# Language keywords and builtins that appear in every repo and are never dictionary words.
CODE_STOP = set("""
def class return self none true false import from as with pass break continue lambda yield raise global del assert
print len range int str float bool list dict set tuple super isinstance type object elif else while for if in is not
and or async await try except finally nonlocal
fn let mut pub impl struct enum match use mod const static crate where dyn ref move unsafe trait loop
some ok err vec string option result box usize isize u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64 char
void long double unsigned signed sizeof typedef include define ifdef ifndef endif pragma template typename namespace
using public private protected virtual override nullptr auto constexpr inline extern static_cast dynamic_cast
reinterpret_cast const_cast noexcept
function export default new this null undefined interface implements extends throw catch switch case do
instanceof typeof console log var require module exports
guard func init deinit weak strong protocol extension throws rethrows nil self any
package go chan defer map make range select
elsif unless begin end then rescue ensure nil
""".split())


def extract_code(until, stats):
    """Identifiers from source files in every repo; each file contributes each token once."""
    repos = find_repos()
    total = 0
    for repo in repos:
        if str(repo).startswith(str(PHONON_SUPPORT)) or repo.name.lower() in REPO_DOC_EXCLUDE:
            continue
        n_repo = 0
        for dirpath, dirnames, filenames in os.walk(repo):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in TREE_SKIP]
            for fname in filenames:
                if n_repo >= CODE_MAX_FILES_PER_REPO or total >= CODE_MAX_FILES:
                    dirnames[:] = []
                    break
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in CODE_EXT or fname.endswith(CODE_SKIP_SUFFIX) or fname.startswith("."):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(path) > CODE_MAX_BYTES:
                        stats["code_files_too_big"] += 1
                        continue
                    with open(path, "r", errors="replace") as f:
                        text = f.read()
                except OSError:
                    continue
                toks = set()
                for m in RE_CODE_TOKEN.finditer(text):
                    t = m.group()
                    if len(t) < 2 or len(t) > 40 or t.lower() in CODE_STOP or not any(c.isalpha() for c in t):
                        continue
                    toks.add(t)
                if not toks:
                    continue
                n_repo += 1
                total += 1
                stats["code_tokens"] += len(toks)
                yield " ".join(sorted(toks))
        stats["code_files"] += n_repo
    stats["code_repos"] = len(repos)


SOURCES = {
    "claude": extract_claude,
    "codex": extract_codex,
    "grok": extract_grok,
    "repos": extract_repos,
    "code": extract_code,
}


def run(sources=None, until=None):
    od = out_dir() / "extract"
    od.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in sources or SOURCES:
        fn = SOURCES[name]
        stats = Counter()
        t0 = time.time()
        n = 0
        uniq = set()
        with open(od / f"{name}.txt", "w") as f:
            for line in fn(until, stats):
                f.write(line + "\n")
                n += 1
                uniq.add(line)
        stats["lines"] = n
        stats["unique_lines"] = len(uniq)
        stats["seconds"] = round(time.time() - t0, 1)
        report[name] = dict(stats)
        print(f"[extract] {name}: {n} lines ({len(uniq)} unique) in {stats['seconds']}s {dict(stats)}", file=sys.stderr)
    counts_path = od / "counts.json"
    prev = {}
    if counts_path.exists():
        prev = json.load(open(counts_path))
    prev.update(report)
    prev["until"] = until
    write_json(counts_path, prev)
    return report
