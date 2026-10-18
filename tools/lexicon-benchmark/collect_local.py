#!/usr/bin/env python3
"""Collect local, technical term candidates for the lexicon benchmark.

The collector deliberately reads inventories and public package metadata, not
user transcripts or source text. The resulting term files are operational data:
keep them under the local research share unless every term is being published
deliberately. The aggregate manifest contains counts and source labels only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent.parent
PROFILE_MINER_ROOT = TOOL_ROOT.parent / "profile-miner"
LEXICON_PATH = PROFILE_MINER_ROOT / "lexicon" / "lexicon.json.gz"
SCHEMA_VERSION = 1
TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{1,47}$")
# A bare package identifier, used after an npm scope is stripped: an npm scope
# is publisher owned and can name a private organization, so only the package
# half of a scoped name is ever retained as a term.
BARE_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{1,47}$")
SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".rb", ".pl", ".js", ".ts"}
GENERIC_TERMS = {
    "app",
    "bin",
    "build",
    "cli",
    "command",
    "core",
    "index",
    "lib",
    "main",
    "run",
    "script",
    "test",
    "tests",
    "tool",
    "tools",
    "utils",
}
SOURCE_WEIGHTS = {
    "toolbelt": 1.0,
    "path-command": 0.85,
    "manpage": 0.78,
    "brew-tap": 0.8,
    "cli-extension": 0.92,
    "cargo-local": 0.72,
    "npm-local": 0.68,
    "pypi-local": 0.7,
}
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "build",
    "Library",
    ".cache",
    ".worktrees",
}
MAN_SECTION_RE = re.compile(r"^(?P<names>.*?)\((?P<section>[^)]+)\)\s*-")
CLEAN_RE = re.compile(r"(?:\x1b\[[0-9;]*m|\\f[BIPR]|\*)")
PUBLIC_CARGO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
PUBLIC_NPM_SOURCE = "https://registry.npmjs.org/"
PUBLIC_PYPI_SOURCES = {
    "registry+https://pypi.org/simple",
    "registry+https://pypi.org/simple/",
}


def load_english() -> set[str]:
    words: set[str] = set()
    for path in (
        Path("/usr/share/dict/words"),
        PROFILE_MINER_ROOT / "profile_miner" / "data" / "english_extra.txt",
    ):
        try:
            words.update(line.strip().lower() for line in path.read_text().splitlines())
        except OSError:
            continue
    return words


def load_known_lexicon(path: Path = LEXICON_PATH) -> set[str]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return {str(key).lower() for key in json.load(handle)}
    except (OSError, ValueError, json.JSONDecodeError):
        return set()


def normalize_term(raw: str) -> str | None:
    term = CLEAN_RE.sub("", raw).strip().strip("`\"'")
    if re.search(r"\s", term):
        return None
    if not TERM_RE.fullmatch(term):
        return None
    if Path(term).suffix.lower() in SCRIPT_SUFFIXES:
        return None
    key = term.lower()
    if key in GENERIC_TERMS or key.removeprefix("lib").isdigit():
        return None
    return term


def normalize_scoped_term(raw: str) -> str | None:
    """Return the bare package half of a scoped name, or None if it is unusable.

    Scopes are dropped rather than kept: they identify the publisher, not the
    package, and a private scope would leak an organization name into a term
    list that is meant to hold public package identifiers only.
    """
    term = CLEAN_RE.sub("", raw).strip().strip("`\"'")
    if "/" not in term:
        return None
    _, _, bare = term.rpartition("/")
    return bare if BARE_TERM_RE.fullmatch(bare) else None


def is_technical(term: str, english: set[str]) -> bool:
    key = term.lower()
    has_shape = bool(
        re.search(r"[_.+-]", term)
        or re.search(r"[a-z][A-Z]", term)
        or re.search(r"[A-Z]{2,}", term)
        or re.search(r"\d", term)
    )
    return has_shape or key not in english


class CandidateStore:
    def __init__(self, english: set[str] | None = None, known: set[str] | None = None):
        self.english = english if english is not None else load_english()
        self.known = known if known is not None else load_known_lexicon()
        self.entries: dict[str, dict[str, Any]] = {}
        self.totals: Counter[str] = Counter()

    def add(self, raw: str, source: str) -> bool:
        self.totals[source] += 1
        term = normalize_scoped_term(raw) or normalize_term(raw)
        if term is None or not is_technical(term, self.english):
            return False
        key = term.lower()
        if key in self.known:
            return False
        entry = self.entries.setdefault(
            key, {"term": term, "sources": [], "observations": {}}
        )
        if source not in entry["sources"]:
            entry["sources"].append(source)
        entry["observations"][source] = int(entry["observations"].get(source, 0)) + 1
        return True

    def ranked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in self.entries.values():
            priority = max(SOURCE_WEIGHTS[source] for source in entry["sources"])
            recurrence_bonus = 0.15 * (
                min(3, max(entry["observations"].values()) - 1) / 3
            )
            breadth_bonus = 0.05 * (len(entry["sources"]) - 1)
            rows.append(
                {
                    "term": entry["term"],
                    "cats": sorted(entry["sources"]),
                    "priority": round(priority + recurrence_bonus + breadth_bonus, 4),
                    "observations": entry["observations"],
                }
            )
        return sorted(rows, key=lambda row: (-row["priority"], row["term"]))


def parse_toolbelt(text: str, store: CandidateStore) -> int:
    before = len(store.entries)
    for line in text.splitlines():
        match = re.match(r"^\s{2}(\S+)\s+(?:/|~).*$", line)
        if match:
            store.add(match.group(1), "toolbelt")
    return len(store.entries) - before


def parse_manpages(text: str, store: CandidateStore) -> int:
    before = len(store.entries)
    for line in text.splitlines():
        match = MAN_SECTION_RE.match(line)
        if not match or not re.match(r"^[145678](?:[a-z]+)?$", match["section"]):
            continue
        for name in match["names"].split(","):
            store.add(name, "manpage")
    return len(store.entries) - before


def parse_gh_extensions(text: str, store: CandidateStore) -> int:
    """Parse native `gh extension list` rows; the CLI has no JSON mode."""
    before = len(store.entries)
    for line in text.splitlines():
        fields = line.split()
        if fields:
            store.add(fields[0], "cli-extension")
    return len(store.entries) - before


def parse_cargo_lock(data: dict[str, Any], store: CandidateStore) -> int:
    before = len(store.entries)
    for package in data.get("package", []):
        if package.get("source") == PUBLIC_CARGO_SOURCE:
            store.add(str(package.get("name", "")), "cargo-local")
    return len(store.entries) - before


def parse_npm_lock(data: dict[str, Any], store: CandidateStore) -> int:
    before = len(store.entries)
    for package_path, metadata in data.get("packages", {}).items():
        if (
            not package_path
            or not isinstance(metadata, dict)
            or metadata.get("link")
            or not str(metadata.get("resolved", "")).startswith(PUBLIC_NPM_SOURCE)
        ):
            continue
        package_name = package_path.rsplit("node_modules/", 1)[-1]
        if not package_name.startswith("@"):
            package_name = package_name.rsplit("/", 1)[-1]
        store.add(package_name, "npm-local")
    return len(store.entries) - before


def parse_uv_lock(data: dict[str, Any], store: CandidateStore) -> int:
    before = len(store.entries)
    for package in data.get("package", []):
        source = str(package.get("source", ""))
        if source in PUBLIC_PYPI_SOURCES:
            store.add(str(package.get("name", "")), "pypi-local")
    return len(store.entries) - before


def run_collector(command: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", repr(error)
    if result.returncode != 0:
        return "", f"exit {result.returncode}"
    return result.stdout, None


def collect_path_commands(store: CandidateStore) -> None:
    seen_paths: set[Path] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory)
        if path in seen_paths or not path.is_dir():
            continue
        seen_paths.add(path)
        try:
            candidates = list(path.iterdir())
        except OSError:
            continue
        for item in candidates:
            try:
                if item.is_file() and os.access(item, os.X_OK):
                    store.add(item.name, "path-command")
            except OSError:
                continue


def collect_cli_extensions(store: CandidateStore) -> dict[str, str]:
    failures: dict[str, str] = {}
    output, error = run_collector(["gh", "extension", "list"])
    if error:
        failures["cli-extension"] = error
    elif output:
        parse_gh_extensions(output, store)
    extensions, code_error = run_collector(["code", "--list-extensions"])
    if code_error and "cli-extension" not in failures:
        failures["cli-extension"] = code_error
    for extension in extensions.splitlines():
        if extension.strip():
            store.add(extension.strip(), "cli-extension")
    return failures


def workspace_files(root: Path, limit: int = 2048) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root):
        directories[:] = sorted(
            name for name in directories if name not in SKIP_DIRECTORIES
        )
        for name in sorted(names):
            if name in {"Cargo.lock", "package-lock.json", "uv.lock"}:
                files.append(Path(current) / name)
                if len(files) >= limit:
                    return files
    return files


def is_safe_output_path(path: Path) -> bool:
    """Permit the ignored build root, but not tracked repo paths."""
    resolved = path.resolve()
    build_root = REPO_ROOT / "build"
    return (
        resolved == build_root
        or build_root in resolved.parents
        or not (resolved == REPO_ROOT or REPO_ROOT in resolved.parents)
    )


def collect_workspace(root: Path, store: CandidateStore) -> None:
    parsers: dict[str, Callable[[dict[str, Any], CandidateStore], int]] = {
        "Cargo.lock": parse_cargo_lock,
        "package-lock.json": parse_npm_lock,
        "uv.lock": parse_uv_lock,
    }
    for path in workspace_files(root):
        parser = parsers[path.name]
        try:
            if path.name == "uv.lock":
                with path.open("rb") as handle:
                    data = tomllib.load(handle)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError):
            continue
        parser(data, store)


def manifest(
    store: CandidateStore,
    rows: list[dict[str, Any]],
    failures: dict[str, str] | None = None,
) -> dict[str, Any]:
    sources: dict[str, dict[str, int]] = {}
    for source, total in sorted(store.totals.items()):
        kept = sum(1 for row in rows if source in row["cats"])
        sources[source] = {"seen": total, "kept": kept}
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "terms": len(rows),
            "sources": len(sources),
            "min_priority": min((row["priority"] for row in rows), default=0.0),
            "max_priority": max((row["priority"] for row in rows), default=0.0),
        },
        "sources": sources,
        "collection_failures": failures or {},
        "non_claims": [
            "This inventory contains command and package names, not user dictation.",
            "A candidate being obscure does not prove that the owner dictates it.",
            "No ASR result is attached until the oracle cache is run separately.",
            "Term rows and oracle outputs are local research data unless explicitly reviewed for publication.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--workspace-root", action="append", type=Path, default=[])
    parser.add_argument("--min-priority", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args(argv)

    output_paths = [
        arguments.out,
        arguments.terms,
        arguments.lexicon,
        arguments.out.with_suffix(".manifest.json"),
    ]
    unsafe_paths = [path for path in output_paths if not is_safe_output_path(path)]
    if unsafe_paths:
        raise ValueError(
            "local inventory outputs must stay outside the repository "
            f"or under ignored build/: {', '.join(str(path) for path in unsafe_paths)}"
        )

    store = CandidateStore()
    failures: dict[str, str] = {}
    brief, brief_error = run_collector(["ah", "brief"])
    if brief_error:
        failures["toolbelt"] = brief_error
    parse_toolbelt(brief, store)
    manpages, man_error = run_collector(["man", "-k", "."])
    if man_error:
        failures["manpage"] = man_error
    parse_manpages(manpages, store)
    taps, tap_error = run_collector(["brew", "tap"])
    if tap_error:
        failures["brew-tap"] = tap_error
    for tap in taps.splitlines():
        store.add(tap.split("/")[-1], "brew-tap")
    failures.update(collect_cli_extensions(store))
    collect_path_commands(store)
    for root in arguments.workspace_root:
        collect_workspace(root.resolve(), store)

    rows = [row for row in store.ranked() if row["priority"] >= arguments.min_priority]
    rows = rows[: arguments.limit] if arguments.limit else rows
    lexicon = {
        row["term"].lower(): {
            "term": row["term"],
            "cats": row["cats"],
            "priority": row["priority"],
        }
        for row in rows
    }
    for path, payload in (
        (arguments.out, rows),
        (arguments.terms, [row["term"] for row in rows]),
        (arguments.lexicon, lexicon),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    report = manifest(store, rows, failures)
    arguments.out.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "terms": len(rows),
                "manifest": str(arguments.out.with_suffix(".manifest.json")),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
