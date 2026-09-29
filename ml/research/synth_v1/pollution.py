"""Path and term pollution filter for synth_v1.

Third-party path classes are vendored libraries, scraped docs, generated
bindings, and decompiled game trees. A term is kept only if it occurs in
user-authored files (source, markdown, scripts, shell-history-like).
"""
from __future__ import annotations

import re
from pathlib import Path

THIRD_PARTY_PARTS = {
    "site-packages",
    "dist-packages",
    "__pypackages__",
    "node_modules",
    "bower_components",
    "vendor",
    "vendored",
    "third_party",
    "third-party",
    "3rdparty",
    ".venv",
    "venv",
    "virtualenv",
    ".tox",
    ".uv",
    ".cargo",
    ".npm",
    ".yarn",
    ".gradle",
    ".swiftpm",
    ".bundle",
    "tmodloader",
    "cuda-oxide",
    "cutile-rs",
    "cuda-bindings",
    "cuda-intrinsics-gen",
    "cuda-core-derive",
    "llvm-project",
    "spiny",
}

THIRD_PARTY_SUBSTR = (
    "/curation/context_resources/",
    "/benchmarks/nccl-tests/",
    "/dev/tools/codex/",
    "/cuda/rust-eval/cuda-oxide/",
    "/cuda/rust-eval/cutile-rs/",
    "/luminite-b0-matrix/cs/tmodloader",
    "/transformers-auto-",
    "/transformers-bert-",
    "/transformers-tokenizers-",
    "/transformers-index-",
    "/nsmbw-wt/",
    "/nsmbw/",
    "/fallcascade/java/",
    "/dev/pufferlib/",
    "/dev/tools/deepseek-harness/packages/",
    "/kernelbench.com/benchmarks/",
    "/cuda-intrinsics-gen/",
    "/cuda-bindings/",
)

GENERATED_RE = re.compile(
    r"(?:_pb2|_pb2_grpc|_generated|bindgen|generated\.(?:rs|py|go|ts)|"
    r"\.d\.ts|autogen|_auto\.py)$",
    re.I,
)

USER_SUBSTR = (
    "/phonon/scripts/",
    "/phonon/src/",
    "/phonon/research/",
    "/phonon/tests/",
    "/phonon/docs/",
    "/phonon/macos/",
    "/phonon/curation/hard_terms",
    "/phonon/mobile/",
    "/experiments/",
    "/box-install/",
)

USER_FILENAMES = {
    "devlog.md",
    "agents.md",
    "spec.md",
    "goal.md",
    "product.md",
    "readme.md",
    "labeling.md",
    "claude.md",
    "changelog.md",
    "contributing.md",
}

USER_EXT = {".sh", ".bash", ".zsh"}
HISTORY_RE = re.compile(r"(?:zsh_history|bash_history|fish_history|\.zshrc|\.bashrc)$")

IDENT_SHAPE = re.compile(r"[a-z][A-Z]|_|[A-Z]{2,8}$|[A-Za-z]\d|\d[A-Za-z]")
DUMP_CHUNK = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]{1,40}(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]{1,40}){3,}"
)
PRIVILEGED_KINDS = {"person", "machine", "cli_tool", "product"}


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def is_third_party_path(path: str) -> bool:
    if not path:
        return True
    n = _norm(path)
    parts = set(Path(n).parts)
    if parts & THIRD_PARTY_PARTS:
        return True
    if any(s in n for s in THIRD_PARTY_SUBSTR):
        return True
    if GENERATED_RE.search(n):
        return True
    return False


def is_user_authored_path(path: str) -> bool:
    if not path or is_third_party_path(path):
        return False
    n = _norm(path)
    if any(s in n for s in USER_SUBSTR):
        return True
    name = Path(n).name
    if name in USER_FILENAMES:
        return True
    if Path(n).suffix in USER_EXT:
        return True
    if HISTORY_RE.search(n):
        return True
    # top-level phonon notes / python entrypoints authored in-tree
    if "/phonon/" in n and n.count("/") <= 5 and Path(n).suffix in {".md", ".py"}:
        if "/curation/context_resources/" not in n:
            return True
    # user scripts dirs outside third-party trees
    if "/scripts/" in n and not is_third_party_path(path):
        return True
    return False


def classify_term(row: dict) -> str:
    """Return 'keep', 'polluted', or 'weak'."""
    files = list(row.get("files") or [])
    src = row.get("source_file") or ""
    if src and src not in files:
        files = [src] + files
    user_files = [f for f in files if is_user_authored_path(f)]
    third = [f for f in files if is_third_party_path(f)]
    kind = row.get("kind") or ""
    count = int(row.get("count") or 0)
    if kind in PRIVILEGED_KINDS:
        return "keep"
    if not user_files:
        return "polluted"
    if count >= 2 and len(user_files) >= 2:
        return "keep"
    if count >= 2 and len(user_files) >= 1 and len(files) >= 2 and len(third) < len(files):
        return "keep"
    if count >= 2 and len(user_files) >= 1:
        return "weak"
    return "polluted"


def is_ident_shape(tok: str) -> bool:
    t = (tok or "").strip(".,;:!?\"'`()[]{}")
    if len(t) < 2:
        return False
    return bool(IDENT_SHAPE.search(t))


def has_identifier_dump(text: str) -> bool:
    for m in DUMP_CHUNK.finditer(text or ""):
        parts = [p.strip() for p in m.group(0).split(",")]
        shaped = sum(1 for p in parts if is_ident_shape(p))
        if shaped >= 3:
            return True
    return False
