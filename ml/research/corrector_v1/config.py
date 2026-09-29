"""Shared paths, prompts, and helpers for corrector v1. Imports v0; does not edit it."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path("/home/user/phonon")
V0_DIR = REPO_ROOT / "research" / "corrector_v0"
OPTION0_DIR = REPO_ROOT / "scripts" / "option0"
RESEARCH_ROOT = REPO_ROOT / "research" / "corrector_v1"
V0_DATA = Path("/data/phonon_corrector_v0")
DATA_ROOT = Path("/data/phonon_corrector_v1")
PARAKEET_JSONL = V0_DATA / "parakeet_v2_wispr.jsonl"
V0_TRAIN_JSONL = V0_DATA / "train.jsonl"
V0_DEV_JSONL = V0_DATA / "dev.jsonl"
V0_ADAPTER = V0_DATA / "adapters" / "qwen3-0.6b"
OPTION0_PREDS = REPO_ROOT / "runs" / "reports" / "option0_sweep_20260915" / "preds"

QWEN_ID = "Qwen/Qwen3-0.6B"
HF_HOME = Path("/data/hf")
HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
SEED = 20260915
MISSING = "<missing>"
AGENT = "grok-multihyp"

NEMO_UNIFIED = "nvidia/parakeet-unified-en-0.6b"
NEMO_LOCAL_V3 = (
    "/home/user/phonon/runs/finetune/"
    "parakeet_v3_a0p4375_x_jointout_lr1em7_interp_fine_save_20260527/"
    "a0p5625/checkpoint.nemo"
)
COHERE_ID = "CohereLabs/cohere-transcribe-03-2026"

# Fixed hypothesis order for every multi-hyp prompt.
LETTER_MODELS = ("parakeet_v2", "parakeet_unified", "local_v3", "cohere")
LETTERS = ("A", "B", "C", "D")
MODEL_LETTER = dict(zip(LETTER_MODELS, LETTERS, strict=True))
LETTER_MODEL = dict(zip(LETTERS, LETTER_MODELS, strict=True))

HYP_PATHS = {
    "parakeet_v2": DATA_ROOT / "hyps_parakeet_v2.jsonl",
    "parakeet_unified": DATA_ROOT / "hyps_parakeet_unified.jsonl",
    "local_v3": DATA_ROOT / "hyps_local_v3.jsonl",
    "cohere": DATA_ROOT / "hyps_cohere.jsonl",
}
OPTION0_TAGS = {
    "parakeet_v2": "parakeet-tdt-0.6b-v2",
    "parakeet_unified": "parakeet-unified-en-0.6b",
    "local_v3": "parakeet_v3_localbest_a0p5625",
    "cohere": "cohere-transcribe-03-2026",
}

SYSTEM_V0 = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
SYSTEM_M3 = (
    "Several automatic transcripts (A-C) of the same dictation are given; "
    "produce the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
SYSTEM_M4 = (
    "Several automatic transcripts (A-D) of the same dictation are given; "
    "produce the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)


def load_v0_common():
    name = "corrector_v0_common"
    if name in sys.modules:
        return sys.modules[name]
    path = V0_DIR / "common.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_data_dirs() -> None:
    for name in ("adapters", "chunks", "logs", "tmp", "eval_predictions"):
        (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)


def set_hf_env() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ.setdefault("HF_TOKEN_PATH", str(HF_TOKEN_PATH))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OVERNIGHT_AGENT", AGENT)
    token = hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def hf_token() -> str | None:
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip() or None
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def heartbeat(ttl: str = "30m") -> None:
    subprocess.run(
        ["overnight-compute", "heartbeat", "--agent", AGENT, "--ttl", ttl],
        check=False,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path, key: str = "id") -> set[str]:
    if not path.exists():
        return set()
    return {str(row[key]) for row in iter_jsonl(path) if row.get(key)}


def format_user(hyps: dict[str, str], letters: tuple[str, ...]) -> str:
    lines = []
    for letter in letters:
        text = hyps.get(letter)
        if text is None:
            text = MISSING
        lines.append(f"{letter}: {text}")
    return "\n".join(lines)


def system_prompt(letters: tuple[str, ...]) -> str:
    if letters == ("A",):
        return SYSTEM_V0
    if letters == ("A", "B", "C"):
        return SYSTEM_M3
    if letters == LETTERS:
        return SYSTEM_M4
    joined = "-".join(letters)
    return (
        f"Several automatic transcripts ({joined}) of the same dictation are given; "
        "produce the exact text the speaker intended. "
        "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
        "do not add or drop content."
    )


def chat_messages(
    user_text: str,
    target: str | None = None,
    letters: tuple[str, ...] = ("A",),
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt(letters)},
        {"role": "user", "content": user_text},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def apply_chat_template(processor, messages: list[dict], add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    for extra in ({"enable_thinking": False}, {}):
        try:
            text = processor.apply_chat_template(messages, **kwargs, **extra)
            if isinstance(text, list):
                text = text[0]
            return str(text)
        except TypeError:
            continue
        except Exception:
            mm = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                mm.append({"role": msg["role"], "content": content})
            try:
                text = processor.apply_chat_template(
                    mm,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
                if isinstance(text, list):
                    text = text[0]
                return str(text)
            except Exception:
                pass
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def max_new_tokens_for(user_text: str) -> int:
    """Per-row generation cap: max(256, 3x input words).

    For labeled A-D prompts the 'input' is the longest provided hypothesis,
    not the concatenated prompt. 3x the four-line blob lets a merger dump
    every transcript and still sit under the cap.
    """
    hyp_words: list[int] = []
    labeled = False
    for line in (user_text or "").splitlines():
        if len(line) >= 3 and line[0] in "ABCD" and line[1:3] == ": ":
            labeled = True
            body = line[3:].strip()
            if body and body != MISSING:
                hyp_words.append(len(body.split()))
    if labeled and hyp_words:
        words = max(hyp_words)
    else:
        words = max(1, len((user_text or "").split()))
    return max(256, 3 * words)


def option0_pred_path(split: str, model_key: str) -> Path:
    return OPTION0_PREDS / split / f"{OPTION0_TAGS[model_key]}.jsonl"


def load_hyp_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path) if row.get("id")}
