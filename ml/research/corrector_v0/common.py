"""Shared paths, JSONL helpers, chat format, and WER for corrector v0."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

DATA = Path(os.environ.get("PHONON_DATA_ROOT", "/data"))
WISPR_ROOT = DATA / "phonon_personal/wispr_20260915"
JULY_ROOT = DATA / "phonon_personal/dictation_eval_20260719"
DATA_ROOT = DATA / "phonon_corrector_v0"
REPO_ROOT = Path(os.environ.get("PHONON_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
RESEARCH_ROOT = REPO_ROOT / "research" / "corrector_v0"

PAIRS_PATH = WISPR_ROOT / "corrector_pairs_v0.jsonl"
PARAKEET_JSONL = DATA_ROOT / "parakeet_v2_wispr.jsonl"
TTS_PAIRS = DATA_ROOT / "tts_pairs.jsonl"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
DEV_JSONL = DATA_ROOT / "dev.jsonl"
ASR_SCORES = DATA_ROOT / "asr_scores.json"
FAILURES_PATH = DATA_ROOT / "failures.jsonl"

HF_HOME = DATA / "hf"
HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"

TRAIN_SPLITS = frozenset({"wispr_train", "wispr_text_train"})
HOLDOUT_SPLITS = frozenset({"wispr_holdout120", "wispr_text_holdout", "wispr_edit25"})

SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)

VOICES = ("af_heart", "am_adam", "bf_emma")
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
QWEN_ID = "Qwen/Qwen3-0.6B"
GEMMA_ID = "google/gemma-4-E2B-it"
QWEN_FALLBACK_ID = "Qwen/Qwen3-1.7B"

_NORM = None


def ensure_data_dirs() -> None:
    for name in ("adapters", "tts_wavs", "chunks", "logs", "tmp"):
        (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)


def hf_token() -> str | None:
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip() or None
    return None


def set_hf_env() -> None:
    import os

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ.setdefault("HF_TOKEN_PATH", str(HF_TOKEN_PATH))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    token = hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path, key: str = "id") -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        value = row.get(key)
        if value:
            done.add(str(value))
    return done


def record_failure(step: str, error: str, attempt: int) -> None:
    append_jsonl(
        FAILURES_PATH,
        {"step": step, "attempt": attempt, "error": error},
    )


def heartbeat(ttl: str = "30m") -> None:
    import os
    import subprocess

    agent = os.environ.get("OVERNIGHT_AGENT", "grok-corrector")
    subprocess.run(
        ["overnight-compute", "heartbeat", "--agent", agent, "--ttl", ttl],
        check=False,
    )


def fair_norm(text: str) -> str:
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORM = EnglishTextNormalizer()
    return _NORM((text or "").strip())


def strict_lc(text: str) -> str:
    return (text or "").strip().lower()


def _safe_pair(ref: str, hyp: str, normalizer: Callable[[str], str]) -> tuple[str, str]:
    nref = normalizer(ref or "")
    nhyp = normalizer(hyp or "")
    if not nref.strip():
        nref = "<empty>"
    if not nhyp.strip():
        nhyp = "<empty>"
    return nref, nhyp


def corpus_wer(
    refs: list[str],
    hyps: list[str],
    normalizer: Callable[[str], str],
) -> float:
    import jiwer

    if not refs:
        return 0.0
    nrefs, nhyps = zip(*[_safe_pair(r, h, normalizer) for r, h in zip(refs, hyps, strict=True)])
    return float(jiwer.wer(list(nrefs), list(nhyps)))


def pair_wer(ref: str, hyp: str, normalizer: Callable[[str], str]) -> float:
    import jiwer

    nref, nhyp = _safe_pair(ref, hyp, normalizer)
    return float(jiwer.wer(nref, nhyp))


def exact_match_rate(
    left: list[str],
    right: list[str],
    normalizer: Callable[[str], str],
) -> float:
    if not left:
        return 0.0
    hits = sum(normalizer(a) == normalizer(b) for a, b in zip(left, right, strict=True))
    return hits / len(left)


def score_lists(refs: list[str], hyps: list[str]) -> dict[str, float]:
    return {
        "n": float(len(refs)),
        "fair_wer": corpus_wer(refs, hyps, fair_norm),
        "strict_lc_wer": corpus_wer(refs, hyps, strict_lc),
        "fair_exact": exact_match_rate(refs, hyps, fair_norm),
        "strict_lc_exact": exact_match_rate(refs, hyps, strict_lc),
    }


def chat_messages(raw: str, target: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def load_wispr_pairs() -> list[dict[str, Any]]:
    return read_jsonl(PAIRS_PATH)


def load_july_rows() -> list[dict[str, Any]]:
    samples = json.loads((JULY_ROOT / "edit_samples.json").read_text(encoding="utf-8"))
    rows = []
    for sample in samples:
        sid = sample["id"]
        wav = JULY_ROOT / "audio" / f"{sid}.wav"
        rows.append(
            {
                "id": sid,
                "asr": (sample.get("asr") or "").strip(),
                "target": (sample.get("edited") or "").strip(),
                "formatted": (sample.get("formatted") or "").strip(),
                "edited": (sample.get("edited") or "").strip(),
                "has_audio": wav.exists(),
                "wav": str(wav),
                "split": "wispr_edit25",
                "duration": float(sample.get("duration") or 0.0),
            }
        )
    return rows


def wispr_wav(row_id: str) -> Path:
    return WISPR_ROOT / "audio" / f"{row_id}.wav"


def asr_jobs() -> list[dict[str, Any]]:
    """All 760 Wispr wavs plus 25 July clips. Empty-label wavs still get transcribed."""
    pairs = {row["id"]: row for row in load_wispr_pairs()}
    jobs = []
    audio_dir = WISPR_ROOT / "audio"
    for wav in sorted(audio_dir.glob("*.wav")):
        row = pairs.get(wav.stem, {})
        jobs.append(
            {
                "id": wav.stem,
                "wav": str(wav),
                "target": row.get("target") or "",
                "wispr_asr": row.get("asr") or "",
                "split": row.get("split") or "wispr_train",
            }
        )
    for row in load_july_rows():
        jobs.append(
            {
                "id": row["id"],
                "wav": row["wav"],
                "target": row["target"],
                "wispr_asr": row["asr"],
                "split": "wispr_edit25",
            }
        )
    return jobs


def strip_thinking(text: str) -> str:
    import re

    cleaned = text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I)
    cleaned = re.sub(r"<\|think\|>.*?(?:<\|/?think\|>|$)", "", cleaned, flags=re.S)
    cleaned = re.sub(r"<\|channel\|>thought.*?<\|channel\|>", "", cleaned, flags=re.S)
    return cleaned.strip()
