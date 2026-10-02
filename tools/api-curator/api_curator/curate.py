"""Structured correction curation through an OpenAI-compatible API."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# Only the fields we need are sent: raw and corrected text. Audio paths,
# timestamps, session IDs, and other local metadata never leave the machine.
SYSTEM = """You are a dictation correction data curator.
Compare the raw ASR transcript with the user's final text.
Return exactly one JSON object fenced by ```json, with:
decision: keep | revise | discard
corrected: final intended text (required for keep/revise; empty string for discard)
reason_code: brief stable tag such as exact, punctuation, terminology, rewrite, unsafe, or ambiguous
confidence: number from 0 to 1
No commentary and no explanation."""
PROMPT = """<pair>
<raw>{raw}</raw>
<final>{final}</final>
</pair>"""


@dataclass(frozen=True, slots=True)
class CuratedRow:
    """A single local row to curate."""

    id: str
    raw: str
    final: str


@dataclass(frozen=True, slots=True)
class CurationJudgment:
    """Validated teacher output."""

    decision: str
    corrected: str
    reason_code: str
    confidence: float
    model: str
    elapsed_s: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


VALID_DECISIONS = {"keep", "revise", "discard"}


def parse_json_block(text: str) -> dict[str, Any]:
    """Parse one fenced or bare JSON object; reject arrays and commentary."""

    match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    value = match.group(1) if match else text.strip()
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError("teacher response must be a JSON object")
    return obj


def validate_judgment(value: dict[str, Any], model: str, elapsed_s: float) -> CurationJudgment:
    decision = str(value.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"invalid decision: {decision!r}")
    corrected = str(value.get("corrected", "")).strip()
    if decision == "discard":
        if corrected:
            raise ValueError("discard requires empty corrected text")
    elif not corrected:
        raise ValueError("keep/revise requires non-empty corrected text")
    reason = str(value.get("reason_code", "unlabeled")).strip() or "unlabeled"
    confidence = float(value.get("confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return CurationJudgment(decision, corrected, reason, confidence, model, elapsed_s)


def make_request(
    endpoint: str,
    api_key: str,
    model: str,
    raw: str,
    final: str,
    timeout: float,
    opener: Callable[..., urllib.request.Request] = urllib.request.Request,
) -> urllib.request.Request:
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": PROMPT.format(raw=raw, final=final),
                },
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
    ).encode()
    request = opener(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return request


def curate_one(
    row: CuratedRow,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float = 180,
    retries: int = 2,
    retry_wait_s: float = 2,
) -> tuple[CurationJudgment, str | None]:
    """Curate one row; the second return value is the raw completion for audit."""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            request = make_request(endpoint, api_key, model, row.raw, row.final, timeout)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            text = body["choices"][0]["message"]["content"]
            judgment = validate_judgment(parse_json_block(text), model, time.monotonic() - started)
            return judgment, text
        except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(retry_wait_s * (2**attempt))
    raise RuntimeError(f"API curation failed for {row.id}: {last_error}") from last_error


def curate_rows(rows: list[CuratedRow], output: Path, *, resume: bool = True, **kwargs: Any) -> Path:
    """Curate rows with JSONL append/resume semantics."""

    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume and output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                done.add(str(json.loads(line)["id"]))
    with output.open("a", encoding="utf-8") as sink:
        for row in rows:
            if row.id in done:
                continue
            judgment, completion = curate_one(row, **kwargs)
            record = {
                "id": row.id,
                "raw": row.raw,
                "final": row.final,
                "judgment": asdict(judgment),
                "completion": completion,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
    return output


VISION_SYSTEM = """You are a screenshot transcription data curator.
The image is a synthetic probe for infrastructure validation.
Return exactly one JSON object fenced by ```json, with:
decision: keep | revise | discard
corrected: exact visible text (empty only for discard)
reason_code: exact | terminology | layout | ambiguous | unsafe
confidence: number from 0 to 1
No commentary."""
VISION_PROMPT = """Transcribe and validate the visible text exactly. Do not infer hidden content."""


def curate_image(
    image_data_url: str,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float = 180,
) -> tuple[CurationJudgment, str]:
    """Validate one screenshot caption candidate without local text leakage."""

    started = time.monotonic()
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    text = body["choices"][0]["message"]["content"]
    judgment = validate_judgment(
        parse_json_block(text),
        model,
        time.monotonic() - started,
    )
    return judgment, text
