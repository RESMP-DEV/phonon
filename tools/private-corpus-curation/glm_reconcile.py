#!/usr/bin/env python3
"""Reconcile technical ASR teachers with GLM through a local API route."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from corpus_curation import (
    CurationError,
    iter_parquet_rows,
    safe_output_path,
    sha256_text,
    validate_output_aliases,
)

SYSTEM_PROMPT = """You reconcile competing automatic speech-recognition hypotheses from technical lecture audio. Return strict JSON only. Choose the most acoustically and contextually likely exact spoken text. Preserve technical identifiers, hardware names, numbers, punctuation, and ordering. Do not invent content, summarize, or continue the lecture. If teachers conflict irreconcilably, choose the most reliable fragments and set confidence low."""
USER_TEMPLATE = """<task>Reconcile the teacher hypotheses for one clip.</task>
<context>Domain tags: {domain_tags}</context>
<teachers>{teachers}</teachers>
<output_schema>{{"decision":"keep|revise|discard","transcript":"...","confidence":0.0,"reason":"..."}}</output_schema>
<rules>Return one JSON object. Use confidence between 0 and 1. Never include IDs, paths, or teacher names in the transcript.</rules>"""
PROMPT_SCHEMA_VERSION = 1
PROMPT_SCHEMA_SHA256 = sha256_text(SYSTEM_PROMPT + USER_TEMPLATE)


def parse_teachers(value: Any) -> dict[str, str]:
    """Parse canonical teacher maps without printing their text."""
    parsed = value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    if not isinstance(parsed, dict):
        raise CurationError("unsupported teacher transcript container")
    result: dict[str, str] = {}
    for name, hypothesis in parsed.items():
        if isinstance(hypothesis, str) and hypothesis.strip():
            result[str(name)] = " ".join(hypothesis.split())
        elif isinstance(hypothesis, dict) and isinstance(hypothesis.get("text"), str):
            result[str(name)] = " ".join(hypothesis["text"].split())
        else:
            raise CurationError("unsupported teacher hypothesis format")
    return result


def json_payload(content: str) -> dict[str, Any]:
    """Extract a strict JSON object from a model response."""
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise CurationError(f"GLM returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise CurationError("GLM JSON response must be an object")
    valid = {"decision", "transcript", "confidence", "reason"}
    if set(value) != valid:
        raise CurationError("GLM JSON fields do not match the schema")
    if value["decision"] not in {"keep", "revise", "discard"}:
        raise CurationError("GLM decision is invalid")
    if not isinstance(value["transcript"], str) or not isinstance(value["reason"], str):
        raise CurationError("GLM transcript and reason must be strings")
    confidence = value["confidence"]
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        raise CurationError("GLM confidence must be numeric between 0 and 1")
    value["confidence"] = float(confidence)
    return value


def existing_ids(path: Path, *, model: str) -> set[str]:
    """Validate resumable provenance and return completed row IDs."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str):
            raise CurationError("reconciliation output lacks id")
        if (
            row.get("schema_version") != PROMPT_SCHEMA_VERSION
            or row.get("model") != model
            or row.get("prompt_schema_sha256") != PROMPT_SCHEMA_SHA256
            or row.get("status") not in {"ok", "error"}
        ):
            raise CurationError(
                f"reconciliation output line {line_number} has mixed provenance"
            )
        if row["id"] in completed:
            raise CurationError(
                f"duplicate reconciliation output id on line {line_number}"
            )
        completed.add(row["id"])
    return completed


def chat(
    endpoint: str, model: str, prompt: str, timeout: float, auth_token: str
) -> tuple[dict[str, Any], str, str]:
    """Make one bounded OpenAI-compatible chat request."""
    base = endpoint.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise CurationError("GLM endpoint returned no message content")
    served_model = payload.get("model") if isinstance(payload, dict) else None
    requested_alias = model.partition(",")[2] or model
    served_alias = (
        str(served_model).partition(",")[2] or str(served_model)
        if served_model is not None
        else None
    )
    if served_model not in {model, requested_alias} or served_alias not in {
        model,
        requested_alias,
    }:
        raise CurationError(f"GLM endpoint served {served_model!r}, not {model!r}")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason != "stop":
        raise CurationError(f"GLM request finished with {finish_reason!r}")
    return (
        json_payload(message["content"]),
        str(finish_reason),
        str(served_model),
    )


def reconcile(
    dataset: Path,
    output: Path,
    *,
    endpoint: str,
    model: str,
    limit: int | None,
    timeout: float,
    auth_token: str,
) -> dict[str, Any]:
    """Run resumable blind reconciliation over teacher-bearing rows."""
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.touch(mode=0o600, exist_ok=True)
    output.chmod(0o600)
    completed = existing_ids(output, model=model)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    input_rows = already_present = generated = failed = 0
    started = time.monotonic()
    for source in iter_parquet_rows(dataset):
        row_id = source.get("id")
        audio_hash = source.get("sha256_audio")
        if not isinstance(row_id, str) or not row_id:
            raise CurationError("dataset row lacks a nonempty string id")
        if not isinstance(audio_hash, str) or not audio_hash:
            raise CurationError(f"row {row_id} lacks sha256_audio")
        if row_id in seen_ids or audio_hash in seen_hashes:
            continue
        seen_ids.add(row_id)
        seen_hashes.add(audio_hash)
        teachers = parse_teachers(source.get("teacher_transcripts"))
        if not teachers:
            continue
        row = {
            "id": row_id,
            "domain_tags": source.get("domain_tags") or [],
            "teachers": teachers,
        }
        input_rows += 1
        if limit is not None and input_rows > limit:
            input_rows -= 1
            break
        if row_id in completed:
            already_present += 1
            continue
        prompt = USER_TEMPLATE.format(
            domain_tags=", ".join(map(str, row["domain_tags"])),
            teachers=json.dumps(row["teachers"], ensure_ascii=True, indent=2),
        )
        record = {
            "id": row["id"],
            "schema_version": PROMPT_SCHEMA_VERSION,
            "model": model,
            "prompt_sha256": sha256_text(prompt),
            "prompt_schema_sha256": PROMPT_SCHEMA_SHA256,
        }
        try:
            result, finish_reason, served_model = chat(
                endpoint, model, prompt, timeout, auth_token
            )
        except (CurationError, OSError, TimeoutError) as error:
            record.update(
                {
                    "status": "error",
                    "error": type(error).__name__,
                    "finish_reason": "error",
                }
            )
            failed += 1
        else:
            record.update(
                {
                    "status": "ok",
                    "served_model": served_model,
                    "finish_reason": finish_reason,
                    **result,
                }
            )
            generated += 1
        with output.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return {
        "endpoint": endpoint,
        "model": model,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_schema_sha256": PROMPT_SCHEMA_SHA256,
        "input_rows": input_rows,
        "already_present": already_present,
        "generated": generated,
        "failed": failed,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "prompt_sha256": sha256_text(SYSTEM_PROMPT + USER_TEMPLATE),
        "blind_to_reference_labels": True,
        "emits_transcripts_in_report": False,
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse reconciliation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:3456")
    parser.add_argument("--model", default="zai,glm-5.3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--auth-token", default=os.environ.get("PHONON_CCR_TOKEN", "ccr-local")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run reconciliation and print aggregate-only output."""
    try:
        arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
        safe_output_path(arguments.out)
        validate_output_aliases(arguments.out, [arguments.dataset])
        receipt = reconcile(
            arguments.dataset,
            arguments.out,
            endpoint=arguments.endpoint,
            model=arguments.model,
            limit=arguments.limit,
            timeout=arguments.timeout,
            auth_token=arguments.auth_token,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - private-input CLI failure boundary
        print(f"FAIL glm-reconcile: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
