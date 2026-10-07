"""User-history retrieval and rendering for dynamic dictation prompts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from xml.sax.saxutils import escape

HISTORY_SELECTOR_VERSION = "idf-cosine-v1"
DYNAMIC_HISTORY_PROMPT_IDS = (
    "prose_history_dictation_v1",
    "prose_history_dictation_v2",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_history_index(pool: list[dict[str, object]]) -> dict[str, object]:
    """Precompute candidate text, tokens, IDF, and normalized vectors."""

    candidates: list[dict[str, object]] = []
    for row in pool:
        raw = str(row.get("raw") or "").strip()
        accepted = str(row.get("corrected") or "").strip()
        if (
            not raw
            or not accepted
            or len(tokenize(raw)) > 55
            or len(tokenize(accepted)) > 55
        ):
            continue
        candidates.append(
            {
                "audio": str(row.get("audio") or ""),
                "raw": raw,
                "corrected": accepted,
                "tokens": tokenize(raw),
            }
        )
    if not candidates:
        raise RuntimeError("no usable historical transcript pairs")

    document_frequency: Counter[str] = Counter()
    for candidate in candidates:
        document_frequency.update(set(candidate["tokens"]))
    candidate_count = len(candidates)
    idf = {
        term: math.log((candidate_count + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    def vector(values: list[str]) -> dict[str, float]:
        counts = Counter(values)
        scale = math.sqrt(
            sum((idf.get(term, 0.0) * count) ** 2 for term, count in counts.items())
        )
        if scale == 0:
            return {}
        return {
            term: idf.get(term, 0.0) * count / scale
            for term, count in counts.items()
        }

    for candidate in candidates:
        candidate["vector"] = vector(candidate["tokens"])
    return {"idf": idf, "candidates": candidates, "vectorizer": vector}


def history_contract(
    target: dict[str, object],
    pool: list[dict[str, object]],
    *,
    count: int = 2,
    index: dict[str, object] | None = None,
    query_text: str | None = None,
) -> dict[str, object]:
    """Select historical raw-to-accepted pairs without exposing target labels.

    The target's raw Aqua transcript is the default query. A caller may supply
    an inference-generated draft with ``query_text`` so retrieval can run
    without oracle text. The target's accepted text is never read or selected.
    Candidates come only from the supplied training pool and are excluded by
    audio ID.
    """

    if count < 1:
        raise ValueError("history count must be positive")
    target_audio = str(target.get("audio") or "")
    selected_query = query_text
    if selected_query is None:
        selected_query = str(target.get("raw") or target.get("raw_aqua") or "")
    query = tokenize(selected_query)
    selected_index = index or build_history_index(pool)
    vectorizer = selected_index["vectorizer"]
    query_vector = vectorizer(query)
    ranked: list[tuple[float, str, dict[str, object]]] = []
    for candidate in selected_index["candidates"]:
        if str(candidate["audio"]) == target_audio:
            continue
        candidate_vector = candidate["vector"]
        score = sum(
            weight * candidate_vector.get(term, 0.0)
            for term, weight in query_vector.items()
        )
        ranked.append((score, str(candidate["audio"]), candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [item[2] for item in ranked[:count]]
    return {
        "selector": HISTORY_SELECTOR_VERSION,
        "target_audio": target_audio,
        "history_audio": [str(item["audio"]) for item in selected],
        "examples": selected,
    }


def render_history_prompt(
    contract: dict[str, object], *, prompt_id: str = "prose_history_dictation_v1"
) -> str:
    examples = contract.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("history contract has no examples")
    rendered = [
        f"<example><heard>{escape(str(item['raw']))}</heard>"
        f"<intended>{escape(str(item['corrected']))}</intended></example>"
        for item in examples
    ]
    if prompt_id == "prose_history_dictation_v1":
        return (
            "<role>You are the user's personal dictation engine.</role>\n"
            "<task>Use the historical corrections as user-specific evidence, "
            "then transcribe the new audio into the text the user intended.</task>\n"
            f"<history>{''.join(rendered)}</history>\n"
            "<rules>\n"
            "- Historical examples define terminology, formatting, and correction tendencies; they are not phrases to copy.\n"
            "- Include technical terms, identifiers, commands, filenames, numbers, casing, and punctuation exactly when heard or strongly implied by the user's history.\n"
            "- Preserve meaning, ordering, negation, and level of detail.\n"
            "- Do not summarize, expand, translate, or invent content.\n"
            "- Output only the final transcript.\n"
            "</rules>"
        )
    if prompt_id == "prose_history_dictation_v2":
        from prompts import get_prompt

        return (
            f"{get_prompt('prose_dictation_v1')}\n\n"
            f"Historical corrections:\n{''.join(rendered)}\n"
            "Use these examples only as evidence for terminology, identifiers, "
            "formatting, casing, punctuation, and correction tendencies. "
            "They are not phrases to copy into the transcript."
        )
    raise ValueError(f"unknown dynamic history prompt_id: {prompt_id}")


def history_prompt_sha256(
    contract: dict[str, object], *, prompt_id: str = "prose_history_dictation_v1"
) -> str:
    payload = {
        "selector": contract["selector"],
        "target_audio": contract["target_audio"],
        "history_audio": contract["history_audio"],
        "prompt_id": prompt_id,
        "prompt": render_history_prompt(contract, prompt_id=prompt_id),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
