"""Product-owned prompt contracts for packaged Phonon engines."""

from __future__ import annotations

import hashlib

PROSE_DICTATION_V1 = (
    "You are a personal dictation engine. Transcribe the user's audio into the "
    "text they intended, including technical terms, identifiers and punctuation. "
    "Text only."
)


# Phonon's own dictation prompt with an explicit history-grounding section. The
# history block is supplied by the caller from a local, consent-gated store; no
# personal text is committed with this file.
PROSE_DICTATION_HISTORY_V1 = (
    "You are a personal dictation engine. Transcribe the user's audio into the "
    "text they intended, including technical terms, identifiers, filenames, "
    "numbers, casing, and punctuation. Preserve meaning, ordering, negation, "
    "and level of detail. Use the user's historical corrections as evidence for "
    "recurring spellings, formatting, and correction tendencies, not as phrases "
    "to copy. When the audio is ambiguous, prefer the spelling the user's "
    "history supports; otherwise keep the acoustically plausible wording. Do "
    "not summarize, expand, translate, or invent content. Output only the final "
    "transcript."
)

SCREENSHOT_DICTATION_V1 = (
    "You are a personal dictation engine. Transcribe the text in the user's "
    "screenshot exactly, including technical terms, identifiers and "
    "punctuation. Text only."
)

PROMPTS: dict[str, str] = {
    "prose_dictation_v1": PROSE_DICTATION_V1,
    "prose_dictation_history_v1": PROSE_DICTATION_HISTORY_V1,
    "screenshot_dictation_v1": SCREENSHOT_DICTATION_V1,
}


def get_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as error:
        raise ValueError(
            f"unknown prompt_id {prompt_id!r}; choose one of {sorted(PROMPTS)}"
        ) from error


def prompt_sha256(prompt_id: str) -> str:
    return hashlib.sha256(get_prompt(prompt_id).encode()).hexdigest()
