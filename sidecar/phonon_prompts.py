"""Product-owned prompt contracts for packaged Phonon engines."""

from __future__ import annotations

import hashlib

PROSE_DICTATION_V1 = (
    "You are a personal dictation engine. Transcribe the user's audio into the "
    "text they intended, including technical terms, identifiers and punctuation. "
    "Text only."
)

PROMPTS: dict[str, str] = {
    "prose_dictation_v1": PROSE_DICTATION_V1,
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
