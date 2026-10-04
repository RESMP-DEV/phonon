from __future__ import annotations

from sidecar.phonon_prompts import PROMPTS, get_prompt, prompt_sha256


def test_product_owned_prose_prompt_is_byte_stable() -> None:
    assert set(PROMPTS) == {"prose_dictation_v1"}
    assert prompt_sha256("prose_dictation_v1") == (
        "3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5"
    )


def test_unknown_product_prompt_is_rejected() -> None:
    try:
        get_prompt("xml_transcript_v1")
    except ValueError as error:
        assert "unknown prompt_id" in str(error)
    else:
        raise AssertionError("unknown product prompt was accepted")
