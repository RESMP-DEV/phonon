from __future__ import annotations

import re

import measure_gpu_budget as probe
import pytest


def test_infonce_matches_manual_ranking_loss() -> None:
    """The probe must compute the official in-batch ranking objective."""
    torch = pytest.importorskip("torch", reason="GPU probe needs the training env")
    torch.manual_seed(0)
    anchor = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    positive = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    negative = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)

    got = probe.infonce(anchor, positive, negative)

    sim_pos = anchor @ positive.T
    sim_neg = anchor @ negative.T
    labels = torch.arange(3)
    want = torch.nn.functional.cross_entropy(
        torch.cat([sim_pos, sim_neg], dim=1), labels
    )
    assert torch.allclose(got, want)


def test_text_tower_targets_exclude_the_audio_tower() -> None:
    """A bare "linear" target matches only the audio tower and yields no gradient."""
    pattern = re.compile(probe.TEXT_TOWER_TARGETS)
    audio = "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear"

    assert pattern.fullmatch(audio) is None
    assert pattern.fullmatch("language_model.layers.0.self_attn.q_proj") is not None
    assert pattern.fullmatch("language_model.layers.23.mlp.down_proj") is not None


def test_parse_cases_accepts_modes_and_rejects_bad_specs() -> None:
    assert probe.parse_cases("8:512:lora, 16:512:full") == [
        (8, 512, "lora", False),
        (16, 512, "full", False),
    ]
    assert probe.parse_cases("8:512:lora:ckpt") == [(8, 512, "lora", True)]

    for bad in ("8:512:bogus", "8:lora", "", "8:512:lora:ckpt:extra"):
        with pytest.raises(probe.MeasureError):
            probe.parse_cases(bad)


def test_probe_is_synthetic_only_for_rented_providers() -> None:
    """The probe must never reference private corpus inputs or paths."""
    with open(probe.__file__, encoding="utf-8") as handle:
        source = handle.read()

    for forbidden in ("private-v2", "aqua-lexicon", "split-manifest", ".jsonl"):
        assert forbidden not in source
    assert "randint" in source  # synthetic ids are the only training input
