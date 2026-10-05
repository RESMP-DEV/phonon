from __future__ import annotations

from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "train_reverse_audio_vl.py").read_text()


def test_batch_size_flags_present() -> None:
    assert '--batch-size' in SOURCE
    assert '--gradient-accumulation' in SOURCE
    assert 'batch_size=args.batch_size' in SOURCE
    assert 'args.gradient_accumulation' in SOURCE


def test_conformer_blocks_flag_present() -> None:
    assert '--conformer-blocks' in SOURCE
    assert '--conformer-lr' in SOURCE
    assert 'conformer_blocks=args.conformer_blocks' in SOURCE
    assert '[-args.conformer_blocks' in SOURCE


def test_collator_import_passes_batch_size() -> None:
    """The imported collator already supports real batch dimensions."""

    assert 'batch_size=args.batch_size' in SOURCE
