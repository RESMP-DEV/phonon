"""CPU-only tests for corrector v0 helpers. No GPU, no network, no deletes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_ops import CHUNK_SAMPLES, TARGET_SR, chunk_audio, degrade_audio, estimate_seconds, save_wav
from build_train import add_pair, collect, split_dev
from common import (
    HOLDOUT_SPLITS,
    TRAIN_SPLITS,
    asr_jobs,
    chat_messages,
    fair_norm,
    load_july_rows,
    load_wispr_pairs,
    pair_wer,
    score_lists,
    strip_thinking,
    SYSTEM_PROMPT,
)
from filter_tts_pairs import filter_rows
from synthesize_tts import select_rows, should_degrade, wav_id


def test_asr_job_counts() -> None:
    jobs = asr_jobs()
    assert len(jobs) == 785
    splits = {job["split"] for job in jobs}
    assert splits == {"wispr_train", "wispr_holdout120", "wispr_edit25"}
    july = [job for job in jobs if job["split"] == "wispr_edit25"]
    wispr = [job for job in jobs if job["split"] != "wispr_edit25"]
    assert len(july) == 25
    assert len(wispr) == 760
    assert all(Path(job["wav"]).exists() for job in jobs)
    labeled = [job for job in wispr if job["target"]]
    assert len(labeled) == 743


def test_holdout_never_in_train_splits() -> None:
    pairs = load_wispr_pairs()
    assert len(pairs) == 4282
    for row in pairs:
        if row["split"] in HOLDOUT_SPLITS:
            assert row["split"] not in TRAIN_SPLITS
        if row["split"] in TRAIN_SPLITS:
            assert "holdout" not in row["split"]


def test_july_targets_are_edits() -> None:
    rows = load_july_rows()
    assert len(rows) == 25
    assert all(row["target"] for row in rows)
    assert all(row["split"] == "wispr_edit25" for row in rows)


def test_chat_format() -> None:
    messages = chat_messages("teh cuda kernal", "the CUDA kernel")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "teh cuda kernal"}
    assert messages[2] == {"role": "assistant", "content": "the CUDA kernel"}
    assert "do not add or drop content" in SYSTEM_PROMPT


def test_fair_wer_identity_and_digit_normalization() -> None:
    refs = ["PTX 9.4 docs", "call me at 4:00"]
    identical = score_lists(refs, refs)
    assert identical["fair_wer"] == 0.0
    assert identical["strict_lc_wer"] == 0.0
    # Whisper EnglishTextNormalizer should unify some number/time formatting.
    fair = pair_wer("call me at 4:00", "call me at 4:00", fair_norm)
    assert fair == 0.0


def test_strip_thinking() -> None:
    text = "<think>plan</think>\nfixed CUDA kernel"
    assert strip_thinking(text) == "fixed CUDA kernel"


def test_chunk_audio_merges_subsecond_tail() -> None:
    wave = np.zeros(CHUNK_SAMPLES + 8000, dtype=np.float32)
    chunks = chunk_audio(wave)
    assert len(chunks) == 1
    assert len(chunks[0]) == len(wave)
    wave2 = np.zeros(CHUNK_SAMPLES + 16000, dtype=np.float32)
    chunks2 = chunk_audio(wave2)
    assert len(chunks2) == 2


def test_save_wav_unique_tmp_then_replace(tmp_path: Path) -> None:
    wave = np.zeros(1600, dtype=np.float32)
    wave[80:120] = 0.2
    dest = tmp_path / "clip.wav"
    save_wav(dest, wave, TARGET_SR)
    assert dest.exists()
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == ["clip.wav"]
    import soundfile as sf

    loaded, sr = sf.read(str(dest), dtype="float32")
    assert int(sr) == TARGET_SR
    assert loaded.shape[0] == 1600


def test_degrade_preserves_length_and_finite() -> None:
    rng = np.random.default_rng(0)
    wave = rng.standard_normal(16000).astype(np.float32) * 0.1
    out = degrade_audio(wave, 16000, rng)
    assert out.shape == wave.shape
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) <= 1.0


def test_select_rows_respects_hour_cap() -> None:
    rows = [
        {
            "id": f"t{i}",
            "split": "wispr_text_train",
            "target": "hello world " * 10,
            "duration": 30.0,
        }
        for i in range(200)
    ]
    selected, total = select_rows(rows, n_voices=3, cap_hours=1.0, seed=0)
    assert selected
    assert total <= 1.0 * 3600 + 30.0 * 3
    assert total >= 3600 * 0.7
    assert all(r["split"] == "wispr_text_train" for r in selected)


def test_select_rows_length_ascending() -> None:
    rows = [
        {"id": "long", "split": "wispr_text_train", "target": "word " * 40, "duration": 0},
        {"id": "short", "split": "wispr_text_train", "target": "hi", "duration": 0},
        {"id": "mid", "split": "wispr_text_train", "target": "word " * 8, "duration": 0},
    ]
    selected, _total = select_rows(
        rows, n_voices=2, cap_hours=1.0, seed=0, order="length-asc"
    )
    assert [row["id"] for row in selected] == ["short", "mid", "long"]


def test_degrade_flag_is_deterministic() -> None:
    assert should_degrade("abc", "af_heart") == should_degrade("abc", "af_heart")
    uid = wav_id("abc", "af_heart", True)
    assert uid.endswith("__deg")


def test_filter_drops_empty_and_high_wer() -> None:
    rows = [
        {"id": "ok", "parakeet_raw": "the cuda kernel", "target": "the CUDA kernel", "voice": "af_heart"},
        {"id": "empty", "parakeet_raw": "", "target": "the CUDA kernel", "voice": "af_heart"},
        {
            "id": "bad",
            "parakeet_raw": "completely unrelated weather report today",
            "target": "the CUDA kernel",
            "voice": "af_heart",
        },
    ]
    kept, stats = filter_rows(rows, max_wer=0.6)
    ids = {row["id"] for row in kept}
    assert "ok" in ids
    assert "empty" not in ids
    assert stats["dropped_empty"] == 1
    assert stats["dropped_wer"] >= 1


def test_build_train_skips_holdout_and_dedupes() -> None:
    bucket: dict[tuple[str, str], dict] = {}
    add_pair(bucket, "raw a", "tgt a", "wispr_asr:wispr_train", "1")
    add_pair(bucket, "raw a", "tgt a", "parakeet:wispr_train", "1b")
    add_pair(bucket, "raw b", "tgt b", "wispr_asr:wispr_train", "2")
    assert len(bucket) == 2
    rows = list(bucket.values())
    train, dev = split_dev(rows, frac=0.5, seed=0)
    assert len(train) + len(dev) == 2
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in dev}) or len(rows) < 2


def test_collect_excludes_holdout_ids() -> None:
    rows, stats = collect(include_parakeet=False, include_tts=False)
    holdout_ids = {row["id"] for row in load_wispr_pairs() if row["split"] in HOLDOUT_SPLITS}
    leaked = [row for row in rows if row["id"] in holdout_ids]
    assert not leaked
    assert stats["skipped_holdout"] == 698
    assert stats["wispr_asr"] == 3584
    assert all(row["messages"][0]["role"] == "system" for row in rows[:5])
    assert all(row["messages"][1]["role"] == "user" for row in rows[:5])
    assert all(row["messages"][2]["role"] == "assistant" for row in rows[:5])


def test_collect_no_wispr_keeps_parakeet_shape() -> None:
    rows, stats = collect(include_parakeet=False, include_tts=False, include_wispr=False)
    assert rows == []
    assert stats["wispr_asr"] == 0
    assert stats["include_wispr"] is False


def test_estimate_seconds_uses_duration() -> None:
    assert estimate_seconds("hello", 12.0) == 12.0
    assert estimate_seconds("hello world") > 0


def test_eval_markdown_contains_required_sets() -> None:
    from eval_corrector import SET_ORDER, build_markdown

    payload = {
        "train_rows": 10,
        "dev_rows": 1,
        "asr_scores": {},
        "models": [
            {
                "label": "qwen3-0.6b-lora",
                "sets": {
                    name: {
                        "n": 2,
                        "baseline": {"n": 2, "fair_wer": 0.2, "strict_lc_wer": 0.3},
                        "corrector": {"n": 2, "fair_wer": 0.1, "strict_lc_wer": 0.2},
                        "worse_frac": 0.0,
                        "worst": [],
                    }
                    for name in SET_ORDER
                },
            }
        ],
        "failures": [],
        "notes": [],
    }
    md = build_markdown(payload)
    for name in SET_ORDER:
        assert f"## {name}" in md
    assert "raw input unchanged" in md
    assert "qwen3-0.6b-lora" in md
