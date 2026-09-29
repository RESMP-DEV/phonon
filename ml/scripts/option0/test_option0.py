"""Small CPU-only checks for gate integrity, recovery, and fair scoring."""

import hashlib
import io
import json
from pathlib import Path
import sys
import wave

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates  # noqa: E402
import materialize_gate_audio as materializer  # noqa: E402
import score_gates as scorer  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_actual_fair_normalizer_unifies_spoken_numbers():
    refs = ["I bought twenty one GPUs for forty two dollars."]
    hypotheses = ["i bought 21 GPUs for 42 dollars"]
    result = scorer.text_metrics(refs, hypotheses)
    assert result["fair_wer"] == 0
    assert result["strict_lowercase_wer"] > 0
    assert result["project_word_wer"] > 0
    assert scorer.text_metrics(["hello"], ["goodbye"])["fair_wer"] == 1


@pytest.mark.parametrize("samples,expected", [
    (480000, [480000]), (480256, [480256]), (495999, [495999]),
    (496000, [480000, 16000]), (960256, [480000, 480256]),
])
def test_chunks_preserve_samples_without_millisecond_tail(samples, expected):
    wave = bytes(samples)
    chunks = gates.chunk_audio(wave)
    assert [len(chunk) for chunk in chunks] == expected
    assert b"".join(chunks) == wave
    assert max(map(len, chunks)) <= 31 * 16000


@pytest.fixture
def scored_fixture(tmp_path):
    rows = [{"id": "a", "reference": "hello CUDA", "insert_text": None,
             "technical_terms": ["CUDA"]}]
    prediction = tmp_path / "model.jsonl"
    write_jsonl(prediction, [{"id": "a", "hypothesis": "hello CUDA"}])
    meta = {"complete": True, "gate": "aqua_new_holdout", "clips": 1,
            "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            "audio_seconds": 10.0, "elapsed_seconds": 2.0, "realtime_factor": 0.2,
            "peak_vram_bytes": 1_000_000_000}
    return rows, prediction, meta


def test_valid_metadata_and_none_insert_label(scored_fixture):
    rows, prediction, meta = scored_fixture
    result, diffs = scorer.score_prediction("aqua_new_holdout", rows, prediction, meta)
    assert result["complete"]
    assert result["fair_wer"] == 0
    assert result["insert_clip_count"] == 0
    assert result["technical_term_error_rate"] == 0
    assert result["realtime_factor"] == 0.2
    assert len(diffs) == 1


@pytest.mark.parametrize("configuration", [
    {"family": "transformers-qwen3asr", "model": "Qwen/Qwen3-ASR-1.7B-hf"},
    {"family": "transformers-granite", "model": "ibm-granite/granite-speech-4.1-2b-plus"},
])
def test_superseded_inference_configuration_is_not_scored(scored_fixture, configuration):
    rows, prediction, meta = scored_fixture
    result, diffs = scorer.score_prediction(
        "aqua_new_holdout", rows, prediction, {**meta, **configuration})
    assert result["status"] == "invalid"
    assert "fair_wer" not in result
    assert not diffs


@pytest.mark.parametrize("configuration", [
    {"family": "transformers-qwen3asr", "model": "Qwen/Qwen3-ASR-1.7B-hf",
     "language_mode": "auto"},
    {"family": "transformers-granite", "model": "ibm-granite/granite-speech-4.1-2b-plus",
     "prompt_policy": "granite_plus_model_card_asr"},
])
def test_final_inference_configuration_is_scored(scored_fixture, configuration):
    rows, prediction, meta = scored_fixture
    result, _ = scorer.score_prediction(
        "aqua_new_holdout", rows, prediction, {**meta, **configuration})
    assert result["complete"]
    assert result["fair_wer"] == 0


def test_finalized_subset_retains_timing_without_full_gate_mean(scored_fixture):
    rows, prediction, meta = scored_fixture
    rows = [*rows, {"id": "missing", "reference": "missing audio", "technical_terms": []}]
    result, diffs = scorer.score_prediction("aqua_new_holdout", rows, prediction, meta)
    assert result["status"] == "partial"
    assert not result["complete"]
    assert result["metadata_valid"]
    assert result["clip_count"] == 1 and result["expected_clips"] == 2
    assert result["missing_prediction_ids"] == ["missing"]
    assert result["fair_wer"] == 0 and len(diffs) == 1
    assert result["realtime_factor"] == 0.2
    assert result["peak_vram_gb"] == 1
    model = scorer.model_record("test", "test", None)
    model["gates"]["aqua_new_holdout"] = result
    scorer.summarize_model(model)
    assert model["realtime_factor"] == 0.2
    assert model["peak_vram_gb"] == 1
    assert model["timed_gates"] == ["aqua_new_holdout"]
    assert model["five_gate_mean_fair_wer"] is None


@pytest.mark.parametrize("change", [
    {"complete": False}, {"clips": 2}, {"gate": "hard77"},
    {"prediction_sha256": "wrong"}, {"prediction_ids": ["wrong"]},
])
def test_stale_metadata_cannot_supply_timing_or_full_mean(scored_fixture, change):
    rows, prediction, meta = scored_fixture
    result, _ = scorer.score_prediction("aqua_new_holdout", rows, prediction, {**meta, **change})
    assert result["status"] == "partial"
    assert result["metadata_errors"]
    assert result["fair_wer"] == 0
    assert result["realtime_factor"] is None
    assert result["peak_vram_gb"] is None
    model = scorer.model_record("test", "test", None)
    model["gates"]["aqua_new_holdout"] = result
    scorer.summarize_model(model)
    assert model["realtime_factor"] is None
    assert model["five_gate_mean_fair_wer"] is None


def test_corrupt_sidecar_is_reported_without_accepting_prediction(tmp_path, monkeypatch):
    pred_dir = tmp_path / "preds"
    gate_dir = pred_dir / "aqua_new_holdout"
    gate_dir.mkdir(parents=True)
    pred = gate_dir / "parakeet-tdt-0.6b-v2.jsonl"
    write_jsonl(pred, [{"id": "a", "hypothesis": "hello"}])
    Path(str(pred) + ".meta.json").write_text("{broken")
    monkeypatch.setattr(scorer, "GATE_NAMES", ("aqua_new_holdout",))
    monkeypatch.setattr(scorer, "load_gate", lambda *args, **kwargs: [
        {"id": "a", "reference": "hello"}])
    monkeypatch.setattr(scorer, "gate_summary", lambda name: {
        "rows": 1, "labeled_rows": 1, "found": 1, "dropped_unlabeled": 0})
    monkeypatch.setattr(sys, "argv", ["score_gates.py", "--pred-dir", str(pred_dir)])
    scorer.main()
    report = json.loads((tmp_path / "results.json").read_text())
    assert any("JSONDecodeError" in error for error in report["errors"])
    assert not report["models"][0]["gates"]
    assert report["models"][0]["realtime_factor"] is None


def test_hidden_fallback_uses_gold_or_consensus_never_teacher(tmp_path, monkeypatch):
    queue = tmp_path / "hidden120.jsonl"
    gold = tmp_path / "youtube_technical_v0_yt_technical_hidden_v0.jsonl"
    ids = [f"youtube:video:seg{i:05}" for i in range(4)]
    write_jsonl(queue, [{"id": row_id, "teacher_text": "machine guessed text",
                         "verbatim_text": "unreviewed teacher" if row_id == ids[2] else "",
                         "label_status": "draft"} for row_id in ids])
    write_jsonl(gold, [
        {"id": ids[0], "label_status": "gold", "verbatim_text": "human gold"},
        {"id": ids[1], "consensus_text": "reviewed consensus"},
        {"id": ids[2], "label_status": "teacher", "verbatim_text": "teacher transcript"},
    ])
    before = queue.read_bytes(), gold.read_bytes()
    monkeypatch.setattr(gates, "QUEUES", {"yt_hidden120": queue})
    monkeypatch.setattr(gates, "QUEUE_ROOT", tmp_path)
    monkeypatch.setattr(gates, "audio_path", lambda row: tmp_path / (row["id"] + ".wav"))
    rows = gates.load_gate("yt_hidden120")
    assert [row["reference"] for row in rows] == ["human gold", "reviewed consensus"]
    assert [row["id"] for row in rows] == ids[:2]
    assert (queue.read_bytes(), gold.read_bytes()) == before


def test_materializer_checks_id_hash_skips_corrupt_shards_and_is_idempotent(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()
    shards = tmp_path / "shards"
    shards.mkdir()
    restored_shards = tmp_path / "restored_shards"
    restored_shards.mkdir()
    report = tmp_path / "report"
    output = storage / "video_seg00001.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    audio_bytes = buffer.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    row_id = "youtube:video:seg00001"
    needed = {"id": row_id, "sha256_audio": digest, "audio_path": str(output)}
    (shards / "000_corrupt.parquet").write_bytes(b"unfinished parquet download")
    schema = pa.schema([
        ("id", pa.string()), ("sha256_audio", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ])
    records = [
        {"id": "youtube:other:seg00001", "sha256_audio": digest,
         "audio": {"bytes": b"invalid audio must not be read", "path": None}},
        {"id": row_id, "sha256_audio": "wrong hash",
         "audio": {"bytes": b"invalid audio must not be read", "path": None}},
        {"id": row_id, "sha256_audio": digest,
         "audio": {"bytes": audio_bytes, "path": "original.wav"}},
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=schema), restored_shards / "001_data.parquet",
                   row_group_size=1)
    monkeypatch.setattr(materializer, "ensure_storage", lambda: storage)
    monkeypatch.setattr(materializer, "QUEUES", {"testgate": tmp_path / "unused.jsonl"})
    monkeypatch.setattr(materializer, "GATE_NAMES", ("testgate",))
    monkeypatch.setattr(materializer, "REPORT", report)
    monkeypatch.setattr(materializer, "raw_gate", lambda name: [needed])
    monkeypatch.setattr(materializer, "audio_path", lambda row: Path(row["audio_path"]))

    def summary(name):
        found = int(materializer.hash_matches(output, digest))
        return {"rows": 1, "found": found, "missing": 1 - found, "dropped_unlabeled": 0}

    monkeypatch.setattr(materializer, "gate_summary", summary)
    monkeypatch.setattr(sys, "argv", ["materialize_gate_audio.py", "--parquet-dirs",
                                    str(shards), str(restored_shards)])
    real_parquet = pq.ParquetFile
    reads = []

    class TrackedParquet:
        def __init__(self, path):
            self.file = real_parquet(path)
            self.num_row_groups = self.file.num_row_groups

        def read_row_group(self, group, columns):
            reads.append((group, tuple(columns)))
            return self.file.read_row_group(group, columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", TrackedParquet)
    materializer.main()
    assert output.read_bytes() == audio_bytes
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert reads == [(0, ("id", "sha256_audio")), (1, ("id", "sha256_audio")),
                     (2, ("id", "sha256_audio")), (2, ("audio",))]
    first_report = json.loads((report / "missing_audio.json").read_text())
    assert first_report["written_this_pass"] == 1
    assert first_report["parquet_dirs"] == [str(shards), str(restored_shards)]
    assert len(first_report["scanned_shards"]) == 2
    assert len(first_report["parquet_errors"]) == 1
    assert "000_corrupt.parquet" in first_report["parquet_errors"][0]["path"]
    assert first_report["gates"]["testgate"]["missing"] == 0
    original_mtime = output.stat().st_mtime_ns
    reads.clear()
    materializer.main()
    assert output.stat().st_mtime_ns == original_mtime
    assert not reads
    assert json.loads((report / "missing_audio.json").read_text())["written_this_pass"] == 0
