"""CPU checks for personal gate mapping and preserving inference across resumes."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates  # noqa: E402
import transcribe_gate as transcriber  # noqa: E402


def test_personal_paths_references_and_eval_only(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, 'PERSONAL_ROOT', tmp_path)
    holdout = tmp_path / 'wispr_20260915'
    edits = tmp_path / 'dictation_eval_20260719'
    for base in (holdout, edits):
        (base / 'audio').mkdir(parents=True)
        (base / 'audio/one.wav').write_bytes(b'original audio bytes')
    source = holdout / 'wispr_holdout120.jsonl'
    source.write_text(json.dumps({'transcriptEntityId': 'one', 'audio_path': 'audio/one.wav',
                                  'reference_text': 'Use CUDA.', 'train_allowed': True}) + '\n')
    edit_source = edits / 'edit_samples.json'
    edit_source.write_text(json.dumps([{'id': 'one', 'audio_wav': '/Users/owner/audio/one.wav',
                                       'edited': '  Use CUDA. \n'}]))
    originals = source.read_bytes(), edit_source.read_bytes()
    for gate, base in [('wispr_holdout120', holdout), ('wispr_edit25', edits)]:
        row, = gates.load_gate(gate)
        assert row['id'] == 'one'
        assert row['reference'] == 'Use CUDA.'
        assert row['audio_path'] == str(base / 'audio/one.wav')
        assert row['train_allowed'] is False and row['eval_allowed'] is True
        assert not row['technical_terms']
        assert gates.gate_summary(gate)['found'] == 1
    assert (source.read_bytes(), edit_source.read_bytes()) == originals


def test_resume_accepts_verified_subset_and_appended_records(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    old = {'id': 'old', 'hypothesis': 'CUDA', 'sha256_audio': 'oldhash'}
    path.write_text(json.dumps(old) + '\n')
    old_bytes = path.read_bytes()
    meta = {'model': 'test', 'gate': 'hard77', 'clips': 1, 'complete': True,
            'merge_tail_below_seconds': 1,
            'prediction_sha256': hashlib.sha256(old_bytes).hexdigest()}
    Path(str(path) + '.meta.json').write_text(json.dumps(meta))
    rows = [{'id': 'old', 'sha256_audio': 'oldhash'}, {'id': 'new', 'sha256_audio': 'newhash'}]
    records, previous = transcriber.resumable_predictions(path, 'hard77', 'test', rows)
    assert records == [old] and previous == meta
    assert path.read_bytes() == old_bytes
    new = {'id': 'new', 'hypothesis': 'GPU', 'sha256_audio': 'newhash',
           'inference_identity': ['hard77', 'test', 1]}
    with path.open('a') as stream:
        stream.write(json.dumps(new) + '\n')
    records, _ = transcriber.resumable_predictions(path, 'hard77', 'test', rows)
    assert records == [old, new]
    rows[0]['sha256_audio'] = 'changed'
    with pytest.raises(ValueError, match='audio hash differs'):
        transcriber.resumable_predictions(path, 'hard77', 'test', rows)


def test_resume_refuses_unknown_or_modified_predictions(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps({'id': 'one', 'sha256_audio': 'hash', 'hypothesis': 'CUDA'}) + '\n')
    rows = [{'id': 'one', 'sha256_audio': 'hash'}]
    with pytest.raises(ValueError, match='Unverified'):
        transcriber.resumable_predictions(path, 'hard77', 'test', rows)
    meta = {'model': 'test', 'gate': 'hard77', 'clips': 1, 'complete': True,
            'merge_tail_below_seconds': 1, 'prediction_sha256': 'wrong'}
    Path(str(path) + '.meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='prefix hash mismatch'):
        transcriber.resumable_predictions(path, 'hard77', 'test', rows)
