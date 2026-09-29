"""Read-only evaluation gates and shared, 16 kHz mono audio decoding."""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
SEGMENTS = Path('/home/user/datasets/phonon/segments/youtube')
REPORT = ROOT / 'runs/reports/option0_sweep_20260915'
QUEUE_ROOT = ROOT / 'datasets/labels/queues'
QUEUES = {
    'novel180': QUEUE_ROOT / 'youtube_real_audio_v2_progress_segments12_novel_core600_probe180_patched_blind_gap_reference_20260525.jsonl',
    'hard77': QUEUE_ROOT / 'youtube_real_audio_v2_progress_segments5_qwen36_hard_exact_terms_top77_redrafts_with_avalon_merged_20260525.jsonl',
    'uncertain48': QUEUE_ROOT / 'youtube_real_audio_v2_progress_segments24_fresh_uncertain_hard_audit_top80_20260525.jsonl',
    'course91': ROOT / 'runs/reports/course_chunks_86FAWCzIe_4_preserve_course/course_agent_preserve_examples_queue.jsonl',
    'yt_hidden120': QUEUE_ROOT / 'youtube_technical_v0_yt_technical_hidden_v0_teacher_scored_top120.jsonl',
}
FIVE_GATES = ('aqua_new_holdout', 'course91', 'novel180', 'hard77', 'uncertain48')
PERSONAL_ROOT = Path('/data/phonon_personal')
PERSONAL_GATES = ('wispr_holdout120', 'wispr_edit25', 'personal_cuda')
GATE_NAMES = (*FIVE_GATES, 'yt_hidden120', *PERSONAL_GATES)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_text(row, label_field='consensus_text'):
    value = (row.get(label_field) or row.get('insert_text') or row.get('verbatim_text')
             or row.get('consensus_text'))
    return ' '.join(str(value or '').split())


def audio_path(row):
    if row.get('audio_path'):
        path = Path(row['audio_path'])
        if not path.is_absolute():
            path = ROOT / path
        old = '/home/user/benchmarks/phonon/'
        if not path.exists() and str(path).startswith(old):
            path = ROOT / str(path)[len(old):]
        return path
    parts = row['id'].split(':')
    if len(parts) == 3 and parts[0] == 'youtube':
        return SEGMENTS / parts[-2] / f'{parts[-2]}_{parts[-1]}.wav'
    return None


def raw_gate(name, include_audio=False):
    if name in {'wispr_holdout120', 'wispr_edit25'}:
        if name == 'wispr_holdout120':
            base = PERSONAL_ROOT / 'wispr_20260915'
            rows = read_jsonl(base / 'wispr_holdout120.jsonl')
            for row in rows:
                row['id'] = row['transcriptEntityId']
                row['audio_path'] = str(base / row['audio_path'])
                row['consensus_text'] = row['reference_text']
        else:
            base = PERSONAL_ROOT / 'dictation_eval_20260719'
            rows = json.loads((base / 'edit_samples.json').read_text())
            for row in rows:
                row['audio_path'] = str(base / 'audio' / Path(row['audio_wav']).name)
                row['consensus_text'] = row['edited'].strip()
        for row in rows:
            row['train_allowed'] = False
            row['eval_allowed'] = True
            path = Path(row['audio_path'])
            if path.is_file():
                row['sha256_audio'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return rows
    if name == 'aqua_new_holdout':
        from phonon.exclusions import filter_excluded_rows
        from phonon.schema import read_parquet_rows
        rows = read_parquet_rows(ROOT / 'datasets/datasets/aqua_voice_seed_v0', name,
                                 include_audio=include_audio)
        rows, _ = filter_excluded_rows(rows, ROOT / 'datasets')
        return rows
    if name == 'personal_cuda':
        paths = sorted((ROOT / 'curation/hard_terms_v0/personal_cuda').rglob(
            'personal_cuda_aligned_holdout_queue.jsonl'))
        if len(paths) > 1:
            raise ValueError(f'Ambiguous personal holdout: {paths}')
        return read_jsonl(paths[0]) if paths else []
    if name not in QUEUES:
        raise ValueError(f'Unknown gate {name}')
    rows = read_jsonl(QUEUES[name])
    if name == 'yt_hidden120':
        fallback = {r['id']: r for r in read_jsonl(
            QUEUE_ROOT / 'youtube_technical_v0_yt_technical_hidden_v0.jsonl')}
        for row in rows:
            if trusted_hidden_label(row) and row_text(row):
                row['reference_eligible'] = True
                continue
            gold = fallback.get(row['id'], {})
            row['reference_eligible'] = False
            if trusted_hidden_label(gold) and row_text(gold):
                for key in ('consensus_text', 'insert_text', 'verbatim_text'):
                    row.pop(key, None)
                for key in ('consensus_text', 'insert_text', 'verbatim_text', 'technical_terms',
                            'term_spans', 'label_status'):
                    if gold.get(key):
                        row[key] = gold[key]
                row['label_source'] = 'hidden_gold_fallback'
                row['reference_eligible'] = True
    return rows


def trusted_hidden_label(row):
    return (bool(str(row.get('consensus_text') or '').strip())
            or row.get('human_locked') is True
            or str(row.get('label_status', '')).lower() in {
                'gold', 'locked', 'human', 'human_verified', 'human_locked'})


def load_gate(name, include_audio=False):
    rows = []
    for original in raw_gate(name, include_audio):
        row = dict(original)
        if row.get('reference_eligible') is False:
            continue
        field = 'verbatim_text' if name == 'aqua_new_holdout' else 'consensus_text'
        ref = row_text(row, field)
        if not ref:
            continue
        row['reference'] = ref
        row['reference_field'] = field
        path = audio_path(row)
        row['audio_path'] = str(path) if path else None
        terms = row.get('technical_terms') or row.get('term_spans') or []
        row['technical_terms'] = [
            str(t.get('term') or t.get('text') or '') if isinstance(t, dict) else str(t)
            for t in terms
        ]
        rows.append(row)
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate ids in gate {name}')
    return rows


def hash_matches(path, expected):
    path = Path(path)
    return path.is_file() and bool(expected) and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def gate_summary(name):
    raw = raw_gate(name)
    rows = load_gate(name)
    missing = []
    for row in rows:
        if name != 'aqua_new_holdout' and not hash_matches(
                row['audio_path'], row.get('sha256_audio')):
            missing.append(row['id'])
    return {'rows': len(raw), 'labeled_rows': len(rows), 'dropped_unlabeled': len(raw) - len(rows),
            'found': len(rows) - len(missing), 'missing': len(missing), 'missing_ids': missing,
            'included': bool(rows) and (name != 'personal_cuda' or not missing)}


def load_audio(row):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio = row.get('audio') or {}
    source = io.BytesIO(audio['bytes']) if audio.get('bytes') else row['audio_path']
    if source is None:
        source = audio.get('path')
    waveform, sr = sf.read(source, dtype='float32', always_2d=True)
    waveform = waveform.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sr // divisor).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f'Invalid audio: {row["id"]}')
    return waveform


def chunk_audio(waveform, chunk_samples=480000, minimum_tail_samples=16000):
    """Keep nominal 30 s chunks, merging a sub-second tail into its predecessor."""
    chunks = [waveform[i:i + chunk_samples]
              for i in range(0, len(waveform), chunk_samples)]
    if len(chunks) > 1 and len(chunks[-1]) < minimum_tail_samples:
        # Some source WAVs are 30.016 s despite 30 s queue metadata. Decoding their
        # 16 ms remainder alone breaks STFT and creates spurious empty-audio output.
        chunks[-2:] = [waveform[(len(chunks) - 2) * chunk_samples:]]
    return chunks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    summaries = {g: gate_summary(g) for g in GATE_NAMES}
    if args.ready:
        print(' '.join(g for g, s in summaries.items() if s['included'] and s['found']))
    else:
        print(json.dumps(summaries, indent=2))
