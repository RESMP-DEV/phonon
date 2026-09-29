"""Build frozen-encoder fusion manifests under /data/phonon_fusion_v0.

Eval uses scripts/option0/gates.py only. YouTube training uses source parquet
flags as written; queues and sidecars are never joined to override permissions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

ROOT = Path('/home/user/phonon')
DATA = Path('/data/phonon_fusion_v0')
AUDIO_DIR = DATA / 'audio'
MANIFEST_DIR = DATA / 'manifests'
WISPR_MANIFEST = Path('/data/phonon_personal/wispr_20260915/manifest_with_audio.jsonl')
WISPR_ROOT = WISPR_MANIFEST.parent
YOUTUBE_DIR = Path('/data/phonon_datasets/youtube_technical_v0/data')
SYNTH_DIR = ROOT / 'datasets/synthetic/tcpgen_tts_v4_20260630'
OPTION0_SEGMENTS = Path('/data/phonon_segments_root/segments')
DEV_FRACTION = 0.02
YOUTUBE_CAP_HOURS = 30.0
SYNTH_CAP_HOURS = 3.0
EVAL_PARQUET_ROOTS = (
    Path('/data/phonon_datasets/option0_supplemental'),
    Path('/data/phonon_datasets/youtube_freecodecamp_real_dev_courses_v0/data'),
    Path('/data/phonon_datasets/youtube_dev_tutorials_real_audio_v0/data'),
    Path('/data/phonon_datasets/youtube_technical_v0/data'),
    ROOT / 'datasets/datasets/aqua_voice_seed_v0/data',
)
YOUTUBE_META_COLUMNS = ('id', 'split', 'train_allowed', 'eval_allowed')
RANKING_DEFINITION = {
    'primary': 'lowest pairwise normalized mutual teacher WER',
    'pairwise_normalized_mutual_wer': (
        'Parse teacher_transcripts as a JSON object, JSON string, or list of '
        '{teacher,text}/{name,transcript} items. For each pair of non-empty '
        'transcripts, score 0.5 * (jiwer.wer(a,b) + jiwer.wer(b,a)). Clip score '
        'is the mean over pairs. Lower is better.'
    ),
    'fallback': (
        'highest existing consensus score among consensus_confidence, '
        'review_score, technical_value_score, consensus_score, teacher_review_score'
    ),
    'labels': 'consensus_text or insert_text from the same parquet row; no fabricated labels',
    'eligibility': (
        'split == train_candidate AND train_allowed is True AND eval_allowed is not True'
    ),
    'permissions': (
        'source parquet flags only; queues and sidecars are not joined to override '
        'train_allowed or eval_allowed'
    ),
    'cap_hours': YOUTUBE_CAP_HOURS,
}

sys.path[:0] = [str(ROOT / 'scripts/option0'), str(ROOT / 'src')]
from gates import (  # noqa: E402
    GATE_NAMES,
    SEGMENTS,
    audio_path,
    gate_summary,
    hash_matches,
    load_audio,
    load_gate,
    raw_gate,
)


def cache_id_for(original_id):
    return hashlib.sha256(str(original_id).encode('utf-8')).hexdigest()


def is_true(value):
    return value is True


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def attempt(name, fn):
    errors = []
    for index in range(2):
        try:
            return fn(), errors
        except Exception as exc:
            item = {
                'attempt': index + 1,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            }
            errors.append(item)
            print(f'FAIL {name} attempt {index + 1}: {item["error"]}', flush=True)
    return None, errors


def terms_of(row):
    terms = row.get('technical_terms') or row.get('term_spans') or []
    out = []
    for term in terms:
        if isinstance(term, dict):
            text = str(term.get('term') or term.get('text') or '').strip()
        else:
            text = str(term).strip()
        if text:
            out.append(text)
    return out


def whitespace(value):
    return ' '.join(str(value or '').split())


def parse_teachers(value):
    if value is None or value == '' or value == {} or value == []:
        return {}
    parsed = value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('utf-8', 'replace')
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {'raw': text}
    if isinstance(parsed, dict):
        return {str(key): whitespace(val) for key, val in parsed.items() if whitespace(val)}
    if isinstance(parsed, list):
        out = {}
        for index, item in enumerate(parsed):
            if isinstance(item, dict):
                text = whitespace(
                    item.get('text') or item.get('transcript') or item.get('hypothesis')
                    or item.get('value'))
                name = item.get('teacher') or item.get('name') or item.get('model') or index
            else:
                text = whitespace(item)
                name = index
            if text:
                out[str(name)] = text
        return out
    text = whitespace(parsed)
    return {'raw': text} if text else {}


def pairwise_normalized_mutual_wer(teachers):
    texts = [text for text in teachers.values() if text]
    if len(texts) < 2:
        return None
    import jiwer
    scores = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            a, b = texts[left], texts[right]
            a_words, b_words = a.split(), b.split()
            if not a_words and not b_words:
                scores.append(0.0)
                continue
            if not a_words or not b_words:
                scores.append(1.0)
                continue
            scores.append(0.5 * (jiwer.wer(a, b) + jiwer.wer(b, a)))
    return sum(scores) / len(scores) if scores else None


def consensus_score(row):
    values = []
    for key in ('consensus_confidence', 'review_score', 'technical_value_score',
                'consensus_score', 'teacher_review_score'):
        value = row.get(key)
        if isinstance(value, (int, float)) and value == value:
            values.append(float(value))
    return max(values) if values else None


def rank_key(item):
    wer = item.get('pairwise_wer')
    score = item.get('consensus_score') or 0.0
    if wer is None:
        return (1, 0.0, -score, item['id'])
    return (0, float(wer), -score, item['id'])


def youtube_label(row):
    return whitespace(row.get('consensus_text') or row.get('insert_text'))


def identity_tokens(row):
    tokens = set()

    def add(value):
        text = whitespace(value)
        if not text or text.lower() in {'youtube', 'true', 'false', 'none', 'null'}:
            return
        tokens.add(text)

    add(row.get('id'))
    add(row.get('transcriptEntityId'))
    add(row.get('sha256_audio'))
    add(row.get('source_id'))
    add(row.get('leakage_group'))
    add(row.get('source_url'))
    add(row.get('speaker_id'))
    add(row.get('channel_group'))
    rid = str(row.get('id') or '')
    parts = rid.split(':')
    if len(parts) >= 3 and parts[0] == 'youtube':
        add(parts[1])
        add(f'youtube:{parts[1]}')
    path = row.get('audio_path')
    if path:
        add(Path(path).name)
        add(Path(path).stem)
    return tokens


def group_key(row):
    for field in ('leakage_group', 'source_id'):
        value = row.get(field)
        if value:
            return str(value)
    rid = str(row['id'])
    parts = rid.split(':')
    if len(parts) >= 3 and parts[0] == 'youtube':
        return parts[1]
    return rid


def is_dev_group(key):
    digest = hashlib.sha256(str(key).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64 < DEV_FRACTION


def parquet_files(*roots):
    files = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        found = sorted(root.glob('*.parquet') if root.is_dir() and root.name == 'data'
                       else root.rglob('*.parquet'))
        if root.is_file() and root.suffix == '.parquet':
            found = [root]
        for path in found:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files


def audio_bytes_from_cell(cell):
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict):
        data = cell.get('bytes')
        if data:
            return bytes(data)
        path = cell.get('path')
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def inspect_duration(path):
    import soundfile as sf
    info = sf.info(str(path))
    if info.samplerate != 16000:
        raise ValueError(f'{path} sample rate {info.samplerate} != 16000')
    if info.channels not in (0, 1):
        raise ValueError(f'{path} channels {info.channels} != 1')
    return info.frames / float(info.samplerate)


def write_waveform(wave, path):
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(wave, dtype=np.float32)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'invalid waveform for {path}')
    tmp = path.parent / (path.stem + '.tmp.wav')
    sf.write(str(tmp), array, 16000, subtype='FLOAT', format='WAV')
    tmp.replace(path)
    duration = inspect_duration(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return duration, digest


def materialize(row, original_id):
    wave = load_audio(row)
    path = AUDIO_DIR / f'{cache_id_for(original_id)}.wav'
    duration, digest = write_waveform(wave, path)
    return path, duration, digest


def candidate_audio_paths(row):
    paths = []
    seen = set()

    def add(path):
        if path is None:
            return
        path = Path(path)
        try:
            resolved = path.resolve() if path.exists() else path
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        paths.append(path)

    add(audio_path(row))
    if row.get('audio_path'):
        add(row['audio_path'])
    rid = str(row.get('id') or '')
    parts = rid.split(':')
    if len(parts) == 3 and parts[0] == 'youtube':
        video, seg = parts[1], parts[2]
        name = f'{video}_{seg}.wav'
        add(SEGMENTS / video / name)
        add(OPTION0_SEGMENTS / 'youtube' / video / name)
    return paths


def usable_source(row, gate):
    audio = row.get('audio') or {}
    if audio.get('bytes'):
        data = audio['bytes']
        expected = row.get('sha256_audio')
        if expected and hashlib.sha256(data).hexdigest() != expected:
            return False, 'hash_mismatch_bytes'
        return True, 'embedded_bytes'
    expected = row.get('sha256_audio')
    for path in candidate_audio_paths(row):
        if not path.is_file():
            continue
        if gate == 'aqua_new_holdout' or not expected:
            return True, str(path)
        if hash_matches(path, expected):
            return True, str(path)
    return False, 'missing'


def bind_source_path(row, path):
    bound = dict(row)
    bound['audio_path'] = str(path)
    bound.pop('audio', None)
    return bound


def manifest_row(original_id, audio_path_out, reference, duration, split, train_allowed,
                 source, terms, sha256_audio, source_split):
    return {
        'id': original_id,
        'audio_path': str(audio_path_out),
        'reference': reference,
        'duration_seconds': duration,
        'split': split,
        'train_allowed': train_allowed,
        'source': source,
        'technical_terms': terms,
        'sha256_audio': sha256_audio,
        'cache_id': cache_id_for(original_id),
        'source_split': source_split,
    }


def empty_gate_summary(error=None):
    return {
        'rows': 0,
        'labeled_rows': 0,
        'dropped_unlabeled': 0,
        'found': 0,
        'missing': 0,
        'missing_ids': [],
        'included': False,
        'errors': 0,
        'error_ids': [],
        'absent': True,
        'load_errors': error or [],
        'materialized': 0,
        'hours': 0.0,
    }


def collect_exclusions():
    keys = set()
    per_gate = {}
    holdout_ids = set()
    holdout_hashes = set()
    errors = []
    for gate in GATE_NAMES:
        def load():
            return list(raw_gate(gate))
        rows, load_errors = attempt(f'raw_gate:{gate}', load)
        if load_errors:
            errors.extend({'gate': gate, **item} for item in load_errors)
        if rows is None:
            per_gate[gate] = {'raw_rows': 0, 'load_errors': load_errors}
            raise RuntimeError(f'Cannot establish train exclusions for raw gate {gate}: {load_errors}')
        tokens = set()
        for row in rows:
            tokens |= identity_tokens(row)
            keys |= identity_tokens(row)
            if gate in {'wispr_holdout120', 'wispr_edit25'} or row.get('eval_allowed') is True:
                holdout_ids.add(str(row.get('id') or ''))
                digest = row.get('sha256_audio')
                if digest:
                    holdout_hashes.add(digest)
        per_gate[gate] = {'raw_rows': len(rows), 'identity_tokens': len(tokens)}
    if WISPR_MANIFEST.is_file():
        for line in WISPR_MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'wispr_holdout120' and row.get('eval_allowed') is not True:
                continue
            rid = str(row.get('transcriptEntityId') or row.get('id') or '')
            holdout_ids.add(rid)
            keys |= identity_tokens({**row, 'id': rid})
            rel = row.get('audio_path')
            path = WISPR_ROOT / rel if rel and not Path(rel).is_absolute() else Path(rel or '')
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                holdout_hashes.add(digest)
                keys.add(digest)
    keys |= holdout_ids
    keys |= holdout_hashes
    keys.discard('')
    return {
        'keys': keys,
        'per_gate': per_gate,
        'holdout_ids': sorted(x for x in holdout_ids if x),
        'holdout_hash_count': len(holdout_hashes),
        'token_count': len(keys),
        'errors': errors,
    }


def option0_name_index():
    index = defaultdict(list)
    seen = set()
    roots = (OPTION0_SEGMENTS, Path(SEGMENTS), Path('/data/phonon_segments_root/option0_recovery'))
    for root in roots:
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for path in root.rglob('*.wav'):
            index[path.name].append(path)
    return index


def recover_from_option0(needed, name_index):
    recovered = {}
    for row_id, row in list(needed.items()):
        expected = row.get('sha256_audio')
        names = []
        parts = str(row_id).split(':')
        if len(parts) == 3 and parts[0] == 'youtube':
            names.append(f'{parts[1]}_{parts[2]}.wav')
        for path in candidate_audio_paths(row):
            names.append(path.name)
        for name in names:
            for path in name_index.get(name, []):
                if not path.is_file():
                    continue
                if expected and not hash_matches(path, expected):
                    continue
                recovered[row_id] = {'path': str(path), 'how': 'option0_materialized'}
                break
            if row_id in recovered:
                break
    return recovered


def recover_from_parquet(needed):
    import pyarrow.parquet as pq
    recovered = {}
    errors = []
    wanted_ids = {row_id for row_id in needed if row_id not in recovered}
    wanted_hashes = {}
    for row_id, row in needed.items():
        digest = row.get('sha256_audio')
        if digest:
            wanted_hashes.setdefault(digest, []).append(row_id)
    files = parquet_files(*EVAL_PARQUET_ROOTS)
    print(f'eval parquet recovery: {len(wanted_ids)} ids, {len(files)} shards', flush=True)
    for path in files:
        if not wanted_ids:
            break
        try:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            if 'id' not in names or 'audio' not in names:
                continue
            columns = ['id']
            if 'sha256_audio' in names:
                columns.append('sha256_audio')
            for group in range(parquet.num_row_groups):
                if not wanted_ids:
                    break
                index_rows = parquet.read_row_group(group, columns=columns).to_pylist()
                matches = []
                for index, row in enumerate(index_rows):
                    rid = row.get('id')
                    digest = row.get('sha256_audio')
                    if rid in wanted_ids:
                        matches.append((index, rid, digest, 'id'))
                    elif digest in wanted_hashes:
                        for target in wanted_hashes[digest]:
                            if target in wanted_ids:
                                matches.append((index, target, digest, 'hash_alias'))
                if not matches:
                    continue
                audio_col = parquet.read_row_group(group, columns=['audio']).column('audio')
                try:
                    for index, target_id, digest, how in matches:
                        if target_id not in wanted_ids:
                            continue
                        expected = needed[target_id].get('sha256_audio')
                        data = audio_bytes_from_cell(audio_col[index].as_py())
                        if not data:
                            continue
                        actual = hashlib.sha256(data).hexdigest()
                        if expected and actual != expected:
                            continue
                        recovered[target_id] = {
                            'bytes': data,
                            'how': how,
                            'parquet': str(path),
                            'sha256_audio': actual,
                        }
                        wanted_ids.discard(target_id)
                finally:
                    del audio_col
            print(f'{path.name}: {len(recovered)} recovered, {len(wanted_ids)} remaining',
                  flush=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
            print(f'SKIP {path}: {exc}', flush=True)
    return recovered, errors


def materialize_eval_row(gate, row, recovered):
    original_id = row['id']
    reference = row.get('reference') or ''
    if not reference:
        return None, 'unlabeled'
    audio = dict(row)
    extra = recovered.get(original_id)
    if extra and extra.get('bytes'):
        audio['audio'] = {'bytes': extra['bytes'], 'path': ''}
        audio['audio_path'] = None
    elif extra and extra.get('path'):
        audio = bind_source_path(audio, extra['path'])
    else:
        ok, _reason = usable_source(row, gate)
        if not ok:
            return None, 'missing'
        for path in candidate_audio_paths(row):
            if path.is_file() and (
                    gate == 'aqua_new_holdout' or not row.get('sha256_audio')
                    or hash_matches(path, row.get('sha256_audio'))):
                audio = bind_source_path(audio, path)
                break
    path, duration, digest = materialize(audio, original_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return manifest_row(
        original_id, path, reference, duration, gate, False, gate, terms_of(row),
        digest, row.get('split') or gate,
    ), 'ok'


def prepare_eval(exclusions):
    summaries = {}
    missing = {}
    manifests = {}
    pending = {}
    step_errors = []
    for gate in GATE_NAMES:
        print(f'gate {gate}: load', flush=True)

        def summarize():
            return gate_summary(gate)

        summary, summary_errors = attempt(f'gate_summary:{gate}', summarize)
        if summary is None:
            summaries[gate] = empty_gate_summary(summary_errors)
            missing[gate] = {
                'missing': 0, 'missing_ids': [], 'errors': len(summary_errors),
                'error_ids': [], 'error_details': summary_errors, 'recovered_ids': [],
            }
            manifests[gate] = []
            step_errors.extend({'gate': gate, **item} for item in summary_errors)
            write_jsonl(MANIFEST_DIR / f'{gate}.jsonl', [])
            continue
        summary = dict(summary)

        def load():
            return load_gate(gate, include_audio=True)

        rows, load_errors = attempt(f'load_gate:{gate}', load)
        if load_errors:
            step_errors.extend({'gate': gate, **item} for item in load_errors)
        if rows is None:
            summary.update(empty_gate_summary(load_errors))
            summary['absent'] = True
            summaries[gate] = summary
            missing[gate] = {
                'missing': 0, 'missing_ids': [], 'errors': len(load_errors),
                'error_ids': [], 'error_details': load_errors, 'recovered_ids': [],
            }
            manifests[gate] = []
            write_jsonl(MANIFEST_DIR / f'{gate}.jsonl', [])
            continue
        ready = []
        missing_ids = []
        error_ids = []
        error_details = []
        for row in rows:
            exclusions['keys'] |= identity_tokens(row)
            ok, _reason = usable_source(row, gate)
            if not ok:
                pending[row['id']] = row
                missing_ids.append(row['id'])
                continue
            result, result_errors = attempt(
                f'materialize:{gate}:{row["id"]}',
                lambda current=row: materialize_eval_row(gate, current, {}),
            )
            if result_errors:
                error_details.extend(result_errors)
            if result is None or result[0] is None or result[1] != 'ok':
                error_ids.append(row['id'])
                continue
            ready.append(result[0])
        summary['materialized'] = len(ready)
        summary['hours'] = sum(r['duration_seconds'] for r in ready) / 3600.0
        summary['errors'] = len(error_ids)
        summary['error_ids'] = error_ids
        summary['absent'] = False
        summary['load_errors'] = load_errors
        summaries[gate] = summary
        missing[gate] = {
            'missing': len(missing_ids),
            'missing_ids': missing_ids,
            'errors': len(error_ids),
            'error_ids': error_ids,
            'error_details': error_details,
            'recovered_ids': [],
        }
        manifests[gate] = ready
        print(f'gate {gate}: labeled={len(rows)} materialized={len(ready)} '
              f'missing={len(missing_ids)} errors={len(error_ids)}', flush=True)
    recovered_map = {}
    parquet_errors = []
    if pending:
        print(f'recovering {len(pending)} missing eval clips', flush=True)
        name_index = option0_name_index()
        option0 = recover_from_option0(pending, name_index)
        still = {row_id: row for row_id, row in pending.items() if row_id not in option0}
        parquet_hits, parquet_errors = recover_from_parquet(still)
        recovered_map.update(option0)
        recovered_map.update(parquet_hits)
        for row_id, payload in recovered_map.items():
            row = pending[row_id]
            gate = None
            for name, info in missing.items():
                if row_id in info['missing_ids']:
                    gate = name
                    break
            if gate is None:
                continue
            result, result_errors = attempt(
                f'recover_materialize:{gate}:{row_id}',
                lambda current=row, extra=payload, name=gate: materialize_eval_row(
                    name, current, {current['id']: extra}),
            )
            if result and result[0] is not None and result[1] == 'ok':
                manifests[gate].append(result[0])
                info = missing[gate]
                info['missing_ids'] = [item for item in info['missing_ids'] if item != row_id]
                info['missing'] = len(info['missing_ids'])
                info['recovered_ids'].append(row_id)
                summaries[gate]['materialized'] = len(manifests[gate])
                summaries[gate]['hours'] = (
                    sum(r['duration_seconds'] for r in manifests[gate]) / 3600.0)
                summaries[gate]['found'] = summaries[gate].get('labeled_rows', 0) - info['missing']
            elif result_errors:
                missing[gate]['error_ids'].append(row_id)
                missing[gate]['errors'] = len(missing[gate]['error_ids'])
                missing[gate]['error_details'].extend(result_errors)
    for gate in GATE_NAMES:
        rows = sorted(manifests.get(gate, []), key=lambda row: row['id'])
        for row in rows:
            if not Path(row['audio_path']).is_file():
                raise FileNotFoundError(row['audio_path'])
        write_jsonl(MANIFEST_DIR / f'{gate}.jsonl', rows)
        manifests[gate] = rows
        info = missing.setdefault(gate, {
            'missing': 0, 'missing_ids': [], 'errors': 0, 'error_ids': [],
            'error_details': [], 'recovered_ids': [],
        })
        summaries.setdefault(gate, empty_gate_summary())
        summaries[gate]['materialized'] = len(rows)
        summaries[gate]['hours'] = sum(r['duration_seconds'] for r in rows) / 3600.0
        print(f'gate {gate}: wrote {len(rows)} rows missing={info["missing"]}', flush=True)
    missing['parquet_errors'] = parquet_errors
    missing['recovered_count'] = sum(len(v.get('recovered_ids', []))
                                     for k, v in missing.items() if isinstance(v, dict))
    return summaries, missing, manifests, step_errors


def prepare_wispr(exclusion_keys):
    if not WISPR_MANIFEST.is_file():
        return [], {
            'status': 'missing_manifest',
            'path': str(WISPR_MANIFEST),
            'rows_read': 0,
            'eligible': 0,
            'excluded_contamination': 0,
            'missing_audio': 0,
            'errors': 1,
        }
    selected = []
    stats = {
        'status': 'ok',
        'path': str(WISPR_MANIFEST),
        'rows_read': 0,
        'eligible': 0,
        'excluded_contamination': 0,
        'missing_audio': 0,
        'errors': 0,
        'error_details': [],
        'label_field': 'reference_text',
    }
    missing_ids = []
    for line in WISPR_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stats['rows_read'] += 1
        if row.get('split') != 'wispr_train' or row.get('train_allowed') is not True:
            continue
        if row.get('eval_allowed') is True:
            continue
        original_id = str(row.get('transcriptEntityId') or row.get('id') or '')
        reference = whitespace(row.get('reference_text'))
        if not original_id or not reference:
            stats['empty_reference'] = stats.get('empty_reference', 0) + 1
            continue
        stats['eligible'] += 1
        probe = {**row, 'id': original_id}
        rel = row.get('audio_path')
        src = WISPR_ROOT / rel if rel and not Path(rel).is_absolute() else Path(rel or '')
        probe['audio_path'] = str(src)
        if src.is_file():
            probe['sha256_audio'] = hashlib.sha256(src.read_bytes()).hexdigest()
        if identity_tokens(probe) & exclusion_keys:
            stats['excluded_contamination'] += 1
            continue
        if not src.is_file():
            stats['missing_audio'] += 1
            missing_ids.append(original_id)
            continue
        result, errors = attempt(
            f'wispr:{original_id}',
            lambda current=probe: materialize(current, original_id),
        )
        if errors:
            stats['error_details'].extend(errors)
        if result is None:
            stats['errors'] += 1
            missing_ids.append(original_id)
            continue
        path, duration, digest = result
        record = manifest_row(
            original_id, path, reference, duration, 'train', True, 'wispr',
            terms_of(row), digest, 'wispr_train',
        )
        record['group_key'] = group_key(probe)
        selected.append(record)
    stats['selected'] = len(selected)
    stats['hours'] = sum(r['duration_seconds'] for r in selected) / 3600.0
    stats['missing_ids'] = missing_ids
    print(f'wispr train: read={stats["rows_read"]} eligible={stats["eligible"]} '
          f'selected={stats["selected"]} excluded={stats["excluded_contamination"]}', flush=True)
    return selected, stats


def prepare_youtube(exclusion_keys):
    import pyarrow.parquet as pq
    stats = {
        'status': 'ok',
        'path': str(YOUTUBE_DIR / '*.parquet'),
        'shards': 0,
        'rows_read': 0,
        'train_candidate': 0,
        'train_allowed_true': 0,
        'eval_allowed_true': 0,
        'permitted': 0,
        'selected': 0,
        'hours': 0.0,
        'label_fields': ['consensus_text', 'insert_text'],
        'permissions': RANKING_DEFINITION['permissions'],
        'ranking_applied': False,
        'error_details': [],
        'split_counts': {},
    }
    files = parquet_files(YOUTUBE_DIR)
    stats['shards'] = len(files)
    permitted_hits = []
    split_counts = defaultdict(int)
    print(f'youtube metadata scan: {len(files)} shards (no audio, no sidecar join)', flush=True)
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            columns = [name for name in YOUTUBE_META_COLUMNS if name in names]
            if not columns:
                continue
            for group in range(parquet.num_row_groups):
                for index, row in enumerate(
                        parquet.read_row_group(group, columns=columns).to_pylist()):
                    stats['rows_read'] += 1
                    split = row.get('split')
                    split_counts[str(split)] += 1
                    if is_true(row.get('train_allowed')):
                        stats['train_allowed_true'] += 1
                    if is_true(row.get('eval_allowed')):
                        stats['eval_allowed_true'] += 1
                    if split != 'train_candidate':
                        continue
                    stats['train_candidate'] += 1
                    if row.get('train_allowed') is True and row.get('eval_allowed') is not True:
                        stats['permitted'] += 1
                        permitted_hits.append({
                            'path': str(path),
                            'row_group': group,
                            'index': index,
                            'id': row.get('id'),
                        })
            print(f'{path.name}: train_candidate={stats["train_candidate"]} '
                  f'permitted={stats["permitted"]}', flush=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            stats['error_details'].append({'path': str(path),
                                          'error': f'{type(exc).__name__}: {exc}'})
            print(f'SKIP {path}: {exc}', flush=True)
    stats['split_counts'] = dict(split_counts)
    if not permitted_hits:
        stats['note'] = (
            'zero source-parquet rows with split=train_candidate and train_allowed=true; '
            'queues/sidecars were not used to override flags'
        )
        print('youtube train: 0 permitted rows', flush=True)
        return [], stats
    # Same-shard labels only. This path stays unused unless parquet flags are true.
    selected = []
    ranked = []
    for hit in permitted_hits:
        parquet = pq.ParquetFile(hit['path'])
        names = set(parquet.schema_arrow.names)
        wanted = [name for name in (
            'id', 'split', 'train_allowed', 'eval_allowed', 'consensus_text', 'insert_text',
            'teacher_transcripts', 'sha256_audio', 'source_id', 'leakage_group',
            'duration_seconds', 'technical_terms', 'term_spans', 'consensus_confidence',
            'review_score', 'technical_value_score', 'consensus_score', 'teacher_review_score',
        ) if name in names]
        rows = parquet.read_row_group(hit['row_group'], columns=wanted).to_pylist()
        row = rows[hit['index']]
        if row.get('split') != 'train_candidate' or row.get('train_allowed') is not True:
            continue
        if row.get('eval_allowed') is True:
            continue
        label = youtube_label(row)
        if not label:
            continue
        if identity_tokens(row) & exclusion_keys:
            continue
        ranked.append({
            'row': row,
            'path': hit['path'],
            'row_group': hit['row_group'],
            'index': hit['index'],
            'id': row['id'],
            'pairwise_wer': pairwise_normalized_mutual_wer(parse_teachers(
                row.get('teacher_transcripts'))),
            'consensus_score': consensus_score(row),
            'duration_seconds': float(row.get('duration_seconds') or 0.0),
        })
    ranked.sort(key=rank_key)
    stats['ranking_applied'] = True
    stats['rankable'] = len(ranked)
    hours = 0.0
    kept = []
    for item in ranked:
        extra = item['duration_seconds'] / 3600.0
        if hours + extra > YOUTUBE_CAP_HOURS and kept:
            break
        hours += extra
        kept.append(item)
    for item in kept:
        parquet = pq.ParquetFile(item['path'])
        if 'audio' not in parquet.schema_arrow.names:
            continue
        audio_col = parquet.read_row_group(item['row_group'], columns=['audio']).column('audio')
        try:
            data = audio_bytes_from_cell(audio_col[item['index']].as_py())
        finally:
            del audio_col
        if not data:
            continue
        expected = item['row'].get('sha256_audio')
        if expected and hashlib.sha256(data).hexdigest() != expected:
            continue
        audio = {'id': item['id'], 'audio': {'bytes': data, 'path': ''}, 'audio_path': None}
        path, duration, digest = materialize(audio, item['id'])
        record = manifest_row(
            item['id'], path, youtube_label(item['row']), duration, 'train', True,
            'youtube_technical_v0', terms_of(item['row']), digest,
            item['row'].get('split') or 'train_candidate',
        )
        record['group_key'] = group_key(item['row'])
        selected.append(record)
    stats['selected'] = len(selected)
    stats['hours'] = sum(r['duration_seconds'] for r in selected) / 3600.0
    return selected, stats


def prepare_synthetic():
    stats = {
        'status': 'omitted',
        'path': str(SYNTH_DIR),
        'exists': SYNTH_DIR.exists(),
        'selected': 0,
        'hours': 0.0,
        'cap_hours': SYNTH_CAP_HOURS,
        'reason': 'directory empty or missing audio+queue text with train_allowed verified',
    }
    if not SYNTH_DIR.exists():
        stats['reason'] = 'directory missing'
        return [], stats
    files = [path for path in SYNTH_DIR.rglob('*') if path.is_file()]
    stats['file_count'] = len(files)
    if not files:
        stats['reason'] = 'directory empty'
        return [], stats
    audio = [path for path in files if path.suffix.lower() in {'.wav', '.flac', '.ogg', '.mp3'}]
    queues = [path for path in files if path.suffix.lower() in {'.jsonl', '.json'}]
    stats['audio_files'] = len(audio)
    stats['queue_files'] = len(queues)
    if not audio or not queues:
        stats['reason'] = 'no verified audio+queue text pair with train_allowed'
        return [], stats
    stats['reason'] = (
        'present files were not used because train_allowed was not verified on this source'
    )
    return [], stats


def split_train_dev(rows):
    train, dev = [], []
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('group_key') or group_key(row)].append(row)
    dev_keys, selected = set(), 0
    for key in sorted(groups, key=lambda value: hashlib.sha256(value.encode()).hexdigest()):
        if selected >= math.ceil(len(rows) * DEV_FRACTION):
            break
        dev_keys.add(key)
        selected += len(groups[key])
    for row in rows:
        key = row.pop('group_key', None) or group_key(row)
        bucket = dev if key in dev_keys else train
        item = dict(row)
        item['split'] = 'dev' if bucket is dev else 'train'
        item['train_allowed'] = True
        bucket.append(item)
    train.sort(key=lambda row: row['id'])
    dev.sort(key=lambda row: row['id'])
    for row in train + dev:
        if not Path(row['audio_path']).is_file():
            raise FileNotFoundError(row['audio_path'])
        if row['train_allowed'] is not True:
            raise ValueError(row['id'])
    return train, dev


def hours_of(rows):
    return sum(row['duration_seconds'] for row in rows) / 3600.0


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    print('collecting raw-gate exclusion identities', flush=True)
    exclusion, exclusion_errors = attempt('exclusions', collect_exclusions)
    if exclusion is None:
        exclusion = {
            'keys': set(), 'per_gate': {}, 'holdout_ids': [], 'holdout_hash_count': 0,
            'token_count': 0, 'errors': exclusion_errors,
        }
    exclusions_verified = exclusion is not None and not exclusion.get('errors')
    keys = exclusion['keys']
    print(f'exclusion tokens={len(keys)} holdout_ids={len(exclusion["holdout_ids"])}', flush=True)
    print('eval gate materialization', flush=True)
    eval_out, eval_errors = attempt(
        'prepare_eval', lambda: prepare_eval(exclusion))
    if eval_out is None:
        summaries, missing, _manifests, step_errors = {}, {}, {}, eval_errors
        for gate in GATE_NAMES:
            write_jsonl(MANIFEST_DIR / f'{gate}.jsonl', [])
            summaries[gate] = empty_gate_summary(eval_errors)
            missing[gate] = {
                'missing': 0, 'missing_ids': [], 'errors': len(eval_errors),
                'error_ids': [], 'error_details': eval_errors, 'recovered_ids': [],
            }
    else:
        summaries, missing, _manifests, step_errors = eval_out
    print('wispr train', flush=True)
    wispr_rows, wispr_stats = [], {}
    wispr_out, wispr_errors = attempt('wispr', lambda: prepare_wispr(keys))
    if wispr_out is None:
        wispr_stats = {'status': 'failed', 'error_details': wispr_errors, 'selected': 0}
    else:
        wispr_rows, wispr_stats = wispr_out
    print('synthetic', flush=True)
    synth_rows, synth_stats = [], {}
    synth_out, synth_errors = attempt('synthetic', prepare_synthetic)
    if synth_out is None:
        synth_stats = {'status': 'failed', 'error_details': synth_errors, 'selected': 0}
    else:
        synth_rows, synth_stats = synth_out
    print('youtube train metadata (source flags only)', flush=True)
    youtube_rows, youtube_stats = [], {}
    youtube_out, youtube_errors = attempt('youtube', lambda: prepare_youtube(keys))
    if youtube_out is None:
        youtube_stats = {'status': 'failed', 'error_details': youtube_errors, 'selected': 0,
                         'permitted': 0}
    else:
        youtube_rows, youtube_stats = youtube_out
    train_candidates = (wispr_rows + youtube_rows + synth_rows) if exclusions_verified else []
    train, dev = split_train_dev(train_candidates)
    write_jsonl(MANIFEST_DIR / 'train.jsonl', train)
    write_jsonl(MANIFEST_DIR / 'dev.jsonl', dev)
    eval_missing = {gate: missing[gate] for gate in GATE_NAMES if gate in missing}
    eval_missing['parquet_errors'] = missing.get('parquet_errors', [])
    eval_missing['recovered_count'] = missing.get('recovered_count', 0)
    write_json(DATA / 'eval_missing.json', eval_missing)
    summary = {
        'gates': summaries,
        'train': {
            'wispr': wispr_stats,
            'youtube_technical_v0': youtube_stats,
            'synthetic_tcpgen_tts_v4_20260630': synth_stats,
        },
        'splits': {
            'train': {'rows': len(train), 'hours': hours_of(train)},
            'dev': {'rows': len(dev), 'hours': hours_of(dev)},
            **{
                gate: {
                    'rows': summaries.get(gate, {}).get('materialized', 0),
                    'hours': summaries.get(gate, {}).get('hours', 0.0),
                    'train_allowed': False,
                }
                for gate in GATE_NAMES
            },
        },
        'exclusions': {
            'token_count': exclusion.get('token_count', len(keys)),
            'holdout_id_count': len(exclusion.get('holdout_ids', [])),
            'holdout_hash_count': exclusion.get('holdout_hash_count', 0),
            'per_gate': exclusion.get('per_gate', {}),
            'wispr_excluded_contamination': wispr_stats.get('excluded_contamination', 0),
            'identities': (
                'raw-gate row id, sha256_audio, source_id, leakage_group, source_url, '
                'youtube video id, audio filename stem; plus all wispr holdout ids/hashes'
            ),
            'errors': exclusion.get('errors', []) + (exclusion_errors or []),
        },
        'missing_audio': {
            gate: {
                'missing': eval_missing.get(gate, {}).get('missing', 0),
                'errors': eval_missing.get(gate, {}).get('errors', 0),
                'recovered': len(eval_missing.get(gate, {}).get('recovered_ids', [])),
            }
            for gate in GATE_NAMES
        },
        'ranking_definition': RANKING_DEFINITION,
        'source_provenance': {
            'wispr': str(WISPR_MANIFEST),
            'youtube': str(YOUTUBE_DIR / '*.parquet'),
            'synthetic': str(SYNTH_DIR),
            'gates': 'scripts/option0/gates.py GATE_NAMES via raw_gate/load_gate',
            'eval_audio_search': [str(path) for path in EVAL_PARQUET_ROOTS] + [
                str(OPTION0_SEGMENTS), str(SEGMENTS),
                '/data/phonon_segments_root/option0_recovery',
            ],
            'audio_output': str(AUDIO_DIR / '<cache_id>.wav'),
            'cache_id': 'sha256(original id) hex',
            'dev_split': (
                'Source groups sorted by SHA256; select until ceil(2% of clips); '
                'leakage_group/source_id/youtube video keep clips together'
            ),
            'waveform': 'full clip decoded to 16 kHz mono float WAV; train labels are not chunked',
        },
        'step_errors': step_errors + (eval_errors or []) + (wispr_errors or [])
        + (synth_errors or []) + (youtube_errors or []),
    }
    write_json(DATA / 'summary.json', summary)
    write_json(MANIFEST_DIR / 'summary.json', summary)
    print(json.dumps({
        'train_rows': len(train),
        'dev_rows': len(dev),
        'train_hours': hours_of(train),
        'dev_hours': hours_of(dev),
        'youtube_permitted': youtube_stats.get('permitted', 0),
        'youtube_selected': youtube_stats.get('selected', 0),
        'synthetic': synth_stats.get('status'),
        'eval_materialized': {
            gate: summaries.get(gate, {}).get('materialized', 0) for gate in GATE_NAMES
        },
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
