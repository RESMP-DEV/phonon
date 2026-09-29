"""CPU cache integrity and timestamp-decode identity audit, after GPU release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/option0'))
from score_gates import NORMALIZE  # noqa: E402

from extract import CACHE, MODELS  # noqa: E402


def main():
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    ready = [row for row in manifest if row['status'] == 'ready']
    result = {'expected_clips': len(ready), 'duration_seconds': sum(
        row['duration_seconds'] for row in ready), 'excluded_audio': [
        row for row in manifest if row['status'] != 'ready'], 'encoders': {}}
    for tag in MODELS:
        report = {'clips': 0, 'frames': 0, 'layers': [], 'errors': [],
                  'missing_clips': [], 'bytes': 0, 'gates': {}}
        for row in ready:
            gate, clip_id = row['gate'], row['id']
            path = CACHE / tag / gate / f'{clip_id}.npy'
            sidecar = path.with_suffix('.json')
            if not path.exists() or not sidecar.exists():
                report['missing_clips'].append([gate, clip_id])
                continue
            meta = json.loads(sidecar.read_text())
            values = np.load(path, mmap_mode='r')
            errors = []
            if values.dtype != np.float16 or values.ndim != 2:
                errors.append('expected fp16 matrix')
            if list(values.shape) != [meta['frames'], meta['dims']]:
                errors.append('shape disagrees with sidecar')
            if not np.isfinite(values).all():
                errors.append('nonfinite final features')
            if sum(chunk['frames'] for chunk in meta['chunks']) != len(values):
                errors.append('chunk count disagrees with frames')
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['feature_sha256']:
                errors.append('feature SHA256 mismatch')
            if meta['wave_sha256'] != row['wave_sha256']:
                errors.append('source waveform hash mismatch')
            for layer in meta['layers']:
                layer_path = CACHE / tag / 'layers' / str(layer) / gate / path.name
                if not layer_path.exists():
                    errors.append(f'layer {layer} missing')
                    continue
                layer_values = np.load(layer_path, mmap_mode='r')
                if layer_values.shape != values.shape or layer_values.dtype != np.float16:
                    errors.append(f'layer {layer} shape/dtype mismatch')
                if not np.isfinite(layer_values).all():
                    errors.append(f'layer {layer} nonfinite')
                report['bytes'] += layer_path.stat().st_size
            if errors:
                report['errors'].append({'gate': gate, 'id': clip_id, 'errors': errors})
            report['clips'] += 1
            report['frames'] += len(values)
            report['layers'] = meta['layers']
            report['bytes'] += path.stat().st_size
            report['gates'][gate] = report['gates'].get(gate, 0) + 1
        report['complete'] = report['clips'] == len(ready) and not report['errors']
        result['encoders'][tag] = report
        print(tag, report['clips'], 'clips;', len(report['errors']), 'errors', flush=True)
    decoded = {}
    for gate in {row['gate'] for row in ready}:
        predicted_path = ROOT / 'runs/reports/option0_sweep_20260915/preds' / gate / 'parakeet-tdt-0.6b-v2.jsonl'
        predicted = {r['id']: r['hypothesis'] for r in map(json.loads, predicted_path.read_text().splitlines())}
        record = {'clips': 0, 'same_normalized_hypothesis': 0, 'word_timestamps': 0,
                  'nonpositive_word_durations': 0, 'outside_clip_times': 0, 'changed_ids': []}
        for row in ready:
            if row['gate'] != gate:
                continue
            sidecar = CACHE / 'parakeet-tdt-0.6b-v2' / gate / f'{row["id"]}.json'
            if not sidecar.exists():
                continue
            meta = json.loads(sidecar.read_text())
            record['clips'] += 1
            if NORMALIZE(meta['hypothesis']) == NORMALIZE(predicted.get(row['id'], '')):
                record['same_normalized_hypothesis'] += 1
            else:
                record['changed_ids'].append(row['id'])
            for word in meta['word_timestamps']:
                record['word_timestamps'] += 1
                record['nonpositive_word_durations'] += word['end'] <= word['start']
                record['outside_clip_times'] += word['start'] < 0 or word['end'] > meta['duration_seconds'] + .081
        decoded[gate] = record
    result['timestamp_decode_identity'] = decoded
    (Path(__file__).parent / 'cache_audit.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
