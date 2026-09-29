"""Recover only gate WAV bytes from available Parquet shards, with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gates import GATE_NAMES, QUEUES, REPORT, audio_path, gate_summary, hash_matches, raw_gate

DEFAULT_PARQUET_DIRS = [Path('/data/phonon_datasets') / dataset / 'data' for dataset in (
    'youtube_technical_v0', 'youtube_freecodecamp_real_dev_courses_v0',
    'youtube_dev_tutorials_real_audio_v0',
)]


def ensure_storage():
    target = Path('/data/phonon_segments_root')
    link = Path('/home/user/datasets/phonon')
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink() and link.exists():
        raise RuntimeError(f'Refusing to replace existing directory: {link}')
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected symlink target: {link} -> {link.resolve()}')
    else:
        link.symlink_to(target, target_is_directory=True)
    return target


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet-dirs', '--parquet-dir', nargs='+', type=Path,
                        default=DEFAULT_PARQUET_DIRS,
                        help='Shard directories to scan (defaults to all three restored datasets).')
    args = parser.parse_args()
    target_root = ensure_storage()
    needed = defaultdict(list)
    for gate in QUEUES:
        for row in raw_gate(gate):
            path = audio_path(row)
            if not path.resolve().is_relative_to(target_root):
                raise ValueError(f'Audio target is outside /data storage: {path}')
            expected = row.get('sha256_audio')
            if not expected:
                raise ValueError(f'No audio hash: {row["id"]}')
            if not hash_matches(path, expected):
                item = (expected, path)
                if item not in needed[row['id']]:
                    needed[row['id']].append(item)
    errors = []
    written = 0
    scanned = []
    for path in [p for directory in args.parquet_dirs for p in sorted(directory.glob('*.parquet'))]:
        if not needed:
            break
        scanned.append(str(path))
        try:
            parquet = pq.ParquetFile(path)
            # First scan only the two cheap columns. Audio is decompressed only in matching groups.
            matches = defaultdict(list)
            for group in range(parquet.num_row_groups):
                index_rows = parquet.read_row_group(group, columns=['id', 'sha256_audio']).to_pylist()
                for index, row in enumerate(index_rows):
                    if row['id'] in needed:
                        for expected, out in needed[row['id']]:
                            if row['sha256_audio'] == expected:
                                matches[group].append((index, row['id'], expected, out))
            for group, items in matches.items():
                audio_rows = parquet.read_row_group(group, columns=['audio']).column('audio')
                for index, row_id, expected, out in items:
                    data = audio_rows[index].as_py()['bytes']
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f'Embedded audio hash mismatch: {row_id}')
                    out.parent.mkdir(parents=True, exist_ok=True)
                    # A complete verified buffer is written, and the on-disk bytes are checked too.
                    out.write_bytes(data)
                    if not hash_matches(out, expected):
                        raise ValueError(f'Written audio hash mismatch: {out}')
                    needed[row_id].remove((expected, out))
                    if not needed[row_id]:
                        del needed[row_id]
                    written += 1
            print(f'{path.name}: restored {sum(map(len, matches.values()))}; '
                  f'{len(needed)} ids remaining', flush=True)
        except (OSError, ValueError, TypeError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
            print(f'SKIP {path}: {exc}', flush=True)
    summaries = {gate: gate_summary(gate) for gate in GATE_NAMES}
    # Also report raw hidden rows whose labels were deliberately excluded from scoring.
    for gate in QUEUES:
        raw_missing = [r['id'] for r in raw_gate(gate)
                       if not hash_matches(audio_path(r), r.get('sha256_audio'))]
        summaries[gate]['raw_missing_ids'] = raw_missing
        summaries[gate]['raw_audio_found'] = summaries[gate]['rows'] - len(raw_missing)
        summaries[gate]['labeled_found'] = summaries[gate]['found']
        summaries[gate]['labeled_missing'] = summaries[gate]['missing']
        summaries[gate]['labeled_missing_ids'] = summaries[gate].get('missing_ids', [])
        summaries[gate]['found'] = summaries[gate]['raw_audio_found']
        summaries[gate]['missing'] = len(raw_missing)
        summaries[gate]['missing_ids'] = raw_missing
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'missing_audio.json').write_text(json.dumps(
        {'gates': summaries, 'parquet_errors': errors, 'written_this_pass': written,
         'parquet_dirs': [str(p) for p in args.parquet_dirs],
         'scanned_shards': scanned}, indent=2) + '\n')
    for gate, summary in summaries.items():
        print(f'{gate}: rows={summary["rows"]} found={summary["found"]} '
              f'missing={summary["missing"]} unlabeled={summary["dropped_unlabeled"]}', flush=True)
        if summary['missing_ids']:
            print(f'{gate} missing IDs: {summary["missing_ids"]}; '
                  f'shard directories: {[str(p) for p in args.parquet_dirs]}', flush=True)


if __name__ == '__main__':
    main()
