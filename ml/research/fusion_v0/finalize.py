"""Refresh final artifacts and append the authorized DEVLOG entry in one write."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/user/phonon')
WORK = ROOT / 'research/fusion_v0'
DATA = Path('/data/phonon_fusion_v0')
TITLE = '2026-09-15: Fusion v0: frozen partner encoder + adapter into Parakeet v2 decoder'


def main():
    from run import make_timing
    finished = DATA / 'finished.json'
    if finished.exists():
        # Reload the current timing implementation after the persistent runner exits.
        make_timing(json.loads(finished.read_text())['systems'])
    subprocess.run(['uv', 'run', '--no-sync', '--with', 'whisper-normalizer', 'python',
                    str(WORK / 'report.py')], check=True)
    result = subprocess.run(['uv', 'run', 'ruff', 'check', 'research/fusion_v0', '--no-cache'],
                            cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, 'UV_NO_SYNC': '1'})
    (DATA / 'logs/ruff_fusion_final.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    summary_path = DATA / 'manifests/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        table = ['## Final manifest sizes', '', '| Split | Clips | Hours |',
                 '| --- | ---: | ---: |']
        for split, values in summary.get('splits', {}).items():
            table.append(f'| {split} | {values["rows"]} | {values["hours"]:.6f} |')
        table += ['', 'These counts precede native TDT memory exclusions. Exact shared '
                  'training exclusions and retained counts appear in `results.md` and '
                  '`/data/phonon_fusion_v0/training_validation.json`.', '']
        readme = WORK / 'README.md'
        text = readme.read_text().split('## Final manifest sizes')[0].rstrip()
        readme.write_text(text + '\n\n' + '\n'.join(table))
    files = sorted(WORK.glob('*.py')) + [WORK / 'gpu_step.sh', WORK / 'README.md',
                                       WORK / 'results.md', WORK / 'scores.json']
    files += sorted((DATA / 'checkpoints').glob('*/best.pt'))
    files += sorted((DATA / 'checkpoints').glob('*/training.json'))
    files += sorted((DATA / 'manifests').glob('*.jsonl'))
    files += [DATA / name for name in ('reproduction.json', 'training_validation.json',
                                      'training_exclusions.json', 'timing.json',
                                      'source_provenance.json') if (DATA / name).exists()]
    files += sorted(Path('/data/hf/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots').glob(
        '*/parakeet-tdt-0.6b-v2.nemo'))
    files += sorted(Path('/data/hf/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots').glob(
        '*/model.safetensors'))
    hashes = {}
    for path in files:
        with path.open('rb') as source:
            hashes[str(path)] = hashlib.file_digest(source, 'sha256').hexdigest()
    provenance = {'files': hashes,
                  'versions': {name: importlib.metadata.version(name)
                               for name in ['torch', 'nemo_toolkit', 'numpy']}}
    (DATA / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    report = (WORK / 'results.md').read_text()
    # The exact reviewed result, including limits and error evidence, is appended once.
    body = report.replace('# Fusion v0 results\n', '', 1).replace('\n## ', '\n### ')
    entry = '\n\n## ' + TITLE + '\n\n' + body
    devlog = ROOT / 'DEVLOG.md'
    if TITLE in devlog.read_text():
        raise RuntimeError('Fusion v0 DEVLOG entry already exists; refusing duplicate append')
    descriptor = os.open(devlog, os.O_WRONLY | os.O_APPEND)
    try:
        encoded = entry.encode()
        count = os.write(descriptor, encoded)
        if count != len(encoded):
            raise RuntimeError(f'DEVLOG short write: {count}/{len(encoded)} bytes')
    finally:
        os.close(descriptor)
    (DATA / 'devlog_append.json').write_text(json.dumps({
        'title': TITLE, 'bytes': count, 'entry_sha256': hashlib.sha256(encoded).hexdigest(),
        'single_os_write': True}, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
