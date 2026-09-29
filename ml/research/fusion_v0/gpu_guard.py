"""Check actual GPU occupancy after a potentially long coordinator wait."""
from __future__ import annotations

import os
import subprocess


def assert_gpu_idle():
    status = subprocess.check_output(['overnight-compute', 'status'], text=True)
    lease_valid = False
    for line in status.splitlines():
        if line.startswith(('agent', 'codex-fusion')):
            print(line, flush=True)
        fields = line.split('\t')
        if len(fields) >= 3 and fields[0] == 'codex-fusion':
            lease_valid = fields[1] in {'gpu0', 'machine'} and fields[2] == 'running'
    if not lease_valid:
        raise RuntimeError('No running codex-fusion GPU lease; use gpu_step.sh before GPU work')
    output = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    other_pids = [int(line.strip()) for line in output.splitlines()
                  if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if other_pids:
        raise RuntimeError(
            f'GPU still occupied after lease acquisition by PIDs {other_pids}; '
            'refusing concurrent extraction, training, or timing')
    print('Post-lease physical GPU occupancy check passed.', flush=True)
