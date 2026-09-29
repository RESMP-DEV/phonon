"""Sequential, leased Fusion v0 experiment with durable state and two-attempt failures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/user/phonon')
WORK = ROOT / 'research/fusion_v0'
DATA = Path('/data/phonon_fusion_v0')
GATES = ['aqua_new_holdout', 'course91', 'novel180', 'hard77', 'uncertain48',
         'yt_hidden120', 'wispr_holdout120', 'wispr_edit25', 'personal_cuda']
UV = ['uv', 'run', '--no-sync', 'python']
PARTNER_UV = ['uv', 'run', '--no-project',
              '/home/user/.cache/uv/archive-v0/<env>/bin/python']
REPORT = ['uv', 'run', '--no-sync', '--with', 'whisper-normalizer', 'python',
          str(WORK / 'report.py')]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def run_step(name, command, gpu=False):
    state_path = DATA / 'state.json'
    done = DATA / 'steps' / f'{name}.json'
    done.parent.mkdir(parents=True, exist_ok=True)
    if (not name.startswith('report') and name != 'stock_reproduction'
            and done.exists() and json.loads(done.read_text()).get('returncode') == 0):
        print(f'verified prior step completion: {name}', flush=True)
        return True
    for attempt in (1, 2):
        log = DATA / 'logs' / f'{name}_attempt{attempt}.log'
        actual = ['bash', str(WORK / 'gpu_step.sh'), *command] if gpu else command
        started = time.time()
        with log.open('w') as output:
            process = subprocess.Popen(actual, cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, env=os.environ.copy())
            state = {'stage': name, 'attempt': attempt, 'pid': process.pid,
                     'started_unix': started, 'command': actual, 'log': str(log),
                     'status': 'running_or_waiting_for_lease', 'gpu_step': gpu}
            save(state_path, state)
            print(json.dumps(state), flush=True)
            returncode = process.wait()
        state.update(returncode=returncode, elapsed_seconds=time.time() - started,
                     status='complete' if returncode == 0 else 'failed')
        save(state_path, state)
        if returncode == 0:
            save(done, state)
            return True
        error = {**state, 'error': log.read_text()[-14000:]}
        with (DATA / 'errors.jsonl').open('a') as output:
            output.write(json.dumps(error) + '\n')
        print(f'failed {name} attempt={attempt} returncode={returncode}', flush=True)
    return False


def engine(action, *args):
    return [*UV, str(WORK / 'engine.py'), action, *args]


def partner(encoder, splits):
    return [*PARTNER_UV, str(WORK / 'partner.py'), '--encoder', encoder, '--splits', *splits]


def make_timing(systems):
    timing = {'scope': 'serial sum of separately measured encoder preprocessing/forward and '
                       'alignment plus cached adapter/native TDT decoding; excludes model load and cache I/O; '
                       'not a directly measured integrated live pipeline', 'gate': 'wispr_holdout120',
              'systems': {}}
    for system, encoder in systems.items():
        pred = WORK / 'preds/wispr_holdout120' / f'{system}.jsonl'
        if not pred.exists():
            continue
        rows = [json.loads(line) for line in pred.read_text().splitlines()]
        audio = sum(r['audio_seconds'] for r in rows)
        if not audio:
            continue
        decode = sum(r['elapsed_seconds'] for r in rows)
        parakeet = sum(r['encoder_seconds'] for r in rows)
        forward = preprocessing = alignment = 0.0
        if encoder:
            summary = DATA / 'feats' / encoder / 'wispr_holdout120/_summary.json'
            if not summary.exists():
                continue
            meta = json.loads(summary.read_text())
            forward = meta.get('forward_seconds', meta.get('total_forward_seconds', 0))
            preprocessing = meta.get('preprocessing_seconds',
                                     meta.get('total_preprocessing_seconds', 0))
            alignment = meta['alignment_seconds']
            if not forward:
                raise RuntimeError(f'Partner forward timing missing: {summary}')
        elapsed = decode + parakeet + forward + preprocessing + alignment
        timing['systems'][system] = {
            'clips': len(rows), 'audio_seconds': audio,
            'parakeet_preprocessing_and_encoder_seconds': parakeet,
            'partner_forward_seconds': forward, 'partner_preprocessing_seconds': preprocessing,
            'partner_alignment_seconds': alignment,
            'adapter_decoder_seconds': decode, 'serial_total_seconds': elapsed,
            'serial_realtime_factor': elapsed / audio,
        }
    save(DATA / 'timing.json', timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after-stock', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    os.environ.update(HF_HOME='/data/hf', PYTHONDONTWRITEBYTECODE='1',
                      UV_CACHE_DIR=str(DATA / 'cache/uv'), TMPDIR=str(DATA / 'tmp'),
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if args.wait_pid:
        while Path(f'/proc/{args.wait_pid}').exists():
            print(f'waiting for existing model probe PID {args.wait_pid}', flush=True)
            save(DATA / 'state.json', {'stage': 'waiting_for_model_probe',
                                     'probe_pid': args.wait_pid, 'pid': os.getpid(),
                                     'updated_unix': time.time()})
            time.sleep(30)
    # Each contiguous extraction or decoding pass is one leased GPU step.
    # On a failed pass, fall back to individual gates so other available data still runs.
    if run_step('cache_parakeet_eval', engine('cache', '--splits', *GATES), gpu=True):
        if not run_step('decode_control_b', engine('decode', '--system', 'control_b',
                                                  '--splits', *GATES), gpu=True):
            for gate in GATES:
                run_step(f'decode_control_b_{gate}', engine(
                    'decode', '--splits', gate, '--system', 'control_b'), gpu=True)
    else:
        for gate in GATES:
            if run_step(f'cache_parakeet_{gate}', engine('cache', '--splits', gate), gpu=True):
                run_step(f'decode_control_b_{gate}', engine(
                    'decode', '--splits', gate, '--system', 'control_b'), gpu=True)
    valid = run_step('stock_reproduction', [*REPORT, '--check-reproduction'])
    if not valid or args.stop_after_stock:
        run_step('report_stock', REPORT)
        return
    cached = run_step('cache_parakeet_train', engine('cache', '--splits', 'train', 'dev'), gpu=True)
    encoder = None
    if cached:
        for candidate in ('qwen', 'cohere', 'granite'):
            if run_step(f'cache_{candidate}_train', partner(candidate, ['train', 'dev']), gpu=True):
                encoder = candidate
                break
    if encoder is not None:
        if not run_step(f'cache_{encoder}_eval', partner(encoder, GATES), gpu=True):
            for gate in GATES:
                run_step(f'cache_{encoder}_{gate}', partner(encoder, [gate]), gpu=True)
    save(DATA / 'primary_partner.json', {'encoder': encoder})
    if encoder is None:
        run_step('report_cache_failure', REPORT)
        return
    validated = run_step('validate_training', engine('validate-train', '--system', 'control_a', '--partner', encoder),
                         gpu=True)
    if not validated:
        run_step('report_training_failure', REPORT)
        return
    systems = {'control_b': None}
    for system in ('control_a', 'fusion'):
        command = ['timeout', '--signal=TERM', '10700', *engine('train', '--system', system, '--partner', encoder)]
        if run_step(f'train_{system}', command, gpu=True):
            if not run_step(f'decode_{system}', engine(
                    'decode', '--system', system, '--partner', encoder, '--splits', *GATES), gpu=True):
                for gate in GATES:
                    run_step(f'decode_{system}_{gate}', engine(
                        'decode', '--system', system, '--partner', encoder, '--splits', gate), gpu=True)
            systems[system] = encoder if system == 'fusion' else None
    make_timing(systems)
    run_step('report_primary', REPORT)
    scores = json.loads((WORK / 'scores.json').read_text())
    models = scores.get('models', {})
    # Conditional follow-up needs the requested FULL five-gate mean and personal gate.
    fusion = models.get('fusion', {})
    control = models.get('control_a', {})
    fmean, cmean = (m.get('five_gate_mean_fair_wer') for m in (fusion, control))
    fg = fusion.get('gates', {}).get('wispr_holdout120', {})
    cg = control.get('gates', {}).get('wispr_holdout120', {})
    wins = (fmean is not None and cmean is not None and fmean < cmean
            and fg.get('complete') and cg.get('complete') and fg['fair_wer'] < cg['fair_wer'])
    followup, encoder = ('cohere_fusion', 'cohere') if wins else ('partner_only', encoder)
    save(DATA / 'followup.json', {'system': followup, 'encoder': encoder,
                                'full_five_gate_gain_established': bool(wins),
                                'reason': 'both required gains established' if wins else
                                'both required gains not established; run partner-only ablation'})
    ready = True
    if encoder == 'cohere':
        ready = run_step('cache_cohere', partner('cohere', ['train', 'dev', *GATES]), gpu=True)
    if ready and (DATA / 'checkpoints/fusion/best.pt').exists():
        command = ['timeout', '--signal=TERM', '10700', *engine(
            'train', '--system', followup, '--partner', encoder)]
        if run_step(f'train_{followup}', command, gpu=True):
            if not run_step(f'decode_{followup}', engine(
                    'decode', '--system', followup, '--partner', encoder, '--splits', *GATES), gpu=True):
                for gate in GATES:
                    run_step(f'decode_{followup}_{gate}', engine(
                        'decode', '--system', followup, '--partner', encoder, '--splits', gate), gpu=True)
            systems[followup] = encoder
    make_timing(systems)
    # Always refresh report; final report is not a resumable expensive step.
    subprocess.run(REPORT, check=True)
    save(DATA / 'finished.json', {'finished_unix': time.time(), 'systems': systems,
                                'results': str(WORK / 'results.md')})


if __name__ == '__main__':
    main()
