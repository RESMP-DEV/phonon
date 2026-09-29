"""Frozen audio encoders and Parakeet word times; run only while holding GPU lease."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['HF_HOME'] = '/data/hf'
os.environ['HF_HUB_CACHE'] = '/data/hf/hub'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/option0'), str(ROOT / 'src')]
from gates import chunk_audio, hash_matches, load_audio, load_gate  # noqa: E402
from transcribe_gate import load_model  # noqa: E402

CACHE = Path('/data/phonon_latent_v0')
GATES = ['novel180', 'hard77', 'uncertain48', 'wispr_holdout120',
         'aqua_new_holdout', 'wispr_edit25']
MODELS = {
    'parakeet-tdt-0.6b-v2': ('nemo', 'nvidia/parakeet-tdt-0.6b-v2'),
    'parakeet-unified-en-0.6b': ('nemo', 'nvidia/parakeet-unified-en-0.6b'),
    'Qwen3-ASR-0.6B-hf': ('transformers-qwen3asr', 'Qwen/Qwen3-ASR-0.6B-hf'),
    'cohere-transcribe-03-2026': ('transformers-cohere', 'CohereLabs/cohere-transcribe-03-2026'),
    'granite-speech-5.0-470m-turboctc': (
        'transformers-granite-ctc', 'ibm-granite/granite-speech-5.0-470m-turboctc'),
    'granite-speech-4.1-2b': ('transformers-granite', 'ibm-granite/granite-speech-4.1-2b'),
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def prepare(extra=False):
    import soundfile as sf
    records = []
    for gate in (['course91', 'personal_cuda'] if extra else GATES):
        for row in load_gate(gate, include_audio=True):
            if gate != 'aqua_new_holdout' and not hash_matches(
                    row['audio_path'], row.get('sha256_audio')):
                records.append({'gate': gate, 'id': row['id'], 'status': 'missing_or_hash_mismatch'})
                continue
            wave = load_audio(row)
            path = CACHE / 'audio' / gate / f'{row["id"]}.wav'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                sf.write(path, wave, 16000, subtype='FLOAT')
            records.append({'gate': gate, 'id': row['id'], 'status': 'ready',
                            'audio_path': str(path), 'duration_seconds': len(wave) / 16000,
                            'sha256_audio': row.get('sha256_audio'),
                            'wave_sha256': hashlib.sha256(wave.tobytes()).hexdigest()})
    write_json(CACHE / ('prediction_manifest.json' if extra else 'manifest.json'), records)
    print(json.dumps({'ready': sum(r['status'] == 'ready' for r in records),
                      'seconds': sum(r.get('duration_seconds', 0) for r in records),
                      'excluded': [r for r in records if r['status'] != 'ready']}), flush=True)


def array(value, nemo=False):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if value.ndim == 3:
        value = value[0]
        if nemo:
            value = value.transpose(0, 1)
    return value.detach().float().cpu().numpy().astype('float16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prepare-extra-predictions', action='store_true')
    parser.add_argument('--encoder', choices=MODELS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.prepare or args.prepare_extra_predictions:
        prepare(extra=args.prepare_extra_predictions)
        return
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    tag = args.encoder
    family, model_id = MODELS[tag]
    local_model = CACHE / 'models' / tag
    load_id = str(local_model) if (local_model / 'config.json').exists() else model_id
    if tag == 'parakeet-tdt-0.6b-v2':
        load_id = str(next(Path('/data/hf/hub').glob(
            'models--nvidia--parakeet-tdt-0.6b-v2/snapshots/*/*.nemo')))
    model, processor = load_model(family, load_id)
    if family == 'nemo':
        model.preprocessor.featurizer.dither = 0.0
    is_v2 = tag == 'parakeet-tdt-0.6b-v2'
    is_qwen = family == 'transformers-qwen3asr'
    encoder = model.encoder if family in ['nemo', 'transformers-granite-ctc'] else (
        model.model.audio_tower if is_qwen else model.model.encoder)
    fps = 50.0 if family == 'transformers-granite' else 12.5
    if family == 'transformers-granite-ctc':
        fps = 50.0 / 2 ** len(encoder.config.subsample_layers)
    captures = {}
    handles = []
    layers = list(range(len(encoder.layers))) if is_v2 or is_qwen else []
    for index in layers:
        def hook(_module, _inputs, output, key=index):
            # NeMo conformer layer output is B,T,D, while encoder output is B,D,T.
            captures[key] = array(output)
        handles.append(encoder.layers[index].register_forward_hook(hook))
    final_capture = {}
    if family == 'nemo':
        def final_hook(_module, _inputs, output):
            final_capture['features'] = array(output[0], nemo=True)
            final_capture['length'] = int(output[1][0])
        handles.append(encoder.register_forward_hook(final_hook))
    model_meta = {'model': model_id, 'encoder_class': type(encoder).__name__,
                  'layers': layers, 'frame_rate': fps, 'dtype': str(next(model.parameters()).dtype),
                  'torch': torch.__version__, 'revision': getattr(
                      getattr(model, 'config', None), '_commit_hash', None)}
    for package in ['transformers', 'nemo_toolkit']:
        try:
            model_meta[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    print(json.dumps(model_meta), flush=True)
    records = [r for r in json.loads((CACHE / 'manifest.json').read_text()) if r['status'] == 'ready']
    if family == 'transformers-granite-ctc':
        records.extend(r for r in json.loads((CACHE / 'prediction_manifest.json').read_text())
                       if r['status'] == 'ready')
    if args.limit:
        records = records[:args.limit]
    start_time = time.monotonic()
    for number, row in enumerate(records, 1):
        out = CACHE / tag / row['gate'] / f'{row["id"]}.npy'
        sidecar = out.with_suffix('.json')
        prediction_sidecar = CACHE / 'ctc_predictions' / row['gate'] / f'{row["id"]}.json'
        if sidecar.exists() and out.exists():
            continue
        if family == 'transformers-granite-ctc' and row['gate'] not in GATES and prediction_sidecar.exists():
            continue
        wave, sr = sf.read(row['audio_path'], dtype='float32')
        assert sr == 16000
        final_parts, layer_parts, chunks, words, hypotheses = [], {i: [] for i in layers}, [], [], []
        offset = 0.0
        for chunk in chunk_audio(wave):
            seconds = len(chunk) / 16000
            captures.clear()
            if family == 'nemo':
                if is_v2:
                    outputs = model.transcribe([chunk], batch_size=1, return_hypotheses=True,
                                               timestamps=True, num_workers=0, verbose=False)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    hyp = outputs[0]
                    hypotheses.append(hyp.text)
                    for word in hyp.timestamp.get('word', []):
                        word = dict(word)
                        word['start'] = float(word['start']) + offset
                        word['end'] = float(word['end']) + offset
                        words.append(word)
                else:
                    signal = torch.from_numpy(chunk).unsqueeze(0).cuda()
                    signal_len = torch.tensor([len(chunk)], device='cuda')
                    processed, length = model.preprocessor(input_signal=signal, length=signal_len)
                    encoder(audio_signal=processed, length=length)
                features = final_capture['features'][:final_capture['length']]
            else:
                if is_qwen:
                    inputs = processor.feature_extractor(chunk, sampling_rate=16000,
                                                         return_tensors='pt', return_attention_mask=True)
                    inputs['input_features_mask'] = inputs.pop('attention_mask')
                    inputs.to(model.device, dtype=model.dtype)
                    features = array(encoder(**inputs).last_hidden_state)
                elif family == 'transformers-granite':
                    inputs = processor.audio_processor(torch.from_numpy(chunk).unsqueeze(0),
                                                         device='cpu')
                    inputs.to(model.device, dtype=model.dtype)
                    features = array(encoder(inputs['input_features']).last_hidden_state)
                else:
                    inputs = processor([chunk], sampling_rate=16000, return_tensors='pt',
                                       **({'language': 'en'} if family == 'transformers-cohere' else {}))
                    inputs.to(model.device, dtype=model.dtype)
                    kwargs = {k: v for k, v in inputs.items() if k in ['input_features', 'attention_mask']}
                    encoded = encoder(**kwargs)
                    if family == 'transformers-granite-ctc':
                        # Identical to native generate(), reusing this encoder pass.
                        sequences = model.ctc_head(encoded.last_hidden_state).argmax(dim=-1)
                        if encoded.attention_mask is not None:
                            sequences[~encoded.attention_mask.bool()] = model.config.pad_token_id
                        hypotheses.extend(processor.batch_decode(sequences, skip_special_tokens=True))
                    features = array(encoded.last_hidden_state)
                    mask = getattr(encoded, 'attention_mask', None)
                    if mask is not None:
                        features = features[:int(mask[0].sum())]
            # Remove any padded positions past the architectural clock, without stretching time.
            n_valid = len(features) if is_qwen else min(len(features), math.ceil(seconds * fps))
            features = features[:n_valid]
            assert features.ndim == 2 and np.isfinite(features).all() and len(features)
            final_parts.append(features)
            for i in layers:
                values = captures[i][:n_valid]
                assert len(values) == n_valid, (i, values.shape, n_valid)
                layer_parts[i].append(values)
            if is_qwen:
                # Qwen restarts the stride-8 CNN every 100 mel frames. 13 frames
                # cover each second, with a short final interval, not a 1.04s drift.
                mel_length = int(inputs['input_features_mask'][0].sum())
                remaining = n_valid
                for mel_start in range(0, mel_length, encoder.n_window * 2):
                    mel_count = min(encoder.n_window * 2, mel_length - mel_start)
                    frame_count = (mel_count + 7) // 8
                    block_start = mel_start / 100.0
                    chunks.append({'start': offset + block_start,
                                   'duration': min(mel_count / 100.0, seconds - block_start),
                                   'frames': frame_count})
                    remaining -= frame_count
                assert remaining == 0
            else:
                chunks.append({'start': offset, 'duration': seconds, 'frames': n_valid})
            offset += seconds
        if family == 'transformers-granite-ctc' and row['gate'] not in GATES:
            write_json(prediction_sidecar, {**row, 'hypothesis': ' '.join(hypotheses),
                                           'chunks': chunks})
            print(json.dumps({'encoder': tag, 'prediction_only_clip': number,
                              'total': len(records), 'id': row['id']}), flush=True)
            continue
        if is_v2 and hypotheses and any(text.strip() for text in hypotheses):
            if not any(word['end'] > word['start'] >= 0 for word in words):
                raise RuntimeError('Parakeet returned no positive-duration word timestamps; '
                                   'refusing an unusable span cache')
        out.parent.mkdir(parents=True, exist_ok=True)
        features = np.concatenate(final_parts)
        np.save(out, features)
        for i in layers:
            layer_path = CACHE / tag / 'layers' / str(i) / row['gate'] / out.name
            layer_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(layer_path, np.concatenate(layer_parts[i]))
        metadata = {**model_meta, **row, 'dims': features.shape[1], 'frames': len(features),
                    'chunks': chunks, 'hypothesis': ' '.join(hypotheses), 'word_timestamps': words,
                    'feature_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
        write_json(sidecar, metadata)
        write_json(CACHE / tag / 'metadata.json', model_meta)
        print(json.dumps({'encoder': tag, 'clip': number, 'total': len(records), 'id': row['id'],
                          'shape': features.shape, 'word_times': len(words),
                          'elapsed_seconds': round(time.monotonic() - start_time, 1)}), flush=True)
    if family == 'transformers-granite-ctc':
        for gate in sorted({row['gate'] for row in records}):
            predictions = []
            for row in records:
                if row['gate'] != gate:
                    continue
                source = (CACHE / tag / gate / f'{row["id"]}.json' if gate in GATES else
                          CACHE / 'ctc_predictions' / gate / f'{row["id"]}.json')
                saved = json.loads(source.read_text())
                predictions.append({'id': row['id'], 'hypothesis': saved['hypothesis'],
                                    'audio_seconds': saved['duration_seconds'],
                                    'chunks': len(saved['chunks']), 'sha256_audio': row.get('sha256_audio'),
                                    'inference_identity': [gate, model_id, 1]})
            prediction_path = Path(__file__).parent / 'preds' / gate / f'{tag}.jsonl'
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            write_json(Path(str(prediction_path) + '.meta.json'), {
                **model_meta, 'gate': gate, 'family': family, 'complete': True,
                'clips': len(predictions), 'prediction_ids': [r['id'] for r in predictions],
                'prediction_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                'merge_tail_below_seconds': 1, 'chunk_seconds': 30,
                'provenance': 'new latent extraction CTC decode; same encoder pass plus native head'})
    for handle in handles:
        handle.remove()
    write_json(CACHE / tag / 'complete.json', {'clips': sum(r['gate'] in GATES for r in records),
                                             'prediction_clips': len(records) if family == 'transformers-granite-ctc' else None,
                                             'elapsed_seconds': time.monotonic() - start_time})


if __name__ == '__main__':
    main()
