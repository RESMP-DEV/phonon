"""GPU transcription only; gate reference text is never supplied to a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HOME'] = '/data/hf'
os.environ['HF_HUB_CACHE'] = '/data/hf/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/hf/hub'
os.environ.setdefault('HF_TOKEN_PATH', str(Path.home() / '.cache/huggingface/token'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from gates import GATE_NAMES, ROOT, chunk_audio, hash_matches, load_audio, load_gate  # noqa: E402


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_model(family, name):
    import torch
    if family == 'nemo':
        import nemo.collections.asr as nemo_asr
        from phonon.eval import _trusted_local_nemo_connector, restore_parakeet_unified
        if name.endswith('.nemo'):
            model = nemo_asr.models.ASRModel.restore_from(
                name, save_restore_connector=_trusted_local_nemo_connector())
        elif 'parakeet-unified' in name:
            model = restore_parakeet_unified(nemo_asr, name)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(name)
        return model.cuda().eval(), None
    import transformers
    processor = transformers.AutoProcessor.from_pretrained(name)
    cls = {
        'transformers-cohere': 'CohereAsrForConditionalGeneration',
        'transformers-qwen3asr': 'AutoModelForMultimodalLM',
        'transformers-granite': 'AutoModelForSpeechSeq2Seq',
        'transformers-granite-ctc': 'AutoModelForCTC',
    }[family]
    kwargs = {} if family == 'transformers-granite-ctc' else {'dtype': torch.bfloat16}
    model = getattr(transformers, cls).from_pretrained(name, device_map='cuda', **kwargs)
    return model.eval(), processor


def transcribe_batch(family, model, processor, waves, paths, model_id=''):
    import torch
    if family == 'transformers-granite-ctc':
        # https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc
        inputs = processor(waves, sampling_rate=processor.feature_extractor.sampling_rate,
                           device=model.device)
        inputs.to(model.device, dtype=model.dtype)
        return processor.batch_decode(model.generate(**inputs), skip_special_tokens=True)
    if family == 'nemo':
        from phonon.eval import _decode_nemo_output
        outputs = model.transcribe(paths, batch_size=len(paths), return_hypotheses=True,
                                   num_workers=0, verbose=False)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return [_decode_nemo_output(item) for item in outputs]
    if family == 'transformers-qwen3asr':
        inputs = processor.apply_transcription_request(
            audio=paths).to(model.device, model.dtype)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = outputs[:, inputs['input_ids'].shape[1]:]
        return processor.decode(generated, return_format='transcription_only')
    if family == 'transformers-cohere':
        inputs = processor(waves, sampling_rate=16000, return_tensors='pt', language='en')
        chunk_index = inputs.get('audio_chunk_index')
        inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return processor.decode(outputs, skip_special_tokens=True,
                                audio_chunk_index=chunk_index, language='en')
    if len(waves) != 1:
        raise ValueError('Granite uses batch size 1')
    if model_id.endswith('granite-speech-4.1-2b-plus'):
        chat = [
            {'role': 'system', 'content':
             "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
             'You are Granite, developed by IBM. You are a helpful AI assistant'},
            {'role': 'user', 'content':
             '<|audio|> can you transcribe the speech into a written format?'},
        ]
    else:
        chat = [{'role': 'user', 'content':
                 '<|audio|>transcribe the speech with proper punctuation and capitalization.'}]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, torch.from_numpy(waves[0]).unsqueeze(0),
                       device='cuda', return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    return processor.tokenizer.batch_decode(
        outputs[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)


def eligible_rows(gate):
    rows = load_gate(gate, include_audio=True)
    if gate == 'aqua_new_holdout':
        return rows
    available = [r for r in rows if hash_matches(r['audio_path'], r.get('sha256_audio'))]
    if gate == 'personal_cuda' and len(available) != len(rows):
        return []
    return available


def compatible_metadata(meta, gate, model):
    return (meta.get('model') == model and meta.get('gate') == gate
            and (not model.endswith('granite-speech-4.1-2b-plus')
                 or meta.get('prompt_policy') == 'granite_plus_model_card_asr')
            and (meta.get('family') != 'transformers-qwen3asr'
                 or meta.get('language_mode') == 'auto')
            and (gate in {'aqua_new_holdout', 'personal_cuda'}
                 or meta.get('merge_tail_below_seconds') == 1))


def resumable_predictions(out, gate, model, rows):
    """Accept a verified prior pass and individually identified records appended since it."""
    if not out.exists():
        return [], {}
    lines = out.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    meta_path = Path(str(out) + '.meta.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    verified = 0
    if meta:
        if not compatible_metadata(meta, gate, model):
            raise ValueError(f'Existing inference policy differs: {out}')
        verified = meta.get('clips', 0)
        if hashlib.sha256(b''.join(lines[:verified])).hexdigest() != meta.get('prediction_sha256'):
            raise ValueError(f'Existing prediction prefix hash mismatch: {out}')
    expected = {row['id']: row for row in rows}
    seen = set()
    for index, record in enumerate(records):
        row_id = record['id']
        if row_id in seen or row_id not in expected:
            raise ValueError(f'Duplicate or unexpected existing prediction: {row_id}')
        if record.get('sha256_audio') != expected[row_id].get('sha256_audio'):
            raise ValueError(f'Existing prediction audio hash differs: {row_id}')
        if index >= verified and record.get('inference_identity') != [gate, model, 1]:
            raise ValueError(f'Unverified existing prediction: {row_id}')
        seen.add(row_id)
    return records, meta


def complete(out, gate, model):
    meta_path = Path(str(out) + '.meta.json')
    if not out.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    rows = eligible_rows(gate)
    try:
        records, _ = resumable_predictions(out, gate, model, rows)
    except (ValueError, KeyError, TypeError):
        return False
    ids = [row['id'] for row in records]
    return (bool(rows) and meta.get('complete') and compatible_metadata(meta, gate, model)
            and meta.get('prediction_sha256') == hashlib.sha256(out.read_bytes()).hexdigest()
            and len(ids) == len(set(ids)) and set(ids) == {r['id'] for r in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=[
        'nemo', 'transformers-cohere', 'transformers-qwen3asr', 'transformers-granite',
        'transformers-granite-ctc'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--gate', required=True, choices=GATE_NAMES)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--check-complete', action='store_true')
    args = parser.parse_args()
    if args.check_complete:
        sys.exit(0 if complete(args.out, args.gate, args.model) else 1)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if complete(args.out, args.gate, args.model):
        print(f'Already complete: {args.out}', flush=True)
        return
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    rows = eligible_rows(args.gate)
    if not rows:
        raise ValueError(f'No labeled audio available for {args.gate}')
    previous, previous_meta = resumable_predictions(args.out, args.gate, args.model, rows)
    previous_ids = {record['id'] for record in previous}
    pending = [row for row in rows if row['id'] not in previous_ids]
    print(f'Reusing {len(previous)} predictions; filling {len(pending)} missing rows', flush=True)
    batch_size = min(args.batch_size, 4 if args.family == 'nemo' else 2)
    if args.family == 'transformers-granite':
        batch_size = 1
    load_start = time.perf_counter()
    model, processor = load_model(args.family, args.model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    parameter_count = sum(p.numel() for p in model.parameters())
    # Common nominal 30-second chunks preserve all audio, including long Aqua clips.
    # Derived inputs live on /data and remain available for reproducibility.
    normalized = Path('/data/phonon_segments_root/option0_16k')
    normalized.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_audio = sum(record['audio_seconds'] for record in previous)
    inference_start = time.perf_counter()
    with args.out.open('a') as output, torch.inference_mode():
        for row in pending:
            torch.cuda.synchronize()
            clip_start = time.perf_counter()
            wave = load_audio(row)
            seconds = len(wave) / 16000
            chunks = chunk_audio(wave)
            hypotheses = []
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                paths = []
                for chunk in batch:
                    digest = hashlib.sha256(chunk.tobytes()).hexdigest()
                    path = normalized / f'{digest}.wav'
                    if not path.exists():
                        sf.write(path, chunk, 16000, subtype='FLOAT')
                    paths.append(str(path))
                result = transcribe_batch(args.family, model, processor, batch, paths, args.model)
                if isinstance(result, str):
                    result = [result]
                if len(result) != len(batch):
                    raise RuntimeError(f'Output length {len(result)} != batch {len(batch)}')
                hypotheses.extend(str(item).strip() for item in result)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - clip_start
            record = {'id': row['id'], 'hypothesis': ' '.join(hypotheses),
                      'audio_seconds': seconds, 'elapsed_seconds': elapsed,
                      'sha256_audio': row.get('sha256_audio'), 'chunks': len(chunks),
                      'inference_identity': [args.gate, args.model, 1]}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            total_audio += seconds
            print(f'{args.gate} {row["id"]}: {seconds:.2f}s audio, {elapsed:.2f}s elapsed',
                  flush=True)
    torch.cuda.synchronize()
    pass_elapsed = time.perf_counter() - inference_start
    previous_elapsed = previous_meta.get('elapsed_seconds', 0) + sum(
        record['elapsed_seconds'] for record in previous[previous_meta.get('clips', 0):])
    elapsed = previous_elapsed + pass_elapsed
    meta = {'model': args.model, 'family': args.family, 'gate': args.gate,
            'torch': version('torch'), 'transformers': version('transformers'),
            'nemo': version('nemo_toolkit'), 'parameter_count': parameter_count,
            'total_elapsed_seconds': time.perf_counter() - total_start,
            'elapsed_seconds': elapsed, 'model_load_seconds': load_seconds,
            'peak_vram_bytes': max(torch.cuda.max_memory_allocated(),
                                   previous_meta.get('peak_vram_bytes', 0)),
            'audio_seconds': total_audio,
            'realtime_factor': elapsed / total_audio, 'clips': len(rows),
            'batch_size': batch_size, 'requested_batch_size': args.batch_size,
            'dtype': str(next(model.parameters()).dtype), 'chunk_seconds': 30,
            'max_chunk_seconds': 31, 'merge_tail_below_seconds': 1,
            'max_new_tokens': (512 if args.family not in {'nemo', 'transformers-granite-ctc'}
                               else None),
            'language_mode': ('auto' if args.family == 'transformers-qwen3asr' else
                              'en' if args.family == 'transformers-cohere' else 'model_default'),
            'language_hint': 'en' if args.family == 'transformers-cohere' else None,
            'prompt_policy': ('granite_plus_model_card_asr' if args.model.endswith(
                'granite-speech-4.1-2b-plus') else 'model_card_default'),
            'hotwords': None, 'complete': True,
            'reused_clips': len(previous), 'new_clips': len(pending),
            'pass_elapsed_seconds': pass_elapsed,
            'prior_pass_metadata': previous_meta or None,
            'prediction_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
            'prediction_ids': [row['id'] for row in rows],
            'timing_scope': 'single pass including preprocessing and cold inference, excluding load',
            'gpu': torch.cuda.get_device_name(), 'repository': str(ROOT),
            'model_revision': getattr(getattr(model, 'config', None), '_commit_hash', None)}
    Path(str(args.out) + '.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
