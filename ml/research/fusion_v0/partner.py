"""Cache frozen partner audio towers on the exact Parakeet chunk/frame grid.

Run with the GPU lease held by gpu_step.sh, in the isolated transformers 5.17
environment. No decoder generation is used. Existing HF snapshots are read only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data/phonon_fusion_v0')
MODELS = {
    'qwen': 'Qwen/Qwen3-ASR-0.6B-hf',
    'cohere': 'CohereLabs/cohere-transcribe-03-2026',
    'granite': 'ibm-granite/granite-speech-5.0-470m-turboctc',
}
POLICY = 'final_tower_chunk_linear_align_corners_false_v1'
os.environ['HF_HOME'] = '/data/hf'
os.environ['HF_HUB_CACHE'] = '/data/hf/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/hf/hub'
os.environ['TMPDIR'] = str(DATA / 'tmp')
os.environ['TORCH_HOME'] = str(DATA / 'cache/torch')
os.environ['NUMBA_CACHE_DIR'] = str(DATA / 'cache/numba')
os.environ['XDG_CACHE_HOME'] = str(DATA / 'cache')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts/option0'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def snapshot(encoder):
    """Resolve the cached revision without network access or writes to HF_HOME."""
    folder = Path('/data/hf/hub') / ('models--' + MODELS[encoder].replace('/', '--'))
    revision = (folder / 'refs/main').read_text().strip()
    path = folder / 'snapshots' / revision
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Model config is not cached: {path}/config.json')
    if not list(path.glob('*.safetensors')):
        raise FileNotFoundError(f'Model weights are not cached: {path}')
    return path, revision


def load_tower(encoder, model_path):
    """Load on CPU, discard the language decoder, and move only the tower to CUDA."""
    import torch
    import transformers
    from gpu_guard import assert_gpu_idle

    assert_gpu_idle()
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True)
    classes = {
        'qwen': transformers.AutoModelForMultimodalLM,
        'cohere': transformers.CohereAsrForConditionalGeneration,
        'granite': transformers.AutoModelForCTC,
    }
    full = classes[encoder].from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa')
    if encoder == 'qwen':
        tower = full.model.audio_tower
    elif encoder == 'cohere':
        tower = full.model.encoder
    else:
        tower = full.encoder
    del full
    gc.collect()
    tower.requires_grad_(False)
    tower.eval().cuda()
    return tower, processor.feature_extractor


def prepare_features(encoder, extractor, wave):
    """Keep source segmentation; do not let Cohere re-segment long training audio."""
    import torch

    if encoder == 'cohere':
        extractor.max_audio_clip_s = max(35.0, len(wave) / 16000 + 1.0)
    features = extractor(
        wave, sampling_rate=16000, padding='longest', truncation=False,
        return_attention_mask=True, return_tensors='pt', device='cpu')
    if encoder == 'qwen':
        mask = features['attention_mask']
        # The processor pads audio shorter than 0.5 s before STFT. Keep that
        # numerical protection but exclude its synthetic tail from the tower.
        valid = max(1, len(wave) // extractor.hop_length)
        mask[:, valid:] = 0
        values = {'input_features': features['input_features'], 'input_features_mask': mask}
    else:
        values = {key: features[key] for key in ('input_features', 'attention_mask')}
        if values['input_features'].shape[0] != 1:
            raise ValueError('Feature extractor changed the source audio segmentation')
    return {key: value.to('cuda', dtype=torch.bfloat16 if value.is_floating_point()
                         else value.dtype) for key, value in values.items()}


def extract_chunk(encoder, tower, extractor, wave, target_frames):
    import torch
    from torch.nn import functional as functional

    torch.cuda.synchronize()
    start = time.perf_counter()
    features = prepare_features(encoder, extractor, wave)
    torch.cuda.synchronize()
    preprocessing_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output = tower(**features, return_dict=True)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start
    hidden = output.last_hidden_state
    if encoder == 'qwen':
        # Qwen packs only valid positions and returns [sum(valid_T), D].
        if hidden.ndim != 2:
            raise ValueError(f'Unexpected packed Qwen shape: {tuple(hidden.shape)}')
    else:
        mask = output.attention_mask
        if mask is None or hidden.shape[0] != 1:
            raise ValueError('Encoder did not return its output padding mask')
        hidden = hidden[0, mask[0].bool()]
    original_frames = hidden.shape[0]
    if not original_frames or target_frames < 1:
        raise ValueError(f'Empty temporal grid: partner={original_frames}, target={target_frames}')
    # Interpolate each matching audio chunk independently, never over a boundary.
    torch.cuda.synchronize()
    start = time.perf_counter()
    aligned = functional.interpolate(
        hidden.float().T.unsqueeze(0), size=target_frames,
        mode='linear', align_corners=False).squeeze(0).T
    torch.cuda.synchronize()
    alignment_seconds = time.perf_counter() - start
    array = aligned.to(dtype=torch.float16).cpu().numpy()
    return array, original_frames, forward_seconds, preprocessing_seconds, alignment_seconds


def source_record(row, split, revision, encoder):
    import numpy as np

    cache_id = str(row['cache_id'])
    if Path(cache_id).name != cache_id or cache_id in {'', '.', '..'}:
        raise ValueError(f'Unsafe cache_id: {cache_id!r}')
    path = DATA / 'feats/parakeet' / split / f'{cache_id}.npy'
    meta_path = path.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    lengths = meta['chunk_lengths']
    if array.ndim != 2 or array.shape[1] != 1024 or sum(lengths) != array.shape[0]:
        raise ValueError(f'Invalid Parakeet feature grid: {path}')
    if not lengths or any(not isinstance(n, int) or n < 1 for n in lengths):
        raise ValueError(f'Invalid Parakeet chunk lengths: {path}')
    stat = Path(row['audio_path']).stat()
    identity = {
        'manifest_row': row, 'source_metadata_sha256': digest(meta_path),
        'source_array_sha256': digest(path), 'audio_size': stat.st_size,
        'audio_mtime_ns': stat.st_mtime_ns, 'model': MODELS[encoder],
        'model_revision': revision, 'policy': POLICY,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return meta, identity_hash, array.shape[0]


def verified(path, identity, frames):
    import numpy as np

    try:
        meta = json.loads(path.with_suffix('.json').read_text())
        if (not meta.get('complete') or meta.get('identity_sha256') != identity
                or 'alignment_seconds' not in meta):
            return None
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if (array.shape != (frames, meta['dimension']) or array.dtype != np.float16
                or not np.isfinite(array).all() or digest(path) != meta['array_sha256']):
            return None
        return meta
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def split_wave(wave, split, source_meta):
    from gates import chunk_audio

    sample_lengths = source_meta.get('chunk_sample_lengths')
    if sample_lengths is None:
        chunks = [wave] if split in {'train', 'dev'} else chunk_audio(wave)
    else:
        if sum(sample_lengths) != len(wave) or any(n < 1 for n in sample_lengths):
            raise ValueError('Source cache sample boundaries do not match decoded audio')
        chunks, start = [], 0
        for length in sample_lengths:
            chunks.append(wave[start:start + length])
            start += length
    if len(chunks) != len(source_meta['chunk_lengths']):
        raise ValueError('Partner and Parakeet have different audio segmentation')
    return chunks


def process_split(args, split, revision, model_path, loaded):
    import numpy as np
    import torch
    import transformers
    from gates import load_audio

    manifest = DATA / 'manifests' / f'{split}.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    output_dir = DATA / 'feats' / args.encoder / split
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        source_meta, identity, frames = source_record(row, split, revision, args.encoder)
        path = output_dir / f'{row["cache_id"]}.npy'
        prior = verified(path, identity, frames)
        if prior is not None:
            records.append(prior)
            continue
        if not loaded:
            loaded.extend(load_tower(args.encoder, model_path))
        tower, extractor = loaded
        audio_start = time.perf_counter()
        wave = load_audio(row)
        audio_loading_seconds = time.perf_counter() - audio_start
        chunks = split_wave(wave, split, source_meta)
        arrays, original_lengths = [], []
        forward_seconds, preprocessing_seconds = 0.0, 0.0
        alignment_seconds = 0.0
        with torch.inference_mode():
            for chunk, target in zip(chunks, source_meta['chunk_lengths'], strict=True):
                array, original, forward, preprocessing, alignment = extract_chunk(
                    args.encoder, tower, extractor, chunk, target)
                arrays.append(array)
                original_lengths.append(original)
                forward_seconds += forward
                preprocessing_seconds += preprocessing
                alignment_seconds += alignment
        array = np.concatenate(arrays)
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite encoder features for {row["id"]}')
        temporary = path.with_suffix('.npy.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
        meta = {
            'complete': True, 'id': row['id'], 'cache_id': row['cache_id'], 'split': split,
            'identity_sha256': identity, 'array_sha256': digest(path),
            'model': MODELS[args.encoder], 'model_revision': revision, 'policy': POLICY,
            'dimension': array.shape[1], 'dtype': 'float16', 'frames': len(array),
            'chunk_lengths': source_meta['chunk_lengths'],
            'chunk_sample_lengths': [len(chunk) for chunk in chunks],
            'original_frame_lengths': original_lengths, 'audio_seconds': len(wave) / 16000,
            'forward_seconds': forward_seconds, 'preprocessing_seconds': preprocessing_seconds,
            'alignment_seconds': alignment_seconds,
            'audio_loading_seconds': audio_loading_seconds,
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'gpu': torch.cuda.get_device_name(), 'tower_dtype': 'bfloat16',
        }
        atomic_json(path.with_suffix('.json'), meta)
        records.append(meta)
        if (index + 1) % 10 == 0 or index == 0 or index + 1 == len(rows):
            print(json.dumps({'split': split, 'clips': index + 1, 'total': len(rows),
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
    audio_seconds = sum(r['audio_seconds'] for r in records)
    forward = sum(r['forward_seconds'] for r in records)
    preprocessing = sum(r['preprocessing_seconds'] for r in records)
    alignment = sum(r['alignment_seconds'] for r in records)
    summary = {
        'encoder': args.encoder, 'model': MODELS[args.encoder], 'model_revision': revision,
        'split': split, 'complete': args.limit is None, 'clips': len(records),
        'audio_seconds': audio_seconds, 'forward_seconds': forward,
        'preprocessing_seconds': preprocessing, 'manifest_sha256': digest(manifest),
        'alignment_seconds': alignment,
        'forward_realtime_factor': forward / audio_seconds if audio_seconds else None,
        'preprocessing_and_forward_realtime_factor': (
            (preprocessing + forward) / audio_seconds if audio_seconds else None),
        'preprocessing_forward_alignment_realtime_factor': (
            (preprocessing + forward + alignment) / audio_seconds if audio_seconds else None),
        'timing_policy': 'CUDA-synchronized tower forward, one measurement per cached clip; '
                         'CPU feature preparation and H2D transfer timed separately; '
                         'CUDA-synchronized float conversion and interpolation timed separately; '
                         'alignment excludes cache D2H transfer and file writes; no decoder',
    }
    summary_name = '_summary.json' if args.limit is None else f'_summary_limit{args.limit}.json'
    atomic_json(output_dir / summary_name, summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=MODELS, default='qwen')
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if any(Path(split).name != split or split in {'.', '..'} for split in args.splits):
        parser.error('Split names must be plain directory names')
    for path in (DATA / 'tmp', DATA / 'cache'):
        path.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(4)
    model_path, revision = snapshot(args.encoder)
    loaded = []
    for split in args.splits:
        process_split(args, split, revision, model_path, loaded)


if __name__ == '__main__':
    main()
