"""Frozen feature extraction and native NeMo TDT training/decoding for Fusion v0."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path('/home/user/phonon')
DATA = Path('/data/phonon_fusion_v0')
sys.path[:0] = [str(ROOT / 'scripts/option0'), str(ROOT / 'src')]
from gates import GATE_NAMES, chunk_audio, load_audio  # noqa: E402


def read_rows(split):
    path = DATA / 'manifests' / f'{split}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_path(encoder, split, row):
    return DATA / 'feats' / encoder / split / f'{row["cache_id"]}.npy'


def load_model():
    from gpu_guard import assert_gpu_idle
    assert_gpu_idle()
    if importlib.metadata.version('nemo_toolkit') != '2.3.0':
        raise RuntimeError('This experiment requires the unchanged repository NeMo 2.3.0 pin')
    import nemo.collections.asr as nemo_asr
    from phonon.eval import _trusted_local_nemo_connector
    paths = sorted(Path('/data/hf/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots').glob(
        '*/parakeet-tdt-0.6b-v2.nemo'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one pinned cached Parakeet v2 snapshot, found {paths}')
    model = nemo_asr.models.ASRModel.restore_from(
        str(paths[0]), map_location='cpu', save_restore_connector=_trusted_local_nemo_connector())
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    print(json.dumps({'checkpoint': str(paths[0]), 'decoding': str(model.cfg.decoding),
                      'loss': str(model.cfg.loss)}), flush=True)
    return model.cuda()


def cache_features(args):
    model = load_model()
    for split in args.splits:
        rows = read_rows(split)
        if args.limit:
            rows = rows[:args.limit]
        for index, row in enumerate(rows):
            path = feature_path('parakeet', split, row)
            meta_path = path.with_suffix('.json')
            identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get('identity') == identity and meta.get('sha256') == digest(path):
                    continue
                raise RuntimeError(f'Existing feature provenance differs: {path}')
            wave = load_audio(row)
            chunks = chunk_audio(wave)
            tensors = []
            elapsed = 0.0
            with torch.inference_mode():
                for wave_chunk in chunks:
                    signal = torch.tensor(wave_chunk, device='cuda')[None]
                    lengths = torch.tensor([len(wave_chunk)], device='cuda')
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    encoded, lens = model(input_signal=signal, input_signal_length=lengths)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    tensors.append(encoded[0, :, :int(lens[0])].T.float().cpu().numpy())
            array = np.concatenate(tensors).astype(np.float16)
            if array.shape[1] != 1024 or not np.isfinite(array).all():
                raise RuntimeError(f'Invalid Parakeet features {row["id"]}: {array.shape}')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
            write_json(meta_path, {'identity': identity, 'sha256': digest(path),
                                  'chunk_lengths': [len(t) for t in tensors],
                                  'chunk_sample_lengths': [len(c) for c in chunks],
                                  'shape': list(array.shape), 'forward_seconds': elapsed,
                                  'audio_seconds': len(wave) / 16000,
                                  'encoder': 'nvidia/parakeet-tdt-0.6b-v2',
                                  'precision': 'fp32 forward; fp16 cache'})
            print(f'cache {split} {index + 1}/{len(rows)} {row["id"]} '
                  f'frames={len(array)} forward={elapsed:.3f}s', flush=True)


class Fusion(nn.Module):
    """Pre-normalized zero projection makes the residual exactly identity at init."""
    def __init__(self, partner_dim, mode):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(1024 + partner_dim)
        self.projection = nn.Linear(1024 + partner_dim, 1024)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, parakeet, partner):
        if self.mode == 'control_a':
            partner = torch.zeros_like(partner)
        if self.mode == 'partner_only':
            parakeet = torch.zeros_like(parakeet)
        return parakeet + self.projection(self.norm(torch.cat([parakeet, partner], dim=-1)))


def decode(args):
    model = load_model()
    adapter = None
    if args.system != 'control_b':
        checkpoint = torch.load(DATA / 'checkpoints' / args.system / 'best.pt',
                                map_location='cpu', weights_only=False)
        adapter = Fusion(checkpoint['partner_dim'], checkpoint['mode']).cuda().eval()
        adapter.load_state_dict(checkpoint['adapter'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
    # Cached features are the only encoder inputs; no frozen encoder remains on GPU.
    model.encoder.cpu()
    torch.cuda.empty_cache()
    for split in args.splits:
        rows = read_rows(split)
        if args.limit:
            rows = rows[:args.limit]
        out = ROOT / 'research/fusion_v0/preds' / split / f'{args.system}.jsonl'
        records = []
        for index, row in enumerate(rows):
            path = feature_path('parakeet', split, row)
            meta = json.loads(path.with_suffix('.json').read_text())
            parakeet = torch.from_numpy(np.load(path).astype(np.float32)).cuda()
            partner = None
            if adapter is not None:
                partner = torch.from_numpy(np.load(feature_path(args.partner, split, row))
                                           .astype(np.float32)).cuda()
                assert partner.shape[0] == parakeet.shape[0]
            torch.cuda.synchronize()
            start = time.perf_counter()
            hypotheses = []
            offset = 0
            with torch.inference_mode():
                for length in meta['chunk_lengths']:
                    features = parakeet[offset:offset + length][None]
                    if adapter is not None:
                        features = adapter(features, partner[offset:offset + length][None])
                    outputs = model.decoding.rnnt_decoder_predictions_tensor(
                        features.transpose(1, 2).contiguous(),
                        torch.tensor([length], device='cuda'), return_hypotheses=False)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    hypotheses.append(str(getattr(outputs[0], 'text', outputs[0])).strip())
                    offset += length
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            records.append({'id': row['id'], 'hypothesis': ' '.join(hypotheses),
                            'audio_seconds': meta['audio_seconds'], 'elapsed_seconds': elapsed,
                            'encoder_seconds': meta['forward_seconds'],
                            'chunks': len(meta['chunk_lengths'])})
            print(f'decode {args.system} {split} {index + 1}/{len(rows)} '
                  f'{elapsed:.3f}s', flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(r) + '\n' for r in records))
        total_audio = sum(r['audio_seconds'] for r in records)
        elapsed = sum(r['elapsed_seconds'] for r in records)
        write_json(Path(str(out) + '.meta.json'), {
            'model': args.system, 'family': 'nemo-cached', 'gate': split,
            'complete': not args.limit, 'clips': len(records),
            'prediction_sha256': digest(out), 'prediction_ids': [r['id'] for r in records],
            'audio_seconds': total_audio, 'elapsed_seconds': elapsed,
            'realtime_factor': elapsed / total_audio if total_audio else None,
            'timing_scope': 'cached fp16 features to text only; encoder time reported separately',
            'feature_precision': 'fp16 cache -> fp32 native decoder', 'chunk_seconds': 30,
            'merge_tail_below_seconds': 1,
        })


def train(args):
    # Native TDT creates large temporary logits; avoid fragmentation across varied clips.
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model()
    model.encoder.cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.decoder, model.joint):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    torch.cuda.empty_cache()
    train_rows, dev_rows = read_rows('train'), read_rows('dev')
    first = np.load(feature_path(args.partner, 'train', train_rows[0]), mmap_mode='r')
    adapter = Fusion(first.shape[1], args.system).cuda()
    optimizer = torch.optim.AdamW([
        {'params': adapter.parameters(), 'lr': 1e-4},
        {'params': list(model.decoder.parameters()) + list(model.joint.parameters()), 'lr': 1e-5},
    ], weight_decay=0.01)
    # Use exactly NeMo training_step's decoder, joint and model-owned TDT loss calls.
    # Calling these directly avoids Lightning logging and unnecessary training WER decode.
    model.joint.set_fuse_loss_wer(False)
    target_dir = DATA / 'checkpoints' / args.system
    target_dir.mkdir(parents=True, exist_ok=True)
    exclusions = []

    def prepare(rows, split):
        prepared = []
        for row in rows:
            assert row['train_allowed'] is True, row['id']
            tokens = model.tokenizer.text_to_ids(row['reference'])
            p = feature_path('parakeet', split, row)
            q = feature_path(args.partner, split, row)
            frames = np.load(p, mmap_mode='r').shape[0]
            if not tokens:
                exclusions.append({'id': row['id'], 'split': split, 'reason': 'empty_tokens'})
                continue
            prepared.append({**row, 'tokens': tokens, 'frames': frames, 'p': p, 'q': q})
        return prepared

    train_rows, dev_rows = prepare(train_rows, 'train'), prepare(dev_rows, 'dev')
    if not train_rows or not dev_rows:
        raise RuntimeError('Training and dev must both contain permitted nonempty labels')

    def batches(rows, shuffle):
        values = list(rows)
        if shuffle:
            random.Random(args.seed + epoch).shuffle(values)
        batch, frames, cells = [], 0, 0
        for row in values:
            projected_frames = max([r['frames'] for r in batch] + [row['frames']])
            projected_tokens = max([len(r['tokens']) for r in batch] + [len(row['tokens'])])
            projected_cells = (len(batch) + 1) * projected_frames * (projected_tokens + 1)
            if batch and (frames + row['frames'] > args.batch_frames
                          or projected_cells > args.max_joint_cells):
                yield batch
                batch, frames, cells = [], 0, 0
            batch.append(row)
            frames += row['frames']
            cells = max(cells, projected_cells)
        if batch:
            yield batch

    def loss_for(batch):
        max_frames = max(row['frames'] for row in batch)
        max_tokens = max(len(row['tokens']) for row in batch)
        p = torch.zeros(len(batch), max_frames, 1024, device='cuda')
        q = torch.zeros(len(batch), max_frames, first.shape[1], device='cuda')
        targets = torch.zeros(len(batch), max_tokens, dtype=torch.long, device='cuda')
        lengths, target_lengths = [], []
        for i, row in enumerate(batch):
            a, b = np.load(row['p']), np.load(row['q'])
            if a.shape[0] != b.shape[0]:
                raise ValueError(f'Feature alignment differs: {row["id"]}')
            p[i, :len(a)] = torch.from_numpy(a.astype(np.float32)).cuda()
            q[i, :len(b)] = torch.from_numpy(b.astype(np.float32)).cuda()
            targets[i, :len(row['tokens'])] = torch.tensor(row['tokens'], device='cuda')
            lengths.append(len(a))
            target_lengths.append(len(row['tokens']))
        lengths = torch.tensor(lengths, device='cuda')
        target_lengths = torch.tensor(target_lengths, device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoded = adapter(p, q).transpose(1, 2).contiguous()
            decoder, target_length, _ = model.decoder(targets=targets, target_length=target_lengths)
            joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        # NeMo's own TDT backend consumes float logits with autocast disabled.
        return model.loss(log_probs=joint.float(), targets=targets,
                          input_lengths=lengths, target_lengths=target_length)

    start = time.monotonic()
    best, step, epoch = float('inf'), 0, 0
    history = []
    skipped_ids = set()
    exclusion_path = DATA / 'training_exclusions.json'
    if exclusion_path.exists():
        exclusions = json.loads(exclusion_path.read_text())
        skipped_ids = {r['id'] for r in exclusions}

    if args.action == 'validate-train':
        model.decoder.train()
        model.joint.train()
        adapter.train()
        # Fail a whole clip twice before excluding it, then use the SAME manifest mask
        # for every arm. No splitting or guessed alignment of its reference text.
        checked = []
        for split, values in [('train', train_rows), ('dev', dev_rows)]:
            for row in sorted(values, key=lambda r: r['frames'] * len(r['tokens']), reverse=True):
                if row['id'] in skipped_ids:
                    continue
                # After the largest feasible shape passes, smaller rows fit the same
                # conservative joint-cell ceiling and need no redundant full pass.
                cells = row['frames'] * (len(row['tokens']) + 1)
                if checked and cells <= min(checked):
                    continue
                for attempt in (1, 2):
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        value = loss_for([row])
                        value.backward()
                        checked.append(cells)
                        print(f'validated {split} {row["id"]} cells={cells}', flush=True)
                        del value
                        break
                    except torch.OutOfMemoryError as exc:
                        error = {'stage': 'native_tdt_memory_validation', 'attempt': attempt,
                                 'id': row['id'], 'split': split, 'joint_cells': cells,
                                 'error': f'{type(exc).__name__}: {exc}'}
                        with (DATA / 'errors.jsonl').open('a') as output:
                            output.write(json.dumps(error) + '\n')
                        print(json.dumps(error), flush=True)
                        if attempt == 2:
                            exclusions.append({**error, 'reason': 'native TDT OOM twice',
                                               'duration_seconds': row['duration_seconds']})
                            skipped_ids.add(row['id'])
                    finally:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                write_json(exclusion_path, exclusions)
        write_json(DATA / 'training_validation.json', {
            'complete': True, 'max_verified_joint_cells': max(checked) if checked else None,
            'exclusions': exclusions,
            'train_retained': sum(r['id'] not in skipped_ids for r in train_rows),
            'dev_retained': sum(r['id'] not in skipped_ids for r in dev_rows)})
        return

    if not (DATA / 'training_validation.json').exists():
        raise RuntimeError('Run validate-train before any training arm')

    def evaluate():
        model.decoder.eval()
        model.joint.eval()
        adapter.eval()
        total, count = 0.0, 0
        random_state = random.getstate()
        random.seed(args.seed + 999)
        with torch.no_grad():
            for batch in batches([r for r in dev_rows if r['id'] not in skipped_ids], False):
                value = loss_for(batch)
                total += float(value) * len(batch)
                count += len(batch)
        random.setstate(random_state)
        return total / count

    while epoch < args.epochs and time.monotonic() - start < args.max_seconds:
        epoch += 1
        model.decoder.train()
        model.joint.train()
        adapter.train()
        for batch in batches([r for r in train_rows if r['id'] not in skipped_ids], True):
            if time.monotonic() - start >= args.max_seconds:
                break
            optimizer.zero_grad(set_to_none=True)
            cells = len(batch) * max(r['frames'] for r in batch) * (
                max(len(r['tokens']) for r in batch) + 1)
            if cells > args.max_joint_cells:
                torch.cuda.empty_cache()
            loss = loss_for(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite native TDT loss at step {step}: {float(loss)}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(model.decoder.parameters())
                + list(model.joint.parameters()), 1.0)
            optimizer.step()
            step += 1
            if step % 10 == 0 or step == 1:
                print(json.dumps({'system': args.system, 'epoch': epoch, 'step': step,
                                  'loss': float(loss), 'elapsed_seconds': time.monotonic() - start}),
                      flush=True)
            del loss
            if args.smoke_steps and step >= args.smoke_steps:
                break
        dev_loss = evaluate()
        if not math.isfinite(dev_loss):
            raise RuntimeError(f'Nonfinite native TDT development loss: {dev_loss}')
        history.append({'epoch': epoch, 'step': step, 'dev_loss': dev_loss,
                        'elapsed_seconds': time.monotonic() - start})
        if dev_loss < best:
            best = dev_loss
            torch.save({'adapter': adapter.state_dict(), 'decoder': model.decoder.state_dict(),
                        'joint': model.joint.state_dict(), 'partner_dim': first.shape[1],
                        'mode': args.system, 'epoch': epoch, 'step': step, 'dev_loss': best,
                        'partner': args.partner}, target_dir / 'best.pt')
        write_json(target_dir / 'training.json', {
            'system': args.system, 'partner': args.partner, 'seed': args.seed,
            'cuda_allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            'large_clip_cache_policy': 'empty_cache before oversized single-clip batches',
            'epochs': epoch, 'steps': step, 'best_dev_loss': best, 'history': history,
            'train_clips': len(train_rows), 'dev_clips': len(dev_rows), 'exclusions': exclusions,
            'elapsed_seconds': time.monotonic() - start, 'max_seconds': args.max_seconds,
            'adapter_lr': 1e-4, 'decoder_joint_lr': 1e-5, 'batch_frames': args.batch_frames,
            'max_joint_cells': args.max_joint_cells, 'optional_transformer_layers': 0,
            'train_manifest_sha256': digest(DATA / 'manifests/train.jsonl'),
            'dev_manifest_sha256': digest(DATA / 'manifests/dev.jsonl'),
            'loss': str(type(model.loss)), 'bf16': True, 'smoke_only': bool(args.smoke_steps),
        })
        print(f'{args.system} epoch={epoch} dev_loss={dev_loss} best={best}', flush=True)
        if args.smoke_steps:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cache', 'decode', 'train', 'validate-train'])
    parser.add_argument('--splits', nargs='+', default=list(GATE_NAMES))
    parser.add_argument('--partner', default='qwen')
    parser.add_argument('--system', default='control_b')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-seconds', type=float, default=9000)
    parser.add_argument('--batch-frames', type=int, default=1600)
    parser.add_argument('--max-joint-cells', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', '/data/hf')
    torch.set_num_threads(4)
    {'cache': cache_features, 'decode': decode, 'train': train, 'validate-train': train}[args.action](args)


if __name__ == '__main__':
    main()
