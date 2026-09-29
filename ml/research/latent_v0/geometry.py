"""CPU geometry and exact linear CKA of cached ASR frame representations.

Native features stay native for PCA. Cross-encoder comparisons use the same
12.5 Hz time bins and identical sampled clip/frame keys for every encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TAGS = (
    "parakeet-tdt-0.6b-v2", "parakeet-unified-en-0.6b", "Qwen3-ASR-0.6B-hf",
    "cohere-transcribe-03-2026", "granite-speech-5.0-470m-turboctc",
    "granite-speech-4.1-2b",
)
LABELS = dict(zip(TAGS, ("Parakeet v2", "Parakeet unified", "Qwen 0.6B",
                       "Cohere", "Granite 5.0", "Granite 4.1"), strict=True))
GATES = ("novel180", "hard77", "uncertain48", "wispr_holdout120",
         "aqua_new_holdout", "wispr_edit25")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def chunks_for(array, metadata):
    """Yield native-array slices and their physical time coordinates."""
    chunks = metadata.get("chunks") or [{"start": 0,
                "duration": metadata["duration_seconds"], "frames": len(array)}]
    offset = 0
    for chunk in chunks:
        count = int(chunk["frames"])
        if count < 0 or offset + count > len(array):
            raise ValueError("Chunk frame counts disagree with cached feature length")
        yield array[offset:offset + count], float(chunk["start"]), float(chunk["duration"])
        offset += count
    if offset != len(array):
        raise ValueError(f"Metadata accounts for {offset} frames, cache has {len(array)}")


def align_frames(array, metadata, target_fps=12.5):
    """Average native frame intervals onto a fixed physical-time grid.

    Each native frame covers [start + i/fps, start + (i+1)/fps), clipped
    at chunk duration. Coarser features are linearly interpolated at bin
    centres. Chunk boundaries are weighted by actual overlap. Uncovered
    output bins are NaN, never invented by stretching the last feature.
    """
    duration = float(metadata["duration_seconds"])
    source_fps = float(metadata["frame_rate"])
    if duration <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Duration and frame rates must be positive")
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected [frames, dims], got {array.shape}")
    count = int(math.ceil(duration * target_fps - 1e-8))
    values = np.zeros((count, array.shape[1]), dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)
    for part, start, chunk_duration in chunks_for(array, metadata):
        if not len(part):
            continue
        end = min(duration, start + chunk_duration, start + len(part) / source_fps)
        indices = np.arange(max(0, int(math.floor(start * target_fps))),
                            min(count, int(math.ceil(end * target_fps - 1e-8))))
        left = np.maximum(indices / target_fps, start)
        right = np.minimum((indices + 1) / target_fps, end)
        valid = right > left
        indices, left, right = indices[valid], left[valid], right[valid]
        overlap = right - left
        data = np.asarray(part, dtype=np.float64)
        if source_fps + 1e-8 >= target_fps:
            integral = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(data, axis=0)))

            def evaluate(times):
                position = np.clip((times - start) * source_fps, 0, len(part))
                whole = np.floor(position).astype(int)
                fraction = position - whole
                return integral[whole] + fraction[:, None] * data[np.minimum(whole, len(part)-1)]

            contribution = (evaluate(right) - evaluate(left)) / source_fps
        else:
            position = ((left + right) / 2 - start) * source_fps - 0.5
            position = np.clip(position, 0, len(part) - 1)
            lo = np.floor(position).astype(int)
            hi = np.minimum(lo + 1, len(part)-1)
            frac = position - lo
            contribution = ((1-frac[:, None]) * data[lo] + frac[:, None] * data[hi]) * overlap[:, None]
        values[indices] += contribution
        weights[indices] += overlap
    np.divide(values, weights[:, None], out=values, where=weights[:, None] > 0)
    values[weights == 0] = np.nan
    return values.astype(np.float32)


def inventory(root, tag):
    result = {}
    for gate in GATES:
        for path in sorted((root / tag / gate).glob("*.npy")):
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                result[(gate, path.stem)] = (path, read_json(meta_path))
    return result


def select_indices(lengths, cap, seed):
    """Uniform frame sampling without replacement across clips."""
    total = sum(lengths)
    chosen = np.sort(np.random.default_rng(seed).choice(total, min(cap, total), replace=False))
    result, offset = [], 0
    for length in lengths:
        lo, hi = np.searchsorted(chosen, (offset, offset + length))
        result.append(chosen[lo:hi] - offset)
        offset += length
    return result


def geometry_stats(records, cap, seed):
    keys = sorted(records)
    shapes = [np.load(records[key][0], mmap_mode="r").shape for key in keys]
    dimensions = sorted({shape[1] for shape in shapes})
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent encoder dimensions: {dimensions}")
    selected = select_indices([shape[0] for shape in shapes], cap, seed)
    samples, norm_sum, frame_count, cosine_sum, pairs = [], 0., 0, 0., 0
    boundary_cosine_sum, boundary_pairs = 0., 0
    for key, indices in zip(keys, selected, strict=True):
        path, meta = records[key]
        array = np.load(path, mmap_mode="r")
        if len(indices):
            samples.append(np.asarray(array[indices], dtype=np.float64))
        previous_last = None
        for part, _, _ in chunks_for(array, meta):
            if len(part) and previous_last is not None:
                first = np.asarray(part[0], dtype=np.float64)
                denominator = np.linalg.norm(previous_last) * np.linalg.norm(first)
                if denominator > 0:
                    boundary_cosine_sum += float(previous_last @ first / denominator)
                    boundary_pairs += 1
            if len(part):
                previous_last = np.asarray(part[-1], dtype=np.float64)
            for start in range(0, len(part), 4096):
                end = min(start + 4096, len(part))
                x = np.asarray(part[start:end], dtype=np.float64)
                norms = np.linalg.norm(x, axis=1)
                norm_sum += float(norms.sum())
                frame_count += len(x)
                if end - start > 1:
                    denominator = norms[:-1] * norms[1:]
                    good = denominator > 0
                    cosine_sum += float((np.einsum("ij,ij->i", x[:-1], x[1:])[good] / denominator[good]).sum())
                    pairs += int(good.sum())
                if start > 0:
                    previous = np.asarray(part[start-1], dtype=np.float64)
                    denominator = np.linalg.norm(previous) * norms[0]
                    if denominator > 0:
                        cosine_sum += float(previous @ x[0] / denominator)
                        pairs += 1
    x = np.concatenate(samples)
    x -= x.mean(axis=0)
    covariance = x.T @ x / max(1, len(x)-1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    total = float(eigenvalues.sum())
    spectrum = eigenvalues / total if total else np.zeros_like(eigenvalues)
    total_duration = sum(float(meta["duration_seconds"]) for _, meta in records.values())
    nominal_rates = sorted({float(meta["frame_rate"]) for _, meta in records.values()})
    return {"clips": len(keys), "native_frames": frame_count,
            "duration_seconds": total_duration,
            "actual_effective_frame_rate": frame_count / total_duration,
            "nominal_frame_rates": nominal_rates,
            "frame_rate_note": "Physical alignment uses nominal within-chunk intervals; actual effective rate is total frames divided by total duration, not a uniform sampling clock.",
            "pca_frames": len(x), "dimension": dimensions[0],
            "frame_rates": nominal_rates,
            "eigenvalues": eigenvalues.tolist(), "explained_variance_ratio": spectrum.tolist(),
            "effective_rank": float(total**2 / np.square(eigenvalues).sum()) if total else 0.,
            "components_for_90_percent": int(np.searchsorted(np.cumsum(spectrum), .9) + 1),
            "mean_norm": norm_sum / frame_count,
            "mean_consecutive_cosine": cosine_sum / pairs if pairs else None,
            "mean_consecutive_cosine_within_chunk": cosine_sum / pairs if pairs else None,
            "mean_consecutive_cosine_all_within_clip": (cosine_sum + boundary_cosine_sum) / (pairs + boundary_pairs) if pairs + boundary_pairs else None,
            "consecutive_pairs": pairs,
            "consecutive_pairs_all_within_clip": pairs + boundary_pairs,
            "chunk_boundary_pairs": boundary_pairs}


def centered(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0)


def cka_self(x):
    covariance = x.T @ x
    return float(np.einsum("ij,ij->", covariance, covariance))


def linear_cka_centered(x, y, xx=None, yy=None):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("CKA needs matching samples and at least two observations")
    cross = x.T @ y
    xx = cka_self(x) if xx is None else xx
    yy = cka_self(y) if yy is None else yy
    denominator = math.sqrt(xx * yy)
    return float(np.square(cross).sum() / denominator) if denominator else None


def sample_plan(inventories, cap, seed):
    keys = sorted(set.intersection(*(set(item) for item in inventories)))
    valid_by_clip = []
    for key in keys:
        valid = None
        for records in inventories:
            path, meta = records[key]
            aligned = align_frames(np.load(path, mmap_mode="r"), meta)
            mask = np.isfinite(aligned).all(axis=1)
            if valid is None:
                valid = mask
            else:
                count = min(len(valid), len(mask))
                valid = valid[:count] & mask[:count]
        valid_by_clip.append(np.flatnonzero(valid))
    selections = select_indices([len(item) for item in valid_by_clip], cap, seed)
    plan = [(key, valid[indices]) for key, valid, indices in
            zip(keys, valid_by_clip, selections, strict=True) if len(indices)]
    serial = [{"gate": key[0], "id": key[1], "frames": indices.tolist()} for key, indices in plan]
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
    return plan, {"sample_frames": sum(len(indices) for _, indices in plan),
                  "common_clips": len(keys), "available_common_frames": sum(map(len, valid_by_clip)),
                  "seed": seed, "sample_key_sha256": digest, "sample_keys": serial}


def gather(records, plan, root=None, tag=None, layer=None):
    arrays = []
    for key, indices in plan:
        path, meta = records[key]
        if layer is not None:
            path = root / tag / "layers" / str(layer) / key[0] / (key[1] + ".npy")
            layer_meta = path.with_suffix(".json")
            if layer_meta.exists():
                meta = read_json(layer_meta)
        aligned = align_frames(np.load(path, mmap_mode="r"), meta)
        arrays.append(aligned[indices])
    return np.concatenate(arrays).astype(np.float64)


def setup_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#111827", "axes.facecolor": "#111827",
                         "savefig.facecolor": "#111827", "font.size": 10})
    return plt


def heatmap(path, matrix, row_labels, col_labels, title, vmin=0, vmax=1):
    plt = setup_plots()
    fig, ax = plt.subplots(figsize=(max(7, len(col_labels)*.4 + 3), max(5, len(row_labels)*.3 + 2)))
    if not row_labels or not col_labels:
        ax.text(.5, .5, "Unavailable: required encoder/layer cache missing", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        values = np.array([[np.nan if x is None else x for x in row] for row in matrix])
        im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(col_labels)), col_labels, rotation=55, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels)
        fig.colorbar(im, ax=ax, label="Linear CKA")
        if values.size <= 100:
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="black" if values[i,j] > .65 else "white")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_geometry(inventories, out, cap, seed):
    result = {}
    for tag, records in inventories.items():
        print(f"PCA {tag}: {len(records)} clips", flush=True)
        result[tag] = geometry_stats(records, cap, seed)
        write_json(out / "geometry.json", result)
    plt = setup_plots()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag, stats in result.items():
        spectrum = stats["explained_variance_ratio"]
        label = f"{LABELS[tag]} (rank {stats['effective_rank']:.1f})"
        axes[0].semilogy(np.arange(1, len(spectrum)+1), spectrum, label=label)
        axes[1].plot(np.arange(1, len(spectrum)+1), np.cumsum(spectrum), label=label)
    axes[0].set(xlabel="Principal component", ylabel="Fraction of variance", title="Native-frame PCA spectra")
    axes[1].set(xlabel="Number of components", ylabel="Cumulative explained variance", title="Concentration of encoder variance", ylim=(0, 1.01))
    axes[1].axhline(.9, color="gray", linestyle=":")
    for ax in axes:
        ax.grid(alpha=.15)
        if result:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pca_spectra.png", dpi=180)
    plt.close(fig)


def run_encoder_cka(inventories, out, cap, seed):
    tags = list(inventories)
    if not tags:
        write_json(out / "cka.json", {"status": "No usable encoder caches"})
        heatmap(out / "cka_encoders.png", [], [], [], "Encoder CKA unavailable")
        heatmap(out / "cka_per_gate.png", [], [], [], "Per-gate CKA unavailable")
        return
    plan, metadata = sample_plan(list(inventories.values()), cap, seed)
    write_json(out / "cka_sample_keys.json", metadata)
    if metadata["sample_frames"] < 2:
        metadata.pop("sample_keys")
        metadata["status"] = "Fewer than two common aligned frames"
        write_json(out / "cka.json", metadata)
        heatmap(out / "cka_encoders.png", [], [], [], "Encoder CKA unavailable")
        heatmap(out / "cka_per_gate.png", [], [], [], "Per-gate CKA unavailable")
        return
    arrays = {tag: gather(inventories[tag], plan) for tag in tags}
    metadata.pop("sample_keys")
    metadata.update({"estimator": "biased centered linear CKA, exact float64 feature covariance",
                     "alignment_fps": 12.5, "tags": tags})
    centered_arrays = {tag: centered(value) for tag, value in arrays.items()}
    norms = {tag: cka_self(value) for tag, value in centered_arrays.items()}
    matrix = [[None]*len(tags) for _ in tags]
    for i, a in enumerate(tags):
        for j in range(i, len(tags)):
            b = tags[j]
            value = linear_cka_centered(centered_arrays[a], centered_arrays[b], norms[a], norms[b])
            matrix[i][j] = matrix[j][i] = value
            print(f"CKA {a} / {b}: {value}", flush=True)
    metadata["matrix"] = matrix
    del centered_arrays
    gates = np.concatenate([np.full(len(indices), key[0], dtype=object) for key, indices in plan])
    per_gate = {}
    if TAGS[0] in tags:
        for gate in GATES:
            mask = gates == gate
            if mask.sum() < 2:
                continue
            base = centered(arrays[TAGS[0]][mask])
            base_norm = cka_self(base)
            per_gate[gate] = {"sample_frames": int(mask.sum()), "cka_vs_parakeet_v2": {}}
            for tag in tags:
                other = centered(arrays[tag][mask])
                per_gate[gate]["cka_vs_parakeet_v2"][tag] = linear_cka_centered(base, other, base_norm)
    metadata["per_gate"] = per_gate
    write_json(out / "cka.json", metadata)
    heatmap(out / "cka_encoders.png", matrix, [LABELS[t] for t in tags], [LABELS[t] for t in tags],
            f"Encoder similarity: {metadata['sample_frames']:,} identical aligned frames")
    heatmap(out / "cka_per_gate.png", [[item["cka_vs_parakeet_v2"][t] for t in tags] for item in per_gate.values()],
            list(per_gate), [LABELS[t] for t in tags], "Similarity to Parakeet v2 by gate")


def available_layers(root, tag, records):
    base = root / tag / "layers"
    return sorted([path.name for path in base.iterdir() if path.is_dir()],
                  key=lambda value: (0, int(value)) if value.isdigit() else (1, value)) if base.exists() else []


def run_layer_cka(root, inventories, out, cap, seed):
    a, b = TAGS[0], TAGS[2]
    rows = available_layers(root, a, inventories.get(a, {}))
    cols = available_layers(root, b, inventories.get(b, {}))
    if not rows or not cols or a not in inventories or b not in inventories:
        write_json(out / "cka_layers.json", {"status": "Required encoder/layer cache missing", "parakeet_layers": rows, "qwen_layers": cols})
        heatmap(out / "cka_layers_parakeet_vs_qwen.png", [], [], [], "Parakeet v2 vs Qwen layer CKA")
        return
    records_by_tag = {}
    for tag, layers in ((a, rows), (b, cols)):
        records_by_tag[tag] = {key: value for key, value in inventories[tag].items()
                               if all((root / tag / "layers" / layer / key[0] / (key[1]+".npy")).exists() for layer in layers)}
    plan, metadata = sample_plan(list(records_by_tag.values()), cap, seed)
    write_json(out / "cka_layer_sample_keys.json", metadata)
    metadata.pop("sample_keys")
    if metadata["sample_frames"] < 2:
        metadata["status"] = "Fewer than two common aligned frames with all layers cached"
        write_json(out / "cka_layers.json", metadata)
        heatmap(out / "cka_layers_parakeet_vs_qwen.png", [], [], [], "Layer CKA unavailable")
        return
    metadata.update({"parakeet_layers": rows, "qwen_layers": cols,
                     "estimator": "biased centered linear CKA, exact float64 feature covariance",
                     "alignment_fps": 12.5})
    matrix = [[None]*len(cols) for _ in rows]
    # Hold Qwen layers in float32 to reuse disk reads; cast one partner at a time.
    qwen = {}
    qwen_norms = {}
    for col in cols:
        x = gather(records_by_tag[b], plan, root, b, col)
        qwen[col] = x.astype(np.float32)
        qwen_norms[col] = cka_self(centered(x))
    for i, row in enumerate(rows):
        x = centered(gather(records_by_tag[a], plan, root, a, row))
        xx = cka_self(x)
        for j, col in enumerate(cols):
            y = centered(qwen[col])
            matrix[i][j] = linear_cka_centered(x, y, xx, qwen_norms[col])
            print(f"Layer CKA {row}/{col}: {matrix[i][j]:.6f}", flush=True)
            metadata["matrix"] = matrix
            write_json(out / "cka_layers.json", metadata)
    heatmap(out / "cka_layers_parakeet_vs_qwen.png", matrix,
            [f"Parakeet {r}" for r in rows], [f"Qwen {c}" for c in cols],
            f"Layer CKA: {metadata['sample_frames']:,} identical aligned frames")


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert np.isclose(linear_cka_centered(centered(x), centered(3*x@q + 9)), 1, atol=1e-12)
    y = rng.normal(size=(80, 5))
    assert np.isclose(linear_cka_centered(centered(x), centered(y)),
                      linear_cka_centered(centered(y), centered(x)), atol=1e-12)
    # Feature-space CKA equals the direct centered Gram-matrix definition.
    xc, yc = centered(x), centered(y)
    gram_x, gram_y = xc @ xc.T, yc @ yc.T
    direct = np.sum(gram_x * gram_y) / np.sqrt(np.square(gram_x).sum() * np.square(gram_y).sum())
    assert np.isclose(linear_cka_centered(xc, yc), direct, atol=1e-12)
    # Exact average of two fine bins into one coarse bin.
    meta = {"duration_seconds": .32, "frame_rate": 25, "chunks": [{"start": 0, "duration": .32, "frames": 8}]}
    values = np.arange(8, dtype=float)[:, None]
    np.testing.assert_allclose(align_frames(values, meta)[:, 0], [.5, 2.5, 4.5, 6.5])
    # Architectural 12.5 Hz survives an incomplete final bin without stretching.
    meta = {"duration_seconds": .18, "frame_rate": 12.5}
    np.testing.assert_allclose(align_frames(np.arange(3)[:, None], meta)[:, 0], [0, 1, 2])
    # Interpolation on a coarser grid.
    meta = {"duration_seconds": .32, "frame_rate": 6.25}
    np.testing.assert_allclose(align_frames(np.array([[0], [1]]), meta)[:, 0], [0, .25, .75, 1])
    # Qwen's 13 frames per one-second CNN block use a 12.5 Hz nominal
    # clock, with the thirteenth interval clipped to 40 ms. Two blocks
    # must remain two seconds and cannot drift to 2.08 seconds.
    meta = {"duration_seconds": 2.0, "frame_rate": 12.5,
            "chunks": [{"start": 0, "duration": 1, "frames": 13},
                       {"start": 1, "duration": 1, "frames": 13}]}
    qwen_values = np.concatenate([np.ones((13, 1)), 3*np.ones((13, 1))])
    qwen_aligned = align_frames(qwen_values, meta)
    assert qwen_aligned.shape == (25, 1)
    np.testing.assert_allclose(qwen_aligned[:12, 0], 1)
    np.testing.assert_allclose(qwen_aligned[12, 0], 2)
    np.testing.assert_allclose(qwen_aligned[13:, 0], 3)
    # A target interval crossing chunks receives the correct overlap weights.
    meta = {"duration_seconds": .16, "frame_rate": 25,
            "chunks": [{"start": 0, "duration": .06, "frames": 2}, {"start": .06, "duration": .10, "frames": 3}]}
    np.testing.assert_allclose(align_frames(np.array([[1], [1], [3], [3], [3]]), meta)[:, 0], [1.5, 3])
    print("Self-tests passed: CKA rotation/scale/translation invariance, symmetry, pooling, partial bins, interpolation, chunk joins.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/data/phonon_latent_v0"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage", choices=("all", "geometry", "encoders", "layers"), default="all")
    parser.add_argument("--pca-frames", type=int, default=200_000)
    parser.add_argument("--cka-frames", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        self_test()
        if args.self_test:
            return
        args.out.mkdir(parents=True, exist_ok=True)
        inventories = {tag: records for tag in TAGS if (records := inventory(args.cache_root, tag))}
        if args.stage in ("all", "geometry"):
            run_geometry(inventories, args.out, args.pca_frames, args.seed)
        if args.stage in ("all", "encoders"):
            run_encoder_cka(inventories, args.out, args.cka_frames, args.seed)
        if args.stage in ("all", "layers"):
            run_layer_cka(args.cache_root, inventories, args.out, args.cka_frames, args.seed)


if __name__ == "__main__":
    main()
