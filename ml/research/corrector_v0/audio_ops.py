"""16 kHz mono I/O, 30 s chunking, and mild room/noise degradation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

TARGET_SR = 16000
KOKORO_SR = 24000
CHUNK_SAMPLES = 480000  # 30 s at 16 kHz
MIN_TAIL_SAMPLES = 16000


def load_wav(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    import soundfile as sf

    wave, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1).astype(np.float32)
    if int(file_sr) != sr:
        wave = resample_to(wave, int(file_sr), sr)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"invalid audio: {path}")
    return wave


def save_wav(path: Path, wave: np.ndarray, sr: int = TARGET_SR) -> None:
    import os
    import tempfile

    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(wave)) + 1e-9)
    out = wave.astype(np.float32)
    if peak > 0.99:
        out = out * (0.99 / peak)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".wav", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # Unique sibling .wav so soundfile can infer format; os.replace is atomic.
        sf.write(str(tmp), out, sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def resample_to(wave: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    if src_sr == dst_sr:
        return wave.astype(np.float32)
    divisor = gcd(src_sr, dst_sr)
    resampled = resample_poly(wave, dst_sr // divisor, src_sr // divisor)
    return np.asarray(resampled, dtype=np.float32)


def peak_normalize(wave: np.ndarray, peak: float = 0.95) -> np.ndarray:
    current = float(np.max(np.abs(wave)) + 1e-9)
    if current == 0:
        return wave.astype(np.float32)
    return (wave * (peak / current)).astype(np.float32)


def chunk_audio(
    waveform: np.ndarray,
    chunk_samples: int = CHUNK_SAMPLES,
    minimum_tail_samples: int = MIN_TAIL_SAMPLES,
) -> list[np.ndarray]:
    """Nominal 30 s chunks; merge a sub-second tail into the previous chunk."""
    chunks = [waveform[i : i + chunk_samples] for i in range(0, len(waveform), chunk_samples)]
    if len(chunks) > 1 and len(chunks[-1]) < minimum_tail_samples:
        chunks[-2:] = [waveform[(len(chunks) - 2) * chunk_samples :]]
    return chunks


def chunk_wav_paths(wave: np.ndarray, cache_dir: Path) -> list[str]:
    import soundfile as sf

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for chunk in chunk_audio(wave):
        digest = hashlib.sha256(chunk.tobytes()).hexdigest()
        path = cache_dir / f"{digest}.wav"
        if not path.exists():
            sf.write(path, chunk, TARGET_SR, subtype="FLOAT")
        paths.append(str(path))
    return paths


def degrade_audio(wave: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Light exponential room IR plus colored noise at SNR 15-30 dB."""
    from scipy.signal import lfilter

    rt60 = float(rng.uniform(0.15, 0.35))
    snr_db = float(rng.uniform(15.0, 30.0))
    wet = 0.22
    ir_len = max(8, int(sr * rt60))
    t = np.arange(ir_len, dtype=np.float32) / float(sr)
    ir = np.exp(-3.0 * t / rt60).astype(np.float32)
    ir *= 0.85 + 0.3 * rng.standard_normal(ir_len).astype(np.float32)
    ir[0] = 1.0
    reverberated = np.convolve(wave, ir, mode="full")[: len(wave)]
    mixed = ((1.0 - wet) * wave + wet * reverberated).astype(np.float32)
    white = rng.standard_normal(len(mixed)).astype(np.float32)
    colored = lfilter([1.0], [1.0, -0.97], white).astype(np.float32)
    colored -= float(colored.mean())
    rms = float(np.sqrt(np.mean(mixed**2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(colored**2) + 1e-12))
    scale = rms / ((10.0 ** (snr_db / 20.0)) * noise_rms)
    out = mixed + colored * scale
    return peak_normalize(out, 0.95)


def estimate_seconds(text: str, duration: float | None = None) -> float:
    if duration and duration > 0:
        return float(duration)
    words = max(1, len((text or "").split()))
    return words / 2.5 + 0.4
