# Receipt: Aqua review-clip audio-quality audit

**Date:** 2026-10-18
**Agent:** Codex

## Question

Determine whether the clips shown in the private Aqua/Cohere review UI are
normalized, why they sound muffled, and whether the limitation is playback,
gain, or source capture.

## Method

The analysis read the 100 WAVs referenced by the private review queue and their
byte-identical copies in the generated review bundle. It reported only aggregate
signal statistics; no transcript text was printed or committed.

## Findings

- All 100 review audio files are byte-identical to Aqua's cached source WAVs.
- The review UI applies no normalization, gain, EQ, or resampling.
- Every file is:
  - mono;
  - PCM16;
  - 16,000 samples/second.
- macOS reports the built-in MacBook Pro microphone at 48,000 samples/second,
  so Aqua's native path downsamples before writing the cache.
- Median RMS is approximately -26.4 dBFS and median peak is 0.36. Only one of
  100 clips reaches digital full scale. This is not a simple low-gain problem.
- Aggregate spectrum:
  - about 57.4% of mean energy is below 300 Hz;
  - about 41.0% is in the 300–3,400 Hz speech band;
  - about 0.8% is from 4–7 kHz;
  - about 0.01% is from 7–8 kHz.
- Median frequency rolloff:
  - 95% of power is below about 1.39 kHz;
  - 99% is below about 3.50 kHz;
  - 99.9% is below about 5.07 kHz;
  - 95 of 100 clips have 99.9% rolloff below 6 kHz.

This is more aggressive than a normal full 16 kHz speech capture, which should
preserve meaningful energy approaching 7.5 kHz. The subjective muffling and
sparse consonant features are therefore consistent with Aqua's native capture or
voice-processing chain, not review-page playback.

## Current Aqua microphone settings

Aqua reports:

- microphone: system default;
- selected device: MacBook Pro built-in microphone;
- `useNativeAudio: true`;
- `muteBackgroundAudio: true`.

## Practical options

1. Use a wired or USB headset/external microphone and select it explicitly in
   Aqua. This is the highest-probability immediate quality improvement.
2. Run a controlled Aqua native-versus-JS-microphone A/B with a few spoken
   calibration sentences, then compare the same aggregate spectrum. Changing the
   setting is not useful without a paired recording test.
3. Check macOS mic mode and disable overly aggressive voice isolation or noise
   suppression for the dictation app if available.
4. Add a consent-gated parallel 48 kHz capture for future Phonon recordings.
   The current Phonon writer also discards the hardware-rate signal and linearly
   resamples to 16 kHz before writing the WAV.
5. Playback EQ or gain can make review more comfortable but cannot restore
   frequencies removed before the WAV was written. Such processed audio must not
   silently replace the original training source.

## Non-claims

- This audit does not prove whether the loss occurs in the native audio API,
  Aqua's voice-processing stack, or the 16 kHz conversion.
- Discord was not measured in this audit.
- No Aqua setting was changed and no new microphone recording was made.
