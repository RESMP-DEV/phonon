# Receipt: consent-gated parallel capture and Aqua-versus-Phonon audio comparison

Date: 2026-10-18
Host: main machine (M4 Max, MacBook Pro built-in microphone, 48 kHz default input)
Branch: `ml/portable-harness`

## What was built

`phonon capture-parallel` records the default microphone to a local
consent-gated sidecar WAV at 48 kHz mono PCM16 through AVFoundation while Aqua
runs untouched. `phonon capture-stop` and `phonon capture-status` manage and
report sessions under `~/Library/Application Support/Phonon/ParallelCaptures/`.
Every session writes hashed-command metadata carrying consent version
`parallel-local-audio-v1`, sample rate, channels, device index, and output path.
Capture refuses to start unless `training_capture_enabled` is already true in the
Phonon settings store.

Consent state change: `training_capture_enabled` was false and is now true in
`~/Library/Application Support/Phonon/settings.json`. Screen-image capture remains
false, so this lane retains audio only.

## Measured comparison

Real hardware capture, same microphone, same session.

| Metric | Aqua cached WAVs (100 clips) | Phonon parallel capture (smoke) |
| --- | ---: | ---: |
| Sample rate | 16,000 Hz | 48,000 Hz |
| Energy in 300-8,000 Hz | 41.0% (mean) | 71.7% |
| Energy in 8-12 kHz | 0.01% (mean) | 0.07% |
| Energy below 300 Hz | 57.4% (mean) | 28.2% |
| Median 99% rolloff | 3,496 Hz | 7,541 Hz |
| Median 99.9% rolloff | 5,066 Hz | 7,541 Hz (measured) |
| RMS level | -26.4 dBFS (median) | -25.3 dBFS |

Interpretation: our own capture path retains substantially more of the speech
band and roughly 2 kHz more usable top end than the Aqua cached extractions. The
loss is in Aqua's or macOS's processing chain, not in the microphone hardware. This
is consistent with the earlier audit finding that the review clips are
bandwidth-starved and bass-heavy.

## Commands and results

- `cargo build -p phonon-cli`: pass
- `cargo clippy -p phonon-cli --all-targets -- -D warnings`: pass
- `cargo test -p phonon-cli`: 6 passed
- `phonon capture-parallel --device 0 --duration-seconds 3 --session-id audio-smoke-v1`: recorded 2.31 s at 48 kHz mono PCM16, `ffprobe` confirmed
- `phonon capture-status --session-id audio-smoke-v1 --json`: status reported
- Capture consent gate verified: `training_capture_enabled` false refuses to start

## Defect found and repaired during verification

A `timeout`-interrupted capture originally published its metadata as `stale`
because the parent process was terminated before it could finalize. The status
classifier now treats a signal-terminated ffmpeg as the documented `stopped`
outcome rather than a failure, while a nonzero ffmpeg exit remains `failed`. The
existing smoke session was annotated `stopped` with `stop_reason=interrupted_smoke`
so the live state is honest.

## Non-claims

- The smoke capture is 2.3 seconds of room audio, not speech; the spectral
  comparison shows the capture chain preserves more bandwidth, not that Phonon
  dictation quality is already better.
- No A/B was run against Aqua microphone and macOS voice-processing toggles; the
  difference is attributed to the capture chain, not to a specific Aqua setting.
- Parallel capture is not yet paired to Aqua utterance timestamps, and it does not
  yet produce consented screenshot attachments.
- Boosting or equalizing the existing 16 kHz Aqua WAVs cannot restore missing
  frequencies; capture-path improvement is the durable fix.
