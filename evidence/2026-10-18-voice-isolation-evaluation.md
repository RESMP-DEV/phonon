# Receipt: voice-isolation A/B for the Phonon dictation path

Date: 2026-10-18
Host: main machine (M4 Max), plus one B550-side scoring lane from the
earlier 100-clip run
Branch: `ml/portable-harness`

## Question

Should Phonon run an offline voice-isolation or noise-suppression stage on its
own dictation audio, and which candidate is worth keeping?

This is deliberately **not** framed as Aqua muffling repair. The prior audit
(`evidence/2026-10-18-aqua-audio-quality-audit.md`) shows the Aqua cached WAVs
are already bandwidth-starved before any processing, so no denoiser can restore
them. The question here is narrower and general: does isolation improve
dictation robustness on Phonon's own capture path, at acceptable bandwidth and
latency cost?

## Candidates

- RNNoise via ffmpeg `arnndn`, four upstream models (`std`, `cb`, `mp`, `bd`),
  model files in `~/.local/share/phonon-research/models/arnndn/`.
- DeepFilterNet 3 with the post-filter, precompiled binary
  `~/.local/share/phonon-research/bin/deep-filter`, version 0.5.6.

No candidate is wired into capture or the dictation runtime. Every artifact is a
derived sidecar; source WAVs are untouched.

## Result 1: 100 real Aqua clips, teacher-transcription agreement

All 100 private Aqua clips were processed by each candidate and re-transcribed
by the same Cohere Transcribe teacher. Comparison artifact:
`~/.local/share/phonon-research/voice-isolation-comparison.json`.

| Variant | Fair WER | Exact | Better rows | Ties | Worse rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untreated | 0.068938 | 0.38 | n/a | n/a | n/a |
| RNNoise std | 0.068127 | 0.41 | 18 | 65 | 17 |
| RNNoise cb | 0.070965 | 0.40 | 19 | 60 | 21 |
| RNNoise mp | 0.066099 | 0.45 | 20 | 65 | 15 |
| RNNoise bd | 0.065693 | 0.44 | 19 | 64 | 17 |
| DeepFilterNet3 + PF | 0.084347 | 0.35 | 19 | 45 | 36 |

`mp` and `bd` are effectively tied on this lane and ahead of untreated by less
than half a WER point. DeepFilterNet is clearly negative on these clips.

## Result 2: native 48 kHz Phonon capture, bandwidth and latency

The smoke capture is 2.315 s of 48 kHz mono PCM16 from the built-in microphone
(no speech). It cannot measure transcription quality; it can measure whether a
candidate destroys the speech band and what it costs. Artifact:
`~/.local/share/phonon-research/parallel-isolation-smoke/smoke-spectral-stats.json`.

| Variant | RMS dBFS | Energy 300-3400 Hz | Energy 3.4-8 kHz | 99% rolloff | 99.9% rolloff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw 48 kHz | -25.29 | 0.7089 | 0.0598 | 5,965 Hz | 7,942 Hz |
| RNNoise mp | -36.12 | 0.6802 | 0.0392 | 6,027 Hz | 10,932 Hz |
| RNNoise bd | -31.53 | 0.7025 | 0.0142 | 3,692 Hz | 8,452 Hz |
| DeepFilterNet + PF | -30.74 | 0.8487 | 0.0037 | 2,822 Hz | 3,668 Hz |
| DeepFilterNet + PF, -a 6 | -28.21 | 0.7736 | 0.0323 | 5,597 Hz | 7,144 Hz |

Findings:

- RNNoise `mp` is the only candidate that keeps the raw 99.9% rolloff (and
  exceeds it) while cutting noise-floor energy, at about -11 dB RMS.
- RNNoise `bd` and DeepFilterNet with the post-filter narrow the top end. On
  DeepFilterNet the 99.9% rolloff collapses from 7.9 kHz to 3.7 kHz, which is the
  same failure mode as the Aqua clips and is disqualifying for a speech-band
  enhancement stage.
- DeepFilterNet's `--pf` default attenuation limit (100 dB) is too aggressive;
  `-a 6` recovers most of the bandwidth, so the model itself is not inherently
  broken, only its default aggressiveness is.

## Cost

On native 48 kHz input, all candidates run far faster than realtime:

- RNNoise mp: 0.046 s for 2.315 s of audio (about 50x realtime).
- RNNoise bd: 0.044 s (about 53x realtime).
- DeepFilterNet: 0.137 s (about 17x realtime).

A typical dictation utterance is a few seconds, so preprocessing latency is not
a constraint at this stage.

## Commands

- `ffmpeg -i <clip> -af arnndn=model=<model> -ar 16000 <out>` (100-clip lane)
- `ffmpeg -i <48k clip> -af arnndn=model=<model> -ar 48000 -ac 1 -sample_fmt s16 <out>`
- `deep-filter --pf -o <dir> <48k clip>` and `--pf -a 6 -o <dir> <48k clip>`
- Cohere Transcribe re-scoring on B550 per variant, same teacher and prompt.

## Non-claims

- Result 1 scores against Aqua final text, which is not verified ground truth.
  These numbers cannot promote a default on their own. Human-approved labels
  from the review UI are required first.
- Result 2 uses a 2.3 s non-speech clip. It falsifies band-destroying candidates
  but cannot establish transcription quality or noise robustness.
- No candidate has been scored through the fused SALM model, which is the
  system that would actually consume the audio.
- Nothing is wired into capture, dictation, training, or export. The raw source
  remains the canonical audio path.
- Real-speech native-capture comparison, including paired noisy/clean
  conditions, is still open.

## Next gate

Record consented real speech through `phonon capture-parallel`, derive
`rnnoise mp` and `rnnoise bd` sidecars from it, and score all three (raw plus
both variants) through the fused SALM model and against human-approved labels.
Promote only if a variant improves both the teacher and SALM scores without
narrowing the band.

## Model file hashes

```
ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b  bd.rnnn
f1357c4e5be9dee8467bead486dfced2d75b640c26ad0b594fa7f102322371d9  cb.rnnn
6b8943dc4a9b6b24425873992a44f29c0577503276456af46a8854774faeb294  std.rnnn
4e84a448a4baf937992aaf4d10c8258007ec5d24219b6647dfd5fb4b563ad231  mp.rnnn
```
