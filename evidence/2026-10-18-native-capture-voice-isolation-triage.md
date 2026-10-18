# Receipt: native-capture voice-isolation triage

Date: 2026-10-18
Hosts: main machine for consented capture and derived sidecars; B550 for two
private ASR teachers
Branch: `ml/portable-harness`

## Question

For general dictation use, not Aqua muffling repair, determine which free
isolation candidates are transparent enough to justify owner review, and reduce
the review burden to the segments where preprocessing actually changes
transcription.

## Capture and derivations

`phonon capture-parallel` recorded a consented local session
`speech-isolation-v2` for 265.92 seconds at 48 kHz mono PCM16. Training audio
consent was already enabled; screen capture remained disabled. WebRTC VAD found
233.3 seconds of speech-positive frames across 93 activity runs. Source SHA-256:
`638bc36209612d28035354041c80fa68035d1c5d96e4b9fcf744717c358a91b4`.

The new `tools/voice-isolation/derive.py` utility derived all candidate WAVs from
the raw source and verified that the source SHA-256 remained unchanged:

- `rnnoise-mp`
- `rnnoise-bd`
- `dfn-pf` (retained as a negative control)
- `dfn-pf-atten6`

The source was segmented into 27 ten-second chunks for teacher scoring. No
accepted transcript existed for this session, so all quality conclusions below
are information-triage conclusions, not WER claims.

## Two-teacher aggregate result

Each arm was transcribed by Cohere Transcribe 03-2026 and Qwen3-ASR 1.7B.
Private aggregate artifact:
`~/.local/share/phonon-research/parallel-isolation-v2/aggregate-comparison.json`.

| Arm | Cohere exact vs its raw | Cohere words | Qwen exact vs its raw | Qwen words |
| --- | ---: | ---: | ---: | ---: |
| Raw | 27/27 | 640 | 27/27 | 659 |
| RNNoise mp | 12/27 | 660 | 14/27 | 663 |
| RNNoise bd | 14/27 | 684 | 13/27 | 665 |
| DeepFilterNet + PF, -a 6 | 19/27 | 641 | 14/27 | 594 |

RNNoise `mp` and `bd` are highly invasive to ASR output. DeepFilterNet with a
6 dB attenuation limit is less invasive for Cohere, but Qwen loses 65 words on
that arm. None can be claimed better without owner-accepted text.

Cross-teacher pseudo-WER is not an accuracy score, but it reinforces the same
separation:

| Arm | Cohere arm vs Qwen raw | Cohere arm vs Qwen same arm |
| --- | ---: | ---: |
| Raw | 0.272604 | 0.272604 |
| RNNoise mp | 0.243519 | 0.250977 |
| RNNoise bd | 0.196523 | 0.165039 |
| DeepFilterNet + PF, -a 6 | 0.274713 | 0.276949 |

## Owner triage

A no-transcript disagreement score was computed for every segment as the maximum
per-teacher WER of each derived candidate against that teacher's raw-audio
hypothesis. The private review bundle is:

`~/.local/share/phonon-research/parallel-isolation-v2/review/index.html`

It contains the highest-scoring 12 of 27 segments, sorted by score, with both
teachers' raw and variant hypotheses available as selectable text. All 12 rows
have local audio copied into the bundle. The owner can approve one canonical
transcript per segment without exporting the full capture.

## Harness repair

Qwen3-ASR initially reported failure after two silent ten-second segments decoded
to empty hypotheses. `completed_keys` treated those legitimate completed records
as missing work. The harness now treats a record containing both `audio` and
`hyp` fields as complete even when `hyp` is empty, records `empty_rows`, and
uses count metadata consistently during progress updates. The preserved raw
output was reused, and all four arms then completed.

## Commands and results

- `PYTHONPATH=tools/voice-isolation uv run --no-project --python 3.12 --with pytest pytest -q tools/voice-isolation/tests`: 2 passed
- `uvx ruff check tools/voice-isolation/derive.py tools/voice-isolation/tests/test_derive.py`: pass
- `uvx ruff format --check tools/voice-isolation/derive.py tools/voice-isolation/tests/test_derive.py`: pass
- `PYTHONPATH=ml/src uv run --project ml --frozen pytest ml/research/asr_teachers/tests/test_transcribe_qwen3_asr.py -q`: 4 passed
- Live `derive.py` execution over all four variants: pass, 265.92-second sidecars
- Cohere arms: 27/27 rows complete each; wall time 7.247-7.481 seconds
- Qwen arms: 27/27 rows complete each; wall time 23.145-26.027 seconds

## Non-claims

- This is not an ASR WER evaluation because there is no accepted transcript.
- No denoising is enabled in capture, dictation, training, or export.
- Raw 48 kHz audio remains canonical; every processed WAV is a derived sidecar.
- The private review bundle and aggregate artifact are local-only and are not
  committed.
- The full SALM gate remains open until owner labels exist.
