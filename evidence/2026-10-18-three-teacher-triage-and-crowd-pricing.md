# Receipt: three-teacher isolation triage and external rating price

Date: 2026-10-18
Hosts: main machine; B550 for the pinned Voxtral MXFP4 vLLM teacher
Branch: `ml/portable-harness`

## Question

Reduce owner-only labeling burden by adding a third independent ASR teacher,
select only high-information native-capture segments, and obtain a live external
rating price for the resulting small batch. No external upload was performed in
this step.

## Third teacher

The third teacher was the existing pinned Voxtral Small 24B MXFP4 vLLM artifact:

`/home/kearm/salm-lora/build/asr-teachers/voxtral-mxfp4-audio256-v1-vllm`

It had previously completed a 500-row historical evaluation. The server was
relaunched on one B550 GPU using the pinned vLLM 0.30.0 environment with
FlashAttention and the FlashInfer sampler disabled where its generated compiler
flags were incompatible with CUDA 12.4. The vLLM audio extras were present after
adding the missing backend dependencies, and 48 kHz WAV inputs were converted to
FLAC sidecars for the server endpoint.

All four arms completed 27/27 rows:

| Arm | Wall seconds | Empty rows |
| --- | ---: | ---: |
| Raw | 20.804 | 0 |
| RNNoise mp | 20.824 | 0 |
| RNNoise bd | 20.887 | 0 |
| DeepFilterNet + PF, -a 6 | 21.335 | 0 |

The Voxtral resume harness was repaired in the same way as Qwen: a completed
record with an empty hypothesis is legitimate silence, not missing work. No
Voxtral row was empty on this capture.

## Three-teacher aggregate

Private aggregate artifact:
`~/.local/share/phonon-research/parallel-isolation-v2/three-teacher-aggregate.json`.

On raw audio, normalized exact agreement between teacher pairs was:

| Teacher pair | Exact rows | Mean pairwise WER |
| --- | ---: | ---: |
| Cohere / Qwen | 4/27 | 0.272604 |
| Cohere / Voxtral | 9/27 | 0.158160 |
| Qwen / Voxtral | 5/27 | 0.259584 |

Exact output stability versus each teacher's raw-audio result:

| Arm | Cohere | Qwen | Voxtral |
| --- | ---: | ---: | ---: |
| RNNoise mp | 12/27 | 14/27 | 12/27 |
| RNNoise bd | 14/27 | 13/27 | 13/27 |
| DeepFilterNet + PF, -a 6 | 19/27 | 14/27 | 16/27 |

This strengthens the earlier conclusion that RNNoise is not a transparent
preprocessor. DeepFilterNet at 6 dB remains the least invasive option for
Cohere, but it still changes a majority of outputs across the three-teacher
matrix and loses words for Qwen. No variant is promoted.

## New triage queue

Each segment was scored by combining mean pairwise teacher disagreement on raw
audio with the largest raw-to-variant change under any teacher. The new queue is:

`~/.local/share/phonon-research/parallel-isolation-v2/review-queue-three-teacher.jsonl`

The top 12 of 27 segments contain 120.233 seconds of audio. The regenerated
review bundle has 12 rows, 12 audio assets, and zero missing audio:

`~/.local/share/phonon-research/parallel-isolation-v2/review-three-teacher/index.html`

A sanitized external-rater package was prepared but not published:

`~/.local/share/phonon-research/parallel-isolation-v2/rater-package/`

It contains the same 12 clips and candidate transcripts, relative audio URLs,
instructions, a manifest, and export-only review JSON. Absolute local paths were
removed from the embedded payload and the HTML was parsed back as valid JSON.
No rater result collector or public host is configured yet.

## Live Prolific pricing

Prolific is the strongest self-serve option found: it exposes verified AI
taskers/domain experts, custom external-study links, transparent corporate
pricing, and no subscription requirement. Amazon Mechanical Turk closed on
2026-09-30. Toloka’s public platform adds a 33% commission, but its old public
crowdsourcing surface is winding down and its current expert pipeline requires a
configured project. Scale, Appen, and SageMaker Ground Truth Plus are primarily
quote-based managed services.

Prolific’s public guidance enforces at least $8/hour, recommends at least
$12/hour, and says specialized AI annotation may require more. Its live
corporate USD calculator response reports a 0.428571 fee multiple on participant
rewards and zero VAT for this default request.

For a conservatively estimated 15-minute task:

| Design | Reward | Equivalent rate | Prolific fee | All-in cost |
| --- | ---: | ---: | ---: | ---: |
| 3 raters | $15 | $20/hour | $6 | $21 |
| 5 raters | $25 | $20/hour | $11 | $36 |

Live API request shape:

```json
{
  "study_type": "SINGLE",
  "is_academic": false,
  "total_available_places": 3,
  "estimated_completion_time": 15,
  "reward": 5,
  "currency_code": "USD"
}
```

Live response for three raters:

```json
{
  "rewards": 15,
  "fees": 6,
  "total_cost": 21,
  "fees_percentage": "0.42857142857142857",
  "vat_rate": "0.00000000000000000"
}
```

No Prolific account, API credential, or wallet is configured on this machine.

## Commands and results

- `PYTHONPATH=ml/src uv run --project ml --frozen pytest ml/research/asr_teachers/tests/test_transcribe_voxtral_vllm.py -q`: 3 passed
- `uvx ruff check ...transcribe_voxtral_vllm.py ...test_transcribe_voxtral_vllm.py`: pass
- `uvx ruff format --check ...`: pass
- Four Voxtral teacher arms: 27/27 rows complete each
- Sanitized package check: embedded JSON parsed, 12 rows, 12 relative audio URLs, no absolute local path

## Non-claims

- Three-teacher disagreement is not transcription accuracy.
- No external rater has seen the data and no third-party upload has occurred.
- The external package has no result collector, completion-code integration, or
  anti-fraud gate yet.
- No preprocessing or runtime behavior was changed.
