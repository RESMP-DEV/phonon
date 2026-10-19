# Ming ASR model survey, 2026-10-19

## Scope

This receipt records the public Hugging Face/API survey, benchmark-card extraction,
local runtime probe, and Phonon relevance assessment for new inclusionAI Ming
audio models. It does not contain Phonon transcripts, row IDs, dictionary values,
private audio paths, or per-row private results. No private Phonon data was sent
to a rented provider.

## Pinned public artifacts

Live Hugging Face metadata was read on 2026-10-19.

| Model | Revision | Safetensors parameters | Card size claim |
| --- | --- | ---:| ---:|
| `inclusionAI/Ming-3.0-asr-tiny` | `b610298956f4d190fa517d396d7e0337a6bf039f` | 8,865,800,992 | 8.6B total, about 2B active |
| `inclusionAI/Ming-3.0-audio-flash` | `e3f257a4b8dbab7811ade4b0bde4fafa9565421c` | 128,181,010,400 | 126B total, 6.5B active |
| `inclusionAI/Ming-3.0-audio-tiny` | `7cbe4a75c979f6ea48f1ee31998eb02bd7012233` | 8,871,306,016 | 9.3B total, 2.7B active |
| `inclusionAI/Ming-UniAudio-16B-A3B` | `ca2b6e788b6d3f099cd6cef4d79654226b1b35af` | 18,275,640,164 | 16B total, A3B |

The safetensors count and card claim can differ when optional components or
counting conventions differ; both are recorded rather than reconciled by guess.

### Ming-3.0-asr-tiny contract

- MIT model weights/project code, with Apache-2.0 and Whisper notices for bundled
  source components.
- Languages: Chinese and English, including dialects.
- Audio-only, single-turn, non-streaming; no image or video input.
- Audio is downmixed/resampled to 16 kHz. The feature extractor uses 128 mel
  features and normally 3,000-frame chunks; Context ASR uses 6,000 frames for
  clips over 30 seconds and up to 60 seconds.
- Context is supplied as a contiguous suffix to the base instruction. Hotwords
  and prior dialogue are hints, not guarantees.
- Text backbone: 24 layers, hidden size 1,536, 128 routed experts plus one
  shared expert, eight experts per token, 8,192 maximum positions.
- Audio encoder: 32 layers, width 1,280, 20 heads.
- The two weight shards total 17,731,608,128 bytes. Local and B550 SHA-256
  values matched:
  - shard 1: `3c3aeb85649b6f64a032d069a2321826191d36fe1413191bddd01835b1570d45`
  - shard 2: `a274b5c624f396a3a2547cbc7cf28aeb33890b61141457b45ac1af85ac6e7c55`

### Ming-3.0-audio-flash contract

- Native audio-language MoE with text output and an optional Talker/speech
  output.
- Audio input plus text instructions; speech recognition, spoken QA, audio
  understanding/reasoning, long-form audio, non-speech sounds, music, acoustic
  scenes, and multi-speaker content.
- 128K text context; 42 layers, hidden size 2,560, 512 routed experts plus one
  shared expert, and eight experts per token.
- BF16 weights are roughly 256 GB. The reference vLLM profile uses eight 80 GB
  H800 GPUs; the validated SGLang profile uses four GPUs.

## Public benchmark evidence

Model-card benchmark figures were extracted from the released images with a
vision model after downscaling; values were not copied from a machine-readable
table. Treat them as vendor self-evaluation until independently reproduced.

### Ming-3.0-asr-tiny overview

Lower is better:

| Category | Ming ASR tiny | Qwen3-ASR 1.7B | Fun-ASR 1.5 | FireRedASR2 8B |
| --- | ---:| ---:| ---:| ---:|
| ASR overall | 4.41 | 6.52 | 5.17 | 5.90 |
| Domain recognition | 1.67 | 2.49 | 4.14 | 1.89 |
| Dialect recognition | 8.42 | 15.30 | 8.00 | 13.56 |
| Chinese average | 3.55 | 3.66 | 4.24 | 2.94 |
| English average | 3.99 | 4.63 | 4.30 | 5.19 |

The repository's serving guide also gives exact ordinary-ASR values for Ming:
AISHELL-1 CER 1.044242%, LibriSpeech test-clean WER 1.169216%, and
LibriSpeech test-other WER 2.394009%.

Context ASR does not make Ming universally best. In the released ContextASR
chart, Ming ASR tiny scored 3.22 WER / 7.55 NE-WER / 2.32 NE-FNR, while Qwen3
ASR 1.7B scored 2.04 / 7.32 / 3.33 and Qwen3-Omni 30B-A3B scored
1.98 / 10.42 / 0.62.

### Ming-3.0-audio-flash versus audio-tiny

Lower is better:

| Benchmark | Audio-flash | Audio-tiny |
| --- | ---:| ---:|
| English ASR average | 3.69 | 4.36 |
| Chinese ASR average | 3.45 | 3.79 |
| Dialect average | 9.40 | 10.29 |
| ContextASR WER | 1.52 | 2.12 |
| Earnings22 long-form WER | 11.60 | 18.29 |

Audio-flash's strongest relative gain is long-form/multi-speaker audio, not
ordinary short single-speaker dictation. It is also not universally best:
Qwen3.8-Omni Flash reports 1.11 ContextASR WER versus audio-flash's 1.52.

## Comparison with the frozen Phonon slice

The registered frozen-slice baselines remain:

| Teacher | Phonon fair WER |
| --- | ---:|
| Aqua real-time raw | 0.032912 |
| Cohere Transcribe 03-2026 | 0.057682 |
| Phonon Voxtral Small MXFP4 | 0.060714 |
| Qwen3-ASR 1.7B | 0.070674 |

The public-board to Phonon-slice degradation factor was 1.216 for Voxtral and
1.639 for Qwen3-ASR. Applying those factors to Ming audio-flash's 3.69% English
average gives only a rough **4.5-6.1% Phonon fair-WER prediction**. Applying the
same factors to Ming ASR tiny's 3.99% average gives roughly **4.9-6.5%**.

Therefore, audio-native architecture alone does not imply that Ming will beat
Aqua raw at 3.29%. It is plausible that Ming beats all independent open teachers
while still losing to Aqua on this short technical dictation distribution.
Context/hotword prompting is the mechanism that could close that gap.

## Live B550 runtime probe

The public model was downloaded to the main research share and transferred to
B550. The official repository verifier passed: audio-only model files,
configuration, and weight shards were present. Both shard hashes matched the
main machine.

A first attempt with the newer EmbeddingGemma environment
(transformers 5.20.2, torch 2.14.1) loaded weights but failed in the custom
linear-attention cache with `AttributeError: 'dict' object has no attribute
'to'`. This was a dependency-contract mismatch, not a model-quality result.

The exact pinned environment then succeeded:

- Python 3.12
- torch 2.11.0+cu130
- torchaudio 2.11.0+cu130
- transformers 5.12.1
- fla-core 0.5.2
- accelerate 1.15.0

The official public CC0 English Common Voice sample completed successfully. With
an existing local service already using about 1.4 GiB, total GPU memory during
model execution was about 19.5 GiB on one RTX 3090 Ti. This proves the pinned
runtime and 24 GB local inference path, not Phonon-slice quality.

A 25-row private frozen-slice pilot was launched with the exact environment, but
B550 became unreachable before a score could be read. No matched WER claim is
made. The pilot output remains local to B550 and was not uploaded.

## Stability and rented-GPU boundary

B550 had already rebooted several times during GPU-provider sizing. During the
Ming pilot it disappeared from ARP and SSH entirely. Until it returns, the
matched no-context and no-leak hotword-context runs are blocked locally.

Ming audio-flash is not a practical local runtime: roughly 256 GB of BF16
weights and a reference eight times 80 GB serving profile. The earlier live
provider survey found Prime's eight-H100 configuration at `$24/hour`; the
wallet balance was `$49.01`. A real run must also budget model staging and
setup, not just inference time.

Phonon private dictation audio must not be uploaded to a rented provider merely
because a GPU is available. Public/synthetic work can use a rented provider
under the standing cloud rules; private audio needs an explicit provider, data
scope, retention, and consent decision.

## Current conclusion

- `Ming-3.0-asr-tiny` is the useful Phonon candidate: it fits one 3090 Ti, has
  a proven pinned runtime, and directly supports hotwords/dialogue context.
- `Ming-3.0-audio-flash` is primarily interesting for long-form, noisy,
  multi-speaker, and audio-reasoning teachers. Its short-dictation advantage
  over the much smaller specialized ASR model is modest and its deployment cost
  is large.
- Architecture does not override the registered matched-slice gate. Promotion
  requires full-slice fair/strict WER, exactness, per-row win/tie/loss against
  Aqua, and a no-leak retrieved-hotword context arm.
- A realistic context arm must derive terms from the Aqua raw hypothesis and
  local lexicon, never from the accepted reference. The staged runtime-shaped
  retriever uses phonetic plus character similarity and returns ten terms.

## Non-claims

This survey does not establish a Phonon WER score for either Ming model, approve
private-data upload, qualify a daily-driver engine, prove streaming behavior,
or establish latency. Vendor benchmark numbers are not matched to Phonon's
protocol and remain self-reported.
