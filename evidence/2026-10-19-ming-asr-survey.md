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

Model-card figures were extracted from the released images with the native
`alt-vision` 0.1.0 and `zai-vision` 0.1.0 CLIs after downscaling. The source
images were public and content-hashed, but the terminal model responses were not
retained verbatim. Treat the consolidated values below as provisional
vendor-self-evaluation plus vision extraction, not acceptance data.

| Source image at the pinned model revision | SHA-256 |
| --- | --- |
| `Ming-3.0-asr-tiny@b610298…/figures/asr.png` | `624d98d65546e12b1763eb157690e51e388f3db6de05ec1980ea457f4122e56d` |
| `Ming-3.0-asr-tiny@b610298…/figures/开源asr.png` | `e05cf1897bd7eb80d64146613e18f3fb83e1e46f763443f42525ee3eddddd795` |
| `Ming-3.0-asr-tiny@b610298…/figures/自建asr.png` | `31ade60fefc3a90aab51439be9d52518775710f262a5a7e5371e89ff56667a40` |
| `Ming-3.0-asr-tiny@b610298…/figures/contextasr.png` | `6924bd1057aa68ef194912995c92d8ca998ef68b15ec55053f5b168da087cd32` |
| `Ming-3.0-audio-flash@e3f257a…/assets/benchmark-asr.png` | `f7558ab421fac9bf78cdc1604762221fc644e63e8ceecf5d43c9c08c17c716da` |

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
1.639 for Qwen3-ASR. These are directional cross-protocol extrapolations, not a
matched prediction: Ming's card averages and the two teacher leaderboard scores
do not share one protocol. They give about **4.5-6.1%** for audio-flash's 3.69%
English average and **4.9-6.5%** for ASR-tiny's 3.99% average. The result is no
evidence that either Ming model beats Aqua; only a matched run can answer that.

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
weights and a reference eight times 80 GB serving profile. Any rented run needs
a fresh provider quote and must budget model staging and setup as well as
inference time.

Phonon private dictation audio must not be uploaded to a rented provider merely
because a GPU is available. Public/synthetic work can use a rented provider
under the standing cloud rules; private audio needs an explicit provider, data
scope, retention, and consent decision.

## Current conclusion

- `Ming-3.0-asr-tiny` is the useful Phonon candidate: it fits one 3090 Ti, has
  a proven pinned runtime, and directly supports hotwords/dialogue context.
- `Ming-3.0-audio-flash` is primarily interesting for long-form, multi-speaker,
  and audio-reasoning teachers. Its short-dictation advantage
  over the much smaller specialized ASR model is modest and its deployment cost
  is large.
- Ming is not being considered merely as another WER engine. Its Phonon role is
  a **judgment / correction-teacher lane** for enhanced dataset construction:
  one or more raw acoustic transcriptions, the Aqua raw hypothesis, local
  lexicon/history, and the user's preference contract enter the judgment; the
  output is an intended-text correction plus enough structured disagreement
  information to rank or reject the row. The private Aqua prompt text is not
  reproduced here. Its semantics include, abstractly, cleanup and formatting
  preferences such as filler handling, spelling, expansion, and technical-term
  recovery rather than an unconditional one-to-one acoustic transcript.
- Dataset construction therefore needs three separable labels: **raw acoustic
  transcription**, **intended-text judgment**, and **correction preference
  enforcement**. Aqua raw and Aqua accepted are evidence for these lanes, but
  neither is treated as a universal acoustic truth: Aqua accepted encodes the
  user's intended-text preference, while the raw lane is the acoustic control.
- Architecture does not override the registered matched-slice gate. The matched
  quality requirements include full-slice fair/strict WER, exactness, per-row
  win/tie/loss against Aqua, and a no-leak retrieved-hotword context arm; all
  registered product, restraint, latency, packaging, and default-path regression
  gates also remain applicable.
- WER remains useful for comparing raw acoustic lanes under a matched protocol.
  It is insufficient for the judgment lane. Judgment acceptance also needs
  preference/correction metrics: intended-text win/tie/loss against accepted
  Aqua output, preservation of semantically important raw acoustic content,
  correct filler/formatting preference application, technical-term recovery,
  and calibrated uncertainty or rejection. Aggregate-only WER cannot promote
  the teacher.
- A realistic context arm must derive terms from the Aqua raw hypothesis and
  local lexicon, never from the accepted reference. The tracked `vocab_v1`
  retriever already provides no-reference phonetic-plus-character ranking with a
  default top-30 list; a ten-term Ming prompt is a proposed experiment and needs
  its own committed harness and receipt.

## Verification

Recorded UTC: 2026-10-19T21:17:58Z. Platforms: macOS main machine and B550
Linux/CUDA. Source revision: `cd603bd2649b38e9ad4da0dc6505fee01714341f`;
worktree dirty state: false when the survey branch was committed. Private-slice
text and audio were not printed in the tracked receipt.

| Check | Sanitized command or procedure | Result |
| --- | --- | --- |
| Public metadata | Python `urllib.request` GETs to `huggingface.co/api/models/...` for the four pinned revisions | exit 0 |
| Public benchmark images | Download the five pinned card images, `sips -Z 2200`, then `alt-vision text` / `zai-vision text` with an exact extraction prompt | extraction completed; values marked provisional above |
| Mac inventory | `python3 Ming-3.0-asr/tools/verify_repository.py --checkpoint <public-model>` | exit 0, PASS |
| Mac shard hashes | `shasum -a 256 model-00001... model-00002...` | exit 0; values recorded above |
| B550 shard hashes | `sha256sum model-00001... model-00002...` | exit 0; byte-identical to Mac |
| Exact environment | pip install torch/torchaudio 2.11 cu130, then the pinned model requirements | exit 0; import versions recorded above |
| Public CC0 smoke | official `examples/transcribe.py` on `english_common_voice.wav`, CUDA 0, BF16/SDPA/greedy | exit 0; seven generated tokens and a nonempty transcript |
| Private 25-row pilot | resumeable teacher launch with no reference-derived prompt | launch exit 0; no score read before B550 outage |
| B550 outage | independent `ssh` and TCP port-22 probes after the pilot | connection failures; host remains unreachable |
| Repository gate | `just quality` at the survey revision | exit 0, PASS |

The B550 runtime logs and any partial pilot output remain on the currently
inaccessible host. They were not copied into Git or a rented provider. When
B550 returns, the decisive receipt must add the pilot's exact completion
metadata and `score_hyps.py` output before any quality claim is made.

## Quantization survey

### Official release state

No official Ming ASR/audio MXFP4, NVFP4, GPTQ, AWQ, GGUF, or compressed-tensors
checkpoint was present in the live 2026-10-19 Hugging Face search. The only
non-BF16 Ming exports found were BF16 MLX conversions:

- `lucasnewman/Ming-3.0-asr-tiny-bf16`
- `mlx-community/Ming-3.0-audio-tiny-bf16`

The MLX exports preserve BF16 model weights. Their audio encoders and MoE
routers remain floating point; neither is a quantized artifact.

There is nevertheless a strong same-family precedent. inclusionAI publishes
mixed FP8/MXFP4 checkpoints for the Bailing V3 language backbone used by Ming:

| Model | Revision | Safetensors size | Quantization |
| --- | --- | ---:| --- |
| `inclusionAI/Ling-3.0-flash-fp4` | `3bae1cf4011b48475b2cc038fff283af49053ebc` | 65.54 GiB | dynamic block-FP8 dense weights, MXFP4 routed experts |
| `inclusionAI/Ling-3.0-flash-VL-fp4` | `8dca6a0d4dde1ddc79cfe79506173bbf375393ac` | 64.80 GiB | same text recipe; vision tower and projector excluded |

The official recipe is **not all-layer NVFP4**. It uses:

- dense/shared projections: dynamic FP8 E4M3 with 128x128 blocks;
- routed experts: MXFP4 E2M1 weights, group size 32, E8M0 scales;
- preserved attention/MLA projections, MoE gates, embeddings, LM head, and the
  VL tower/projector.

The official Ling FP4 quality table reports:

| Benchmark | BF16 | Mixed FP8/MXFP4 |
| --- | ---:| ---:|
| GPQA Diamond | 84.97 | 83.68 |
| IFBench | 74.49 | 73.87 |
| SciCode | 41.24 | 38.43 |
| ArcPrize | 68.75 | 68.06 |

This is same-family evidence only. It is not Ming ASR evidence and does not
measure technical-vocabulary recovery.

### Runtime support and caveat

The pinned Ming SGLang branch registers
`BailingMoeV3AudioForConditionalGeneration` in its Bailing V3 model override.
When a mixed checkpoint reports FP8 quantization plus `is_fp4_experts`, that
override selects the FlashInfer MXFP4 MoE runner on SM100 and SM120. The audio
model constructs its Whisper audio encoder and `linear_proj_audio` outside the
language model's `quant_config`, so the intended boundary is mixed-precision
language backbone with an unquantized audio path.

The relevant SGLang unit test explicitly covers the causal and VL architectures,
not the audio architecture. This is a real gap: an audio mixed-FP4 checkpoint is
plausible but not proved until a live Ming audio model loads and transcribes.

NVFP4 is not the closest proven path for this family. No Bailing V3 NVFP4
artifact was found. A new NVFP4 export would additionally require loader and
MoE-runner validation on this audio architecture. MXFP4 is the evidence-backed
route on Tetra's SM120 hardware.

### Size estimates

The local Ming ASR-tiny safetensors headers were grouped without reading tensor
values:

| Group | Parameters | BF16 bytes |
| --- | ---:| ---:|
| routed experts | 7,247,757,312 | 13.500 GiB |
| everything else | 1,618,043,680 | 3.014 GiB |

Applying the official family recipe gives a conservative **6.6 GiB** if all
non-expert tensors remain BF16, or about **5.1 GiB** if eligible dense tensors
use dynamic FP8. This excludes runtime, KV/cache, and file-metadata overhead.
The current BF16 checkpoint is 16.52 GiB.

BF16 Ming audio-flash is 241.52 GiB. The same Bailing V3 text backbone in
official Ling FP4 form is 65.54 GiB; adding the BF16 audio path suggests an
unmeasured audio-flash mixed-FP4 checkpoint around **67-72 GiB**, not including
runtime or KV cache. This is an estimate, not a claim that such an export
currently exists.

### Live Tetra state

Read-only SSH probe on 2026-10-19:

- four `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` GPUs;
- compute capability 12.0 (SM120-class Blackwell);
- driver 615.71.09;
- 97,887 MiB visible per GPU;
- PCIe/NODE topology, no reported NVLink;
- 1.6 TB free on `/home`.

All four GPUs are currently occupied by a root vLLM TP4 deployment of
`/work/heretic-native`, served as `deepseek-v41-local`, using about 89.1 GiB
per GPU. A separate embedding service uses about 1.7 GiB. Tetra is therefore
not currently available for a Ming quantization smoke test, and this survey did
not interrupt either process.

### Recommended experiment

When Tetra becomes idle, the first target should be **Ming ASR-tiny mixed
FP8/MXFP4 on one GPU**, not audio-flash. Its quantized weight floor is small
enough to leave ample runtime space. The export should copy the official Ling
FP4 exclusion policy, add the Ming audio encoder and `linear_proj_audio` to the
exclusions, and preserve routers and sensitive MLA/attention projections.
The architecture rationale is favorable: the Bailing V3 backbone is an MLA/MoE
transformer, audio is an alternate conditioning modality, and the recipe
quantizes routed experts while preserving the modality path, router, MLA,
embeddings, and output head. That makes severe modality-induced sensitivity
unlikely, but it is still an unmeasured hypothesis until matched outputs are
compared.

Acceptance must be paired against the already proven BF16 runtime:

1. strict checkpoint inventory and tensor-dtype audit;
2. same public CC0 English and Chinese samples;
3. exact-output comparison where possible;
4. the frozen 25-row Phonon pilot;
5. a preference-aware judgment pilot over disagreement-rich rows, with no
   private Aqua prompt text or private example in its receipt;
6. only then the full 500-row raw-lane fair/strict WER gate and intended-text
   per-row win/tie/loss gate.

An audio-flash export is a later experiment. BF16 audio-flash may already fit
Tetra's four-GPU aggregate for short-context ASR, but a mixed-FP4 export would
reduce pressure dramatically and potentially permit a one-GPU text-only teacher.
Neither path should be claimed until the SGLang audio model loads and the
matched quality gate passes.

## Non-claims

This survey does not establish a Phonon WER score for either Ming model, approve
private-data upload, qualify a daily-driver engine, prove streaming behavior,
or establish latency. Vendor benchmark numbers are not matched to Phonon's
protocol and remain self-reported.
