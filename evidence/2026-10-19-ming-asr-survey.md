# Ming ASR model survey, 2026-10-19

## Scope

This receipt records two dated passes over the new inclusionAI Ming audio
models. The first pass (2026-10-19T21:17:58Z, revision `cd603bd`) covers the
public Hugging Face/API survey, benchmark-card extraction, the local B550
runtime probe, and the Phonon relevance assessment. The second pass (the
`## Quantization survey` section, this revision) adds the quantization-artifact
survey, the pinned same-family Ling FP4 recipe, a pinned SGLang audio-branch
source inspection, a local safetensors-header arithmetic check, and read-only
hardware probes of Tetra and B550. It does not contain Phonon transcripts, row
IDs, dictionary values, private audio paths, or per-row private results. No
private Phonon data was sent to a rented provider.

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

**Recovered diagnosis (2026-10-19T22:35Z, B550 back online).** The pilot never
produced output. Its `run.log` shows an immediate
`ModuleNotFoundError: No module named 'metaphone'` at import of
`asr_teachers/vocab_retrieval.py`, so zero of the 25 rows were transcribed and
no partial JSONL exists. The first pass's "no score read before outage" was
therefore optimistic: the correct statement is that the pilot crashed at import
and produced nothing to score. The fix is to install `metaphone` (the DoubleMetaphone
package the retriever imports) into the pinned environment and relaunch; both
3090 Ti GPUs were idle at the recovery probe. The model, scorer
(`score_hyps.py`), and teacher script remain staged on B550.

## Stability and rented-GPU boundary

B550 had already rebooted several times during GPU-provider sizing, and during
the first Ming pilot pass it disappeared from ARP and SSH entirely. It returned
by the second pass (2026-10-19T22:35Z; reachable via the `B550` SSH alias, both
RTX 3090 Ti GPUs idle). The matched no-context and no-leak hotword-context runs
are therefore no longer blocked by reachability; they are blocked only by the
missing `metaphone` dependency in the pinned environment, diagnosed above.

> **Superseded (2026-10-19T23:15Z).** B550 returned, the missing `metaphone` /
> `rapidfuzz` dependencies were installed into the pinned env, and both 25-row
> arms completed and were scored. The matched aggregate result is recorded in
> `evidence/2026-10-19-ming-asr-slice-score.md`; the "zero rows / no score"
> status above applies only to the first pass.

Ming audio-flash is not a practical local runtime: roughly 256 GB of BF16
weights and a reference eight times 80 GB serving profile. Any rented run needs
a fresh provider quote and must budget model staging and setup as well as
inference time.

Phonon private dictation audio must not be uploaded to a rented provider merely
because a GPU is available. Public/synthetic work can use a rented provider
under the standing cloud rules; private audio needs an explicit provider, data
scope, retention, and consent decision.

## Quantization survey

### Official release state

**Search scope.** The negative result below is bounded to one dated Hub query
snapshot, not an exhaustive existence proof. On 2026-10-19 a
`GET https://huggingface.co/api/models?search=Ming-3.0&limit=100` (snapshot
`ming-search.json`, SHA-256
`08e71d9d9a973d59ec725dbe064723ef97b306fb51c9873e91b3dbdcc9a2a6a9`) returned
only `inclusionAI/*` and `ml-resources/*` mirrors plus `mlx-community/*`
conversions; no `inclusionAI`-published MXFP4, NVFP4, GPTQ, AWQ, GGUF, or
compressed-tensors Ming ASR/audio checkpoint appeared.

**Correction to the first pass.** The first pass stated the only non-BF16 Ming
exports were BF16 MLX conversions. That is stale. As of the 2026-10-19T22:19Z
refresh, `mlx-community` publishes genuinely quantized Ming exports alongside
the BF16 ones, all pinned and verified by metadata this pass:

| Export | Revision | Reported dtype mix | Note |
| --- | --- | --- | --- |
| `mlx-community/Ming-3.0-asr-tiny-4bit` | `89acb1a5f240a15119f175065ba8e2356f59d7a1` | 649,876,896 BF16 + 7,890,665,472 U32-packed | 4-bit affine, `group_size` 64 |
| `mlx-community/Ming-3.0-audio-tiny-4bit` | `af0238886184f15fb81e28858df2a3bf43983145` | 664,035,744 BF16 + 7,890,665,472 U32-packed | 4-bit affine |
| `mlx-community/Ming-3.0-audio-tiny-8bit` | `b92d0d10226ffb45bf0c73ad86b30e273c243914` | 664,035,744 BF16 + 7,890,665,472 U32-packed | 8-bit |
| `mlx-community/Ming-3.0-asr-tiny-bf16` | `2732edd6f66295941a0e9d055f9eaf25eafe3aef` | all BF16 | unquantized conversion |

These are **MLX-format** artifacts (`quantization: {bits: 4, mode: affine,
group_size: 64}` in the 4-bit config; architecture
`MingV3ASRForConditionalGeneration`). They are real weight-only quantizations,
not BF16 passthroughs, but they target Apple Silicon MLX and cannot be loaded by
the CUDA SGLang/vLLM path on B550 or Tetra. They therefore open a **Mac-side
quantized lane** for local judgment-teacher prototyping, and they are independent
evidence that Ming's routed-expert weights tolerate 4-bit packing without the
export being withheld for quality reasons. The earlier `lucasnewman/…-bf16`
export did not appear in this refresh's author search and is not relied upon.

There is nevertheless a strong same-family precedent on the CUDA side. inclusionAI publishes
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

**Pinned source.** The Ming SGLang work is on the fork
`inclusionAI/sglang_ling_v3`, branch `ling_v3_audio_for_pr`, commit
`bba862771a9e3413a062dad33e8b494ea9f7755b` ("[Model]: Fix Ming audio RoPE and
typed config overrides", 2026-10-16). This pass fetched and hashed the decisive
files from that commit (SHA-256 recorded in the Verification table):
`python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py`,
`python/sglang/srt/models/bailing_audio_v3.py`, and
`python/sglang/srt/configs/bailing_audio_v3.py`.

**What the override actually does.** `_bailing_moe_v3_overrides` is registered
(`@_register_for`) for exactly three architectures:
`BailingMoeV3ForCausalLM`, `BailingMoeV3VLForConditionalGeneration`, and
`BailingMoeV3AudioForConditionalGeneration`. It returns
`{"moe_runner_backend": "flashinfer_mxfp4"}` only when the platform is CUDA and
SM90/SM100/SM120, `moe_runner_backend == "auto"`, `moe_a2a_backend == "none"`,
and the checkpoint reports `quantization == "fp8"` with `is_fp4_experts` true.
The audio model builds its `WhisperAudioEncoder` and audio projector outside the
language model's `quant_config`, so the intended boundary is a mixed-precision
text MoE backbone with an unquantized audio path.

**Decisive caveat for ASR-tiny.** The registered audio class is
`BailingMoeV3AudioForConditionalGeneration` (`model_type` `bailing_moe_v3_audio`),
which is what `Ming-3.0-audio-tiny` and `Ming-3.0-audio-flash` report. But
`Ming-3.0-asr-tiny` reports `architectures: [MingV3ASRForConditionalGeneration]`
and `model_type: ming_v3_asr`, and loads through Hugging Face `auto_map`
custom code (`configuration_ming_v3_asr` / `modeling_ming_v3_asr`), not through
this SGLang model file. The `ling_v3_audio_for_pr` tree contains **no**
`MingV3ASR` / `ming_v3_asr` registration at all. Therefore the SGLang MXFP4
override does **not** currently cover Ming ASR-tiny. The first pass's framing
("run ASR-tiny through the SGLang MXFP4 lane") was unsupported: the SGLang FP4
path applies to the Bailing-V3 *audio* MoE (audio-tiny/flash), while ASR-tiny
runs on the transformers custom-code path that this branch does not serve.

**Test coverage.** The pinned tree's Ming audio tests
(`test/registered/unit/models/test_bailing_audio_plain_rope.py`,
`.../configs/test_bailing_audio_rope_config.py`,
`.../managers/test_bailing_modality_metadata.py`) are CPU RoPE/config/routing
regressions. No test in the tree exercises an FP4/MXFP4 *audio* load-and-
transcribe; the MXFP4 e2e coverage targets DeepSeek-V4 and other models. So even
for the audio-tiny/flash class, a mixed-FP4 audio checkpoint is plausible but
unproved until a live model loads and transcribes, and for ASR-tiny the CUDA FP4
path is not wired at all.

NVFP4 is not the closest proven path for this family. No Bailing V3 NVFP4
artifact was found. A new NVFP4 export would additionally require loader and
MoE-runner validation on this audio architecture. On the CUDA side MXFP4 is the
evidence-backed recipe for the *audio* MoE on Tetra's SM120 hardware; the ASR-
tiny quantization lane on CUDA instead needs either a transformers-side
compressed-tensors/GPTQ export or a custom-code MXFP4 loader that does not yet
exist. On Apple Silicon the `mlx-community` 4-bit/8-bit exports above already
provide a loadable quantized ASR-tiny.

### Size estimates

The local Ming ASR-tiny safetensors headers were grouped without reading tensor
values:

| Group | Parameters | BF16 bytes |
| --- | ---:| ---:|
| routed experts | 7,247,757,312 | 13.500 GiB |
| everything else | 1,618,043,680 | 3.014 GiB |

Applying the official family recipe to those groups, verified by recomputation
from the safetensors headers this pass (routed experts = tensors matching
`\.experts\.\d+\.(gate|up|down)_proj`, excluding `shared_expert`):

- routed experts in MXFP4 (0.5 byte/param E2M1 + 1 byte E8M0 scale per group of
  32) = 3.586 GiB;
- **conservative estimate 6.6 GiB** (3.586 + 3.014 GiB of BF16 non-experts), the
  realistic floor given the recipe preserves MLA/attention, routers, embeddings,
  LM head, and the audio path in higher precision;
- **5.1 GiB is an absolute lower bound only**, reached if *every* non-expert
  tensor were FP8, which the recipe does not do. Treat 6.6 GiB as the planning
  number and 5.1 GiB as a non-achievable-with-this-recipe bound.

These exclude runtime, KV/cache, and file-metadata overhead. The BF16
weight-data total is 17,731,608,128 bytes = **16.51 GiB** (the first pass's
"16.52 GiB" was a rounding error; on-disk the two shards with headers are
16.515 GiB).

BF16 Ming audio-flash is roughly 241.52 GiB. The same Bailing V3 text backbone
in official Ling FP4 form is 65.54 GiB. Adding the BF16 audio encoder/projector
gives a **broad feasibility bound of about 67-72 GiB** for a hypothetical
audio-flash mixed-FP4 export, not an arithmetic derivation: the audio-path
parameter count was not separately grouped from the audio-flash headers this
pass, so the upper end is unsigned slack, not a computed figure. No such export
currently exists.

### Live Tetra state

Read-only SSH probe, 2026-10-19T22:34:07Z (`nvidia-smi`
query-gpu/query-compute-apps and `df -h /home`; snapshot
`tetra-probe.txt`, SHA-256
`9bd609b9986ea87fb351afaab6f0892d65530ff350df571d5d4ff06f87aaa569`). GPU
occupancy is inherently a point-in-time observation:

- four `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` GPUs;
- compute capability 12.0 (SM120-class Blackwell);
- driver 615.71.09 (NVRM Open Kernel Module, 2026-09-04 build);
- 97,887 MiB visible per GPU;
- 1.6 TB free on `/home` (`/dev/nvme3n1p2`, 56% used).

At probe time all four GPUs were occupied (89.9 / 89.2 / 89.2 / 91.1 GiB used)
by a root vLLM TP4 deployment of `/work/heretic-native` served as
`deepseek-v41-local` (PIDs 1602786-1602789, ~89.1 GiB each) plus a small
embedding service (~1.7 GiB). Tetra is therefore not currently available for a
Ming quantization smoke test, and this survey did not interrupt either process.
Recheck idle state before any run.

### Recommended experiment

Because ASR-tiny and the audio models are *different architectures on different
runtimes*, the quantization experiment must be routed per class rather than
assuming one SGLang MXFP4 path covers both:

- **ASR-tiny judgment teacher on Apple Silicon (available now).** Load
  `mlx-community/Ming-3.0-asr-tiny-4bit` (revision `89acb1a…`) through MLX on the
  Mac and compare its outputs to the proven BF16 CUDA path on the public CC0
  English/Chinese samples, then on the frozen private slice. This is the fastest
  route to a quantized ASR-tiny judgment teacher and needs no new export. It
  does not run on B550/Tetra (MLX-only).
- **ASR-tiny on CUDA (needs new work).** The transformers custom-code path
  (`MingV3ASRForConditionalGeneration`) has no MXFP4 loader in the pinned SGLang
  branch. A CUDA quantized ASR-tiny would require either a compressed-tensors /
  GPTQ-pack export loadable by transformers, or a custom-code MXFP4 loader. Treat
  this as unstarted, not a small config change.
- **audio-tiny / audio-flash on Tetra SGLang (the real MXFP4 lane).** When Tetra
  is idle, the SGLang `flashinfer_mxfp4` override *does* apply, because those
  models report `BailingMoeV3AudioForConditionalGeneration`. The export should
  copy the official Ling FP4 exclusion policy and additionally preserve the Ming
  Whisper audio encoder and audio projector. The architecture rationale is
  favorable (MLA/MoE backbone, audio as an alternate conditioning modality,
  routed experts quantized while MLA/router/embedding/head/modality path are
  preserved), which makes severe modality-induced FP4 sensitivity unlikely, but
  this is an unmeasured hypothesis and the branch has no FP4-audio load test.

Acceptance, whichever lane, must be paired against the proven BF16 runtime:

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

This receipt has two dated verification passes. The first covers the ASR survey
and B550 runtime probe; the second covers the quantization survey and the
hardware re-probes. Each states its own revision, timestamp, and platforms.

### First pass (ASR survey and B550 runtime)

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
| Private 25-row pilot | resumeable teacher launch with no reference-derived prompt | launch exit 0; **superseded**: recovered `run.log` shows an import crash (`metaphone` missing), zero rows produced |
| B550 outage | independent `ssh` and TCP port-22 probes after the pilot | connection failures at first-pass time; **superseded**: host reachable again by the second pass |
| Repository gate | `just quality` at the survey revision | exit 0, PASS |

The B550 runtime logs remain on the host. They were not copied into Git or a
rented provider. The second pass recovered the pilot `run.log` and found the
import crash described above; the decisive receipt must still add a completed
pilot's `score_hyps.py` output before any matched quality claim is made.

### Second pass (quantization survey and hardware re-probes)

Recorded UTC: 2026-10-19T22:34Z-22:36Z. Platforms: macOS main machine, Tetra
Linux/CUDA (read-only), and B550 Linux/CUDA (read-only). Source revision: this
PR head; worktree dirty state: false at commit. All raw snapshots, source files,
and probe outputs are retained under ignored `build/quant-provenance/` and only
sanitized hashes and aggregates are recorded here. Private-slice text and audio
were not printed.

| Check | Sanitized command or procedure | Result |
| --- | --- | --- |
| Hub search snapshot | `curl 'https://huggingface.co/api/models?search=Ming-3.0&limit=100'` | exit 0; `ming-search.json` SHA-256 `08e71d9d…a2a6a9`; found only inclusionAI/ml-resources mirrors and mlx-community conversions |
| MLX quant metadata | `curl …/api/models/mlx-community/Ming-3.0-{asr,audio}-tiny-{4bit,8bit,bf16}` + 4-bit `config.json` | exit 0; 4-bit/8-bit report U32-packed weights (`bits:4/8, mode:affine, group_size:64`); 4-bit config SHA-256 `216e5b05…24460e` |
| Ling FP4 recipe | `curl` README + `config.json` at the two pinned revisions | exit 0; `quantization_config` shows `quant_method: fp8`, `routed_experts_quant_method: mxfp4`, `routed_experts_weight_dtype: e2m1`, `routed_experts_scale_dtype: e8m0`, `routed_experts_group_size: 32`; README SHA-256 `37753fff…9f72ff`, config `1af5e442…9956dc` |
| SGLang branch pin | `curl …/api.github.com/repos/inclusionAI/sglang_ling_v3/branches/ling_v3_audio_for_pr` | exit 0; commit `bba862771a9e3413a062dad33e8b494ea9f7755b` |
| SGLang override source | fetch `bailing_moe_v3.py`, `bailing_audio_v3.py` (model + config) at that commit | exit 0; override SHA-256 `768218fa…eaa4622`, audio model `2afee3c0…44272`, audio config `df67338f…fdb694`; `_register_for` lists Causal/VL/**Audio**, not `MingV3ASR` |
| ASR-tiny architecture | read local `config.json` `architectures`/`auto_map` | `MingV3ASRForConditionalGeneration`, `model_type ming_v3_asr`, HF custom code; absent from the SGLang audio tree |
| Safetensors arithmetic | Python header grouping (routed experts vs rest) | routed 7,247,757,312 params (13.500 GiB BF16, 3.586 GiB MXFP4), rest 1,618,043,680 (3.014 GiB); data total 17,731,608,128 B = 16.51 GiB; conservative 6.6 GiB, absolute floor 5.1 GiB |
| Tetra re-probe | read-only `ssh tetra` `nvidia-smi` + `df -h /home` at 22:34:07Z | exit 0; all four GPUs 89-91 GiB used by root vLLM TP4; snapshot SHA-256 `9bd609b9…aaa569` |
| B550 re-probe | read-only `ssh B550`: pilot dir, `run.log`, staged scripts, `nvidia-smi` | exit 0; pilot crashed at import (`metaphone`), zero output rows; both GPUs idle |
| Repository gate | `just quality` at this PR head | exit 0, PASS |

Non-claims for the second pass: no Ming model was quantized, loaded, or
transcribed this pass; the SGLang FP4-audio path and the MLX 4-bit ASR-tiny path
were inspected from pinned source/metadata only, not executed. The Tetra and B550
GPU states are point-in-time and must be rechecked before any run.

## Non-claims

This survey does not establish a Phonon WER score for either Ming model, approve
private-data upload, qualify a daily-driver engine, prove streaming behavior,
or establish latency. Vendor benchmark numbers are not matched to Phonon's
protocol and remain self-reported. The quantization pass did not execute any
quantized Ming model: the MLX 4-bit ASR-tiny export, the SGLang MXFP4 audio
override, and the Ling FP4 recipe were inspected from pinned metadata/source
only. It does not prove that a quantized Ming matches BF16 output, and it
establishes that the pinned SGLang MXFP4 lane does **not** cover Ming ASR-tiny
(`MingV3ASRForConditionalGeneration`), only the Bailing-V3 audio class.
