---
license: other
license_name: lfm1.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
language:
- en
base_model:
- LiquidAI/LFM2.5-Audio-1.5B
- LiquidAI/LFM2.5-VL-1.6B
tags:
- phonon
- reverse-salm
- lfm2.5
- speech-recognition
- dictation
- lora
- research
- alpha
private: true
---

# Phonon reverse SALM — v0.1.0-alpha.1 (provisional prototype)

**Status: alpha research prototype. Weights are not published in this repository.**

This private repository carries the model identity, prompt contract, evaluation
protocol, pinned revisions, and aggregate results for the Phonon reverse SALM
lane. It intentionally contains no adapter, checkpoint, tokenizer payload, audio,
or transcript content.

- Version: `v0.1.0-alpha.1`
- Card date: 2026-10-06
- Provisional prototype adapter SHA-256:
  `44a379ee1762461077319fa28173d2da782ad1ea5f29b0303c8fde3218e033a4`
- Public checkpoint this lane resumes from SHA-256:
  `511c6f1ecae7281c039c63fda52a5cd83b692b1e09909506ad0fbfadf0d9bb20`

## What this is

Phonon is a local-first voice authoring system: microphone in, intended text out.
This lane transplants the audio conformer and audio adapter of
`LiquidAI/LFM2.5-Audio-1.5B` onto the language stack of
`LiquidAI/LFM2.5-VL-1.6B`, then trains a language LoRA plus the audio adapter.
Vision stays intact and unused in this lane. Inference is single-stage
audio-to-intended-text behind an explicit, non-default engine selection.

The model is not a verbatim transcriber. It is trained to output the text the
speaker intended, including technical terms, identifiers, casing, and
punctuation, while preserving meaning, ordering, and negation.

## Provisional prototype configuration

The recommended alpha configuration is one forward pass with a fixed user-history
card, so no retrieval query and no reference transcript are needed at inference
time:

| Setting | Value |
| --- | --- |
| Prompt ID | `prose_history_dictation_v2` |
| Prompt schema SHA-256 | `664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6` |
| Context length | 768 |
| LoRA rank | 32 |
| LoRA dropout | 0.033694424584220395 |
| Language LoRA LR | 6.505720091093967e-05, linear warmup 50 then cosine |
| Audio adapter LR | 4.943429131224935e-05, static |
| History examples | 8 fixed raw-to-accepted pairs from the owner's own history |
| Kernels | FlashAttention 2.8.3.post1, causal-conv1d 1.7.0, torch 2.14.0+cu130 |

The fixed history card is built locally from the owner's own consented
correction pairs, sampled evenly across the training timeline, with every
evaluation-slice audio ID excluded. The rendered card contains personal
transcript text, is stored only on controlled compute storage, and is never
uploaded here.

## Prompt contracts

Prompt formatting is part of the training contract. Training packs, adapters, and
scores are only valid together with the same prompt ID and byte-identical text,
the same context length, and the same LoRA rank.

| Prompt ID | SHA-256 | Context | Use |
| --- | --- | --- | --- |
| `prose_dictation_v1` | `3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5` | 512 | Static-prose lane; currently installed product-side selection |
| `prose_history_dictation_v1` | `4d7ea7b4687a8f82ddc910b76c2c529fbf17755dcfc6649d2c6be02007c3670f` | 768 | First history layout (XML scaffolding); rejected |
| `prose_history_dictation_v2` | `664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6` | 768 | Alpha lane; preserves `prose_dictation_v1` as a byte-exact prefix, then appends history pairs |

Dynamic per-row history selection uses a deterministic IDF-cosine selector over
training-only raw-to-accepted pairs. The current target is excluded by audio ID,
and the target's accepted text is never used for retrieval or prompt rendering.

## Data

- Public CPT: `espnet/yodas-granary@969944574ea3f37890beaf67ea651e160cfaf043`,
  English `asr_only` stream, embedded 16 kHz audio, CC-BY-3.0 backing audio;
  `nvidia/Granary` CC-BY-4.0 manifests.
- Consented personal adaptation: 12,855 dictation rows, 2.0 to 20.0 seconds,
  contributed by the owner of this project. Only aggregate counts, durations, and
  hashes of internal manifests are used in receipts; no audio, transcript, or
  accepted text appears here.
- Frozen evaluation slices: 500-row selection slice and an untouched 750-row
  future split recorded later in the timeline than the adaptation rows.

## Evaluation protocol

- Metrics: fair corpus WER (lower is better), strict lowercase corpus WER (lower
  is better), and exact match (higher is better).
- Reference-kernel training is the quality reference. Kernel mode is recorded for
  every run because optimized FlashAttention/causal-convolution training changed
  full-slice quality on a matched repeat.
- Aggregate scores are never pooled across different prompt IDs, context lengths,
  or LoRA ranks.

### Frozen 500-row selection slice

| Candidate | Fair WER | Strict WER | Exact |
| --- | ---: | ---: | ---: |
| Static prose, 4,000-step adaptation | 0.117616 | 0.154049 | 0.316 |
| Static prose, full epoch | 0.110774 | 0.142857 | 0.352 |
| Static prose, equal 8k + full-epoch soup | 0.109302 | 0.140673 | 0.350 |
| History, full epoch, per-row retrieval | 0.113459 | 0.145223 | 0.350 |
| History, full epoch, K4 fixed card | 0.116144 | 0.148954 | 0.336 |
| **History, full epoch, K8 fixed card** | **0.114499** | **0.149045** | **0.328** |
| History, full epoch, two-pass self-retrieval | 0.123593 | 0.156688 | 0.324 |
| Static soup with the K4 history card | 0.390525 | 0.418289 | 0.318 |

The K8 fixed card is within 0.0011 fair WER of the per-row retrieval result while
removing the retrieval query entirely.

### Untouched 750-row future slice

| Candidate | Fair WER | Strict WER | Exact |
| --- | ---: | ---: | ---: |
| Static prose, full epoch | 0.145212 | 0.193407 | 0.187 |
| Static prose, equal 8k + full-epoch soup | 0.143926 | 0.191308 | 0.187 |
| **History, full epoch, per-row retrieval** | **0.144452** | **0.192975** | **0.205** |
| Aqua Voice real-time raw transcript | 0.072606 | 0.135070 | 0.364 |

For reference, the same history adapter also scored 0.113459 / 0.145223 / 0.350
on the selection slice under the optimized runtime, matching its reference-kernel
scores exactly on both splits.

## Runtime

The history adapter was evaluated under both the reference kernel mode and the
optimized runtime with FlashAttention and causal-conv1d installed. Aggregate
quality matched exactly on both the 500-row selection slice and the 750-row
future slice.

Research-evaluator generation time under the optimized runtime was 0.9088 seconds
mean (p50 0.805, p95 1.721) on the selection slice and 0.9295 seconds mean
(p50 0.860, p95 1.570) on the future slice. These are research-evaluator numbers
on a single `RTX 3090 Ti`, not resident product-protocol or macOS end-to-end
latency.

## Usage

This is an internal prototype lane. It requires the LFM2.5 audio and VL bases,
the reverse graft code, and the adapter above on controlled compute:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=<repo>/ml/research/final_sweep:<repo>/ml/research/reverse_vl_v0 \
python <repo>/ml/research/reverse_vl_v0/transcribe_reverse_audio_vl.py \
  --slice <slice.jsonl> \
  --out <hyps.jsonl> \
  --adapter <adapter.safetensors> \
  --lora-rank 32 \
  --audio-root <audio-root> \
  --device cuda \
  --prompt-id prose_history_dictation_v2 \
  --prompt-file <local-fixed-history-card.txt>
```

`--prompt-file` exists so a personal history card can be supplied from controlled
storage without being committed. Producing that card is a local research step;
it is not distributed.

## Artifacts and release state

- Weights published: **no**
- Tokenizer published: **no**
- Training data published: **no**
- Fixed history cards published: **no**

The adapters were trained partly on personal dictation audio and accepted text.
Publishing any adapter, even privately, is a separate explicit release decision
and has not been taken. Candidate adapter hashes are recorded above so a future
release can be matched to a verified artifact.

## Privacy

No audio, transcript, accepted text, dictionary value, screenshot content, or
personal history card is present in this repository. Personal dictation data is
not uploaded to any endpoint, artifact, or telemetry provider. Internal receipts
record only audio IDs, hashes, counts, and aggregate metrics, and those receipts
stay on controlled compute storage.

## Honest limitations

- This prototype does not beat the owner's real-time Aqua Voice raw transcript on
  the untouched future slice. It improves the prior checkpoints within this lane,
  and it is not a claim of overall dictation superiority.
- The K8 card result is one configurable sample of one user's history. Card size,
  card composition, and recency weighting are not yet tuned or held out as
  independent factors.
- The fixed history card is user-specific. A card from a different user, or an
  empty card, has not been measured and should not be assumed to transfer.
- The static prose adapter collapses under the history prompt (0.390525 fair WER).
  History conditioning requires the history-trained adapter; it is not a
  prompt-only upgrade.
- Two-pass self-retrieval, which needs no reference transcript, is worse than both
  the fixed card and per-row retrieval, and roughly doubles generation.
- Aggregate WER is corpus-level. It is not byte-for-byte transcript parity, and
  no per-row confidence calibration is published.
- Kernel mode is part of the receipt. Optimized and reference modes matched here,
  but that is a measured result for these artifacts, not a general guarantee.
- The reverse engine remains an explicit non-default option. The shipped default
  engine is unchanged and no product promotion has occurred.

## Version history

| Version | Date | Change |
| --- | --- | --- |
| `v0.1.0-alpha.1` | 2026-10-06 | Provisional history-conditioned prototype: fixed-card configuration, parity results, and honest non-claims |

## License and credit

This card and the described artifacts are derivatives of Liquid AI's LFM2.5
family and inherit the upstream LFM1.0 license terms. Upstream credit to Liquid AI.
