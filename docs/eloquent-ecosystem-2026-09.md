# Google AI Edge Eloquent: ecosystem review and local assets

Survey date: 2026-09-28. Everything here was verified live on the main
machine (app version, databases, model files) or against primary sources
(GitHub releases API, Hugging Face API, iTunes API) on that date. Facts
measured against Eloquent v1.5.1 in July 2026 are marked as such.

## Why this matters for phonon

Eloquent is Google's shipping answer to the problem phonon solves, and
its measured behavior is evidence for choices we are already making:

- **Two-model cascade vs snappiness.** Eloquent ships ASR + a Gemma
  polish pass and pays multi-second latency for it (Wispr Flow measured
  e2e p50 0.65 s / p90 1.16 s; Eloquent's cascade is multi-second, with
  an "Instant Transcript (Skip polishing)" setting as the escape hatch).
  This is production-scale evidence for the single-model SALM directive.
- **The polish pass corrupts.** Measured A/B: the Gemma pass fixes
  near-misses (llama.CPP -> llama.cpp) but inverts subject/object, drops
  words, introduces `don'm`, and truncates. Phonon's
  worse-than-input-fraction gate is the right metric; Google ships
  without one.
- **Context biasing evidence.** Eloquent's dictionary biasing reaches
  raw ASR decoding for non-decomposable tokens (raw output contained
  exact `MXFP4`, `GPTQ`) but fails for English-decomposable terms
  (`NV switch`) and phonetically distant ones (`cuDNN` -> `QDN`).
  "Failures are exact-form decisions" is the same diagnosis as our
  LATENT_CONTEXT_BIASING / vocabulary-prompting work; the unified SALM
  subsumes the candidate-conditioning role.
- **Serving lane.** LiteRT-LM v0.17 runs `.litertlm` containers with
  Metal residency on Apple Silicon and now ships C API prebuilts. If a
  Parakeet/SALM-class model ever converts to that container, it gets a
  Google-supported serving path with no build effort.

## App timeline and current state

| Date | Event |
| --- | --- |
| 2026-04-06 | iOS launch. Free, offline-first dictation, on-device Gemma. |
| 2026-06-03 | macOS port, shipped with the Gemma 4 12B announcement. Added interactive polish/rewrite and Voice Edit. |
| 2026-08-21 | iOS v1.4.1: dictation in 15 languages (14 added), UI refresh, microphone selection. |
| 2026-09-28 (verified) | Local macOS app is v1.6.1, ahead of the iOS listing. No public changelog exists for 1.5/1.6. |

Two apps can share the `com.google.AIEdgeEloquent` bundle id after an
update leaves `Edge Eloquent.app` behind; `/Applications/Eloquent.app` is
the active one on this machine. Always launch by explicit path.

New in the official docs (not present at reverse-engineering time in
July): Eloquent can optionally read workspace data such as Gmail to
auto-generate a vocabulary list. That is a cloud-side equivalent of the
local mining pipeline in RESMP-DEV/ASR (`sync_asr_dictionary.py`), which
stays on-device and covers sources Google cannot see.

## Reverse-engineered internals (v1.5.1, re-checked 2026-09-28)

All under `~/Library/Application Support/com.google.AIEdgeEloquent/`:

- `databases/eloquent_database.db`
  - `dictionary_words(word, dictionary_id, timestamp)`, PK `(word, dictionary_id)`.
    `dictionary_id` is the literal UI collection name; no registry exists.
    Write protocol: quit the app first; journal mode delete.
  - `history(timestamp, rawTranscription, polishedTranscription, duration, cache_json)`:
    raw/polished pairs are directly usable as polish-pass eval data.
  - Also `app_settings`, `contacts`.
- `key_value_store.sqlite`, table `key_value_entries` with typed columns
  (`text_value`, `int_value`, `real_value`, `bool_value`, `blob_value`).
  Notable keys: `downloaded_model_path` (JSON mapping model slots
  3/7/9/12 to filenames), usage counters
  (`total_words_dictated`, `total_edits_applied`,
  `total_recording_duration_seconds`).

## Local model assets

Plain, unencrypted `.litertlm` in the app-support directory (both usable
outside the app via LiteRT-LM):

| File | Size | Class |
| --- | --- | --- |
| `e450f245265502c7760788f.litertlm` | 6.1 GiB | 12B-class streaming ASR + LLM (streaming audio encoder) |
| `e450f245265502c7760785e.litertlm` | 2.4 GiB | E4B-class, static audio encoder + audio adapter (the polish pass) |

Plus two encrypted fallbacks (`*.tflite.enc`, 166 MB and 120 MB) that are
not usable outside the app, and per-model XNNPACK / mldrift caches. The
app bundle itself ships `libLiteRt.dylib`, `libLiteRtMetalAccelerator.dylib`,
and TopK Metal/WebGPU sampler dylibs.

Dictionary state on this machine: 996 words (Aqua Voice 776, Curated 159,
Wispr Flow 61), managed by RESMP-DEV/ASR.

## LiteRT-LM runtime: releases since July

`google-ai-edge/LiteRT-LM` is very active. The three releases that
matter for local interop:

- **v0.15.0 (2026-08-04)**: Apple Foundation Framework adapter (native
  Apple backend for text and multimodal), CLI `config.json` system with
  strict parameter precedence, JavaScript API, Gemma 4 web support.
- **v0.16.0 (2026-08-11)**: first versioned C API prebuilt shared
  libraries for all supported platforms. Natively integrating
  `.litertlm` no longer requires building anything.
- **v0.17.0 (2026-09-09)**: Metal residency on Apple Silicon, optimized
  local attention (lower memory, longer contexts), Gemma 4 12B
  multimodal + multi-token prediction. v0.17.1 (2026-09-16) fixed a
  tool-call int type bug.

PyPI `litert-lm` tracks the releases (0.17.1 as of the survey date), so
the CLI/Python API is one `uv tool install litert-lm` away.

## Open model ecosystem (litert-community, Hugging Face)

The org publishes open models in the same LiteRT container format as
Eloquent's. ASR/speech-relevant, all refreshed within the week of the
survey:

| Model | Notes |
| --- | --- |
| `VibeVoice-ASR-Streaming-1.5B` | Streaming ASR as `.litertlm` (fp32 + int8 variants, with manifest) |
| `VibeVoice-ASR-BitNet` | BitNet-weight ASR variant |
| `Fun-ASR-Nano-2512` | Small ASR |
| `Nemotron-3-Diarization-LiteRT` | Speaker diarization |
| `Audio8-TTS-Preview-0.6b`, `Qwen3-TTS-12Hz-0.6B-Base`, `Matcha-TTS`, `sopro-v2-turbo` | TTS |
| `gemma-4-{12B,26B-A4B,31B}-it-litert-lm` | The polish-model family, including web/CPU variants |

## Interop options

1. Load the app's own unencrypted models in standalone LiteRT-LM to probe
   their real contracts (tokenizer behavior, context limits, prompt
   format) without the app in the loop.
2. Assemble a fully open Eloquent-like pipeline: VibeVoice streaming ASR
   plus a Gemma 4 litert-lm polish model, with vocabulary injected
   wherever biasing turns out to be supported.
3. For phonon specifically: evaluate `.litertlm` as a distribution
   container for the SALM refiner lane once quant recipes stabilize.

## Open questions (probe before assuming)

- Does the standalone runtime expose any dictionary-biasing hook? The
  app's biasing was observed at the app layer; nothing in the public
  C API documents it.
- Do the app's two `.litertlm` files load in v0.17.1 as-is? Format
  versioning between the app's bundled runtime and the open releases is
  unverified.
- The exact identity of the 6.1 GiB streaming ASR model is not publicly
  documented; VibeVoice-ASR-Streaming-1.5B is the closest public relative,
  not a confirmed match.
- What the encrypted `.tflite.enc` slots (3 and 7) contain, and whether
  the keys are derivable locally.

## References

- App docs: https://developers.google.com/edge/eloquent
- Runtime releases: https://github.com/google-ai-edge/LiteRT-LM/releases
- Open models: https://huggingface.co/litert-community
- Dictionary schema and sync tooling: RESMP-DEV/ASR README
