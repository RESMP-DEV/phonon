# Phonon architecture

## Philosophy

Phonon's non-negotiable priority order is privacy and user sovereignty first, then final-text fidelity, then interaction latency, then personality and convenience. Every architectural decision below is subordinate to that order. The product is a local-first voice authoring system: microphone in, intended text out, with personal context retained under explicit consent and no cloud dependency.

## Ownership table

| Boundary | Owner | Current position |
| --- | --- | --- |
| Product specification | `SPEC.md` | Canonical product requirements and policy; needs status links after sections are audited |
| Architecture and roadmap | `docs/architecture.md` | This document; the single cross-component plan |
| macOS app and capture path | `bar/`, `crates/` | Shaped as the local product surface |
| Screen context and vision input | `bar/Sources/PhononBar.swift`, `crates/phonon-core` | OCR-only today; image path and sidecar protocol still open |
| Audio front end | `MicRecorder`, `phonon-audio` | Hardware-rate capture with independent streaming and final paths |
| Final-training Optuna sweep | `ml/research/final_sweep` | Prompt-baked, context-keyed packs; live study running on B550 |
| ML correction and research | `ml/` | Canonical text-refiner research pipeline |
| Profile and vocabulary mining | `tools/profile-miner` | Local consent-gated personal context lane |
| Synthetic persona evaluation | `tools/persona-gym` | Synthetic-user regression tool for mining |
| Distribution | `DISTRIBUTION.md`, `scripts/`, platform crates | Packaging, signatures, and pinned runtime assets |
| Public site | `website/` | Separate web surface; consumes stable claims only |
| Audio/vision SALM experiments | local `phonon-eval` workspace, B550 `~/salm-lora`, and the committed product seam on `ml/portable-harness` | Research adapters exercised and measured; product integration is committed and the remaining path is gated on multimodal and distribution work |

## Product planes

1. **Speech plane.** Capture audio locally, recognize it, and preserve the acoustic signal until final text is committed or deliberately discarded.
2. **Intent plane.** Turn a raw transcript into the words the user intended, using local dictionaries, a profile, and restraint guards. A poor correction must never replace a better raw transcript.
3. **Context plane.** Mine, review, retain, and expire personal vocabulary and style facts locally. Every source is opt-in, reviewed where used for candidate generation, and journaled.
4. **Parody plane.** Transform text only after the user has selected a parody mode. It is an explicit mode, never a hidden personality applied to ordinary dictation.
5. **Evaluation plane.** Freeze fixtures, register protocols, and gate every user-visible model or transform on measured fidelity, restraint, latency, and privacy invariants.

## Parody architecture

### Goals and non-goals

The goal is a local, deterministic-first transform layer that can parody the user's own style and named external styles while preserving semantic content. It should support short voice commands such as “in the style of …” or “make that sound like …”, preview the result, and return to the intended text without losing either version.

The non-goals are impersonation of private individuals without their consent, unattributable public impersonation, network model calls, hidden always-on personality, and allowing style transformations to weaken the accuracy of normal dictation.

### Architecture

`ParodyRequest` is the sole entry point:

```text
ParodyRequest {
  source_text: IntendedText,
  style: StyleId,
  intensity: 0..=3,
  user_profile: ProfileView,
  interaction: Selected | PreviewOnly,
}
```

A style resolves to a signed `StyleContract` containing protected-content rules, allowed and forbidden transformations, source/licensing metadata, sample pairs, and calibration controls. Built-in styles are versioned assets. User-derived styles are generated only from consented local history, are marked `derived: true`, and name their source as the user rather than a private third party.

The transform is a four-stage local pipeline:

1. **Meaning lock.** Extract protected entities, code identifiers, URLs, numeric facts, quotations, negation, ordering, and semantically load-bearing adjectives. These form a structured meaning contract.
2. **Style synthesis.** Apply the style contract. The implementation may begin as deterministic template and lexical transforms, then graduate to the local refiner or SALM once its base fidelity is proven. The model path must receive the meaning lock and style contract, not just a free-form prompt.
3. **Restraint pass.** Reject or repair changes to protected content, unsupported factual claims, first-person identity changes, attribution changes, and intensity above the selected level. Deterministic checks are mandatory even when a model produced the candidate.
4. **Safety and policy pass.** Enforce consent, attribution, protected-character controls, transformation disclosure, and style policy before a candidate can leave preview.

Each accepted request writes a sanitized local `ParodyEvent`. The event records the style version, selected intensity, transform implementation, latency, user action, and evaluation labels. It never stores audio or full personal text merely for telemetry. Trace-derived tuning data is local, reviewed, and revocable; it is not automatically added to a training corpus.

### Feature sequence

The first implementable slice is deliberately not a model feature:

1. Define `ParodyRequest`, `StyleContract`, `IntendedText`, `ParodyResult`, and `ParodyEvent` as typed, tested platform-neutral interfaces.
2. Implement two built-in styles with deterministic transforms: `pirate` and `noir`. These are unambiguously fictional parody styles, require no external personality rights, and exercise the full preview, accept, reject, and undo path.
3. Add a meaning lock with protected spans and sentence-level semantic assertions. Start with entities, identifiers, numbers, URLs, quotations, negation, and ordering.
4. Add the restraint and policy passes as pure functions with adversarial tests. They must catch entity substitution, identifier mutation, number drift, dropped negation, reordered prerequisites, and intensity escalation.
5. Add explicit UI state: source text, parody preview, mode, undo, and disclosure that a transformation was applied. No parody text is inserted by a background path.
6. Add user-derived style extraction as a separate opt-in stage, using consented history and a review sheet. It initially outputs a lexical/style card, not a trainable persona.
7. Only after those slices pass should model-backed synthesis use the same interfaces and be compared against the deterministic baseline.

### Evaluation and gates

A parody transform is not accepted on vibes. It requires four registered suites:

1. **Semantic preservation.** Protected spans are exact; entities and identifiers are unchanged; numbers and units match; quotations and attributions remain verbatim; negation and ordering survive. A semantic judge may supplement deterministic checks, but cannot replace them.
2. **Style effect.** A frozen, human-labeled style corpus scores recognizable style change, intensity calibration, humor, and non-repetition. Fresh outputs are never pooled with historical replay scores.
3. **Harm and rights control.** Tests cover private-person impersonation, false attribution, protected-character confusion, impersonation of official communications, deceptive formatting, and consent-denied styles. Refusal and disclosure are expected outcomes.
4. **Product invariants.** Source remains recoverable after insert, preview never modifies history, no network is used, disabled styles disappear from all UI states, and latency remains inside the mode's budget.

Minimum release gates for the deterministic slice are zero protected-span mutations, zero dropped negations, zero number or ordering changes, all adversarial policy tests passing, and source recovery passing. Model-backed synthesis additionally requires semantic preservation at or above the deterministic baseline on the same frozen set, no degradation in the normal dictation benchmark, and equal or better end-to-end latency for the selected plane.

### Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Transform engine | Deterministic transforms first; model path deferred | Model path must beat deterministic semantic/style gates without affecting normal dictation |
| Interface owner | Typed platform-neutral contracts in the core | Implement Rust core structs plus Swift command serialization without UI coupling |
| Style pack format | Versioned signed local assets | Signature, revocation, and migration tests once distribution begins |
| User-derived styles | Opt-in reviewed style cards | Consent, review, deletion, and re-identification audit on real local histories |
| Profanity and edge humor | Allow only where explicit user policy permits, never in default styles | Policy table plus adversarial locale tests |
| Trace use | Sanitized local `ParodyEvent` only | Human review and deletion/export semantics before any training corpus exists |
| Voice-trigger scope | Selected command changes the immediately preceding dictation | Live capture-state tests prove no background transformation |

## Unified audio-head and vision-head transplant

### Current position

The committed architectural direction is a single-model multimodal SALM lane: audio or an image in, intended text out, replacing the separate ASR, retrieval, and polish cascade for an experimental engine selection. This is motivated by measured production behavior: the two-model polish cascade adds latency and can corrupt otherwise acceptable raw text, while a local SALM LoRA can improve the same local 500-clip slice.

The repository contains the first committed product integration seam on `ml/portable-harness`: `sidecar/salm_server.py`, ASR-sidecar selection in `crates/phonon-asr`, benchmark wiring in `crates/phonon-cli`, and README instructions. It targets `LiquidAI/LFM2.5-Audio-1.5B`, speaks the existing JSONL protocol for whole utterances, and intentionally does not support streaming partials.

The completed experiments live in the Claude Code workspace and on B550 rather than this Git checkout:

- Work directory: `/Users/kearm/agent_workspace/phonon-eval` locally and `~/salm-lora` on B550.
- Local 500-clip slice aggregate metrics and hypothesis files: `hyps_grpo_v1.jsonl`, `hyps_v4full_*.jsonl`, and `results-salm-vs-parakeet.md`.
- B550 final audio adapter: `salm-grpo-v1/lora_adapter.safetensors`, SHA-256 `d157a38be7ebc801000a6bac13146b8f7333b4a89f8dcda1eb9e26ad5b54f830`.
- B550 vision CPT adapter: `salm-vision-cpt-v1/cpt_adapter.safetensors`, SHA-256 `a43fac977d6fa3bf8150257ac26db8a9cea39d8eb937db17d64ecc493260d6dc`.
- B550 vision evaluation: `vision_eval_cpt.json`, SHA-256 `900042c309f8fc3305dc42f666a5b51f6834308f0a13785437210d13d08b7287`.
- Vision surgery provenance: `vision-transplant/manifest.json`, audio snapshot `c362a0625dfe45aa588dce5f0ada28a7e5707628`, VL snapshot `919fde3d022e3f90a4716006f993938ee8c2eb97`, 105 target IDs, OMP `k=64`, and mean donor-relative residual `0.17129524052143097`.

### Measured state

The experiments are implemented and exercised, but not shipped or integrated into the app:

| Capability | Result | Boundary |
| --- | --- | --- |
| Full-audio SFT checkpoint | `v4full@2000`: 0.0851 fair WER, 0.1157 strict WER, 0.370 exact on the local 500-clip slice | Experimental local slice; product protocol and packaging are absent |
| GRPO pilot | 300 steps from `v4full@2000`: 0.0839 fair WER, 0.1132 strict WER, 0.360 exact, unchanged 0.66 s/clip | Passed its preregistered improvement-over-anchor gate; not yet a release gate |
| Vision surgery and CPT | 24,000 steps in 56 minutes; final lane means audio 0.442, corrector 0.225, vision 0.034; projector remained frozen | Research adapter, not integrated with the app sidecar |
| Vision reading | 200 held-out screenshots: real-image WER 0.0250, exact 79.5 percent | Research-only screenshot transcription |
| Vision blank control | WER 0.9970, exact 0.0 with white images preserving token layout | Rules out caption recitation as the source of the vision score |
| Audio after vision CPT | Fair WER 0.0902 versus 0.0874 for the v2 mixed SFT baseline | About +0.003 audio cost while learning the vision lane |
| Baselines | Parakeet 0.0807; Eloquent E4B 0.1356 on the same local slice | Parakeet still leads fair WER; SALM narrows the lead |

The vision architecture uses the same-family `LFM2.5-VL-1.6B` SigLIP2 tower and projector. No vocabulary growth is required: the audio model already has 105 reserved rows that map to VL image-token IDs 396 through 500. The surgery transfers those rows with OMP, keeps the audio modality IDs, injects image embeddings at the existing modality-flag seam, and trains a LoRA in the language model while the tower and projector remain frozen. Generation must truncate pack rows at the first supervised position; feeding prompt plus caption samples post-caption pretraining boilerplate and is not a valid inference path.

### Work packages

1. **Preserve the committed product seam.** Keep custom/default ASR process selection independent of model claims, with tests proving the custom script/runtime selection and that the default Parakeet command remains unchanged.
2. **Bring the experiment under durable source control.** Review and commit the local experiment scripts, omitting personal data, then record adapter hashes and score receipts in an evidence index. The existing local workspace is not a release process.
3. **Protocol contract tests.** Exercise ready, ping, transcribe, warmup, unsupported streaming, missing adapter, malformed JSON, and shutdown behavior without loading the model by injecting a fake engine boundary into `salm_server.py`.
4. **Create the audio release gate.** Repeat the GRPO result with a frozen, manifest-hashed fixture and compare SALM, Parakeet, and the two-model refiner under one registered protocol. The present local slice is strong evidence, not a product ship gate.
5. **Add the multimodal protocol.** Extend the sidecar contract with an explicit image field, screenshot preprocessing metadata, and a model capability declaration. Do not encode screenshots as an audio command.
6. **Run a vision release gate.** Expand beyond the 200-row research set, add non-Phonon screenshot distributions and blank/degraded controls, and define protected-content and privacy handling for screen context. Keep audio regression as a simultaneous gate.
7. **Product path.** Route an explicit experimental engine selection through the same capture, undo, telemetry, readiness, and offline checks as Parakeet. Vision input must remain opt-in and include retention, provenance, and screen-context policy.
8. **Distribution decision.** Quantize and package the chosen adapter plus the vision tower/projector, pin every component, and measure latency and resident memory on target platforms. A LiteRT-LM container remains an option, not a chosen destination.

### Acceptance gates

The product seam can be called implemented only when its code and protocol tests are committed and pass the relevant Rust and Python checks. The research adapters are already exercised on real audio and screenshots, but the lane can be called product-ready only after the frozen-fixture audio and vision gates pass, sidecar multimodal behavior is tested, offline artifact pinning works, streaming behavior is explicitly represented, and the default Parakeet path has no regression. The current vision adapter must not be described as shipping in Phonon merely because its research evaluation passed.

## Screen context and the vision-head image path

### Current position

Screen context is implemented in `bar/Sources/PhononBar.swift` as
`ScreenContextCapture.recognizeAllDisplays()`, and it is OCR-only. The captured
image is discarded before the audio finishes; only recognized text survives, and
only the dictionary terms that the transcript already resembles are forwarded to
the correction stage. `crates/phonon-core/src/data.rs::screen_confirmed_terms`
keeps a term only when the normalized screen text contains the canonical form
and some known form of that entry is contextually similar to the raw
transcript, so full screen text never reaches the correction prompt.

The vision-head product path is not implemented. Productizing it requires
forwarding the image itself, or a normalized representation of it, alongside the
audio, which is a protocol change rather than a capture change.

### Measured capture contract

The capture behavior below was read from the current source, not from a
specification.

| Property | Current behavior | Owner |
| --- | --- | --- |
| Permission | `CGPreflightScreenCaptureAccess`, then `CGRequestScreenCaptureAccess`; denial returns empty text silently | `ScreenContextCapture` |
| Scope | Every display returned by `SCShareableContent`, not the active display | `ScreenContextCapture` |
| Self-exclusion | Phonon's own application is excluded from every filter | `ScreenContextCapture` |
| Filter | One `SCContentFilter` per display over the whole display, no window selection | `ScreenContextCapture` |
| Pixel size | `SCStreamConfiguration.width` and `.height` set to `display.width` and `display.height`, the display's native logical point size | `ScreenContextCapture` |
| Cursor | `showsCursor = false` | `ScreenContextCapture` |
| Capture API | `SCScreenshotManager.captureImage`, a still image, not a stream | `ScreenContextCapture` |
| Timing | Started at recording start, concurrently with audio capture, not at correction time | `PhononBar` dictation start |
| OCR | `VNRecognizeTextRequest` with `recognitionLevel = .accurate` and language correction on | `ScreenContextCapture` |
| Output | Text only, displays joined with newlines; the image is released | `ScreenContextCapture` |
| Failure | Any capture or OCR error is logged and degrades to empty context | `ScreenContextCapture` |
| Blocking | If OCR is still running when ASR returns, the final polish waits for it | `PhononBar` pending final polish |

### Model-side image requirements

The image requirements come from the pinned donor
`LiquidAI/LFM2.5-VL-1.6B` snapshot `919fde3d022e3f90a4716006f993938ee8c2eb97`,
read from its `config.json` and its `Lfm2VlImageProcessor` implementation, then
confirmed by running the processor locally over synthetic images at each display
size, using Transformers 5.17.0 to match the B550 training environment. The
model snapshot has no `preprocessor_config.json`, so the image processor is
constructed from model-config values rather than loaded as a standalone file.

| Property | Value | Source |
| --- | --- | --- |
| Tile size | 512 x 512 pixels | `tile_size` |
| Encoder patch size | 16 pixels | `encoder_patch_size`, SigLIP2 |
| Downsample factor | 2, giving 16 x 16 = 256 language-model placeholders per full 512px tile | `downsample_factor` |
| Tiles per image | 2 to 10 chosen by aspect ratio, plus one thumbnail when the grid is larger than one tile | `min_tiles`, `max_tiles`, `use_thumbnail` |
| Splitting | Enabled for images above roughly 256 tokens of pixels; smaller images are resized to a single image | `do_image_splitting`, `max_image_tokens`, `max_pixels_tolerance` |
| Resize filter | Bilinear with antialiasing | `resample` |
| Normalization | Channel-wise mean 0.5 and standard deviation 0.5, that is `(x / 255 - 0.5) / 0.5` | `image_mean`, `image_std` |
| Image token id | 396, with the audio model reserving rows 396 through 500 for the same range | `image_token_id` |
| Text | 65536 rows, 128k positions, 16 layers, 2048 hidden | `text_config` |

Measured tile behavior for realistic display sizes, produced locally with the
real processor:

| Display size | Tile grid | Processor resize | LM image placeholders |
| --- | --- | --- | ---: |
| 1440 x 900 | 3 x 2 | 1536 x 1024 | 1,776 |
| 1512 x 982 | 3 x 2 | 1536 x 1024 | 1,764 |
| 1920 x 1080 | 4 x 2 | 2048 x 1024 | 2,300 |
| 2056 x 1329 | 3 x 2 | 1536 x 1024 | 1,764 |
| 2560 x 1440 | 4 x 2 | 2048 x 1024 | 2,300 |
| 3008 x 1692 | 4 x 2 | 2048 x 1024 | 2,300 |
| 3456 x 2234 | 3 x 2 | 1536 x 1024 | 1,764 |
| 3840 x 2160 | 4 x 2 | 2048 x 1024 | 2,300 |
| 5120 x 2880 | 4 x 2 | 2048 x 1024 | 2,300 |

The counts come from tokenizing the complete processor expansion with the pinned
tokenizer, not from hand arithmetic. The composition is 256 placeholders per
512px tile plus a separately `smart_resize`-budgeted thumbnail, which is why a
3x2 grid does not cost exactly 6 x 256 placeholders.

Two consequences follow from this measurement, and both are budget problems
rather than quality problems. First, every measured display size collapses to a
3x2 or 4x2 grid, so a 1440p screenshot and a 5K screenshot cost within 30 percent
of each other: nominal resolution buys almost nothing, and active-display versus
all-display is a context, latency, and privacy decision. Second, a single display
costs roughly 1,800 to 2,300 language-model placeholders. That already exceeds
the 512-1024 context lengths in the current reverse-graft sweep and must be
treated as a real per-request context cost, which is a further argument for one
active display rather than unconditional all-display capture.

Separately, the current capture request is expressed in display logical
dimensions, but the backing scale of the `CGImage` that `SCScreenshotManager`
actually returns still needs to be measured on a live Retina system before any
small-text fidelity claim is made.

### Image protocol for the vision head

The sidecar protocol is a JSONL request and response stream. A multimodal
request must add image fields without disturbing the existing audio fields, and
must declare capability so a single-stage engine can refuse cleanly:

```text
transcribe {
  cmd: "transcribe",
  id: "...",
  path: "/path/to/audio.wav",
  images: [
    {
      path: "/path/to/screenshot.png",
      display_id: 1,
      width: 5120,
      height: 2880,
      scale_factor: 2.0,
      captured_at_ms: 1759400000123,
      origin: "screen_context",
    },
  ],
}
```

The engine must answer `ready` with a capability declaration such as
`{"audio": true, "image": true, "max_screenshot_tokens": 2300}` so the app can hide
the screen-context affordance when the selected engine cannot consume images.
Screenshots are never encoded into an audio command, never base64'd into the
`pcm16` field, and never written into the corpus unless retention is separately
consented.

### Work packages

1. **Add the image field and capability declaration** to the sidecar protocol and
   the Rust `AsrSidecar` request builder, with the audio-only path unchanged when
   no image is attached.
2. **Contract-test the multimodal request** without loading a model, using the
   existing fake-engine seam: image accepted, image refused by an audio-only
   engine, missing file, malformed image record.
3. **Return a normalized image to the bar.** Capture the CGImage, encode a
   bounded-resolution PNG, write it to a per-session temporary directory, and
   delete it on session end exactly as the audio WAV is deleted today.
4. **Decide display scope.** Active display only, or all displays, with the token
   cost and privacy cost stated for each.
5. **Keep the OCR path.** OCR remains useful without the vision head, since it
   is what ranks dictionary candidates today. It should not be removed when the
   image path is added.
6. **Gate the vision lane** with real held-out display captures, blank and
   degraded controls, a simultaneous audio regression check, and a privacy review
   of what a retained screenshot can contain.

### Acceptance gates

The image path is implemented when the protocol carries images, a model capability
declaration is reported, the capture and deletion lifecycle is tested, an
audio-only engine still passes the default Parakeet path unchanged, and the
measured token and latency cost of one and of all displays is recorded. It is
product-ready only after a registered vision gate on real display captures passes
alongside the audio gate.

## Audio capture and normalization contract

### Current position

Audio capture, streaming, and final-WAV writing all live in `MicRecorder` in
`bar/Sources/PhononBar.swift`. The audio language model is trained and evaluated
on 16 kHz mono, and the current implementation already delivers that. What does
not exist is a single declared contract: resampling is ad hoc linear
interpolation, normalization is absent by design, and the streaming and final
paths duplicate the resample call without sharing a filter or a provenance
record.

### Measured capture contract

| Property | Current behavior | Owner |
| --- | --- | --- |
| Source | `AVAudioEngine` input tap on bus 0 at the hardware input format | `MicRecorder` |
| Rate | Hardware rate, re-read and reinstalled whenever the device renegotiates its format | `MicRecorder` |
| Tap buffer | 1024 frames | `MicRecorder` |
| Channels | Averaged to mono at ingest | `MicRecorder` |
| Level meter | RMS dBFS mapped from -55 to 0 dB, smoothed 0.18 past and 0.82 present, every second frame published; UI only, not applied to samples | `MicRecorder` |
| Duration cap | At most 120 seconds retained, oldest samples dropped | `MicRecorder` |
| Streaming | Accumulate about 0.4 s, resample to 16 kHz, encode PCM16, send as base64 `stream_chunk` | `MicRecorder`, `PhononBar` |
| Final path | Resample the whole captured buffer to 16 kHz, write mono PCM16 WAV under `Corpus/<id>/audio.wav` | `MicRecorder` |
| Minimum clip | Below about 80 ms no WAV is written, which suppresses a cold-start stub | `MicRecorder` |
| Resampling | Linear interpolation, passthrough when the rates are within 0.5 Hz | `MicRecorder` |
| Quantization | Clamp to [-1, 1], scale by `Int16.max`, round to nearest | `MicRecorder` |
| Speech gate | Rust-side adaptive gate on the written WAV before ASR | `phonon-audio` |

The speech gate in `crates/phonon-audio` requires mono PCM16, so the WAV
contract is already load-bearing: a future change to the WAV format breaks the
gate before it breaks the model. It frames audio at 50 Hz, takes the 20th
percentile frame energy as the noise floor, requires energy above three times
that floor with an absolute floor of 180, caps the voiced zero-crossing rate at
0.20, and requires four consecutive voiced frames, that is about 80 ms of
sustained speech-like energy.

### Target contract for the audio head

The audio language model consumes 16 kHz mono float audio through
`ChatState.add_audio`, with the sample rate passed explicitly, so the sidecar
trusts the file's declared rate. That makes the writer, not the model, the owner
of correctness. The target contract, which the current code satisfies only
partially, is:

1. One resampler instance and one filter chain shared by the streaming path and
   the final path, so a streamed partial and the final WAV of the same utterance
   carry identical signal treatment.
2. `AVAudioConverter` with an explicit quality setting instead of linear
   interpolation, or a documented measured justification for keeping linear
   interpolation when the source rate is 48 kHz and the target is 16 kHz.
3. No automatic gain applied to the samples. The meter is a display concern.
   Record the capture level as metadata so a training or evaluation pass can
   restrict or stratify by level.
4. DC-offset removal before resampling, since a single-tap `AVAudioEngine` path
   can carry a small DC bias that a mean-removed signal avoids.
5. Clipping provenance: record whether the clip reached full scale, so a
   truncated clip is visible in evaluation instead of being silently scored.
6. Provenance in model input metadata: hardware rate, channel count, selected
   device name, resampler identity, target rate, and whether normalization ran.

### Work packages

1. Extract a single `AudioFrontEnd` in Swift that owns downmix, DC removal,
   resampling, and quantization, and route both the streaming and the final path
   through it.
2. Add a unit test that a 48 kHz synthetic tone and its 16 kHz conversion agree
   on frequency and level within a stated tolerance, and that a stereo input
   downmixes to the expected mono amplitude.
3. Extend the corpus `metadata.json` with the front-end provenance fields and a
   clipping flag, then make the evaluation report stratify by level.
4. Replace the linear resampler with `AVAudioConverter` and measure the WER delta
   on the frozen 500-clip slice before adopting it.
5. Decide whether the streaming path stays Parakeet-only. The SALM sidecar
   currently rejects `stream_start`, `stream_chunk`, and `stream_stop`, so the
   0.4-second chunk contract is a Parakeet feature, not a shared one.

### Acceptance gates

The audio contract is implemented when both paths share one front end, tests
cover downmix, resample, DC removal, and quantization, and metadata records
provenance. It is product-ready when the resampler change is measured on the
frozen slice without regressing fair WER, and when the speech gate is proven
against real clips that contain no speech.

## Work-log

### 2026-10-02: screen and audio architecture audit

Audited the live screen-context and microphone paths from source rather than intent. `ScreenContextCapture` captures every display with ScreenCaptureKit, excludes Phonon itself, resizes to native logical dimensions, hides the cursor, and immediately discards the image after Vision OCR; only transcript-relevant dictionary terms survive into the correction prompt. Measured the exact LFM2.5-VL image processor on common display sizes: all realistic captures collapse to 3x2 or 4x2 grids of 512px tiles plus a thumbnail, consuming 1,764 to 2,300 language-model image placeholders regardless of nominal resolution. A 5K capture costs barely more than a 1440p one, so active-display versus all-display is a context, latency, and privacy decision rather than a fidelity one.

Audited `MicRecorder` and the Rust speech gate. Current capture is hardware-rate stereo averaged to mono, retained up to 120 seconds, and separately resampled by linear interpolation to 16 kHz for streaming and final PCM16 WAV. The level meter is UI-only, no automatic gain is applied, and the final WAV is required to be mono PCM16 before the adaptive speech gate runs. Added architecture sections for the multimodal image protocol, exact vision preprocessing, and the target shared audio front end with resampler, DC-offset, clipping, and provenance gates.
### 2026-10-01: reverse Audio-to-VL graft and public audio CPT

Implemented the reverse graft under `ml/research/reverse_vl_v0`: retain the VL language stack and vision path, transplant the Audio conformer and audio adapter, reserve token id 14 for continuous audio slots, and scatter projected audio embeddings into the VL language stream. The trainer updates a VL LoRA plus the audio adapter while freezing the VL base and conformer; the evaluator loads a saved graft adapter and performs greedy audio-to-text generation.

A synthetic end-to-end probe succeeded before training. On the first 25 rows of the frozen Aqua slice, a 2,000-step Aqua-only reverse graft scored 0.1458 fair WER / 0.2029 strict / 0.28 exact. Public audio pretraining used the open `espnet/yodas-granary` English stream (embedded 16 kHz WAV, CC-BY-3.0 backing audio; `nvidia/Granary` CC-BY-4.0 manifests). A 2,000-row/2,000-step public stage alone scored 0.1840 fair WER, but public pretraining followed by 1,000 Aqua steps improved to 0.1337 fair WER / 0.1755 strict / 0.40 exact. A larger 10,000-row, 10,000-step public run followed by 1,000 Aqua steps scored 0.0938 fair WER / 0.1335 strict / 0.36 exact on the same 25-row slice, showing a strong public-data scaling direction. This lane remains research-only and behind the native Audio student. The first 800-step high-quality sequential arm scored 0.0877 fair WER on the frozen 500-clip slice; its first interleaved score was discarded because the evaluator used sequential generation; the corrected interleaved result appears below.

### 2026-10-02: persistent Optuna and prompt-baked final-run harness

Added `ml/research/final_sweep`. It provides a persistent Optuna study for the eventual reverse Audio-to-VL production run, optimizing LoRA rank/dropout, learning rates, context length, and warmup while sampling or fixing a registered prompt ID. Public YODAS-Granary and Aqua packs are rebuilt with that exact prompt, the evaluator uses the same prompt ID, and Optuna records the prompt hash and score receipts. The public dataset revision is pinned to `969944574ea3f37890beaf67ea651e160cfaf043`; Aqua manifests are content-hashed. A dry-run mode prints the complete command plan without training, and three contract tests verify prompt propagation and checkpoint naming.

The corrected interleaved HQ evaluation scored 0.1135 fair WER / 0.1455 strict / 0.338 exact with 3.76 seconds mean generation, so the sequential HQ arm remains better at 0.0877 fair WER and 0.66 seconds generation. A first sequential-versus-interleaved score produced 1.4135 fair WER and was discarded because the interleaved adapter was incorrectly evaluated in sequential generation mode.

### 2026-10-01: API-first correction curation

Corrected the model-selection boundary: larger third-party models are API teachers and evaluators, not local Phonon runtimes. Stopped and removed the experimental B550 Voxtral download and environment. A live synthetic vision probe through the OCR/review OpenAI-compatible service exactly recovered `PHONON_API_VISION_42`; the configured fallback served `glm-5.3-flash`, proving image transport through the local API tier. Direct fully qualified free OpenRouter Gemma and Nemotron Omni probes timed out, so the immediately reliable teacher path is the configured tier rather than a hand-selected free model ID.

Added `tools/api-curator`, a resumeable CLI that sends only raw and final dictation text to an OpenAI-compatible endpoint. It validates a strict JSON contract (`keep`, `revise`, or `discard`, corrected text, reason code, confidence), never sends audio paths or local session metadata, and stores a completion for audit without requesting or storing reasoning. Five offline tests pass. A three-row synthetic live pilot through `glm-5.3-flashx` succeeded, and a 25-row real-data pilot returned valid judgments for every row: 14 keep, 11 revise, 0 discard, mean request latency 3.688 seconds, max 15.461 seconds.

The vision-head graft process is the already-measured research path, now reframed for screenshot conditioning: donor `LFM2.5-VL-1.6B`; copy its SigLIP2 tower and 2048-dimensional projector into `LFM2.5-Audio-1.5B`; reuse audio-vocab reserved IDs 396-500 as image-token rows; initialize those rows by reserved/copy/OMP race; inject image embeddings at the existing modality flag seam; train a LoRA in the audio LM with tower and projector frozen; and always truncate generation at the first supervised position. The measured adapter reads real held-out screenshots at 0.0250 WER / 79.5 percent exact and collapses on blank controls. Productization requires an explicit image field, consented screen capture, API-teacher curation of screenshot-caption pairs, retention policy, and simultaneous audio regression.

### 2026-10-01: higher-quality audio data and model survey

Measured two longer GRPO arms after the 300-step pilot. The 500-step group-8 control scored 0.0848 fair WER and the 500-step group-12/SFT-init-KL arm scored 0.0840, while GRPO v1 remained best at 0.0839. This closes the "just run GRPO longer" hypothesis for the current recipe.

Audited the Aqua packs without emitting transcript contents. The full interleaved pack is valid and substantial: 31,652 rows with 2,619,605 text slots, 3,258,794 audio-in slots, and 3,292,509 audio-out slots. The newer v3/v4 mixed packs use sequential audio-in rows plus text correction rows and do not include the interleaved read-along lane. A matched high-quality rebuild therefore provides new signal rather than a duplicate experiment.

The manifest has 34,246 rows, mean duration 9.44 seconds, and 32,813 correction-positive rows. A higher-quality selection keeps only correction-positive clips between two and twenty seconds, excludes the frozen 500-clip slice, and uses a stable audio-ID hash split: 12,855 train, 1,458 development, and 750 future rows. Work began on matched sequential and interleaved packs over exactly those training rows.

A public model survey ranked Voxtral Mini 3B, Voxtral Small 24B, Voxtral Mini 4B Realtime, Gemma 4 E4B/12B, Qwen3 Omni 30B-A3B, and LFM2.5-Audio as the strongest practical audio-text candidates. Voxtral Mini 3B is the first external baseline to download and run because it combines an Apache-2.0 license, 32k context, roughly 30-minute transcription support, a Whisper-family audio head, and an 8.72 GiB selected-weight footprint. Voxtral Mini 4B Realtime is the streaming-specific follow-up; Gemma 4 is a strong long-context but chunked-audio candidate; Qwen3 Omni and Voxtral Small are quality ceilings rather than routine local baselines.

### 2026-10-01

Recovered ownership of the SALM and vision transplant thread from the Phonon Claude Code session. Reconciled the architecture with its surviving local and B550 artifacts, adapter hashes, and aggregate metrics.

Implemented the first takeover slices. Added `AsrEngineSelection` as one tested launch plan for default Parakeet and custom SALM selection, wired both `AsrSidecar` and the ASR benchmark to it, and added five no-model JSONL protocol tests for `sidecar/salm_server.py`. Added deterministic-first parody contracts in `phonon-core` with protected URL, identifier, and number spans, ordered and duplicate-count preservation checks, negation and source recovery tests, and two fictional built-in styles. The Rust workspace, Python sidecar tests, formatting, and Clippy pass locally. These slices do not wire either capability into the UI or ship the measured research adapters.

Added the canonical architecture after auditing the repository. Encoded the parody plane as a deterministic-first, consent-gated, meaning-locked transform architecture with registered semantic, style, rights, and product gates. Classified the SALM transplant as implementation-first and measurement-gated, and defined preservation, protocol, stock-load, adapter, benchmark, product, and distribution gates. No application code or model behavior was changed in this edit.

## Contracts to preserve

- Local execution and user sovereignty outrank quality, personality, and performance.
- Normal dictation accuracy must not regress to obtain a style or model experiment.
- Raw or intended text must remain recoverable when a transform or experimental engine is selected.
- Evaluated outputs stay within their registered workflow and are never pooled with unrelated protocols.
- Personal data is local, consent-gated, journaled, reviewable, and deletable.
- Unverified implementation is never reported as a shipped or working capability.
