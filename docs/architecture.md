# Phonon architecture

## Philosophy

Phonon's non-negotiable priority order is privacy and user sovereignty first, then final-text fidelity, then interaction latency, then personality and convenience. Every architectural decision below is subordinate to that order. The product is a local-first voice authoring system: microphone in, intended text out, with personal context retained under explicit consent and no cloud dependency.

## Ownership table

| Boundary | Owner | Current position |
| --- | --- | --- |
| Product specification | `SPEC.md` | Canonical product requirements and policy; needs status links after sections are audited |
| Architecture and roadmap | `docs/architecture.md` | This document; the single cross-component plan |
| macOS app surface | `bar/Sources/*.swift` | Native app, settings, retention, backup mirror, microphone and screen capture |
| Engine process | `crates/phonon-core`, `phonon-cli` | Warm JSONL engine, speech gate, dictionary retrieval, correction orchestration |
| ASR sidecar | `sidecar/asr_server.py`, `phonon-asr` | Pinned Parakeet MLX process; batch and streaming; reverse SALM remains explicit non-default |
| Correction sidecar | `sidecar/polish_server.py`, `phonon-llm` | Pinned Gemma MLX process, prefix cache and MTP |
| Screen context and vision input | `bar/Sources/PhononBar.swift`, `crates/phonon-core` | OCR-only today; image path and sidecar protocol still open |
| Audio front end | `MicRecorder`, `phonon-audio` | Hardware-rate capture with independent streaming and final paths |
| Local user data | `phonon-core::data`, `AppData.swift` | Dictionary, settings, paired corpus, retention, and explicit export |
| Training-corpus capture | `phonon-core::data`, `phonon-cli` | Core consent/hash/retention/export path; macOS automatic capture wiring remains unimplemented |
| Final-training Optuna sweep | `ml/research/final_sweep` | Prompt-baked, context-keyed packs; live study running on B550 |
| ML correction and research | `ml/` | Canonical text-refiner research pipeline |
| Profile and vocabulary mining | `tools/profile-miner` | Local consent-gated personal context lane |
| Synthetic persona evaluation | `tools/persona-gym` | Synthetic-user regression tool for mining |
| Distribution | `DISTRIBUTION.md`, `scripts/`, platform crates | Packaging, signatures, and pinned runtime assets |
| Public site | `website/` | Separate web surface; consumes stable claims only |
| Audio/vision SALM experiments | local `phonon-eval` workspace, B550 `~/salm-lora`, and the committed product seam on `ml/portable-harness` | Research adapters exercised and measured; product integration is committed and the remaining path is gated on multimodal and distribution work |

## Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Dictation engine | Parakeet plus the local corrector ships; the single-stage SALM lane is experimental | Frozen-fixture audio gate beats or matches the shipped cascade on fair WER at equal or better latency, with the default Parakeet path unchanged |
| Final-training format and hyperparameters | Rank 32/context 512 trial-4 recipe fixed; `prose_dictation_v1` won the matched full-slice format bake-off | A repeat/final-run gate showing the selected family beats the prior native-audio lane under the same frozen protocol |
| Training kernel stack | Reference execution remains the training-quality winner; active FlashAttention/causal kernels degraded the full repeat, Liger and warm Triton are slower | A kernel mode must match the full-slice quality of its reference-trained adapter before product or final-training promotion |
| Reverse graft versus native audio student | The promoted reverse graft improved to 0.109302 fair WER on the frozen slice but remains behind the native Audio student | Reverse graft reaches or beats 0.0877 fair WER on the frozen slice at comparable generation latency, then passes an untouched future-set gate |
| Screen context input | OCR-only today; no image is retained | An explicit image field, a model capability declaration, and a consent-gated retention and deletion policy ship together |
| Screenshot scope | All displays are captured for OCR | Measured token, latency, and privacy cost of one display versus all displays, on real captures |
| Screenshot resolution | Capture is requested in logical display points | Live measurement of the backing scale returned on a Retina display, plus a small-text fidelity check |
| Audio normalization | No automatic gain; UI meter only; linear-interpolation resampling | A shared streaming and final front end with an explicit resampler, measured without fair-WER regression |
| Training data curation | Aqua accepted text remains the ground truth; API reconciliation has not beaten Aqua raw on the full slice | A teacher or reconciliation lane that improves the full 500-clip slice rather than a 25-row pilot |
| Aqua baseline receipts | `tools/aqua_baseline.py` scores explicit local JSONL inputs with redaction | Registration of the actual frozen slice and live candidate/baseline comparison receipts |

## Product planes

1. **Speech plane.** Capture audio locally, recognize it, and preserve the acoustic signal until final text is committed or deliberately discarded.
2. **Intent plane.** Turn a raw transcript into the words the user intended, using local dictionaries, a profile, and restraint guards. A poor correction must never replace a better raw transcript.
3. **Context plane.** Mine, review, retain, and expire personal vocabulary and style facts locally. Every source is opt-in, reviewed where used for candidate generation, and journaled.
4. **Parody plane.** Transform text only after the user has selected a parody mode. It is an explicit mode, never a hidden personality applied to ordinary dictation.
5. **Evaluation plane.** Freeze fixtures, register protocols, and gate every user-visible model or transform on measured fidelity, restraint, latency, and privacy invariants.

## Aqua Voice parity program

### Scope and target baseline

This program targets a macOS daily-driver replacement for the owner's current
Aqua Voice installation. It does not reproduce Aqua's account, subscription,
cloud orchestration, telemetry, mobile clients, or team administration. Those
are deliberately replaced by local execution, explicit data ownership, and
offline distribution. Parity is claimed only for interaction and final-text
quality on the owner's real dictation distribution.

The target is the installed Aqua desktop 0.20.10, observed locally on
2026-10-05 at settings schema 89. Its live local cache reports deep context on,
the fast-LLM selector on automatic, correction enabled, 782 dictionary values,
65 replacements, and a 12,830-character instruction sheet. Only counts and
settings were inspected for this plan; no instruction, dictionary, replacement,
history text, or audio was copied into the repository. Aqua's rendered controls
still require a live accessibility walk because account gating can hide bundle
features.

The existing frozen 500-clip Aqua slice remains the quality ledger. Its current
Aqua raw baseline is 0.032912 fair WER, 0.055232 strict WER, and 0.674 exact.
That is the number to beat or credibly match; the selected reverse-SALM
adapter at 0.124372 fair WER is not close enough to promote. No single-model
directive waives this gate. The default cascade can remain the safe product
path while the SALM lane catches up.

### Parity matrix

| Capability | Aqua target | Phonon current position | Parity work |
| --- | --- | --- | --- |
| Push to talk | Hold the activation key, then release and insert | Hold Globe or Right Option, release to finalize | Qualify key ownership and insertion against Aqua on the same target apps |
| Hands-free mode | Latch or realtime session with endpointing | Double-tap latch exists; stopping still requires the key | Add a measured silence/endpoint mode without regressing the hold path |
| Streaming preview | Realtime transcript while speaking | Parakeet partials exist and are shown when enabled | Preserve partial state, cancellation, and supersession under long utterances |
| Custom dictionary | Cloud-synced dictionary of names and jargon | Local dictionary UI and CLI exist | Add a read-only, idempotent Aqua import and reconciliation report |
| Exact replacements | Spoken phrase to exact written form | Replacement field exists in the same local dictionary format | Map Aqua replacements without duplicating entries and preserve manual edits |
| Style instructions | A long correction instruction sheet sent with each request | `profile/user.md` and `profile/vocab.md` exist behind a default-off setting | Import to a local reviewed profile, enforce length/versioning, and measure correction restraint |
| Deep context | Screen context contributes to cloud correction | Local OCR ranks only relevant dictionary terms | Preserve OCR as default; vision image conditioning remains a separate consented model gate |
| Edit mode | Select text, speak an edit instruction, replace selection safely | Not implemented | Add selection capture, replacement/undo policy, failure rollback, and edit fixtures |
| History | Search, inspect, delete, and replay source behavior | Local paired corpus, intended text, search, deletion, and stats exist | Add audio playback/rerun affordances only after retention and deletion remain proven |
| Network/status surface | Account, network, server model, and usage reporting | Local doctor, model readiness, and local stats | Expose offline/runtime status without implying cloud health |
| Clipboard sovereignty | Avoid clipboard history where possible | Short output types directly; long output uses clipboard and restores it | Measure and honor an avoid-clipboard-history mode on supported systems |
| Languages | Aqua offers multiple dictation languages | English product path | Language parity is a separate model/protocol gate, not a UI relabel |
| Cloud/mobile account | Login, plans, sync, and mobile clients | Explicit non-target | Local export/backup and pinned runtime artifacts are the replacement |

### Work order

1. **Freeze the owner baseline.** Record the rendered Aqua 0.20.10 controls with a live accessibility walk, then run one matched macOS interaction probe across representative apps. Register the existing frozen slice, term subsets, end-to-end timing fields, and Aqua/Phonon output identifiers. Personal rows stay local; only aggregate metrics and hashes enter evidence.
2. **Make migration reversible.** Add `phonon dictionary import-aqua --dry-run` and a real run that reads only the Aqua settings cache, writes a dated backup, maps dictionary and replacement entries, reports collisions/changes, and never contacts Aqua. Keep the instruction sheet as a versioned local profile asset with explicit review and disable controls.
3. **Close daily interaction gaps.** Implement endpointing as an optional hands-free mode, edit mode with selected-text recovery and undo, and an avoid-clipboard-history insertion mode. Each must retain the existing raw transcript and fail closed when accessibility or selection ownership is uncertain.
4. **Requalify the shipped cascade.** Run current Parakeet plus Gemma and Aqua raw over the complete frozen slice under one scorer, then measure term hit, worse-than-input rate, fair/strict WER, exactness, and p50/p95 keyboard-to-insertion latency. A correction may not be promoted if it creates more harm than it repairs.
5. **Improve the single-model lane from the real gap.** The reverse SALM remains the architectural destination, but next training must target the diagnosed difference between 0.124372 and 0.032912 fair WER: onset/clipping, named entities, punctuation/formatting, correction restraint, and long-utterance state. Repeat adapters only through prompt-, context-, rank-, and kernel-aware protocols.
6. **Gate vision context separately.** The image protocol and consented retention policy must pass before a screenshot reaches a model. Screen OCR remains useful even after vision qualification because it supplies auditable dictionary confirmation.
7. **Qualify daily-driver status.** Require at least five consecutive working days where the owner chooses Phonon for real dictation, all failures are journaled, no Aqua fallback is needed for a blocked capability, and the registered quality/latency gates remain green.
8. **Package the selected runtime.** Pin every model and adapter, package offline weights or a verified first-run fetch, measure cold and warm launch on the target Mac, and retain the default cascade until a single-model candidate wins the same gate.

### Acceptance gates

Functional parity requires the matrix rows marked present to pass their existing
Rust, Swift, and Python checks plus the migration and interaction gates above.
Quality parity requires no regression against the registered Aqua raw baseline
on the full 500-clip slice and a matched live-use sample; no small pilot may be
extrapolated. A single-model promotion additionally requires the audio gate,
restraint gate, latency gate, offline packaging gate, and default-path
regression gate to pass at the same pinned adapter and runtime. Until then,
Phonon can be a usable local alternative or experimental engine, but not a
claimed Aqua replacement.

## Training-corpus capture

### Consent and capture modes

The existing paired corpus is audio-first and text-supervised: a recording may
be retained locally, but no screenshot is retained, and an intended transcript
is optional. Training collection is a separate product mode, never a side effect
of ordinary dictation or screen context. It therefore has two independent
default-off settings:

1. `training_capture_enabled`, which authorizes creation of a training candidate
   for one explicit purpose and consent version.
2. `include_screen_images`, which additionally authorizes retention of the
   already-captured screen image. When it is false, the OCR text and confirmed
   dictionary terms may still be used according to ordinary screen-context
   policy, but the image is deleted at the existing capture boundary.

Both must be true before `screenshot.png` can exist in a corpus item. The
capture record identifies the session, purpose, consent version, consent time,
capture origin, display ID, pixel dimensions, backing scale, capture timestamp,
retention deadline, SHA-256, and model capability declaration. Consent is
scoped and reviewable; it does not transfer to a new purpose, model family,
endpoint, or training run.

Retention is finite from capture time. `screenshot_retention_seconds` must be
set before image capture, and expiry removes only the expired screenshot. It
does not silently delete audio, text, metadata, or the corpus item. Deletion
failure is an error and marks the candidate unusable for export; it is never
treated as successful deletion. Explicit corpus deletion removes the entire
item, including screenshot, audio, metadata, sidecars, and export links, but
not independent copies the user explicitly exported.

### Candidate format and integrity

A training candidate remains local under the existing corpus root:

```text
Corpus/<id>/
├── audio.wav
├── metadata.json
├── screenshot.png
└── screenshot.json
```

`screenshot.json` is the provenance manifest, not a replacement for
`metadata.json`. It records schema version, corpus ID, display/capture fields,
consent version and time, retention deadline, file name, SHA-256, byte length,
and deletion state. Every load hashes the image again; a mismatch invalidates
the candidate. The audio receives the same SHA-256 discipline at export.
Raw and final transcripts remain inference observations. An intended transcript
or an explicit reviewed label is required before a row can enter a training
manifest. Manual edits record who/what reviewed the label without storing the
reviewer's personal identity when no review was human.

Export is deliberately not training. `phonon corpus export` writes outside the
personal corpus directory and creates a manifest of IDs, hashes, durations,
sources, timestamps, consent versions/provenance, and requested modalities. It
omits transcript text unless `--include-text` is explicit; even then it
includes only reviewed intended text, not raw ASR. Screen images are copied
only with `--include-screen-images` and `--only-consented`; a missing consent
record, expired image, hash mismatch, or unresolved scope fails the export.
No exporter contacts a model provider, cloud service, telemetry provider, or
public artifact destination.

### Work packages and gates

1. Add the settings, schema-defaulted metadata, atomic capture registration,
   expiry, deletion, and load-time hash validation, with no change to default
   capture behavior.
2. Extend the Swift stop path to retain the already-captured image only under
   the two flags, write its provenance record, and enforce expiry without
   blocking ordinary dictation on a storage error.
3. Add auditable CLI inspection and consent review verbs, followed by the
   redacted exporter above.
4. Add a pack builder only after export integrity passes. It must preserve the
   active prompt ID/hash, context length, LoRA rank, modality order, and pack
   schema, and must never select rows by reading the frozen evaluation slice.
5. Register an end-to-end privacy gate: consent provenance, expiry, deletion,
   export redaction, hash mismatch, malformed manifest, and refusal to export
   raw ASR or unlabeled text.

The capture path is implemented only when default-off behavior, explicit
consent, finite retention, provenance, integrity, and deletion are covered by
tests. It is product-ready only after a real macOS capture, a retention expiry,
an explicit export, a malformed-image rejection, and a full corpus deletion are
recorded in local evidence without exposing personal contents. The existing
vision-head image protocol remains a separate inference gate; a training
candidate never authorizes model calls by itself.

## Dataset creation and teacher curation

### Current output flow

The Aqua export starts with two text observations for every clip:
`raw_text`, the latency-constrained real-time transcript, and `normalized_text`,
the accepted or normalized text. `ingest_aqua` converts audio to 16 kHz mono,
hashes it, and stores `verbatim_text` from the raw transcript,
`insert_text`/`normalized_text` from the accepted text, and a
`teacher_transcripts` map initially containing only `aqua_raw` and
`aqua_normalized`. The label status is `model_assisted`; it is not represented
as human-audited truth.

There are three uses of text after ingestion:

1. **Evaluation.** A hypothesis file carries `ref`, `raw_aqua`, and `hyp`; the
   registered fair/strict scorer compares `hyp` to `ref` over the complete
   frozen slice.
2. **Training.** `build_aqua_prompt_pack.py` currently takes
   `row["corrected"]` directly as the assistant target. It verifies audio
   presence, duration, manifest hash, prompt hash, and context length, but it
   does not read `teacher_transcripts`. This is the principal curation gap.
3. **Review.** The general Phonon teacher lane can append NeMo or Whisper
   transcripts to the canonical dataset and rank disagreement for review, but
   that lane has not been connected to the final reverse-graft pack builder.

### Dedicated offline ASR teachers

Two non-real-time ASR teachers were run over the same frozen 500-clip slice on
B550 after the Optuna study completed. Neither was given reference text. Both
output a pinned model/revision, per-row generation time, Aqua raw text, and the
hypothesis used for scoring.

| Teacher | Fair WER | Strict WER | Exact | Rows | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aqua real-time raw | 0.03291183093712108 | 0.055232029117379434 | 0.674 | 500 | existing slice |
| Qwen3-ASR 1.7B `bcd2b5b7…` | 0.07067382643339684 | 0.11847133757961784 | 0.398 | 500 | 418 s |
| Cohere Transcribe 03-2026 `b1eacc26…` | 0.05768231422137537 | 0.09526842584167425 | 0.454 | 500 | 108 s |
| Phonon Voxtral Small MXFP4 audio256 `f979da7e…` | 0.06071366707084705 | 0.09181073703366698 | 0.436 | 500 | 360.399 s |

Cohere is the stronger independent offline teacher and is much faster, but both
dedicated teachers remain behind Aqua raw. The agreement analysis is more
important than either aggregate: normalized Qwen and Cohere agree on 264 rows,
including 46 where they differ from Aqua raw, but they both beat Aqua raw on
only four rows and at least one beats Aqua raw on only 22 of 500. A naive
majority-or-Aqua vote scores 0.04524 mean row WER, worse than Aqua raw's 0.03513.
The per-row oracle over Aqua raw, Qwen, and Cohere is only 0.03065. Therefore
offline ASR is useful as selective evidence, not as a replacement ground truth
and not as an unconditional ensemble vote.
The Phonon-owned Voxtral result does not change that boundary. It wins on 11
rows, ties Aqua raw on 305, loses on 184, and has only five Voxtral-only exact
rows versus 124 Aqua-only exact rows. Its mean row WER against accepted text is
0.07701 versus Aqua raw's 0.03513. It is therefore weaker than Cohere in exact
recall and cannot replace Aqua as dataset truth under the current quantization
and prompt. A BF16 Voxtral reference remains the decisive diagnostic for
whether this gap is model/domain mismatch or MXFP4 calibration damage.

### Market-survey boundary

The first two offline teacher runs were feasibility probes, not a claim that
1.7–3B models represent the market ceiling. Two primary leaderboards were
checked on 2026-10-03.

Artificial Analysis AA-WER v2 non-streaming weights three real-world datasets:
AA-AgentTalk at 50%, VoxPopuli-Cleaned-AA at 25%, and Earnings22-Cleaned-AA at
25%. Its current accuracy leaders are:

| Rank | Model | Availability | AA-WER |
| ---: | --- | --- | ---: |
| 1 | Alibaba Fun-Realtime-ASR-preview / Fun-ASR-Flash | Preview; no established public price or local credential in this environment | 1.7% |
| 2 | ElevenLabs Scribe v2 | Hosted API | 2.2% |
| 3 | Microsoft MAI-Transcribe-1.5 | Azure API | 2.4% |
| 4 | Smallest AI Pulse Pro | Hosted API | 2.4% |
| 5 | Google Gemini 3.5 Transcribe | Hosted API | 2.6% |

AA-WER currently ranks Mistral Voxtral Small 24B as the strongest open-weights
teacher at 2.8%, followed by Thinking Machines Inkling at 3.5% and Voxtral Mini
Transcribe 2 at 3.6%. By contrast, the Hugging Face Open ASR Leaderboard's
latest 02-10-2026 English short-form table uses ten public/private datasets and
reports Zoom Scribe v2 Pro first overall at 3.5925%, Azure Speech 07-2026 second
at 3.8112%, and ElevenLabs Scribe v2 fourth at 3.9688%. Its strongest open
checkpoint is Qwen3-ASR-1.7B-hf at 4.3113%, with Hojo-ASR-V1 at 4.3338%,
Higgs Audio v3 STT at 4.3925%, Canary-Qwen-2.5B at 4.4275%, and Voxtral Small
24B farther down at 4.9938%.

These rankings do not contradict one another; they answer different questions.
AA-WER gives more weight to agent-style speech, while the HF board macro-averages
ten English datasets. Neither replaces the frozen Phonon slice. The next teacher
selection must therefore be gated on access and on measured Phonon-slice WER:
first a hosted ElevenLabs Scribe v2 synthetic probe, then MAI-Transcribe-1.5 if
Azure access is configured, and in parallel an open-weights Voxtral Small 24B
run on a rented or multi-GPU host. The Alibaba Fun preview is the nominal
accuracy leader but is not yet a practical candidate without a usable account
endpoint and price.

### Voxtral Small on the 3090 Ti pair

MXFP4 makes memory sense for Voxtral Small 24B, but the 3090 Ti execution mode
must be stated precisely. The RTX 3090 Ti is SM 8.6 and has no native FP4 tensor
path. Current vLLM compressed-tensors MXFP4 supports Ampere as **W4A16
weight-only through Marlin**; true W4A4 is reserved for newer hardware. That is
still useful for a batch teacher: roughly 24 billion MXFP4 weights plus E8M0
group scales should be around 13-15 GiB before runtime and KV cache, allowing a
single 24 GB GPU to hold the language model while the second GPU remains free.

No public Voxtral Small MXFP4 checkpoint was found in the 2026-10-03 Hub survey.
The nearest existing compressed checkpoint is
`ghecko78/Voxtral-Small-24B-2507-W4A16` at revision
`8ed9721e8cae74f5ed21dfd504fb78008e0b8ce1`. It is 14.46 GiB, uses INT4
GPTQ W4A16 rather than MXFP4, was calibrated on 256 English C4 text samples,
and keeps the audio tower, multimodal projector, and LM head unquantized. Its
quality is therefore not evidence for or against an audio-calibrated MXFP4
checkpoint.

A live vLLM 0.30.0 probe on B550 resolved the checkpoint to compressed-tensors
W4A16, selected `MarlinLinearKernel`, selected FlashAttention v2, and allocated
14,432 MiB on one RTX 3090 Ti. Two configuration defects were found before
weights loaded: Voxtral requires `--tokenizer-mode mistral`, and the third-party
HF config omitted audio fields that current vLLM reads directly. The pinned
snapshot was repaired with the official Mistral values: `downsample_factor: 4`,
`d_model: 1280`, `sampling_rate: 16000`, `hop_length: 160`, and
`window_size: 400`. Weight download was still in progress when this receipt was
written, so transcription quality and latency remain unmeasured.

A Phonon-specific quantization must calibrate on actual audio-text pairs from
the training/public corpus, never the frozen evaluation slice, and must keep the
audio tower and projector unquantized. Its acceptance gate is not "it fits";
it must remain close to the BF16 Voxtral Small reference on the frozen 500-row
slice, preserve technical terms, and beat or supplement Aqua raw before any
curated target is allowed into training.

### API correction contract

The approved GLM-5.3-FlashX curation request contains text only. Depending on
the arm it may include Aqua raw, named offline ASR hypotheses, and the accepted
historical text; the independent arm omits the accepted text. It never contains
audio bytes, audio paths, timestamps, session IDs, machine identifiers, or
screen content. The reply must remain a strict `keep`/`revise`/`discard` JSON
decision with corrected text, reason code, confidence, model, and elapsed time.

A targeted diagnostic input selects the 22 rows where Qwen or Cohere beat Aqua
raw, disagreement controls where the two offline teachers agree but differ from
Aqua, and exact controls. This selection intentionally concentrates the small
possible gain; it is not a full-slice quality estimate. One input row has an
empty Aqua raw field and is correctly skipped by the curator, leaving 49 calls
in each API arm.

| API arm | Decisions | Mean row WER | Exact | Versus baseline | Mean latency |
| --- | --- | ---: | ---: | --- | ---: |
| accepted text included | 40 keep, 9 revise | 0.01587 | 0.857 | 0 wins, 42 ties, 7 losses versus accepted | 6.65 s |
| accepted text omitted | 20 keep, 29 revise | 0.08295 | 0.347 | 12 wins, 26 ties, 11 losses versus Aqua raw; raw mean 0.07519 and exact 0.408 | 5.35 s |

The first arm cannot beat an already-accepted label by construction and mostly
copies it; its seven revisions are losses. The independent second arm is the
meaningful test: it occasionally recovers terminology, but net WER and exactness
are worse than Aqua raw on this deliberately favorable selection. The full
500-row API pass is therefore not justified. The accepted-input output SHA-256
is `89c663938129f47b457888e9665a6c80573c8ec64952b0eade952bbf373b7a5f`; the
independent output SHA-256 is
`b7419a9b3ebafc43ceb70c8acf692c567d658ab4506e53933c5457b4f9263331`.

### Curation gates

1. A teacher transcript is evidence with model, revision, prompt/language mode,
   and runtime metadata. It is never silently promoted to `insert_text`.
2. API output is a candidate label. It must preserve the original ID and retain
   all source hypotheses for audit.
3. A replacement training target must improve fair WER and exactness on the
   full registered slice, not only a selected pilot.
4. A curation pass must report wins, losses, reversions, empty outputs,
   confidence calibration, and latency. Conservative behavior that merely keeps
   accepted text is not an improvement.
5. The pack builder must gain an explicit versioned target-field choice before
   curated labels can enter training. Its pack metadata must hash the curation
   sidecar and preserve the existing prompt/context/rank contracts.
6. A failed targeted pilot must stop scale-up. Repeating the same prompt on all
   500 rows would consume API time without testing a new hypothesis.

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

## Runtime architecture

### Process topology

Phonon on macOS is four cooperating pieces, and the boundaries between them are
the architecture:

```text
Phonon.app (Swift/AppKit)        bar/Sources/*.swift
  ├─ AppStore, settings, retention, backup mirror
  ├─ MicRecorder: CoreAudio tap, streaming chunks, 16 kHz WAV
  └─ ScreenContextCapture: ScreenCaptureKit stills, Vision OCR
        │  JSONL over stdio (line-delimited JSON, both directions)
        ▼
phonon engine (Rust)             crates/phonon-cli/src/main.rs (Commands::Engine)
  ├─ speech gate, corpus metadata, dictionary, correction prompt
  └─ two sidecars, one per weight stream, each a separate uv process
        ├─ sidecar/asr_server.py     parakeet-mlx, 16 kHz, batch and streaming
        └─ sidecar/polish_server.py  mlx-lm, prefix cache, MTP drafter
```

The bar never loads a model. It speaks the same line protocol the CLI exposes,
so `phonon engine` is the only owner of engine state, and the benchmark path
consumes the same object the app does. The workspace is nine Rust crates:
`phonon-core` owns the engine and data model, `phonon-asr` and `phonon-llm` own
one sidecar each, `phonon-audio` owns the speech gate, `phonon-hotkey` owns the
hold/tap latch, `phonon-cli` is the only binary entry point, `phonon-mine` and
`phonon-profile` are developer tools, and `phonon-win` is the Windows port.

### Protocol contract

One JSON object per line, on stdin and stdout. Requests are
`status`, `transcribe`, `polish`, `reload_dictionary`, `shutdown`, plus the
Parakeet-only `stream_start`, `stream_chunk`, and `stream_stop`. Events are
`stream` for per-stack loading progress, a single `ready` when every required
stream is warm, `result` for a transcript or correction, and `error`.

Two rules keep the protocol safe to extend. Every result carries the request
`id` so a superseded pass can be dropped rather than inserted, and a sidecar
that cannot serve a request answers with an explicit `error` rather than
silently degrading. That is exactly how the SALM sidecar refuses streaming
partials, and it is the precedent the multimodal image field must follow.

The sidecar child is killed on drop, and its last twelve stderr lines are
retained so a sidecar that dies before `ready` reports why instead of hanging
the loader at whatever percentage it reached.

### Lifecycle and readiness

`Engine::start` spawns the ASR and correction sidecars in parallel and then
refuses to report ready until a startup gate passes on all four axes. Parakeet
must transcribe `assets/startup.wav` in batch mode, must transcribe the same
fixture through the streaming path, the correction model must answer a text
prime, and the correction model must round-trip the actual ASR output of that
fixture. A transcript mismatch on any axis is an error, not a warning. The
correction prompt is shipped inside the executable, not read from disk at
runtime, so a first launch does not depend on a loose file.

A first dictation frequently lands before the weights are warm. The bar keeps
the capture, queues the transcribe until `ready`, and shows a loading state
rather than dropping the utterance. This is why the app has a loading
presentation at all.

### Pinned runtime and offline policy

| Piece | Pin |
| --- | --- |
| ASR model | `mlx-community/parakeet-tdt-0.6b-v2` at `8ae155301e23d820d82aa60d24817c900e69e487` |
| ASR runtime | `parakeet-mlx==0.5.2` |
| Correction model | `mlx-community/gemma-4-e2b-it-4bit` at `238767527555cb75a05732a84dff5d6ba0dd6809` |
| Correction runtime | `mlx-lm==0.31.3` |
| Python | exactly `3.12`, not an open range |

`phonon doctor` reports whether each runtime is already in uv's cache, because
that cache is what makes a second launch work with no network at all. An
unprimed install is not a broken install, and SoX is reported as optional since
the native app records through CoreAudio and only the terminal path needs it.

### Local data, retention, and the backup mirror

All user state is one directory, `~/Library/Application Support/Phonon`:
`settings.json`, `dictionary.json`, and `Corpus/<id>/` with `audio.wav` and
`metadata.json`. A recording's metadata carries the schema version, source,
microphone name, duration, the speech-gate verdict, the raw transcript, the
final corrected text, an optional intended transcript, every applied
dictionary correction, the screen-confirmed terms, and the correction model's
latency, time to first token, and tokens per second.

Retention is explicit and opt-in by schema, not by documentation. Schema 2 made
local history and screen context default off, and an existing schema 1 install
keeps whatever its owner had already been shown. The default window keeps
recordings until the owner deletes them; 7, 30, and 90-day windows prune on
change. Deletion is always to the Trash, never an unlink, so a mistaken clear is
recoverable by the person who made it.

Because uninstallers match on the app name and then sweep `~/Library`, a small
set of irreplaceable text files is mirrored to `~/.phonon/backup`:
`dictionary.json`, `settings.json`, legacy `History.json`, `Vocabulary.txt`,
and `WordReplacements.json`. The new paired `Corpus/` is deliberately not
mirrored because it contains audio and grows without bound; Settings' explicit
export copies it in full. The mirror counts corpus entries in its manifest but
does not copy them, so restore cannot reconstruct a modern recording. Model
weights are also excluded: they live in the Hugging Face cache, are named after
their models rather than this app, and no uninstaller should be able to remove
them. A capture that would replace a non-empty mirror with an empty store is
refused, restore never overwrites an existing file, and restoration is offered
only when the live store is empty.

### CLI surface

`phonon` with no arguments launches the native app. The rest are one verb each,
and every mutating verb is a file operation the user can undo or inspect:
`bar`, `engine`, `bench`, `doctor`, `profile` with `kernel`, `model`, and `e2e`
subcommands, `dictionary` with `list`, `add`, `learn`, `import-wispr`,
`import-txt`, `test`, and `evaluate`, `corpus` with `path`, `list`, `show`,
`set-intended`, `migrate-legacy`, `delete`, and `polish-eval`, and `stats`.
The dictionary and corpus verbs are what make the training set auditable from
a terminal, so they are product surface, not developer tooling.

### Distribution

macOS ships as a signed app bundle with the Rust binary and the pinned weights
resolved through uv, and the Windows port is a self-contained executable with
its own manifest. Both Windows weight streams differ from macOS because the
runtime is not Metal: speech uses
`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` at
`1ab9323565ddb038682214b292f588070a538ce2` through sherpa-onnx `1.13.6` rather
than the MLX Parakeet build, and correction uses the GGUF QAT build of the same
correction model, `google/gemma-4-E2B-it-qat-q4_0-gguf` at
`675cff42a74c774d6cb76f76d8eacb49b48c9b93`, through a pinned llama.cpp build
`b10726`. Every manifest URL is pinned and every asset hash is a SHA-256, and a
fetch verifies the hash before the file is used.

The first-run Windows download is roughly four gigabytes. That is a product
decision with a measured cost, not an implementation detail, and the same
correction model is quantized differently per platform rather than shipped
identically.

### Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Startup gate cost | Four axes, so a heavy first launch is paid before readiness | Measured time-to-first-dictation on target hardware against a lighter gate |
| Deletion semantics | Trash, not unlink, for recordings and clear-all | A decision to support secure deletion would need its own privacy review |
| Backup mirror scope | Small text files and legacy history only; the paired Corpus and weights are excluded | Include corpus metadata without audio, or keep corpus recovery dependent on explicit export, after measuring size and privacy impact |
| Windows parity | Different runtimes on both streams: sherpa-onnx int8 Parakeet and GGUF q4 Gemma instead of MLX | A measured quality comparison between the macOS and Windows stacks on one registered protocol |
| Streaming ownership | Parakeet-only today; the SALM sidecar refuses streaming | Either a streaming SALM path with partial-state contract, or a documented product decision that single-stage engines are whole-utterance only |

## Work-log

### 2026-10-03: Phonon-owned Voxtral MXFP4 W4A16 harness

The third-party INT4 checkpoint is closed as an input. Phonon's teacher path now
builds from `mistralai/Voxtral-Small-24B-2507` at revision
`da5b42409f279fdd92febee0511a6c32828569c1` and quantizes it with LLM
Compressor GPTQ, `scheme="MXFP4A16"`, `targets="Linear"`, and compressed-tensors
packing. `lm_head`, every `audio_tower` linear, and every
`multi_modal_projector` linear are ignored. Calibration is selected only from
training/public audio-text rows, de-duplicates audio IDs, excludes every ID in
`slice-eval-500.jsonl`, hashes both audio and accepted text, and never copies
accepted text into the calibration receipt. The first full recipe is 256 stable
selected rows of 2-20 second clips; a two-row smoke is allowed before that.

Added `ml/research/asr_teachers/quantize_voxtral_mxfp4.py`. It downloads only
the eleven official sharded BF16 safetensors files and required config assets;
`consolidated.safetensors` and any `original/` tree are blocked. Calibration
uses the actual `VoxtralProcessor.apply_transcription_request(..., language="en")`
path. Token IDs are cached in Arrow while the batch-size-one collator regenerates
`input_features` live and checks that the token IDs did not drift. The official
audio values (`downsample_factor: 4`, `d_model: 1280`, `sampling_rate: 16000`,
`hop_length: 160`, `window_size: 400`) are baked into both the model and output
config. Completion requires a receipt with base/model/runtime/calibration
hashes, a config audit, uint8 packed/scale tensors, proof that ignored components
remain unquantized, and SHA-256 for every output file.

Local verification passed: `uvx ruff check
ml/research/asr_teachers`; `uv run --python 3.12 python -m py_compile
ml/research/asr_teachers/quantize_voxtral_mxfp4.py`; nineteen tests in
`ml/research/asr_teachers/tests` under `python -m pytest . -q -p
no:cacheprovider --noconftest`; and the CLI dry-run selected one synthetic
training row, reported MXFP4 W4A16 and all ignored layers, and reported eleven
expected shards with the duplicate file blocked. The official revision, eleven
shards, duplicate single-file weight, config, and nested audio parameters were
also verified live through the authenticated HF CLI/API. No 24B model was loaded
and no quality is claimed.

B550 was not reachable for takeover: both the configured `B550` SSH route and a
direct `kearm@192.168.1.171` attempt timed out, and ICMP received no reply. No
process, GPU, cache, or hardware conclusion can be drawn from that result. The
next two-row run must use a fresh output directory and write its receipt under
`/home/kearm/salm-lora/build/asr-teachers/`; only after serialization, vLLM
Marlin loading, and two real training-clip transcriptions should the 256-row
recipe run.

The route recovered long enough to verify live B550 state and seed the run. Both
RTX 3090 Ti GPUs were idle (1 MiB and 33 MiB used, zero percent utilization)
with no compute processes. `/home/kearm/salm-lora/hq-train-manifest.jsonl`
contains 12,855 training rows, the frozen slice contains 500 rows, and the first
manifest audio resolves under `/home/kearm/aqua-training-data`. The successful
synthetic serialization artifact is present and its config is exactly
compressed-tensors `mxfp4-pack-quantized`, four-bit float weights, group size 32,
and `torch.uint8` scales. A clean `/home/kearm/phonon` checkout was cloned from
a Git bundle at pushed revision `5b0d3ccddb0c0351d6c723ea22bb7956a0f25b16`;
no credentials or authoritative Git state was copied. The network then dropped
again through five bounded SSH retries, so no base download or GPTQ process has
been started and no run is live.

The route later stabilized enough to finish source-side setup. A dedicated
Python 3.12 environment on B550 now contains LLM Compressor 0.14.0,
compressed-tensors 0.19.0, Transformers 5.17.0, Torch 2.14.0 with CUDA 13.0,
Datasets 5.0.1, Accelerate 1.15.0, Mistral Common 1.12.0, librosa 0.11.0, and
safetensors 0.8.0; both SM 8.6 devices are visible. The real-manifest dry-run
selected two rows from 12,855 eligible training rows. An official processor
preflight on those two clips produced 383 input IDs and attention values per
30-second-padded clip and a `[1, 128, 3000]` BF16 `input_features` tensor, with
the cached and live token IDs in agreement.

The first two `huggingface_hub`/Xet download attempts were preserved as failed
receipts after stalls and low throughput. The live transfer now uses canonical
`hfd` 0.1.1 with aria2, sixteen connections per file, four concurrent jobs, and
full verification. Its dry-run resolved the pinned revision, selected exactly 18
files, excluded the duplicate consolidated weight, reported 48,542,538,856 bytes,
and recorded manifest SHA-256
`23661eb5a89f3f6b3b42306e38e3722d12ccfae4e851fa0556f930a839bf1ebd`. At 09:04
PDT it was transferring the first four shards at about 10 MiB/s. A tmux waiter
will launch the two-row GPTQ runner only if `hfd` exits zero; no model weights
have been loaded and no quantization quality is claimed yet.

The deployment-path audit corrected two further traps before the 24B load. LLM
Compressor 0.14's argument default is `independent`, so the harness now requests
`pipeline="sequential"` explicitly, keeps BF16 weights on CPU, onloads subgraphs
to GPU 0, and offloads intermediate activations to GPU 1. The official HF index
uses `audio_tower` and `multi_modal_projector`, while vLLM 0.30's default loader
expects the third-party layout's `model.layers` language names plus Mistral
audio names. A range request read the official `consolidated.safetensors` header
without downloading its 48.5 GB body and confirmed all 488 audio/projector
names. `convert_voxtral_mxfp4_vllm.py` now drops the fixed HF sinusoidal
position table, maps all 488 audio/projector tensors exactly, removes the
`language_model.` prefix from language tensors, and adds both `lm_head` and
`output` ignore aliases. The mapping is a separate audited variant; it does not
rewrite the canonical HF artifact.

A second B550 environment contains vLLM 0.30.0 with CUDA Torch 2.13.0 and
Transformers 5.18.0. The verifier starts that server on one 3090 Ti with
`--tokenizer-mode mistral`, checks health, sends exactly the two selected clips
to `/v1/audio/transcriptions`, rejects empty JSON, stops the server by PID, and
hashes the local response files. It never prints or uploads transcript text. At
09:43 PDT the canonical base transfer had reached 32 GiB, shards 5-8 were
73-78% complete, and all three tmux waiters remained healthy.

The canonical `hfd` transfer completed at 10:05 PDT with all 18 selected files
fully verified, the pinned revision, and the expected 48,542,538,856 bytes. A
stale background fragment from the earlier malformed nested waiter launched one
old `smoke2` command after the download; its failed receipt and markers are
preserved. The clean durable runner then launched `voxtral-mxfp4-smoke4`.
Sequential GPTQ used CPU-resident BF16 base weights, GPU 0 for subgraphs, and
GPU 1 for activation offload. It completed in 376.419 seconds with two training
clips, 280 packed uint8 tensors, 280 uint8 E8M0 scale tensors, no quantized
ignored component, and the official audio fields. The canonical HF artifact is
`voxtral-mxfp4-smoke4/model.safetensors` (15 GiB), SHA-256
`17791a6adb4b341be56bf00f8fd0439e704275a8307b6d756d80de482ca684a1`; its config
SHA-256 is `573fb8ad0e49459e47f575d579dc55f81d8906e90fd44e2cafc15a06d1f6f174`.

The audited vLLM name variant is separate. It contains 1,131 tensors, maps all
486 non-position audio/projector tensors to Mistral names, leaves zero audio
tensors quantized, and completes conversion in 22.288 seconds. Its model and
config SHA-256 values are
`87f83e65d03b9c25ba5bbae3265bba5881e7fee609dcd94f7dfb7d9633533ac8` and
`819bdaeb3f2c7d44d5263c35c51170064bbea275d1a7317cd87fa1c712ffc9ba`. Three
startup repairs are preserved as receipts: `soundfile==0.14.0` was missing,
the official unset `global_log_mel_max` had to be serialized as null, and
FlashInfer sampling had to be disabled because its JIT invoked an nvcc option
unsupported by B550. The final server run selected
`MarlinMxFp4LinearKernel` and FlashAttention v2, loaded the model in 14.89 GiB,
allocated 5.48 GiB KV cache (35,904 tokens), accepted both real clips, and
stopped cleanly. Response SHA-256 values are
`48422210e0b7e79e0810c73a6476708b9a79e74040cc1af97bd0eed79a07fb93` and
`4940f1a6660dba2f7a06d34e1e24aef6c7061a8305d64960976c971fcdcbdd4e`; their
local-only sizes are 215 and 178 bytes. A two-row sanity score against accepted
text was 0.10909 fair WER / 0.21569 strict WER / 0 exact. This is explicitly
not a full-slice quality estimate.

The requested 256-row run `voxtral-mxfp4-audio256-v1` completed from the same
fully verified base in 1,117.66 seconds. It selected 256 training rows, used
the same sequential CPU/GPU plan, and produced the same 280 packed and 280
scale tensor audit with no protected component quantized. Its model and config
SHA-256 values are
`f979da7e5d571f28317a872839f4c161b078307b3f6534573d3510c30c551fe3` and
`e72506870b44892ea68f36decdb178c07bf0c4b1eda62bfba0436aef5c3f0ba5`. GPTQ
Hessian inversion failed for one of 280 modules,
`model.language_model.layers.2.mlp.down_proj`, and that module fell back to
round-to-nearest while retaining MXFP4 packing. The chained vLLM run again
selected `MarlinMxFp4LinearKernel` and FlashAttention v2, loaded 14.89 GiB,
allocated 5.48 GiB KV cache, transcribed two bounded clips, and stopped
cleanly. Its conversion receipt SHA-256 is
`f6019950656563f8c3c9cbb8e17326f2ef643fa9ddd9cb35dda5b28ba894e86e`.

The full frozen-slice evaluation then completed all 500 rows in 360.399
seconds with mean request generation time 0.720568 seconds and maximum 2.23
seconds. It scored 0.06071366707084705 fair WER,
0.09181073703366698 strict WER, and 0.436 exact. The hypothesis, score, and
server-log SHA-256 values are
`bc999fdfb90bb2526c18ed2766f96fdf0aa0f0b7d21055e30be07f088e952b8f`,
`0b57993a39ed629874bffa01737f5551c14ee13198095c29055b2e3da881acb2`, and
`657b743c1daf72b8da2c940f8b97c506cc24d314647f2023cb0c51444a39ad32`. Against
Aqua raw per-row fair WER, Voxtral wins 11 rows, ties 305, and loses 184; it
has five Voxtral-only exact rows versus 124 Aqua-only exact rows. This is a
complete registered quality result, not a pilot, but it does not justify
replacing or unconditionally ensemble-voting against Aqua raw. No BF16
Voxtral reference or technical-term damage audit has yet been run, so the
remaining decision is whether the gap comes from model/domain mismatch or
MXFP4/GPTQ damage.

### 2026-10-02: offline ASR teachers and independent GLM reconciliation

Added pinned, resumeable ASR teacher harnesses for Qwen3-ASR 1.7B and Cohere
Transcribe 03-2026. Both completed all 500 rows on B550 without reference text.
Qwen scored 0.07067382643339684 fair WER / 0.11847133757961784 strict /
0.398 exact in 418 seconds; its hypothesis SHA-256 is
`c916da3da92be9f2604df1bb3571b5a3de789d9888aeddc190bed9faf854d503`. Cohere
scored 0.05768231422137537 fair WER / 0.09526842584167425 strict / 0.454 exact
in 108 seconds; its hypothesis SHA-256 is
`9aa6dc5ae558c2fd81669394cadc7cbf1987747444bbe99c3877d460d8ef3f52`. Aqua raw
remains substantially better at 0.03291183093712108 fair WER and 0.674 exact.

The API curator now accepts named ASR hypothesis files, XML-escapes them, and
supports a no-accepted-final mode that sends only Aqua raw plus named offline
hypotheses. A synthetic FlashX request verified the local route and prompt
contract. On the targeted 49-row real-data diagnostic, including accepted text
produced no wins, 42 ties, and seven losses. Omitting accepted text produced an
independent candidate with 12 wins, 26 ties, and 11 losses against Aqua raw,
but worse mean WER (0.08295 versus 0.07519) and worse exactness (0.347 versus
0.408). This negative result stops a full 500-row API pass under the current
prompt; offline ASR remains evidence for selective review, not a replacement
training truth.

### 2026-10-02: runtime architecture audit

Audited the shipped runtime from source rather than from the README and added a
Runtime architecture section covering the four-process topology (Swift app, Rust
engine, two uv sidecars), the line protocol and its error discipline, the
four-axis startup readiness gate, the pinned macOS and Windows artifacts, the
opt-in retention model, the CLI surface, and distribution.

Three claims were corrected against the code while writing it. The README and
the first draft of the section implied the backup mirror covers the paired
corpus; `PhononDataMirror.mirroredFiles` mirrors only dictionary, settings,
legacy history, vocabulary, and word replacements, counts corpus entries in its
manifest, and leaves recovery of modern recordings to the explicit export. The
Windows port was described as a correction-only difference; `phonon-win` also
swaps Parakeet to a sherpa-onnx int8 build. And screen context and local history
are opt-in by settings schema 2, defaulting off on a fresh install, while the
Rust-side `SettingsFile` still defaults them to true because the native app owns
that file and the engine must not rewrite it.

### 2026-10-02: Optuna first-stage receipts and rank-aware recovery

The first real eight-trial request ran on B550 with the fixed
`xml_dictation_v1` prompt, 10,000 public YODAS-Granary rows, 10,000 public
steps, 1,000 Aqua adaptation steps, and all 500 frozen evaluation rows. The
persistent study is `final-reverse-vl-xml-v1` in
`/home/kearm/salm-lora/build/optuna/final-reverse-vl-xml-v1.db`. Its prompt
SHA-256 is
`a805969f94dd4075a6f8abfae95ba0117a824a5ac8f9321a74f6d59ac514bf88`.

| Optuna trial | State | Fair WER | Strict WER | Exact | Key parameters |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | complete | 0.12974190195738783 | 0.16833484986351227 | 0.262 | context 1024, rank 16, LR 4.7537e-5, adapter LR 1.2466e-4, warmup 25 |
| 1 | complete | 0.1454183266932271 | 0.18061874431301184 | 0.274 | context 768, rank 16, LR 1.2482e-4, adapter LR 1.4816e-4, warmup 175 |
| 2 | complete | 0.13563138749350426 | 0.17343039126478618 | 0.276 | context 512, rank 16, LR 1.4443e-4, adapter LR 6.9040e-5, warmup 50 |
| 3 | failed evaluation | not scored by first run | not scored by first run | not scored by first run | context 512, rank 32, LR 6.5057e-5, adapter LR 4.9434e-5, warmup 50 |
| 4 | recovered completion of trial 3; best | 0.12783648016629134 | 0.16642402183803456 | 0.294 | identical to trial 3 |
| 5 | complete | 0.15156764247358392 | 0.18817106460418562 | 0.280 | context 1024, rank 32, LR 3.1521e-5, adapter LR 1.4771e-4, warmup 200 |
| 6 | complete | 0.1335527455395808 | 0.17033666969972702 | 0.294 | context 512, rank 32, LR 6.5806e-5, adapter LR 4.2540e-5, warmup 200 |
| 7 | complete | 0.19539234366880304 | 0.2310282073413285 | 0.246 | context 512, rank 8, LR 3.0438e-5, adapter LR 3.3352e-5, warmup 150 |
| 8 | complete | 0.136497488307639 | 0.17370336689699726 | 0.282 | context 768, rank 8, LR 6.1324e-5, adapter LR 4.7486e-5, warmup 150 |

Trial 3 completed both training phases, but its first evaluation reconstructed
the model at LoRA rank 16 even though the adapter was trained at rank 32. The
failure is recorded in `trial-0003/logs/evaluate-failed-rank16.log`; the final
adapter was intact. Commit `d7b4db3` passes the trial rank explicitly and can
infer rank from adapter A-matrix shapes when omitted. The recovered evaluation
completed all 500 rows with rank 32. Its hypothesis file is
`trial-0003/hyps.jsonl`, SHA-256
`cf738a1260c3e9ac89abc2a81a34835a4f3a577bfcd27c679988bce775f318d1`; its score
file is `trial-0003/score.json`, SHA-256
`e108e64b282f318c53d15ba0ce879bd6ab89515f9650bd3342992f39c99778d7`. Trial 4
is a separate Optuna completion with `recovered_from_trial=3`; failed trial 3
was not rewritten or deleted.

The B550 machine audit found no failed systemd unit, kernel XID/NVRM message,
OOM kill, storage/filesystem error, hardware memory error, thermal fault, GPU
remapped row, or pending GPU repair. The visible `rxe0 qp not ready to send`
messages occurred while the persistent Soft-RoCE link on `enp7s0` transitioned
to link-up during the public-data phase; current RDMA counters showed
`send_err=0`. Hugging Face also emitted transient DNS and IPv6 unreachable
retries before the 10,000-row pack download recovered. These were not CUDA or
filesystem failures.

The study resumed four additional trials through `run_final_optuna_resume.sh`.
Trials 5 through 8 all completed, and `OPTUNA-RESUME-DONE` appeared at 09:53
PDT. The final database SHA-256 is
`62f5297a0d5fda3d5bbfa028d5355f8a4edc005520f703b4da97a33e449add94`. Trial 4,
the recovered rank-32/context-512 run, remains best at 0.12783648016629134 fair
WER with 0.294 exact; its final adapter SHA-256 is
`542588cf601a31b5376a4f9b13f583fab64ce646386fd6a84da465b1a1e07b59`. Mean
generation across completed trials was 0.858-0.889 seconds. Rank 32 supplies
the best point, but rank 16 trial 0 is close at 0.12974190 fair WER and rank
32 trial 5 is worse at 0.15156764, so this is a promising direction rather than
proof that rank alone caused the win. The current selected candidate for the
next controlled training stage is trial 4's parameter family; it is not a
product adapter.

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

### 2026-10-04: matched prompt bake-off, telemetry, kernel boundary, and repair receipt

Live host selection was measured rather than inferred. Tetra was running an
external `train_puffer.py --envs 1024` workload with load average above 320,
three heavily resident GPUs, and GPU 3 management-dead, so it was not used for
Phonon. B550 remained the research boundary.

The matched Liquid format bake-off fixed the recovered Optuna trial-4 recipe
(LoRA rank 32, context 512, LR `6.505720091093967e-5`, adapter LR
`4.943429131224935e-5`, dropout `0.033694424584220395`, warmup 50), 10,000
public rows/steps, 1,000 Aqua steps, and all 500 frozen evaluation rows. The
results at this receipt were:

| Trial | Prompt | Status | Fair WER | Strict WER | Exact |
| ---: | --- | --- | ---: | ---: | ---: |
| 0100 | `prose_dictation_v1` | complete | 0.1243720769097523 | 0.15959963603275706 | 0.296 |
| Optuna 4 | `xml_dictation_v1` | complete baseline | 0.12783648016629134 | 0.16642402183803456 | 0.294 |
| 0102 | `xml_dictation_guarded_v1` | complete | 0.14368612506495756 | 0.18089171974522292 | 0.296 |
| 0101 | `xml_transcript_v1` | public complete; Aqua repaired below | pending | pending | pending |

Trial 0101 completed its public phase at step 10,000, but the first Aqua
process failed during model import after an investigation installed published
FlashAttention and causal-convolution wheels built against a different Torch
C++ ABI. Transformers selected FlashAttention merely because the package was
present and failed on `c10_cuda_check_implementation`. The incompatible wheels
were removed, a real `ReverseAudioVL.from_pretrained` load succeeded, and Aqua
was resumed from the intact public checkpoint in a separate tmux lane. The
failed `format-receipt.json`, logs, and public checkpoint remain unchanged; the
completion uses `hyps-repair.jsonl`, `score-repair.json`, and
`resume-aqua-repair.log`. This incident is the reason kernel packages must be
installed only when import and a real model load pass against the exact Torch
and CUDA build.

The reverse SALM application seam was exercised on B550 with the existing rank-32
XML adapter SHA-256
`542588cf601a31b5376a4f9b13f583fab64ce646386fd6a84da465b1a1e07b59` and prompt
SHA-256
`a805969f94dd4075a6f8abfae95ba0117a824a5ac8f9321a74f6d59ac514bf88`.
One warmup request took 1.9759 seconds; two warm product-protocol requests took
0.7869 and 0.7750 seconds (0.7810 seconds mean). This is a resident sidecar
protocol measurement, not an end-to-end macOS application or UI latency claim.
The packaged product still needs a product-owned prompt asset and an actual Rust
`AsrSidecar::spawn_engine` exercise.

NSys profiling also established a tooling boundary. System `/usr/bin/nsys`
2023.4 recorded no CUDA kernels with Torch 2.14/CUDA 13; the CUDA-13-capable
NSys 2024.6.2 install did. Over 300 training steps, the baseline took 77
seconds wall time, 1,307,347 kernel instances, and 20.1011 seconds of total GPU
kernel time. Torch Inductor/Triton took 370 seconds compile-inclusive, but
reduced kernels to 670,943, total GPU kernel time to 17.8731 seconds, and showed
163,820 Triton launches. The first-step compile delay was about 315 seconds.
This proves kernel-count and GPU-time reduction, not yet a warm-cache wall-time
win; the decisive next profile must warm compilation and then time a separate
300-1,000-step phase in one process.

Optional W&B telemetry is now part of the training contract. B550 writes
offline transactions only; it receives no W&B credential. The trainer records
scalar loss, both learning rates, gradient norm, step rate, wall time, prompt
and pack identity hashes, runtime versions, and adapter SHA-256. It never logs
audio bytes, transcript text, accepted text, raw ASR text, or screen content.
The format orchestrator uses one run ID per full format trial across public,
Aqua, and final-score transactions. W&B 0.30 warns that offline resume is
ignored and creates separate same-ID local transaction files; this is expected.
`ml/research/final_sweep/sync_wandb_offline.py` runs only on the main machine,
orders those transactions by timestamp, and passes their common ID to
`wandb sync --id`. A three-transaction public/Aqua/score smoke test merged into
one cloud run with two history rows and a final score summary at
`https://wandb.ai/retis_labs/phonon-smoke/runs/offline-two-stage-smoke-v2`.
Local checks passed with `ruff check ml/research/final_sweep
ml/research/reverse_vl_v0` and nine passing tests in
`ml/research/final_sweep/tests/test_optuna_plan.py`. These checks prove the
telemetry and command contract, not model quality.

The CUDA kernel work is deliberately left unclaimed. `wandb==0.30.0` is
installed on B550. The published `flash-attn==2.8.3.post1` and
`causal-conv1d==1.7.0` wheels are not usable with Torch `2.14.0+cu130`; both
were removed after the failed import. A CUDA 13.0 toolkit is being provisioned
under the B550 user account so the extensions can be rebuilt against the live
Torch headers for SM 8.6. FlashAttention, causal convolution, and any fused
cross-entropy implementation require import, real model-load, and matched
profile evidence before they can be reported as active.

### 2026-10-04: prose promotion and offline-telemetry completion

The separate trial-0101 repair resumed at step 10,000, completed all 1,000 Aqua
steps, and evaluated all 500 frozen rows. `xml_transcript_v1` scored
0.1449852762861597 fair WER, 0.1827115559599636 strict WER, and 0.294 exact.
Its repaired hypothesis SHA-256 is
`1316afb8844e1f8492fb9c8470e5189b19684df9537906e198efdb27743e0f58`, score
SHA-256 is
`3a3a0285f3875d7d9c63b8f61b27b9a75c0b4ac2c84e4e156371fbeb7d6e636f`, adapter
SHA-256 is
`dc2a78926431de8e7d7cd6cfe4e92651346782349cf3abe61ed4d29018771c28`, and repair
receipt SHA-256 is
`6966c6712f6abb3344af59e97c86c683b04ef7d2b5f05e8a3f3586050367d30f`. The
orchestrator's failed receipt remains untouched. `prose_dictation_v1` is the
full-slice winner at 0.1243720769097523 fair WER, 0.15959963603275706 strict
WER, and 0.296 exact; guarded XML and verbatim XML are both worse.

The product now owns that winning prompt in `sidecar/phonon_prompts.py`, with
prompt SHA-256
`3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`. The reverse
sidecar passes the prompt text directly instead of importing the research
registry. The prose trial-0100 adapter was copied to B550's controlled product
path, `/home/kearm/.local/share/phonon/reverse-salm/lora_adapter.safetensors`,
and its SHA-256 is
`74bd889172b886b07bc658426df1de047a4185dda2f8ec593e6a60c3888e6f7e`. The engine
remains an explicit non-default selection.

The repaired trial's three same-ID offline W&B transactions were copied back to
the Mac and synchronized in timestamp order. The live run is
`https://wandb.ai/retis_labs/phonon/runs/format-0101-xml_transcript_v1`; it has
100 Aqua history points and final score, adapter-hash, loss, and wall-time
summaries. The first transaction records the pre-training path failure, the
second records Aqua, and the third records the score. Public-stage steps were
not invented retroactively. No W&B credential was present on or copied to B550.
The two format trials that completed before telemetry existed were then added
as score-only transactions, without fabricated historical steps:
`format-0100-prose_dictation_v1` and
`format-0102-xml_dictation_guarded_v1` are now live in the same project with
their full-slice scores and adapter hashes.

The real Rust launch gate now exists and ran on B550. With the reverse engine,
CUDA 0, the product adapter, and a local audio file, `AsrSidecar::spawn_engine`
reached ready and returned a non-empty result; the opt-in test passed in 13.22
seconds after the runtime cache was populated. It was rerun with `uv` absent
from PATH to prove the Linux `$HOME/.local/bin/uv` fallback. The test does not
print or persist transcript text. Local Python tests passed 16/16, Cargo Clippy
passed with warnings denied, and the full Rust workspace test suite passed.

Kernel work advanced from package presence to real execution evidence. A
user-local CUDA 13.0 toolkit was installed at `/home/kearm/cuda-13.0` (nvcc
13.0.48); the installer SHA-256 is
`c64969f35ad99bf3f9e8acb8e3d22355150c6ca07acc16a853778600a9b65ba6`. CUDA 13.0
has a known glibc `rsqrt`/`rsqrtf` declaration mismatch, so a two-declaration
local `noexcept` compatibility patch was recorded with pre/post header hashes
and `build/cuda-13.0-glibc-noexcept.patch`. `causal-conv1d==1.7.0` was then
built from source against Torch 2.14/CUDA 13; import exposed
`causal_conv1d_fn`, and a real `ReverseAudioVL.from_pretrained` model load
succeeded. `liger-kernel==0.8.4` was installed, its fused linear cross entropy
matched a reference CUDA probe within `9.54e-7`, and the trainer now has an
explicit `--liger-cross-entropy` hidden-state path. A real one-step training
smoke completed with that option and produced adapter SHA-256
`992d6892701ceea72e602e4a9bb79278e6a90028686d6dab88eb7b2e9ecb34e8`; this is
execution evidence only, not a speed or quality claim.

FlashAttention's published wheel was ABI-incompatible, so the source build
patched the emitted architecture from SM 80 to SM 86 for the RTX 3090 Ti and
moved host/device compilation from C++17 to C++20 for Torch 2.14 headers. Both
patches are recorded under `/home/kearm/salm-lora/build/flash-attn-*.patch`.
The 25m45s build succeeded. A direct `flash_attn_func` probe on CUDA returned
the same output as scaled dot-product attention with maximum absolute difference
0.0, and a real `ReverseAudioVL.from_pretrained` load succeeded with both
FlashAttention and causal convolution present.

The matched 300-step, 500-context, rank-32 profile at source `acb7be3` then
ran the current baseline and Liger variants with NSys and offline W&B. Both
launched named FlashAttention forward/backward kernels and causal-convolution
forward/backward kernels; hierarchical NVTX recorded 300 instances of each
load, transfer, forward, backward, and optimizer range.
The profile metadata SHA-256 is
`123c0effaa7965b0968268144c7261724cb32690bb0ceee5d4ee1798b2dcfba0`; baseline
NSys report/SQLite hashes are `0fe7fbae5ad2a43264123def07dfc1cdcd8f80116b0fc8c3d9e2fbcde5236939`
and `a52c645348f69dda66d1ef5665f1fa181c6d8eec79d0678ab17a652f1d23eb6f`; Liger
report/SQLite hashes are `256f86ff056f5c5e3bba8d4783f7175ff1451f4a2bb4b9b4fa79fbb946845e24`
and `08f838703c997d3e572101c68e9177b7c5818fffd34c600648fc14a6cb5e21d2`.

| Variant | Wall time | Kernel instances | Total GPU kernel time | Notes |
| --- | ---: | ---: | ---: | --- |
| Current baseline | 77.3354 s | 1,301,432 | 19.9120 s | FlashAttention and causal convolution active |
| Liger fused loss | 80.7633 s | 1,348,594 | 24.5041 s | 9,600 `liger_cross_entropy_kernel` launches |

Liger reduced forward NVTX time from 57.6291 to 56.4887 seconds but increased
backward time from 16.1556 to 20.6844 seconds. It is numerically correct and
wired as an explicit option, but it is not selected for the next full run. The
live telemetry runs are
`https://wandb.ai/retis_labs/phonon/runs/reverse-profile-baseline-v1` and
`https://wandb.ai/retis_labs/phonon/runs/reverse-profile-liger-v1`. The current
optimized stack is active, but its 300-step wall time is still effectively the
pre-kernel baseline; GEMMs and elementwise work remain the dominant costs.

### 2026-10-04: final prose repeat and Triton amortization control

The controlled final prose repeat was trial 0200 at source `39ea165`, on GPU 0,
with the exact validated public and Aqua packs, prompt SHA-256
`3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`, rank 32,
context 512, 10,000 public steps, 1,000 Aqua steps, and all 500 evaluation
rows. It used FlashAttention 2.8.3.post1 and causal-convolution 1.7.0, but did
not use Liger or Torch Inductor. Public CPT took 1,193.4 seconds and Aqua took
165.5 seconds.

The parent orchestrator was mistakenly launched through its `python3` shebang
rather than the research Python, so after public training, Aqua training,
evaluation, and scoring had all completed, only its score-logger import failed.
The orchestrator receipt remains in `failed` state. A separate completion
receipt records the model result and the research-Python score transaction; no
training or evaluation was rerun. The completion receipt SHA-256 is
`056f9d4cd9700b240f348aeb075168d13de3cdb060cb62f944e03cf34ccd378c`.

Trial 0200 scored 0.1377100294474277 fair WER, 0.17288444040036396 strict WER,
and 0.298 exact. Its public checkpoint SHA-256 is
`d96882817aad3c099cb10866ce2fbb2a86193b32ec4d806cdced9050d89510c5`, adapter
SHA-256 is
`089883eab0d4ca8953f71c304ce7668bbc1d49989f7e816495d8d89b9fa35a90`,
hypothesis SHA-256 is
`02cb826ca52c3236606dcbf1319fc933ae2ea8a1b3218a4263987307170f398f`, and score
SHA-256 is
`ff6456769707f43768a87b76f680fb3c44093af54b83f10eda6a65e4e594d9cc`. It does
not replace trial 0100, which remains the selected 0.124372 fair-WER adapter.
This is evidence that the optimized execution mode is not training-quality
neutral even when hyperparameters, data, prompt, rank, and context are fixed.

The unchanged trial-0100 product adapter was then evaluated under the same
active FlashAttention/causal runtime on all 500 rows. It scored exactly
0.1243720769097523 fair WER, 0.15959963603275706 strict WER, and 0.296 exact,
matching its reference-runtime result. Its active-runtime hypothesis SHA-256 is
`0ae5b6ae5c510eb77af900f1eaa57ca5448fb14cb59c26cc2d9a6c211a27bfb5`, score
SHA-256 is
`00e5c5f4e4990db13f9aa441a404fd91eaeeac4424b8259f45d2ffba8fdf94bb`, and receipt
SHA-256 is `d0c8e9c162f205cd030188a71ac5d254f77d4441339801557af71d58b34bdbe3`.
Therefore the installed trial-0100 adapter remains valid for current inference,
while future training must record kernel mode and cannot assume quality
equivalence. The W&B receipt is
`https://wandb.ai/retis_labs/phonon/runs/trial-0100-active-runtime-v1`.

W&B run `format-0200-prose_dictation_v1` contains 1,101 history points across
public and Aqua plus the repaired score summary:
`https://wandb.ai/retis_labs/phonon/runs/format-0200-prose_dictation_v1`.

GPU 1 ran an isolated warm-cache experiment while trial 0200 trained. Cold
Triton took 268.767 seconds for 300 steps; the warm cache reduced that to
136.293 seconds. A 1,000-step warm control then took 224.703 seconds with
2,174,022 kernel instances, 57.238491 seconds of GPU kernel time, and 581,000
Triton launches. The like-for-like baseline took 208.547 seconds with
4,266,544 kernels and 65.691366 GPU seconds. Triton reduced kernel count by
about 49 percent and GPU time by about 13 percent, but increased forward
range time from 142.262 to 180.726 seconds and lost by 7.8 percent in wall
time. Baseline execution therefore remains the production choice; Inductor is
useful for kernel analysis but not this trainer's dispatch path.

The baseline 1,000-step NSys report SHA-256 is
`d56b25cdd0ab4c1e620de0e642eb8b7f3313768aef315f3915ef932a08e9983a`; the
1,000-step Triton report SHA-256 is
`0c24d30a0f31a2f3ae0f09f8c1fbb2a42d17893c58350111e930356df40067c4`. Live
telemetry is recorded under `reverse-profile-baseline-1000-v1`,
`reverse-profile-triton-warm1000-v1`, and `reverse-profile-triton-cache-v1`.
The profile runner now refuses to overwrite an existing variant directory and
force-refreshes a stale SQLite export at `7336865`.

### 2026-10-05: Aqua Voice daily-driver parity plan

Added the repository's Aqua parity program to this architecture document rather
than creating a parallel roadmap. The program targets a macOS local daily
driver, not Aqua's account, cloud, mobile, or team-administration surface. It
keeps the frozen 500-clip slice as the quality ledger and makes the current
reverse-SALM promotion boundary explicit: the selected 0.124372 fair-WER
adapter does not meet the locally observed Aqua raw 0.032912 baseline.

The plan records eight milestones: baseline freeze, reversible Aqua migration,
interaction gaps, cascade requalification, single-model improvement, separate
vision qualification, five-day owner dogfood, and distribution packaging. Its
parity matrix marks streaming preview, local history, stats, screen OCR, and
dictionary/replacement management as present or partial; edit mode, language
support, optional endpointing, and instruction migration as gaps.

Target evidence was local only. The installed desktop package reports Aqua
0.20.10. Its local settings cache reports schema 89, deep context on, fast-LLM
selection automatic, correction enabled, 782 dictionary values, 65
replacements, and 12,830 instruction characters. No Aqua audio, transcript,
instruction text, dictionary value, or replacement text was copied into the
checkout. A bundle/settings inspection is structural evidence only; milestone 1
still requires a live accessibility walk because server and account gating can
hide packaged features. This entry is a plan, not a claim that any parity gate
has run.

### 2026-10-05: consented training-corpus capture and implementation fan-out

Added a training-corpus capture plan to this document. It separates ordinary
dictation and OCR-only screen context from a new default-off training purpose.
Two independent consents control candidate creation and screenshot retention.
Every retained screenshot must carry explicit scope, consent version/time,
provenance, dimensions/scale, retention deadline, and SHA-256; a hash mismatch
or expired image cannot enter export. Export writes outside the personal corpus
root, omits raw ASR, includes reviewed intended text only on an explicit
request, and never contacts a provider or telemetry destination. Image training
is a later pack-builder gate, not automatic inference authorization.

Three independent implementation slices were launched in isolated worktrees:
reversible Aqua dictionary/replacement/instruction migration, consented
screenshot/audio training-candidate capture and redacted export, and a
no-model Aqua baseline/score harness. Their results are not accepted at launch;
the parent must inspect each diff and rerun decisive checks. No gate, migration,
capture, export, or quality measurement in this plan has run yet.

### 2026-10-05: first Aqua parity implementation slices

Three worker slices completed after parent review and repair. The worker runs
themselves timed out before final self-reports, so none was accepted on worker
output alone; the parent inspected every diff, repaired the corpus privacy
boundary, reran the decisive checks, merged the slices, and ran one combined
synthetic CLI integration pass.

`phonon dictionary import-aqua` is implemented with `--settings`, `--dry-run`,
and `--json`. It reads only Aqua's local cache, maps dictionary values and
from/to replacements, counts instruction characters without copying their text,
creates a dated dictionary backup when a dictionary already exists, and reports
first-run and idempotent-repeat counts. The synthetic integration imported
three values on first run, zero on repeat, counted two invalid values, and
created the expected backup.

The training-corpus core is implemented, but automatic macOS screenshot capture
is deliberately not wired into the bar yet. `phonon corpus attach-screenshot`
requires both default-off settings, a PNG, explicit display/provenance fields,
and a TCC preflight declaration. It writes `screenshot.png` plus a
`screenshot.json` sidecar with hash, dimensions, scale, consent, capability,
and deletion deadline. `phonon corpus expire-screenshots` removes only an
expired verified image and its sidecar; `phonon corpus export` refuses an
existing destination, verifies hashes and sidecars, copies consented images only
on request, and omits transcript text unless explicitly requested. The worker's
Swift stop-path diff was rejected and preserved outside the tree because it
left an untracked image copy and could block ordinary dictation; app wiring
remains open.

`tools/aqua_baseline.py` registers the no-model scoring protocol for explicit
local JSONL inputs. It supports field mappings, stable-ID validation, optional
manifest and baseline inputs, fair/strict WER, exactness, wins/ties/losses,
worse-than-input rate, latency percentiles, input hashes, and allowlisted
metadata. Its receipt excludes references, hypotheses, and per-row timings.
The synthetic two-row run scored 0.2 fair WER with one win and one loss; it
proves the harness, not product quality.

Combined verification passed: `cargo fmt --all -- --check`; `cargo clippy
--workspace --all-targets -- -D warnings`; full `cargo test --workspace`; Ruff
and 13 focused baseline tests; the existing 16 root Python tests; and the
synthetic integration receipt at
`~/.cache/phonon-parity-integration-20261005T105937/receipt.json`. The Rust run
included 40 core tests, four Aqua-import CLI tests, and the opt-in reverse-SALM
spawn test. Non-claims: no real Aqua settings were imported, no real screenshot
or personal text entered Git, no live Aqua UI walk ran, no macOS app capture
path is enabled, and no full-slice quality gate or five-day dogfood claim is
made.

### 2026-10-05: reverse-SALM training-option sweep

The first controlled option addressed an underuse defect in the promoted
recipe: only 1,000 of 12,855 eligible Aqua rows were consumed during
adaptation. All arms below resume the same trial-0100 public checkpoint, use
prompt SHA-256
`3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`, rank 32,
context 512, and the reference FlashAttention/causal-convolution execution
mode that produced trial 0100. The optimized wheels were preserved as local
SM86/reference-compatible wheel files, removed only while training or scoring
reference arms, and restored afterward.

| Additional Aqua steps | Fair WER | Strict WER | Exact |
| ---: | ---: | ---: | ---: |
| 1,000 baseline | 0.1243720769097523 | 0.15959963603275706 | 0.296 |
| 2,000 | 0.12272648536289624 | 0.15787079162875342 | 0.290 |
| 4,000 | 0.11761649055950113 | 0.154049135577798 | 0.316 |
| 8,000 | 0.11311276632600034 | 0.14585987261146496 | 0.332 |
| 10,000 | 0.14593798718170795 | 0.18152866242038215 | 0.342 |
| 12,855 one epoch | 0.11077429412783649 | 0.14285714285714285 | 0.352 |

The 10,000-step dip followed by the one-epoch recovery means this is not a
monotonic overfitting curve; learning-rate schedule, row order, and checkpoint
position remain entangled. A full epoch is the best single checkpoint, but the
non-monotonicity is a non-claim against predicting a second epoch.

A 30,000-row public-diversity arm scanned 42,826 pinned YODAS source rows and
trained 10,000 public steps plus 4,000 Aqua steps. It scored 0.119954962757665
fair WER, 0.1564149226569609 strict WER, and 0.304 exact, worse than the
original 10,000-row public pack at the same 4,000-step Aqua adaptation. This
does not prove larger public packs are useless; it only rejects the specific
30k-row/10k-step combination.

Three model soups were tested. Equal averaging of the 1k/2k/4k adapters scored
0.11605750909405854 fair WER and 0.15159235668789808 strict WER; averaging the
2k/4k adapters scored 0.11683699982677984 and 0.15177434030937215. The
decisive soup equally averaged the 8,000-step and one-epoch adapters. Under the
current optimized product runtime it scored 0.10930192274380737 fair WER,
0.14067333939945406 strict WER, and 0.350 exact on the frozen 500-row slice.
Its adapter SHA-256 is
`9a70586a682fe2c694896efe3ac5aa5c467874ab6a9f5e0c2dc847eff31a3f5c`.

The repeated selection slice is no longer sufficient by itself, so the stable
750-row future split was converted into an evaluation slice with
`corrected -> ref`; its SHA-256 is
`5f71c33816151f363be727622e48ac75c3ebef982a14c71aa0951d722554e728`. An initial
future score of 23.165 fair WER was invalid because the raw manifest's
`corrected` field was read as an empty `ref`; that failed artifact and its
cause remain recorded rather than being treated as model evidence.

| Adapter | Future fair WER | Future strict WER | Future exact |
| --- | ---: | ---: | ---: |
| Trial 0100 | 0.17531860166023616 | 0.22834742885363293 | 0.164 |
| 8,000-step | 0.16082076464398457 | 0.21001296376319525 | 0.18933333333333333 |
| One epoch, optimized runtime | 0.14521220624342335 | 0.19340700043212544 | 0.18666666666666668 |
| 8k/full soup | 0.1439261077984333 | 0.19130810543860732 | 0.18666666666666668 |
| Aqua raw | 0.07260610312171167 | 0.13507006605346009 | 0.364 |

The soup improves fair and strict WER over the one-epoch checkpoint on both
selection and future slices. It is the promoted explicit reverse-SALM adapter,
but it still does not beat Aqua raw or the native Audio student and must not be
described as overall dictation superiority.

The 8,000-step adapter matched its reference aggregate scores under optimized
inference on both 500 and 750 rows. The one-epoch adapter matched its reference
score on the 500-row slice and its optimized-runtime future result is recorded
above. The soup was evaluated directly in the product runtime.

The soup adapter was installed at
`/home/kearm/.local/share/phonon/reverse-salm/lora_adapter.safetensors`; the
prior trial-0100 adapter remains beside it as
`lora_adapter.trial0100.safetensors`. A resident sidecar warmup took 1.9924
seconds, two warm requests took 0.7815 and 0.7574 seconds (0.7694 seconds
mean), and the opt-in Rust `AsrSidecar::spawn_engine` test passed in 24.32
seconds. The engine remains non-default and the shipped Parakeet path is
unchanged. W&B runs include
`https://wandb.ai/retis_labs/phonon/runs/aqua-length-8000-reference-v1`,
`https://wandb.ai/retis_labs/phonon/runs/aqua-full-epoch-reference-v1`,
`https://wandb.ai/retis_labs/phonon/runs/full-epoch-active-runtime-v1`, and
`https://wandb.ai/retis_labs/phonon/runs/soup-8000-full-active-v1`.

### 2026-10-05: private model placeholder and training-control implementation

Created the private Hugging Face placeholder
`https://huggingface.co/RESMP-DEV/phonon-reverse-salm`. The Hub API confirmed
`private: true`; it contains only `README.md` and `config.json`. The card records
the prompt contract, pinned public-data revision, aggregate scores, hashes, and
non-claims. It intentionally contains no weights, tokenizer, audio, transcript,
accepted text, dictionary value, screenshot, or other personal data. The current
trained adapter remains on controlled compute storage pending an explicit
release decision.

The trainer now exposes real `--batch-size`, `--gradient-accumulation`,
`--conformer-blocks`, and `--conformer-lr` controls. Batching uses the native
LFM2 collator's real batch dimension, accumulation averages microbatch losses,
and selective conformer tensors are included in checkpoints and restored on both
resume and inference. A real one-step CUDA smoke with batch 4, accumulation 2,
and the last four conformer layers unfrozen completed successfully. Timing
probes showed batch 2 and batch 4 are wall-clock regressions on one RTX 3090 Ti
for this model, so the first long structural arms use batch 1 while preserving
the new controls for quality experiments.

The first two long structural arms completed. A 30,000-row public pack with
20,000 public steps followed by one full Aqua epoch scored 0.11908886194353023
fair WER, 0.15304822565969062 strict WER, and 0.312 exact; it does not replace
the promoted soup. Unfreezing the last four conformer layers for 8,000 public
steps followed by one full Aqua epoch scored 0.11406547722154858 fair WER,
0.14804367606915378 strict WER, and 0.354 exact. Its exact score beats the
promoted soup, but fair and strict WER remain worse; it is retained as a
candidate for souping rather than promoted.

The learning-rate audit found a real schedule asymmetry in the winning runs:
the language LoRA used linear warmup followed by cosine decay, while the audio
adapter learning rate remained static. Optuna tuned the two initial learning
rates, rank, context, and warmup; it did not tune schedule shape. Two full-epoch
controls then scaled the adapter learning rate with the existing cosine. The
Optuna initial adapter LR scored 0.11458513771002944 fair WER, 0.1475887170154686
strict WER, and 0.340 exact. Half that initial adapter LR scored 0.1144119175472025,
0.14749772520473156, and 0.338. Both are worse than the static-adapter
full-epoch checkpoint and promoted soup, so the static adapter LR remains the
quality winner. W&B histories confirm the new adapter LR actually decayed from
approximately `2.96e-5` or `1.48e-5` to near zero.

### 2026-10-05: dynamic Aqua history prompts

The prompt contract was extended so each target can carry user-specific
historical Aqua evidence without exposing the current target's accepted label.
The new prompt ID is `prose_history_dictation_v1`, with base SHA-256
`4d7ea7b4687a8f82ddc910b76c2c529fbf17755dcfc6649d2c6be02007c3670f`.
For each row, the current raw Aqua transcript queries an IDF-cosine index
(`idf-cosine-v1`) over training-only raw-to-corrected pairs. Two distinct
historical pairs are rendered in the system prompt; the current audio ID is
excluded, the current corrected text is never consulted, and every target has a
separate dynamic prompt hash. The evaluator reconstructs the same prompts from
the receipted training manifest.

The first full-pack attempt rebuilt IDF and candidate vectors independently for
every row and was stopped before producing a usable pack. Commit
`ac536758d01258926307acef9e87d9546743ebed` introduced one reusable sparse index
for both pack construction and evaluation. Local and B550 checks both passed
ruff, six focused tests, and `git diff --check`.

The completed v2 pack is at
`/home/kearm/salm-lora/build/history-prompt-full-v2/pack`, with its sanitized
receipt at `receipt.json` and tracked local receipt at
`evidence/2026-10-05-history-prompt-full-v2.md`. The manifest SHA-256 is
`9378e4532e54c8816ea962a515ac833d60efbab0ec1e1d102b0296e81d490f08`; the
history-contract SHA-256 is
`21eccc7415d93548a347f816aa8b71ef18709ae2b45ef0f77da9c2441d623053`. It has
12,855 target contracts, 12,855 unique dynamic hashes, all self-exclusions and
two-history invariants passing, and 13 Arrow shards totaling 6,127,290,118
bytes. Dataset preprocessing retained 12,854 examples because one 781-token
target exceeded context 768. Context 768 remains intentional: the 32-row pilot's
minimum, median, P95, and maximum packed lengths were 296, 478, 578, and 649.

The first matched training pilot resumed the verified trial-0100 public
checkpoint SHA-256
`511c6f1ecae7281c039c63fda52a5cd83b692b1e09909506ad0fbfadf0d9bb20`, added
4,000 Aqua steps at rank 32 and context 768, and retained the tuned initial
LRs with static audio-adapter behavior from the winning recipe. It completed in
609.4 seconds under reference kernels, produced adapter SHA-256
`7f2d103385e699ab4c36c37c156f04058ed925c022c0d9cb5cbebcafd0897194`, and logged
offline W&B run `history-aqua-4000-reference-v4`. Three wrapper-only preflight
failures are preserved under sibling `aqua-4000-reference-v*-launch-failed` or
failed roots; they never started a training process and the EXIT trap restored
both optimized wheels each time.

The 500-row selection result is negative:

| Adapter | Fair WER | Strict WER | Exact |
| --- | ---: | ---: | ---: |
| Static-prompt 4,000-step arm | 0.11761649055950113 | 0.154049135577798 | 0.316 |
| Dynamic-history 4,000-step arm | 0.17218084184999133 | 0.20937215650591448 | 0.278 |

The hypothesis SHA-256 is
`008036013fd5192a52057ac4f3589749e2c2c4ab37ad0906bd8742121f8579a9`, and the
score SHA-256 is
`2812b9cbf3c197ce749f170561f7f69474b72e57e1215dd54fdea1d040106d17`. Paired
row analysis found 84 improved rows, 272 unchanged rows, and 144 worsened rows;
median row delta was zero while the p95 regression was 0.36534391534391514.
History outputs increased mean candidate unigram overlap only from 0.40194 to
0.42038 and bigram overlap from 0.17925 to 0.20391; six of 500 outputs exactly
copied a candidate. This rejects scaling this exact two-history prompt recipe
to a full epoch and suggests long-system-prompt disruption is at least as
plausible as history copying. The result does not reject all historical
conditioning, and no future-750 claim is made.

An unchanged trial-0100 control clarified the failure mode. With no additional
history adaptation, XML-history v1 scored 0.8041746059241296 fair WER,
0.8413102820746133 strict WER, and 0.128 exact. The prompt had replaced the
short trained prose prompt with XML scaffolding, so the checkpoint was far
outside its trained system-prompt distribution even though audio and labels were
unchanged.

Prompt `prose_history_dictation_v2` therefore preserves `prose_dictation_v1`
byte-for-byte as its prefix, appends the same two selected examples compactly,
and retains the anti-copy instruction. Its schema SHA-256 is
`664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6`. The
compact pack at `/home/kearm/salm-lora/build/history-prompt-compact-v2/pack`
uses the same manifest and retrieval hash as v1, but all 12,855 targets fit
context 768; it has 12,855 Arrow examples, 13 shards, 6,113,806,478 bytes, and
checksum-manifest SHA-256
`82a82b0257aaa9e3f27cc407acd2d349f2e58d8906c68a88fcd902b602444673`.

The unchanged trial-0100 control improved to 0.7024943703447081 fair WER,
0.7403093721565059 strict WER, and 0.128 exact under compact v2, confirming
that preserving the trained prefix helps but does not make history injection
free. A matched 4,000-step compact adaptation completed in 613.3 seconds with
adapter SHA-256
`16ab70ac07c49749835a7613e888d58a9f1b45c04ece23f87a9799921e5d58d8`. It scored
0.1627403429759224 fair WER, 0.2002729754322111 strict WER, and 0.280 exact;
paired static comparison found 85 improved, 275 tied, and 140 worsened rows.
This is better than XML v1 but still worse than static prose, so the full
compact epoch is the next decisive quality gate rather than promotion.

The full compact-history epoch ran all 12,855 examples over 22,855 total steps
from the same public resume point. It completed in 1,829.2 seconds with adapter
SHA-256
`44a379ee1762461077319fa28173d2da782ad1ea5f29b0303c8fde3218e033a4` and offline
W&B run `history-compact-full-epoch-v1`. On selection 500 it scored 0.11345920665165425
fair WER, 0.14522292993630573 strict WER, and 0.350 exact. On future 750 it
scored 0.14445223898047468 fair WER, 0.19297487499228347 strict WER, and
0.20533333333333334 exact.

Compared with static full epoch, history is slightly worse on selection but
better on every future metric; exact rises from 0.18666666666666668 to
0.20533333333333334. Row-level selection comparison finds 104 history wins, 85
static wins, and 311 ties. The promoted static 8k/full soup remains slightly
better on future fair/strict WER (`0.1439261077984333` and
`0.19130810543860732`), while history full has the better future exact score.
This validates compact user-history conditioning as a competitive research
direction, not a product promotion. Aqua raw remains substantially stronger,
and optimized-runtime parity/latency remain unmeasured for this adapter.

Two adapter-soup controls rejected averaging as the next step. Equal static
full plus history full scored 0.13554477741209076 fair WER,
0.16560509554140126 strict WER, and 0.356 exact on selection. Equal static 8k,
static full, and history full scored 0.208470465962238 fair WER,
0.24340309372156507 strict WER, and 0.310 exact. The first improves exact but
damages WER; the second damages all headline metrics. Neither was sent to the
future split or promoted.

The same full-history adapter later matched every reference aggregate metric
exactly under the optimized runtime (Torch `2.14.0+cu130`, FlashAttention
`2.8.3.post1`, and causal-conv1d `1.7.0`): selection scored 0.11345920665165425
fair WER, 0.14522292993630573 strict WER, and 0.350 exact; future scored
0.14445223898047468, 0.19297487499228347, and 0.20533333333333334. The receipt
is `/home/kearm/salm-lora/build/history-compact-full-epoch-v1/optimized-runtime-v1/receipt.json`.
Research-evaluator mean generation was 0.9088 seconds on selection and 0.9295
seconds on future. This proves optimized-kernel quality parity only; it is not
a resident product-protocol or macOS end-to-end latency claim.

The oracle dependency was then tested directly. A two-pass, same-adapter
evaluator first generated a static-prompt draft, used that draft as the only
retrieval query, and generated the final history-conditioned transcript; the
slice row's Aqua raw text was not passed to retrieval. The draft scored
0.1268837692707431 fair WER and the final improved to 0.123592586177031, but
remained well behind the oracle-query result of 0.11345920665165425. Mean
two-pass time was 1.8611 seconds while a peer calibration used both GPUs, so
that timing is diagnostic rather than a clean latency gate. This rejects the
current two-pass self-retrieval design for product use: the second pass helps,
but the draft query loses too much retrieval quality and doubles generation.

A one-pass alternative replaces retrieval with a fixed history card: a
deterministic quantile sample of real corrected pairs from the training
timeline, rendered once into a system prompt. All selection and future audio IDs
were excluded. The K8 card reaches 0.11449852762861597 fair WER, 0.14904458598726114
strict WER, and 0.328 exact, close to the oracle 0.11345920665165425/0.14522292993630573/0.350
result without a retrieval query. The K4 card is slightly easier to package but
scores 0.11614411917547203/0.14895359417652412/0.336. The same K4 prompt collapses
the installed static soup to 0.39052485709336565 fair WER, proving that the fixed-card
prototype requires the history-trained adapter and is not a prompt-only upgrade.

### 2026-10-06: reverse-SALM v0.1.0-alpha.1 private adapter release

The private Hugging Face placeholder
`RESMP-DEV/phonon-reverse-salm` was promoted from a bare stub to a proper alpha
model card and then, by explicit owner instruction, updated from metadata-only to
a private runnable adapter release for collaborators. The card records the provisional fixed-history prototype, prompt and
kernel contracts, aggregate selection/future results, exact optimized-runtime
parity, runtime non-claims, release state, privacy boundaries, and honest
limitations. It follows the house card convention used by the ASR and calibrated
LFM releases: pinned upstream and public-data revisions, matched metrics, and
explicit statements about what the artifact is not.

- Card: `README.md`
- Sanitized local receipt copy:
  `evidence/huggingface/2026-10-06-phonon-reverse-salm-alpha.md`
- Card commit:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm/commit/b9329d42078bccc493d2b58e3de9277bed9366e8`
- Config commit:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm/commit/686c15dd7ee02a2c5e284568e1c909e0daae56a8`
- Alpha adapter commit:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm/commit/899e8a21a1389c4df8bc13e93018df942c9dd9a6`
- Static-soup control commit:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm/commit/850ff731eee028daedb2903f23affccc47f1b502`
- Artifact documentation/config/checksum commits:
  `50f7f6252c7412dfcd9eaeea2b4295b5a188986b`,
  `b4b4323fc7ef253ace39a5adbf732acb5753ea2f`, and
  `c3d37ce12a1a106ff65aa73093ede5ca8fc409d8`
- Hub API verification: `private: true`
- Repository files after release: `.gitattributes`, `README.md`, `config.json`,
  `ADAPTER_SHA256SUMS`, and two private research adapters
- Private research adapters published: yes
  - `adapters/v0.1.0-alpha.1.safetensors`, SHA-256
    `44a379ee1762461077319fa28173d2da782ad1ea5f29b0303c8fde3218e033a4`
  - `adapters/static-soup-v0.safetensors`, SHA-256
    `9a70586a682fe2c694896efe3ac5aa5c467874ab6a9f5e0c2dc847eff31a3f5c`
- Base weights and tokenizer published: no
- Personal audio, accepted text, dictionary values, screenshots, history cards,
  audio IDs, session IDs, and internal manifest hashes published: no

The remote card was re-downloaded and scanned for prohibited optimization-method
vocabulary, local absolute paths, Aqua audio IDs, and session IDs; all checks
were negative. Both adapters were subsequently re-downloaded from the Hub and
their SHA-256 values matched the source adapters byte-for-byte. This private
adapter release does not authorize public redistribution and does not change the
default product engine.

### 2026-10-06: reverse-SALM adapter merge/agent contract

Added a private root `AGENTS.md` to the Hugging Face adapter repository so
collaborators cannot treat the 190-tensor graft as a normal single-base PEFT
checkpoint. The file documents the required two-base load, namespace split,
runtime injection parameters, fixed-history prompt contracts, adapter-hash and
rank checks, privacy rules, and the in-place eager merge procedure.

The eager path was verified on B550 rather than copied from generic PEFT
documentation. The alpha adapter injected correctly, generated output, then all
92 inserted PEFT target layers were merged in place with `.merge()`; a second
generation produced an identical non-empty output. The generic
`PeftModel.merge_and_unload()` path is invalid for this architecture because
`inject_adapter_in_model` leaves the original VL class in place. The audio
adapter is six full replacement tensors, not LoRA tensors, and must remain a
separate graft component.

- Hub commit:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm/commit/1f8530f8fd8d579413ade427f78a95253c518255`
- Local tracked copy:
  `evidence/huggingface/phonon-reverse-salm-AGENTS.md`
- Remote re-download and leak scan: clean
- Smoke receipt: 92 merged targets, identical output, no transcript printed

### 2026-10-06: native SALM prototype becomes the tolerable Phonon model

The evidence audit redirected the prototype from the reverse graft to the
best-measured native audio lane. `salm-grpo-v1` (GRPO 300 steps from
`v4full@2000`) scores 0.0839 fair WER, 0.1132 strict WER, and 0.360 exact on the
local 500-clip slice, substantially better than the promoted reverse graft at
0.109302/0.140673/0.350. It is also a normal single-base LoRA on
`LiquidAI/LFM2.5-Audio-1.5B`, rank 16, adapter SHA-256
`d157a38be7ebc801000a6bac13146b8f7333b4a89f8dcda1eb9e26ad5b54f830`.

The adapter was verified through the real Rust `AsrSidecar` on macOS, not just a
research evaluator. On `RTX 3090 Ti` CUDA the sidecar reported 13.29 seconds to
ready, 2.631 seconds warmup, and 0.358 seconds mean across five warm requests.
On M4 Max MPS the real spawn test completed in 20.46 seconds with 5.6 seconds to
ready, 0.465 seconds warmup, and 0.309 seconds for the second request. These are
sidecar timings, not keyboard-to-insertion end-to-end latency.

The adapter is installed as the explicit experimental native SALM model at
`~/.local/share/phonon/salm/lora_adapter.safetensors`; the previous
`salm-lora-v2-2000s` adapter at SHA-256
`343a9e9ec19c11b0531e39ac54855e458c80059a4e937333cee9f9415956fd61` remains
preserved as `lora_adapter.v2-2000s.safetensors`. Selecting it is explicit and
non-default via `PHONON_ASR_ENGINE=salm`. `AsrEngineSelection::salm()` pins the
runtime to liquid-audio 1.3.0, peft, safetensors, soundfile, and transformers
5.x; the default Parakeet command is unchanged.

The private Hugging Face repository now includes this adapter as
`adapters/native-salm-grpo-v1.safetensors`, and its root `AGENTS.md` recommends
it as the Phonon prototype. The reverse graft and static-soup controls remain
published for comparison, not for daily use.

The product promotion gate is unchanged. This is a tolerable research prototype
and explicit experimental engine, not a shipped Aqua replacement. It still needs
the frozen-protocol release gate, keyboard-to-insertion latency, offline
packaging, default-path regression, and five-day dogfood acceptance.

### 2026-10-07: vision-on-audio post-CPT ASR SFT and LoRA fuse

The vision-capable lane is audio-native SALM with a transplanted SigLIP2 tower
and VL projector, not a reverse graft. Its research implementation, the CPT/SFT
trainer, a fuse script, and a JSONL product-sidecar prototype were migrated from
the uncontrolled B550 workspace into `ml/research/salm_vision_v0` and
`sidecar/salm_vision_server.py`. The trainer gained `--init-adapter`, so task
tuning can start from an existing LoRA instead of a fresh one. The fuse script
folds the language LoRA into `lfm.*` weights by merging each inserted PEFT target
layer, then writes the merged language state, optional projector, the vision
rows, and a hashed receipt.

Two post-CPT supervised fine-tuning arms were run from the same CPT adapter
`a43fac977d6fa3bf8150257ac26db8a9cea39d8eb937db17d64ecc493260d6dc` at rank 16,
with no GRPO.

| Arm | Steps | LR | Lanes | Audio fair WER | Audio exact | Vision WER | Vision exact | Blank WER |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CPT baseline | - | - | - | 0.0902 | 0.350 | 0.0250 | 0.795 | 0.9970 |
| Mixed SFT | 2,000 | 2e-5 | 4:1:1 | 0.09024770483284254 | 0.350 | 0.0248 | 0.795 | 1.2973 |
| Audio-only SFT | 2,000 | 1e-4 | audio | 0.08531093019227438 | 0.356 | 0.0252 | 0.770 | 1.0298 |
| Audio-only GRPO, no vision | 300 | - | audio | 0.0839 | 0.360 | n/a | n/a | n/a |

Strict WER was 0.11637852593266607 for the audio-only arm. The mixed-lane arm
held vision but did not improve audio. The audio-only arm at the original ASR
learning rate recovered 0.0049 fair WER while retaining screenshot reading and
keeping the blank-image control at zero exact match, so the model still reads
pixels rather than reciting. The audio-only SFT adapter is the best
vision-capable audio model measured so far, still 0.0014 fair WER behind the
audio-only GRPO adapter.

A fuse smoke on the CPT adapter produced `merged_lfm.safetensors` with SHA-256
`a0400d2db955a4bfa7d6a60d02bd2102345685d2e6947624eee329487c95ef15`, folding 184
LoRA tensors across 92 target layers, and the audio-only SFT fuse produced
`9fa33fc51c7c6ee40c1254fbe939ecc5366809da1e541b52ce93aa9c4e1c4659`.

Non-claims: the fused weights were not scored as a standalone model, only the
LoRA-injected path was; no future-split or product latency measurement exists for
either arm; one operating point is not a swept optimum.

### 2026-10-07: Optuna owns vision SFT learning rates

The vision-on-audio SFT path now has a persistent Optuna harness rather than
hand-picked learning-rate follow-ups. `optuna_vision_sft.py` fixes the graft,
rank 16, 4:1:1 lane ratio, 768/768/1024 contexts, and initialization adapter,
then samples learning rate over a persistent SQLite study. Each trial trains,
scores the frozen 500-row audio slice, scores held-out screenshots, records
source revision, environment manifest, kernel mode, commands, adapter hashes,
hypothesis hashes, and score hashes. The objective is audio fair WER plus a
penalty if vision WER exceeds its ceiling, preventing catastrophic forgetting.

The pinned environment is recorded in `environment-b550.json`: Python 3.12,
Torch `2.14.0+cu130`, liquid-audio `1.3.0`, PEFT `0.21.0`, Transformers
`5.17.0`, datasets `5.0.1`, Optuna `5.0.0`, FlashAttention `2.8.3.post1`, and
causal-conv1d `1.7.0`. Kernel mode is recorded from actual imports and package
versions, not package resolution alone. Any new source revision, prompt, pack,
context, initialization adapter, rank, or kernel mode requires a new named
study and study root.

A first one-trial smoke exposed and preserved a real evaluator defect: training
completed, but the harness passed the wrong B550 audio root and the evaluator
failed before scoring. The harness now pins `/home/kearm/aqua-training-data`,
runs the vision evaluator from its owning directory, forces `CUDA_VISIBLE_DEVICES=0`,
records kernel mode, and reports no-complete-trial states instead of raising
from `best_trial`.

A corrected one-trial smoke completed end to end. The real four-trial, 600-step
study then completed with the pinned optimized kernels active:

| Trial | LR | Objective | Audio fair WER | Vision WER |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.3598e-4 | 0.153504 | 0.088862 | 0.072321 |
| 1 | 6.2685e-5 | 0.087216 | 0.087216 | 0.027262 |
| 2 | 1.3466e-5 | 0.087996 | 0.087996 | 0.021583 |
| 3 | 3.8238e-5 | 0.087823 | 0.087823 | 0.021204 |

Best trial 1 selected LR `6.268486284584557e-05`; adapter SHA-256 is
`996feaacbdd41ed06a5fe14b463db8f2d6658ce89bd059554d0c04f76e7d23d3`. The receipt
is `evidence/2026-10-07-vision-sft-optuna-lr.md`. This LR is contract-specific
and must not be reused after a source, pack, prompt, context, rank, init, or
kernel change.

### 2026-10-07: prompt-baked SFT retention tempering contract

The workable vision-on-audio SFT path now has an explicit capability-retention
contract. All tensors outside the language LoRA are frozen. The transplanted
SigLIP2 vision tower is always frozen, and the VL projector is frozen unless a
separate stage explicitly enables `--unfreeze-projector` with its low learning
rate. This keeps the existing audio and vision capability anchored while the
language LoRA learns task-specific correction behavior.

The mistaken external-correction prompt was removed after the owner clarified
that the target is Phonon's own prompt-baked model, not a separate external
layer. Prompt-baked training remains the active direction: future packs must
carry the exact Phonon prompt, history examples, dictionary constraints, and
optional screen-context fields as part of the training contract. History must
be evidence for spelling, formatting, and correction tendencies, never phrases
to copy. Uncertain wording must remain unchanged. Personal transcript text
stays in controlled training storage and never enters Git or public artifacts.

### 2026-10-07: Phonon history-prompt SFT Optuna study

The mistaken external prompt correction was removed and replaced with
Phonon-owned history prompts. A full 12,855-row dynamic-history pack was built
with prompt `prose_history_dictation_v2`, and because the pack contract changed,
a new four-trial Optuna study selected the learning rate. Trial 2 won with LR
`8.619877702306163e-05`, audio fair WER 0.08678330157630348, strict WER
0.11683348498635122, exact 0.356, vision WER 0.022718667171525937, and exact
0.780. Its adapter SHA-256 is
`d3397f013161d8d38244b3d8ef4c77b2e0395d57681830d97c4ac733b44a664c`. Full results
are in `evidence/2026-10-07-phonon-history-sft-optuna-lr.md`.

### 2026-10-07: Phonon history-prompt 2,000-step SFT study

A full-length Optuna study on the history prompt pack completed three 2,000-step
trials. Trial 0 selected learning rate `1.0231901903347211e-04` with audio fair
WER `0.08626364108782263`, strict WER `0.11546860782529572`, exact `0.358`,
vision WER `0.027262400605831124`, and vision exact `0.770`. Its adapter
SHA-256 is
`d0bd203510dae234839e2afa5fdd337ef382a5406bd9acd5b38ac0f1f38515f5`. The LoRA was
fused into the language weights as `merged_lfm.safetensors` with SHA-256
`e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7`, merging 184
LoRA tensors across 92 target layers with injected cross-entropy
`0.12680479884147644`.

This is the best measured audio result among vision-capable adapters, and it
remains `0.0024` behind the audio-only GRPO adapter. The fused weights were not
scored standalone and the future split was not scored. Receipt:
`evidence/2026-10-07-phonon-history-sft-2000-study.md`.

### 2026-10-08: stage-two history SFT control

A second Optuna round initialized from the best 2,000-step history adapter, ran
1,000 additional mixed SFT steps, fused every trial, and scored the standalone
fused model. The best stage-two trial used learning rate
`6.0245784940505935e-05` and scored 0.08669669149489001 fair WER,
0.11565059144676978 strict WER, 0.368 exact, and 0.03445664521014767 vision WER.
It did not improve the existing fused checkpoint, which remains at 0.08548415035510133
fair WER, 0.11474067333939945 strict WER, 0.374 exact, and 0.025369178341537298
vision WER. Stage two is retained as a negative control and the existing fused
2,000-step artifact remains the selected vision-capable model. Receipt:
`evidence/2026-10-08-phonon-history-sft-stage2.md`.

### 2026-10-08: history-training epoch audit and upload boundary

The history-prompt runs were audited by lane rather than by global step count.
The selected 2,000-step run plus the rejected 1,000-step stage-two control
exposed 2,002 history-audio rows of 12,855, 499 corrector rows of 17,496, and
499 vision rows of 8,000. The new history-prompt contract therefore consumed
only 15.6 percent of its audio lane, 2.9 percent of corrector, and 6.2 percent
of vision. The older 24,000-step CPT exceeded one vision epoch but did not
complete audio or corrector epochs and used a different audio pack.

Accordingly, the 2.3 GiB fused language file
`e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7` was not
uploaded. The private Hugging Face repository continues to carry adapter
artifacts and metadata. Receipt:
`evidence/2026-10-08-history-training-epoch-audit.md`.

## Contracts to preserve

- Local execution and user sovereignty outrank quality, personality, and performance.
- Normal dictation accuracy must not regress to obtain a style or model experiment.
- Raw or intended text must remain recoverable when a transform or experimental engine is selected.
- Evaluated outputs stay within their registered workflow and are never pooled with unrelated protocols.
- Personal data is local, consent-gated, journaled, reviewable, and deletable.
- Unverified implementation is never reported as a shipped or working capability.
- The prompt format, context length, and LoRA rank are part of a training contract. A pack, checkpoint, adapter, or score is only valid with the same prompt ID and hash, the same context length, and the same rank that produced it.
- A failed trial is preserved, never rewritten. Repaired work is recorded as a separate completion carrying the original trial, the cause, and the evidence path.
- Screen images are never a by-product of capture. Image conditioning requires an explicit protocol field, a declared capability, and a consent-gated retention and deletion policy.
- The final audio path stays mono PCM16 at 16 kHz. Any change to that format must clear the speech gate, not just the model.
