# Phonon architecture

## Philosophy

Phonon's non-negotiable priority order is privacy and user sovereignty first, then final-text fidelity, then interaction latency, then personality and convenience. Every architectural decision below is subordinate to that order. The product is a local-first voice authoring system: microphone in, intended text out, with personal context retained under explicit consent and no cloud dependency.

## Ownership table

| Boundary | Owner | Current position |
| --- | --- | --- |
| Product specification | `SPEC.md` | Canonical product requirements and policy; needs status links after sections are audited |
| Architecture and roadmap | `docs/architecture.md` | This document; the single cross-component plan |
| macOS app and capture path | `bar/`, `crates/` | Shaped as the local product surface |
| ML correction and research | `ml/` | Canonical text-refiner research pipeline |
| Profile and vocabulary mining | `tools/profile-miner` | Local consent-gated personal context lane |
| Synthetic persona evaluation | `tools/persona-gym` | Synthetic-user regression tool for mining |
| Distribution | `DISTRIBUTION.md`, `scripts/`, platform crates | Packaging, signatures, and pinned runtime assets |
| Public site | `website/` | Separate web surface; consumes stable claims only |
| Audio/vision SALM experiments | local `phonon-eval` workspace, B550 `~/salm-lora`, and uncommitted product seam on `ml/portable-harness` | Research adapters exercised and measured; product integration remains uncommitted and untested |

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

The repository contains only the first product integration seam, currently uncommitted on `ml/portable-harness`: `sidecar/salm_server.py`, ASR-sidecar selection in `crates/phonon-asr`, benchmark wiring in `crates/phonon-cli`, and README instructions. It targets `LiquidAI/LFM2.5-Audio-1.5B`, speaks the existing JSONL protocol for whole utterances, and intentionally does not support streaming partials.

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

1. **Preserve the dirty product seam.** Inspect and commit custom/default ASR process selection independently from model claims. Add tests proving the custom script/runtime selection and that the default Parakeet command remains unchanged.
2. **Bring the experiment under durable source control.** Review and commit the local experiment scripts, omitting personal data, then record adapter hashes and score receipts in an evidence index. The existing local workspace is not a release process.
3. **Protocol contract tests.** Exercise ready, ping, transcribe, warmup, unsupported streaming, missing adapter, malformed JSON, and shutdown behavior without loading the model by injecting a fake engine boundary into `salm_server.py`.
4. **Create the audio release gate.** Repeat the GRPO result with a frozen, manifest-hashed fixture and compare SALM, Parakeet, and the two-model refiner under one registered protocol. The present local slice is strong evidence, not a product ship gate.
5. **Add the multimodal protocol.** Extend the sidecar contract with an explicit image field, screenshot preprocessing metadata, and a model capability declaration. Do not encode screenshots as an audio command.
6. **Run a vision release gate.** Expand beyond the 200-row research set, add non-Phonon screenshot distributions and blank/degraded controls, and define protected-content and privacy handling for screen context. Keep audio regression as a simultaneous gate.
7. **Product path.** Route an explicit experimental engine selection through the same capture, undo, telemetry, readiness, and offline checks as Parakeet. Vision input must remain opt-in and include retention, provenance, and screen-context policy.
8. **Distribution decision.** Quantize and package the chosen adapter plus the vision tower/projector, pin every component, and measure latency and resident memory on target platforms. A LiteRT-LM container remains an option, not a chosen destination.

### Acceptance gates

The product seam can be called implemented only when its code and protocol tests are committed and pass the relevant Rust and Python checks. The research adapters are already exercised on real audio and screenshots, but the lane can be called product-ready only after the frozen-fixture audio and vision gates pass, sidecar multimodal behavior is tested, offline artifact pinning works, streaming behavior is explicitly represented, and the default Parakeet path has no regression. The current vision adapter must not be described as shipping in Phonon merely because its research evaluation passed.

## Work-log

### 2026-10-01: higher-quality audio data and model survey

Measured two longer GRPO arms after the 300-step pilot. The 500-step group-8 control scored 0.0848 fair WER and the 500-step group-12/SFT-init-KL arm scored 0.0840, while GRPO v1 remained best at 0.0839. This closes the "just run GRPO longer" hypothesis for the current recipe.

Audited the Aqua packs without emitting transcript contents. The full interleaved pack is valid and substantial: 31,652 rows with 2,619,605 text slots, 3,258,794 audio-in slots, and 3,292,509 audio-out slots. The newer v3/v4 mixed packs use sequential audio-in rows plus text correction rows and do not include the interleaved read-along lane. A matched high-quality rebuild therefore provides new signal rather than a duplicate experiment.

The manifest has 34,246 rows, mean duration 9.44 seconds, and 32,813 correction-positive rows. A higher-quality selection keeps only correction-positive clips between two and twenty seconds, excludes the frozen 500-clip slice, and uses a stable audio-ID hash split: 12,855 train, 1,458 development, and 750 future rows. Work began on matched sequential and interleaved packs over exactly those training rows.

A public model survey ranked Voxtral Mini 3B, Voxtral Small 24B, Voxtral Mini 4B Realtime, Gemma 4 E4B/12B, Qwen3 Omni 30B-A3B, and LFM2.5-Audio as the strongest practical audio-text candidates. Voxtral Mini 3B is the first external baseline to download and run because it combines an Apache-2.0 license, 32k context, roughly 30-minute transcription support, a Whisper-family audio head, and an 8.72 GiB selected-weight footprint. Voxtral Mini 4B Realtime is the streaming-specific follow-up; Gemma 4 is a strong long-context but chunked-audio candidate; Qwen3 Omni and Voxtral Small are quality ceilings rather than routine local baselines.

### 2026-10-01

Recovered ownership of the SALM and vision transplant thread from the Phonon Claude Code session. Reconciled the architecture with its surviving local and B550 artifacts, adapter hashes, and aggregate metrics.

Implemented the first takeover slices. Added `AsrEngineSelection` as one tested launch plan for default Parakeet and custom SALM selection, wired both `AsrSidecar` and the ASR benchmark to it, and added five no-model JSONL protocol tests for `sidecar/salm_server.py`. Added deterministic-first parody contracts in `phonon-core` with protected URL, identifier, and number spans, ordered and duplicate-count preservation checks, negation and source recovery tests, and two fictional built-in styles. The Rust workspace, Python sidecar tests, formatting, and Clippy pass locally. These slices do not wire either capability into the UI or ship the measured research adapters.

Added the canonical architecture after auditing the repository. Encoded the parody plane as a deterministic-first, consent-gated, meaning-locked transform architecture with registered semantic, style, rights, and product gates. Classified the visible SALM transplant as uncommitted and unverified, and defined preservation, protocol, stock-load, adapter, benchmark, product, and distribution gates. No application code or model behavior was changed in this edit.

## Contracts to preserve

- Local execution and user sovereignty outrank quality, personality, and performance.
- Normal dictation accuracy must not regress to obtain a style or model experiment.
- Raw or intended text must remain recoverable when a transform or experimental engine is selected.
- Evaluated outputs stay within their registered workflow and are never pooled with unrelated protocols.
- Personal data is local, consent-gated, journaled, reviewable, and deletable.
- Unverified implementation is never reported as a shipped or working capability.
