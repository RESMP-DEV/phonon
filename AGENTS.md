# Phonon agent instructions

## Authority and scope

Phonon is a local-first voice authoring system for macOS, with a beta Windows
runtime. Privacy and user sovereignty outrank final-text fidelity, latency,
personality, and convenience. `docs/architecture.md` is the single
cross-component plan; `SPEC.md` owns product requirements; this file and
`CONTRIBUTING.md` own contributor procedure. `README.md` routes current status.

Act through implementation and verification within the assigned scope. An
investigation does not authorize feature wiring, and a quality or documentation
task does not authorize runtime changes. Preserve unrelated dirty work and use
one worktree per implementation agent under ignored `.worktrees/`.

`AGENTS.md` is the canonical instruction file. `CLAUDE.md` must remain the
literal relative symlink `AGENTS.md`; the common quality runner checks this.

## Product boundaries

- Personal dictation audio, transcripts, accepted text, OCR, screenshots, and
  traces are local data. Never place them in Git, logs, public artifacts,
  telemetry, or an unapproved remote provider.
- Consent is specific: microphone permission is not screen consent; screen OCR
  is not image-model consent; ordinary dictation is not corpus retention. Screen
  capture must fail closed when macOS permission is absent.
- Training capture defaults off. Any retained screenshot requires explicit
  capture consent, image consent, provenance, retention, review, and deletion.
- A failed or malformed personal-data store is evidence. Preserve it, surface
  recovery, and never silently overwrite it.
- Raw or intended text must remain recoverable when an experimental transform is
  selected. A correction may not replace a better acoustic transcript.
- A green unit test does not qualify a model adapter, full-slice quality claim,
  latency claim, or daily-driver readiness. Apply the registered gates in
  `docs/architecture.md`.

## Testing and evidence procedure

Use `CONTRIBUTING.md` as the shared procedure and required-check matrix. Start
with `just quality`, then add each affected Rust, Swift, native-UI,
permissions, packaging, research, or privacy gate. A receipt must state its
revision, dirty state, commands, outcomes, scope, and non-claims; startup must
invalidate any prior PASS.

Workers are accelerators, not acceptance. The integrating contributor reviews
their diffs, reruns decisive parent checks, and records one consolidated
receipt. Native debugging uses JSON diagnostics and deterministic synthetic
fixtures, never owner-screen capture or UI-coordinate automation.

## Parallel work

Assign disjoint file scope and one build/report owner per checkout. Workers
hand off changed files, contracts, commands, evidence, limitations, and
remaining work. Shared contracts or architecture edits require coordinator
review before dependent branches start. Do not reset, clean, or overwrite
another contributor's checkout.
