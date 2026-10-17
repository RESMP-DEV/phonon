# Receipt: fused checkpoint release status and next-build options

**Date:** 2026-10-17
**Agent:** Codex
**Scope:** live private-Hugging-Face inventory, local fused-artifact integrity, branch integration state, and next product-build options. No model or repository state was changed.

## Live findings

- Authenticated HF identity: `Kearm`.
- Private repository `RESMP-DEV/phonon-reverse-salm` was live at commit
  `2a083fc199e3bac8d10ad0c7afe265ff6309b01f`.
- Its files are `.gitattributes`, `ADAPTER_SHA256SUMS`, `AGENTS.md`, `README.md`,
  `config.json`, and three adapters:
  `native-salm-grpo-v1`, `static-soup-v0`, and `v0.1.0-alpha.1`.
- The selected fused vision checkpoint is **not present on the Hub**.
- The controlled local checkpoint remains intact:
  - `~/.local/share/phonon/salm-vision/merged_lfm.safetensors`
  - SHA-256 `e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7`
  - approximately 2.2 GiB
- Its companion rows remain intact:
  - `~/.local/share/phonon/salm-vision/rows_omp.safetensors`
  - SHA-256 `18a2f5cbc05a88f5da3f490e65eae7212de50c51c8003de4b350c22c00d12117`

The earlier upload was deliberately withheld because the release condition was
lane-level dataset exhaustion, while the measured exposure was only 15.6% audio,
2.9% corrector, and 6.2% vision. That condition has not changed.

## Integration state

- The tested product-sidecar branch is `ml/portable-harness` at
  `5eb2e9a07259b8a45e91bd30d0572554002f1097`.
- There is currently no PR for this branch.
- Current `origin/main` and the branch diverged. A three-way merge preview
  reported overlapping changes in 12 paths, including `AGENTS.md`, `README.md`,
  `phonon-core::data`, CLI command wiring, the architecture plan, and Aqua
  baseline tooling.
- Therefore the fused lane is implemented and verified on its branch, but not yet
  integrated with the newer consent/retention and quality-workflow mainline.

## Options assessed

1. **Integrate the branch first.** Reconcile `ml/portable-harness` with current
   main while preserving PR #18 consent/retention semantics and the quality
   runner. This is the prerequisite for a coherent collaborator release.
2. **Private fused release.** A clean private repository such as
   `RESMP-DEV/phonon-salm-vision-alpha` can carry the fused language state,
   vision rows, checksums, pinned bases/runtime, and loading contract. This is
   turnkey for internal testing, but the artifact is not standalone: the sidecar
   also loads the pinned audio base and VL donor/tower.
3. **Adapter-only release.** Upload the selected pre-fusion adapter and rows,
   then have collaborators reconstruct the fused state. This is smaller but no
   longer matches the now-default direct-fused sidecar path and adds a
   reconstruction failure mode.
4. **Reuse `phonon-reverse-salm`.** Fastest existing channel, but mixes the
   reverse-graft, native audio, and fused vision contracts in a misleadingly
   named repository.
5. **Swift vision-context integration.** Add a separate image-model consent
   setting, temporary no-retention screenshot request, `EngineClient.caption`,
   handling for `kind:"image"`, and fusion of image text with existing OCR
   context. Existing screen-OCR consent must not silently authorize image-model
   consent.
6. **Real-context measurement.** Before promotion, compare OCR-only,
   vision-only, and combined context on consented real captures across active
   and all-display policies, with p50/p95 latency and failure/privacy controls.
7. **Packaging/offline pinning.** Pin the audio base, VL donor, fused state,
   rows, Python runtime, and cache layout; then measure cold/warm launch and
   resident memory.
8. **More same-recipe SFT.** Not recommended: stage-two loss fell while audio
   and vision quality worsened. A new data/recipe direction requires its own
   registered study.

## Commands

- `hf auth whoami`: authenticated as `Kearm`.
- `hf models info RESMP-DEV/phonon-reverse-salm --format json`: private repo and
  file inventory above.
- `shasum -a 256 ~/.local/share/phonon/salm-vision/*`: both artifact hashes
  matched the prior receipts.
- `gh pr list --repo RESMP-DEV/phonon --head ml/portable-harness --state all`:
  no PR.
- `git merge-tree $(git merge-base origin/main ml/portable-harness) origin/main
  ml/portable-harness`: overlapping paths identified above.

## Non-claims

- No upload was performed and this receipt is not release authorization.
- No new model quality, real-screenshot, latency, packaging, or dogfood claim is
  made.
- The branch-merge preview identifies overlap, not the difficulty of semantic
  resolution; no integration branch was created in this audit.
