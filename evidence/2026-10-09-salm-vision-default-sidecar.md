# Receipt: fused SALM vision becomes the default image sidecar

**Date:** 2026-10-09
**Agent:** Codex
**Scope:** macOS product engine selection, fused inference sidecar, JSONL image protocol, controlled local model installation, and no-model protocol tests.

## What changed

- `AsrEngineSelection::salm_vision()` is now the default when neither an explicit engine nor a custom script override is present. `PHONON_ASR_ENGINE=parakeet|salm|reverse_salm` remains a compatibility override.
- `sidecar/salm_vision_server.py` now loads `merged_lfm.safetensors` directly with the PEFT-key normalization used by the corrected fused evaluator, installs the transplanted SigLIP2 tower/projector, and serves both `transcribe`/`warmup_stream` and `caption`.
- The image request requires `capability="screen_image_model"` and `consent=true`, is rejected if persistence is requested, and has no corpus-registration side effect. The sidecar reads the local image and does not copy it.
- `phonon engine` exposes `caption` and emits a distinct `kind:"image"` result with image feature count.
- The exact vision-lane prompt is pinned as `screenshot_dictation_v1`; audio keeps `prose_dictation_v1`, with an optional one-or-two-example local history extension matching the trained prompt shape.
- The runtime contract includes `torchvision`; the first macOS attempt failed because Transformers' LFM2-VL image processor requires it.

## Pinned local artifacts

Source host: B550 (`compute`).

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Fused language model | `~/.local/share/phonon/salm-vision/merged_lfm.safetensors` | `e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7` |
| Transplanted vision rows | `~/.local/share/phonon/salm-vision/rows_omp.safetensors` | `18a2f5cbc05a88f5da3f490e65eae7212de50c51c8003de4b350c22c00d12117` |

The resumable `rsync --partial` transfer completed after 2.27 GiB / 22 minutes; both local hashes matched the B550 source.

## Live inference

### B550 CUDA control

The sidecar loaded the base audio model, applied `148/332` merged language tensors, installed the vision graft, and reported the fused-model hash above.

- Synthetic screenshot request returned `CUDA kernels bf16 tensors` in `0.9542384219821543` s with 72 image features. The fixture text was `CUDA kernels use bfloat16 tensors`; this is a non-personal smoke, not a quality score.
- The same image without capability and consent returned: `image inference requires capability=screen_image_model and consent=true`.
- `assets/startup.wav` returned exactly `Hello, Fluid Voice.` in `1.052202678984031` s.

### Mac MPS direct sidecar

The exact uv dependency set (including `torchvision`) loaded the controlled local artifacts on MPS.

- `assets/startup.wav` returned exactly `Hello, Fluid Voice.` in `0.37939429201651365` s.
- The synthetic screenshot returned `CUDA kernels bf16 tensors` in `0.5762355419574305` s with 72 image features.
- The no-consent control was rejected.
- Transformers reported that `causal_conv1d` is absent and the reference PyTorch implementation is used. This is a known slower Mac fallback, not a failed check.

### Mac product engine

`cargo run -q -p phonon-cli -- engine` used the new default without an engine environment override.

- It emitted `starting the fused SALM vision engine`.
- The ASR reported `salm-vision-fused`, applied 148/332 merged tensors, passed the batch startup transcript, and passed the warmup request.
- The correction model reached ready and the process emitted top-level `{"type":"ready"}` in about 11.4 s on the second run.
- A consented synthetic `caption` returned `CUDA kernels bf16 tensors` with 72 features in `0.7163533340208232` s as `kind:"image"`.
- The no-consent request was rejected before sidecar dispatch.
- A clean `shutdown` emitted `{"msg":"bye","type":"status"}`.

## Checks

All commands were run in `/Users/kearm/phonon/.worktrees/train-adapt-v4`:

- `cargo fmt --all -- --check`: pass.
- `cargo clippy -p phonon-asr -p phonon-core -p phonon-cli --all-targets -- -D warnings`: pass.
- `cargo test -p phonon-asr`: 10 tests pass (8 engine-selection tests plus the two opt-in real spawn tests that skip without fixture variables).
- `cargo test -p phonon-core`: 41 tests pass, including the new image consent/no-persistence gate.
- `uvx ruff check sidecar tests`: pass.
- `PYTHONPATH=. uv run --no-project --python 3.12 --with pytest --with numpy pytest -q tests/test_prompt_contracts.py tests/test_salm_vision_server.py tests/test_salm_server.py`: 12 tests pass.
- `git diff --check`: pass.

## Non-claims

- This is not a new audio or vision quality result; the selected checkpoint's previously registered scores are unchanged.
- The synthetic screenshot is not a accuracy benchmark and its abbreviated output is not treated as one.
- Keyboard-to-insertion latency, real screenshot capture wiring in the Swift bar, packaging/offline distribution, five-day dogfood status, and public/collaborator artifact upload remain unproven.
- The optional history examples are caller-supplied and local. No personal audio, screenshot, accepted text, or history example was committed.
