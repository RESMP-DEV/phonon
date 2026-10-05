# Phonon agent instructions

## Authority and boundaries

- This repository is the canonical Phonon product source. `docs/architecture.md` is
  the single implementation plan; do not create a parallel roadmap or component
  summary.
- Keep changes scoped to the requested product plane. A research result is not
  permission to wire an experimental model into the app, and documentation work is
  not permission to alter runtime behavior.
- Preserve unrelated dirty work. Inspect `git status` before edits and never reset,
  checkout, clean, or overwrite it to make a branch tidy.
- Publishing targets remain private repositories under `RESMP-DEV/phonon`. Do not
  enable GitHub Actions or author workflow files; verification runs locally and on
  the named compute host.
- B550 is the CUDA research boundary at `~/salm-lora`. The main machine remains the
  authority for source, credentials, Git, and acceptance. Do not turn B550 into a
  control plane or copy authoritative state there.

## Data and privacy

- Aqua dictation audio and accepted text are personal data. Keep them on
  main-machine or B550-controlled storage; never upload them to a public endpoint,
  public artifact, model card, issue, PR, or telemetry provider.
- API teachers may receive only the minimum fields explicitly approved by the
  active architecture document. OpenRouter is not an approved dataset-generation
  provider for this thread.
- The Hugging Face placeholder
  `RESMP-DEV/phonon-reverse-salm` is private and metadata-only. Do not upload
  adapter weights, tokenizer payloads, Aqua audio, transcripts, accepted text,
  dictionaries, or screenshots without a new explicit release decision.
- Screen context is OCR-only in the product today. A screenshot may be passed to a
  local vision head only through an explicit opt-in image protocol with declared
  retention, provenance, capability, and deletion behavior. Do not persist it merely
  because capture succeeded.
- W&B on B550 is offline-only. Never copy `/Users/kearm/.netrc`, a W&B API key,
  or an online W&B environment to the compute host. Copy only offline run
  transaction directories back to the Mac and sync them with
  `ml/research/final_sweep/sync_wandb_offline.py` from this machine.
- W&B telemetry may contain prompt IDs, hashes, scalar metrics, package
  versions, and adapter hashes. It must never contain Aqua audio, transcript or
  accepted text, raw ASR text, screen content, or unnecessary personal paths.
- The packaged reverse SALM prompt is owned by `sidecar/phonon_prompts.py`.
  Its current sole contract is `prose_dictation_v1` with SHA-256
  `3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`;
  training, evaluation, and runtime must remain byte-identical.
- A small pilot cannot be presented as a full-slice improvement. Record the row
  count, frozen manifest, protocol, and non-claims with every quality claim.

## Research receipts

- Before claiming a B550 run is live, record the process command, SQLite trial
  state, GPU utilization/memory, and relevant log tail. A wrapper PID file alone is
  not evidence.
- Before claiming a run completed, record the marker, trial state and value,
  hypothesis/score paths, SHA-256 hashes, and prompt hash. For resumed or repaired
  work, preserve the failed trial and add a separate provenance-rich completion.
- Distinguish transient network retries from hardware failure. Check
  `journalctl -k`, XID/NVRM messages, OOM kills, storage errors, GPU remapped rows,
  memory, and disk before calling a machine fault.
- Optuna packs are prompt- and context-length-specific. Never reuse a pack path or
  metadata across either dimension. Evaluation must reconstruct the exact trained
  LoRA rank.
- Prompt formatting is part of the training contract. Public CPT, Aqua adaptation,
  and evaluation must all record and use the same prompt ID and SHA-256.

## Current B550 research locations

- Study database: `/home/kearm/salm-lora/build/optuna/final-reverse-vl-xml-v1.db`
- Trial roots: `/home/kearm/salm-lora/build/optuna/trial-*`
- First-run Optuna log:
  `/home/kearm/salm-lora/build/optuna/final-reverse-vl-xml-v1.log`
- Resumed four-trial log: `/home/kearm/salm-lora/build/optuna/final-resume.log`
- Resumed markers: `OPTUNA-RESUME-DONE` and `OPTUNA-RESUME-FAILED`
- Frozen evaluation slice: `/home/kearm/salm-lora/slice-eval-500.jsonl`
- Untouched future evaluation slice:
  `/home/kearm/salm-lora/hq-future-eval-v1.jsonl`, SHA-256
  `5f71c33816151f363be727622e48ac75c3ebef982a14c71aa0951d722554e728`.
  It maps the manifest's `corrected` field to the evaluator's `ref` field.
- Offline ASR teacher roots:
  `/home/kearm/salm-lora/build/asr-teachers/{qwen3-asr-1p7b,cohere-transcribe-03-2026}`
- Phonon-owned Voxtral teacher:
  `ml/research/asr_teachers/quantize_voxtral_mxfp4.py`. Build only from
  `mistralai/Voxtral-Small-24B-2507@da5b42409f279fdd92febee0511a6c32828569c1`,
  never from the third-party C4-calibrated INT4 checkpoint. Keep B550 base,
  calibration, output, and receipt artifacts under
  `/home/kearm/salm-lora/build/asr-teachers/`.
  The canonical verified base is
  `/home/kearm/salm-lora/build/asr-teachers/voxtral-small-24b-hfd-da5b424`;
  transfer it with pinned `hfd ... --verify full`, not the stalled Xet path.
  The full-calibration HF artifact is
  `voxtral-mxfp4-audio256-v1` and its vLLM Mistral-name variant is
  `voxtral-mxfp4-audio256-v1-vllm`; preserve both receipts. The full-slice
  result is 0.060714 fair WER and does not replace Aqua raw.
- Public dataset revision:
  `espnet/yodas-granary@969944574ea3f37890beaf67ea651e160cfaf043`
- Matched format bake-off root:
  `/home/kearm/salm-lora/build/format-bakeoff/trial-*`
- Reverse-training NSys profile root:
  `/home/kearm/salm-lora/build/nsys/reverse-training-v1`
- Current kernel-stack profile root:
  `/home/kearm/salm-lora/build/nsys/reverse-training-kernels-v1`
- B550 reverse SALM product adapter:
- B550 reverse SALM product adapter:
  `/home/kearm/.local/share/phonon/reverse-salm/lora_adapter.safetensors`,
  currently the equal 8k/full-epoch prose soup SHA-256
  `9a70586a682fe2c694896efe3ac5aa5c467874ab6a9f5e0c2dc847eff31a3f5c`.
  The preceding trial-0100 adapter remains at
  `lora_adapter.trial0100.safetensors` in the same directory.
- CUDA extension packages are not active just because a wheel resolved. Record
  the Torch/CUDA build, import result, real model load, and matched profile
  before reporting FlashAttention, causal-convolution, or fused-loss kernels.
- Kernel mode is part of a training and inference receipt. Trial 0200 showed
  that active FlashAttention/causal kernels can change full-slice training
  quality, but the unchanged trial-0100 adapter subsequently matched its exact
  full-slice inference score under the current runtime. Record kernel mode with
  both training and runtime quality claims.
- Trial-0200 final-repeat root:
  `/home/kearm/salm-lora/build/format-bakeoff/trial-0200`
- Warm-cache and 1,000-step controls:
  `/home/kearm/salm-lora/build/nsys/reverse-triton-warm-cache-v1`,
  `/home/kearm/salm-lora/build/nsys/reverse-triton-warm1000-v1`, and
  `/home/kearm/salm-lora/build/nsys/reverse-baseline-1000-v1`
- Trial-0100 current-runtime parity receipt:
  `/home/kearm/salm-lora/build/product-runtime-eval/trial-0100-active-kernels-v1/receipt.json`
- Aqua adaptation-length roots:
  `/home/kearm/salm-lora/build/aqua-length-reference-v1`,
  `/home/kearm/salm-lora/build/training-options-v2`, and
  `/home/kearm/salm-lora/build/aqua-length-bracket-v3`.
- Promoted soup and installation receipts:
  `/home/kearm/salm-lora/build/aqua-length-bracket-v3/soup-8000-full-receipt.json`
  and
  `/home/kearm/.local/share/phonon/reverse-salm/installed-adapter-receipt.json`.
- Private HF placeholder:
  `https://huggingface.co/RESMP-DEV/phonon-reverse-salm`

## Required local checks

Run the scope appropriate to the change, and state exactly what ran:

```bash
uvx ruff check ml/research/asr_teachers ml/research/final_sweep ml/research/reverse_vl_v0
cd ml/research/asr_teachers/tests && \
  uv run --python 3.12 --with pytest python -m pytest . -q -p no:cacheprovider --noconftest
shellcheck ml/research/asr_teachers/*.sh
shfmt --diff ml/research/asr_teachers/*.sh
PYTHONPATH=ml/research/final_sweep uv run --python 3.12 --with optuna \
  pytest ml/research/final_sweep/tests/test_optuna_plan.py -q -p no:cacheprovider
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
PYTHONPATH="$PWD" uv run --python 3.12 pytest tests -q -p no:cacheprovider
```

A green unit test does not make a research adapter product-ready. Product claims
also require the registered audio/vision gates and the architecture document's
privacy, latency, and regression requirements.
