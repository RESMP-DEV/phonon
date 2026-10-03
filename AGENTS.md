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
- Screen context is OCR-only in the product today. A screenshot may be passed to a
  local vision head only through an explicit opt-in image protocol with declared
  retention, provenance, capability, and deletion behavior. Do not persist it merely
  because capture succeeded.
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
- Public dataset revision:
  `espnet/yodas-granary@969944574ea3f37890beaf67ea651e160cfaf043`

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
