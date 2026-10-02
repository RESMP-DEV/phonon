# Final-training Optuna sweep

This lane prepares the eventual reverse Audio-to-VL production training run.
It is deliberately not launched by default: pass `--execute` only when the
compute budget and data/prompt choices are final.

## Contracts

- Prompt formatting is baked into every public and Aqua training row.
- The evaluator receives the same `--prompt-id`; hypothesis rows record the ID
  and SHA-256 without storing the prompt text.
- Optuna user attributes record prompt hash, trial root, fair WER, strict WER,
  exact rate, and score path.
- The public backing dataset revision is pinned to
  `969944574ea3f37890beaf67ea651e160cfaf043`
  (`espnet/yodas-granary`, English `asr_only`).
- Aqua manifests are content-hashed into pack paths.
- The study uses a persistent SQLite storage URL and resumes with
  `load_if_exists=True`.
- Every trial trains public-first and Aqua-second, matching the measured
  scaling recipe.

## Prompt candidates

`prompts.py` defines:

- `prose_dictation_v1`: prior SALM prompt.
- `xml_dictation_v1`: house XML format with exact technical-term and detail rules.
- `xml_dictation_guarded_v1`: conservative uncertainty and no-invention rules.
- `xml_transcript_v1`: verbatim public-audio pretraining prompt.

A production run can fix the chosen format with, for example,
`--prompt-ids xml_dictation_v1`; leaving the default samples all registered IDs.

## Dry run

From the B550 research checkout after syncing this directory and sibling
`reverse_vl_v0`:

```bash
uv run --python 3.12 --with optuna python optuna_sweep.py \
  --study final-reverse-vl --dry-run \
  --research-python /home/kearm/envs/salm-lora/bin/python \
  --work-root /home/kearm/salm-lora/build/optuna \
  --manifest /home/kearm/salm-lora/hq-train-manifest.jsonl \
  --audio-root /home/kearm/aqua-training-data \
  --exclude /home/kearm/salm-lora/slice-eval-500.jsonl \
  --eval-slice /home/kearm/salm-lora/slice-eval-500.jsonl
```

The dry run prints every command without loading models or training.

## Execute

```bash
uv run --python 3.12 --with optuna python optuna_sweep.py \
  --study final-reverse-vl --trials 24 --execute \
  --storage sqlite:////home/kearm/salm-lora/build/optuna/final-reverse-vl.db \
  --research-python /home/kearm/envs/salm-lora/bin/python \
  --work-root /home/kearm/salm-lora/build/optuna \
  --manifest /home/kearm/salm-lora/hq-train-manifest.jsonl \
  --audio-root /home/kearm/aqua-training-data \
  --exclude /home/kearm/salm-lora/slice-eval-500.jsonl \
  --eval-slice /home/kearm/salm-lora/slice-eval-500.jsonl \
  --public-rows 10000 --public-steps 10000 --aqua-steps 1000 \
  --eval-limit 500 --prompt-ids xml_dictation_v1
```

The objective is fair WER on the frozen evaluation slice. The study is intended
to run serially on one GPU. Increase data/steps only after the architecture and
prompt format have been selected; do not spend the full budget exploring all
prompts unless that is an explicit experiment.

## Measured scaling context

On the first 25 frozen rows, the reverse graft measured:

| recipe | fair WER | strict WER | exact |
| --- | ---: | ---: | ---: |
| Aqua-only 2k | 0.1458 | 0.2029 | 0.28 |
| public 2k only | 0.1840 | 0.2322 | 0.28 |
| public 2k + Aqua 1k | 0.1337 | 0.1755 | 0.40 |
| public 10k + Aqua 1k | **0.0938** | **0.1335** | 0.36 |

The 10k public run used a different exact-transcript public prompt and prose
Aqua prompt. The Optuna harness fixes that inconsistency by using one selected
prompt across both stages and evaluation.
