# Receipt: Aqua dynamic-history prompt pack v2

Date: 2026-10-05
Agent: Codex
Source revision: `ac536758d01258926307acef9e87d9546743ebed`
Host: B550 (`/home/kearm/phonon`)
Output: `/home/kearm/salm-lora/build/history-prompt-full-v2/pack`

## What changed

- Added prompt `prose_history_dictation_v1`, base prompt SHA-256
  `4d7ea7b4687a8f82ddc910b76c2c529fbf17755dcfc6649d2c6be02007c3670f`.
- Retrieval uses the current Aqua raw transcript as the query, the training-only
  manifest as the candidate pool, and two historical `raw -> corrected` pairs
  selected by `idf-cosine-v1`.
- Retrieval excludes the target by audio ID and never reads the target's
  corrected text. Each target receives a dynamic prompt SHA-256.
- Replaced per-row IDF/vector reconstruction with one precomputed history index
  shared by pack construction and evaluation.

## Verification

Local and B550 both ran:

```bash
uvx ruff check ml/research/final_sweep ml/research/reverse_vl_v0
PYTHONPATH=ml/research/final_sweep \
  uv run --python 3.12 --with pytest \
  pytest ml/research/final_sweep/tests/test_history_prompt.py \
    ml/research/reverse_vl_v0/tests/test_train_adaptations.py \
    -q -p no:cacheprovider
git diff --check
```

Result: ruff passed, six tests passed, and whitespace checks passed on both
hosts.

## Full-pack receipt

- Started: `2026-10-05T15:32:57-07:00`
- Completed: `2026-10-05T15:47:58-07:00`
- Wall time: 901 seconds
- Manifest SHA-256:
  `9378e4532e54c8816ea962a515ac833d60efbab0ec1e1d102b0296e81d490f08`
- History-contract SHA-256:
  `21eccc7415d93548a347f816aa8b71ef18709ae2b45ef0f77da9c2441d623053`
- Contract targets: 12,855
- Dynamic prompt hashes: 12,855 unique values
- Target self-exclusions: all 12,855 passed
- Two distinct history IDs per target: all 12,855 passed
- Packed Arrow examples: 12,854
- Arrow shards: 13
- Pack size: `6127290118` bytes
- Per-file SHA-256 manifest:
  `/home/kearm/salm-lora/build/history-prompt-full-v2/SHA256SUMS`
- SHA-256 of that checksum manifest:
  `4f293cc3ae2352bcdc3300fc8949e2e533318433ed9ac17db159beb6bef0eddd`

The metadata row count is 12,855 because every eligible manifest target has a
history contract. Dataset preprocessing skipped one target whose packed length
was 781 tokens at context 768, leaving 12,854 trainable Arrow examples. Context
768 was retained because the 32-row pilot had minimum/median/P95/maximum packed
lengths of 296/478/578/649; context 512 would truncate many history prompts.

The earlier full-v1 attempt was stopped before producing a usable pack because
per-row IDF/vector reconstruction made retrieval unnecessarily quadratic. The
v2 root is a separate completion; the failed attempt remains preserved.

## Non-claims

- This receipt proves pack structure and reproducibility, not model quality.
- It does not score the 500-row selection slice or 750-row future split.
- It does not establish optimized-runtime inference parity or product latency.
- Aqua audio and transcript contents remain personal data on B550-controlled
  storage. Only IDs, hashes, counts, sizes, and aggregate metrics are receipted.
- Downstream checkpoints and scores are valid only with prompt
  `prose_history_dictation_v1`, context 768, and the reconstructed history
  index from the receipted training manifest.
