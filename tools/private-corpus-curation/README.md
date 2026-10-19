# Private corpus curation

This research lane turns private YouTube teacher hypotheses into a
deduplicated, leakage-safe, content-balanced training/evaluation plan before
any permission promotion.

## Components

- `corpus_curation.py` imports Aqua dictionary values locally, computes lexical
  fingerprints, removes duplicate IDs/audio hashes, blocks cross-group
  near-duplicate components, and assigns whole leakage groups to train/dev/test
  with a deterministic local-search balance objective.
- `embedding_runner.py` emits resumable normalized vectors. It supports the live
  local Nomic endpoint and pinned `google/embeddinggemma-2` text/audio inputs.
- `glm_reconcile.py` sends only teacher hypotheses and coarse domain tags to a
  configured GLM endpoint. It is blind to reference labels and requires strict
  JSON output.

## Pinned EmbeddingGemma 2

```text
model=google/embeddinggemma-2
revision=914f7f89142e33e77833254d9c9b90c3cef7303b
dimensions=768
context=8192 shared text/audio/vision tokens
audio_input=mono PCM16 16 kHz
license=Apache-2.0
```

The model supplies a unified audio/text space, so the same cosine thresholds can
be calibrated for transcript and acoustic deduplication. Text uses the
`SentenceSimilarity` task prefix; audio uses the `<|audio|>` placeholder.

## Privacy

Never place row-level split manifests, embedding vectors, teacher hypotheses,
GLM transcripts, Aqua phrases, or source URLs in Git. Keep those files only in
the local research share with mode `0600`. Reports printed to the terminal and
tracked evidence must be aggregate-only.

A passing plan does not promote a row for training. Every source row retains its
original `train_allowed` and `eval_allowed` flags in the local manifest. A later
promotion must be a separate, provenanced decision after frozen evaluation
contamination checks.
