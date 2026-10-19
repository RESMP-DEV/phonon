# Private corpus curation pilot, 2026-10-19

## Scope

This receipt records the first implementation and live pilot for the private
YouTube curation lane. No source row was promoted for training, no private audio
or transcript was committed, and no Aqua phrase appears below.

## Live Aqua lexical signal

The local Aqua settings cache was read without printing entry values:

- Dictionary observations: 782
- Replacement observations: 65
- Unique normalized lexical targets: 796
- Sources represented across unique targets: dictionary and replacement
- Local lexicon SHA over normalized phrases:
  `e0bd7118e08f6a719a7eb7df9aa4aa7e0873440c997e487a164bbb260efbcfeb`

The full lexicon remains local under
`~/.local/share/phonon-research/private-corpus-curation/20261019/`.

## EmbeddingGemma 2 pilot

The pinned model was:

- `google/embeddinggemma-2`
- revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
- 740M parameters
- 768-dimensional unified text/audio/vision space
- 8K shared context
- Apache-2.0 license

Live local inference succeeded for both text and audio. On the same first 32
consensus rows:

- Text: 32 vectors, 768 dimensions, 9.456 seconds total
- Audio: 32 vectors, 768 dimensions, 16.963 seconds total
- Audio-to-text retrieval matched the same row 32/32 times
- Mean matching audio/text cosine: 0.802126
- Fifth-percentile matching audio/text cosine: 0.773948

The same 32-row Nomic baseline produced 768-dimensional text vectors in 0.963
seconds. On this technical slice:

- Nomic maximum pair cosine: 0.818027
- EmbeddingGemma 2 maximum text pair cosine: 0.887755
- Both providers had zero text pairs at the initial 0.98 threshold
- Both placed 25/32 nearest text neighbors in the same leakage group
- Top-10 neighborhood overlap: 0.628125

These numbers do not yet prove a global winner. They justify a full-pass
EmbeddingGemma 2 comparison because it adds native audio embeddings and stronger
same-group semantic separation.

## Reconciliation pilot

A blind GLM reconciliation request used teacher hypotheses and coarse domain
tags only. It did not include reference text, source URLs, audio bytes, paths,
or row IDs. The first two rows completed in 16.419 seconds with:

- Strict JSON outputs: 2/2
- Finish reason `stop`: 2/2
- Decisions: 2 keep
- Mean confidence: 0.975

The outputs remain local and mode `0600`.

## Real split-plan pilot

The planner processed the 1,700-observation consensus mirror:

- Input observations: 1,700
- Kept unique rows: 1,442
- Duplicate IDs removed: 258
- Duplicate audio hashes after ID deduplication: 0
- Leakage groups/components: 18
- Rows with pilot text vectors: 32
- Rows with pilot audio vectors: 32
- Cross-group text or audio pairs above initial thresholds: 0

The deterministic optimizer assigns whole leakage-group components. With the
32-row EmbeddingGemma 2 fingerprint pilot, the exploratory plan had:

- Train: 1,151 rows, 7 groups, 9.58 hours
- Dev: 141 rows, 7 groups, 1.17 hours
- Test: 150 rows, 4 groups, 1.25 hours

This is an implementation pilot, not a frozen split. Full text and audio
vectors are required before balancing or near-duplicate thresholds are final.
Every row remains `train_allowed=false,eval_allowed=false`.

## Non-claims

This pilot does not authorize training, does not prove that EmbeddingGemma 2
outperforms Nomic on the full corpus, does not calibrate final duplicate
thresholds, and does not establish GLM reconciliation quality. It also does not
remove the private YouTube permission gate or compare source groups against
every frozen evaluation manifest.
