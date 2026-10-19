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

## Execution provenance

The pilot began at repository revision
`51f728fa4298569962020c4759dc917e5b5d0012` with the new files uncommitted, as
expected for a first live implementation. The following aggregate-only live
commands completed successfully before the first implementation commit:

- Aqua lexicon import: exit 0, mode `0600`
- Nomic text pilot: exit 0, 32 rows
- EmbeddingGemma 2 text pilot: exit 0, 32 rows
- EmbeddingGemma 2 audio pilot: exit 0, 32 rows
- GLM-5.3 reconciliation pilot: exit 0, 2 rows
- Real split planning: exit 0, aggregate report and local row manifest

The first implementation commit is
`a23e907ffe3df6ce78b74b8f882074fd8ab3f045`; its clean-head `just quality` gate
passed. A later live run exposed duplicate embedding work, fixed by commit
`8f5de513b5e6f8fc94bfaeb1e6f1ab3b9371cc42`; the final clean-head gate for that
repair is recorded under the worktree's ignored
`build/reports/quality/receipt.json`. Review hardening is recorded separately
and does not reinterpret the pilot numbers above.

## Full EmbeddingGemma 2 pass

After review hardening, the resumable local job completed successfully:

- Text rows: 1,442 / 1,442
- Audio rows: 1,442 / 1,442
- Dimensions: 768 for both modalities
- Model: `google/embeddinggemma-2`
- Revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- Local weight SHA-256:
  `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`
- Output permissions: `0600`
- Text/audio job exit statuses: 0 / 0

A provenance-checked resume returned 1,442 already-present rows, generated zero,
and validated the local artifact contract.

The first full-coverage, whole-leakage-group exploratory plan retained 1,442
rows and 18 groups:

- Train: 1,113 rows, 10 groups, 9.26 hours
- Dev: 191 rows, 2 groups, 1.59 hours
- Test: 138 rows, 6 groups, 1.15 hours
- Text/audio embedding coverage: 100% / 100%
- Initial 0.98 text and 0.995 audio cross-group near-duplicate pairs: 0

The split remains exploratory because final thresholds and frozen-evaluation
contamination checks are incomplete. `freeze_eligible` remains false because
every source row still carries both permission flags false.

The stricter GLM prompt schema initially produced two preserved `CurationError`
rows because the endpoint reports served model `glm-5.3` while the request uses
`zai,glm-5.3`. After alias validation, a fresh two-row run produced 2/2
successful strict-JSON records with served model `glm-5.3`, finish reason
`stop`, and zero failures.
