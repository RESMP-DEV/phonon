# EmbeddingGemma 2 fine-tuning dataset pilot, 2026-10-19

## Scope

This receipt records the first public-seed inventory and private scrub for an
EmbeddingGemma 2 contrastive adaptation dataset. Public seed text is stored in
the local research share. Private transcripts, dictionary values, row IDs, and
audio remain absent from this receipt and Git.

## Official training contract

The pinned model remains:

- `google/embeddinggemma-2`
- revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
- 768-dimensional text/audio/vision space

Google's current SentenceTransformers guide specifies
`MultipleNegativesRankingLoss`, `(anchor, positive, negative)` records, the
same task prompt at training and inference, and column-specific prompts for
multimodal records. The generated schema follows that contract.

## Public seed inventory

Ten pinned source files were downloaded locally:

- AllNLI train/dev/test triplets
- Quora duplicate-question triplets
- STSB train pairs
- sentence-compression pairs
- Simple Wiki pairs
- SQuAD train/validation
- CodeSearchNet Python validation

The deterministic converter sampled up to 1,000 records per source:

- Written public triplets: 9,996
- `public_text`: 6,996
- `public_qa`: 2,000
- `public_code`: 1,000
- Invalid records: four
- Duplicate triplets: zero
- Public output size: 13,402,023 bytes

Every source revision and file SHA-256 is recorded in the local
`public-triplets-v1.report.json`.

## Private scrub

The scrubber used:

- 1,700 source observations
- 1,442 unique audio/text rows
- 258 removed duplicate observations
- Complete 1,442-row text and audio vector coverage
- The regenerated 797-target Aqua lexical lexicon
- The current whole-leakage-group split manifest

It wrote:

- Train: 2,440 triplets
- Dev: 410 triplets
- Test: 311 triplets
- Total: 3,161 triplets

By kind:

- `teacher_consistency`: 1,442
- `text_audio`: 1,442
- `term_context`: 244
- `replacement`: 33

Quality observations:

- Rows represented by at least one dictionary term: 91 unique terms
- Replacement sources available: 36; 33 passed distinctness filtering
- Hard-negative cosine range: 0.717187 to 0.906652
- Mean hard-negative cosine: 0.829769
- Invalid generated triplets: 0
- Rows containing a URL or home-path marker: 0
- Hash-addressed audio files: 1,442
- Audio bytes: 1,383,060,878
- JSONL and report permissions: `0600`

## Non-claims

The private scrub does not promote YouTube rows, remove the source permission
gate, establish frozen-evaluation disjointness, or qualify a fine-tuned model.
No training run has begun. The public audio corpora FLEURS, LibriSpeech,
VoxPopuli, and People's Speech were cataloged but not bulk-downloaded because
their smallest useful embedded-audio shards begin around 400 MiB; the existing
private audio already supplies the first cross-modal lane.

## Verification

The implementation's focused test suite passed eight tests. The live public
converter completed with exit code 0 and wrote 9,996 validated records. The live
private scrubber completed with exit code 0, wrote 3,161 validated records, and
reported zero invalid generated triplets. A subsequent full-file validation pass
found zero URL/home-path markers in the private JSONL. The repository common
quality gate passed with 122 Python tests plus repository contracts, Ruff,
formatting, diff hygiene, and shell checks; the machine receipt records the
exact final head under `build/reports/quality/receipt.json`.
