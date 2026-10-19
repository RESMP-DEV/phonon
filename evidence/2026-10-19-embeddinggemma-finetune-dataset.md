# EmbeddingGemma 2 fine-tuning dataset pilot, 2026-10-19

## Scope

This receipt records the public-seed inventory, private scrub, and review
repair for an EmbeddingGemma 2 contrastive adaptation dataset. Public seed text
is stored in the local research share. Private transcripts, dictionary values,
row IDs, and audio remain absent from this receipt and Git.

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
- Public output size: 10,380,485 bytes

Every source revision and file SHA-256 is recorded in the local
`public-triplets-v2.report.json`.

Review remediation:

- STSB is filtered before sampling so only pairs with normalized similarity
  `>= 0.8` can become positives; 4,343 low-similarity source rows were filtered.
- SQuAD and CodeSearchNet negatives rotate across differing titles or
  repositories. The generated QA lane has 1,733 distinct explicit negatives
  (maximum reuse four); the code lane has 991 (maximum reuse two).
- Output paths are checked for self-collision and pinned-source collision.
- Local filesystem paths were removed from public records and replaced by
  relative seed paths.

## Private scrub

The scrubber used:

- 1,700 source observations
- 1,442 unique audio/text rows
- 258 removed duplicate observations
- Complete 1,442-row text and audio vector coverage
- The regenerated 797-target Aqua lexical lexicon
- The current whole-leakage-group split manifest

It wrote:

- Train: 2,482 triplets
- Dev: 428 triplets
- Test: 328 triplets
- Total: 3,238 triplets

By kind:

- `teacher_consistency`: 1,442
- `text_audio`: 1,442
- `term_context`: 321
- `replacement`: 33

Quality observations:

- Rows represented by at least one dictionary term: 122 unique terms
- Replacement sources available: 36; 33 passed distinctness filtering
- Replacement negative targets: 24 distinct values; maximum reuse two
- Hard-negative cosine range: 0.717187 to 0.906652
- Mean hard-negative cosine: 0.830375
- Invalid generated triplets: 0
- Rows containing a URL or home-path marker: 0
- Hash-addressed audio files: 1,442
- Audio bytes: 1,383,060,878
- JSONL and report permissions: `0600`; output directory permissions: `0700`
- Text-audio records use portable `audio://<sha256>.wav` references

Review remediation:

- The split manifest now rejects a leakage group that spans multiple splits.
- Per-term caps continue instead of aborting all later terms in the same row.
- Empty teacher hypotheses cannot be selected.
- Replacement negatives use deterministic overlap plus varied hash ranking.
- Private report privacy flags distinguish aggregate report contents from the
  intentionally local training JSONL.

## Non-claims

The private scrub does not promote YouTube rows, remove the source permission
gate, establish frozen-evaluation disjointness, or qualify a fine-tuned model.
No training run has begun. The public audio corpora FLEURS, LibriSpeech,
VoxPopuli, and People's Speech were cataloged but not bulk-downloaded because
their smallest useful embedded-audio shards begin around 400 MiB; the existing
private audio already supplies the first cross-modal lane.

## Verification

The repaired implementation's focused test suite passed 14 tests. The live
public v2 converter completed with exit code 0, wrote 9,996 validated records,
and emitted no home-path markers. The live private v2 scrubber completed with
exit code 0, wrote 3,238 validated records, and reported zero invalid generated
triplets. A subsequent full-file aggregate validation pass found zero URL or
home-path markers in the private JSONL, verified all text-audio references were
portable hash URIs, and verified the file/directory permission matrix. The
repository common quality gate result for the final reviewed head is recorded
under `build/reports/quality/receipt.json`.
