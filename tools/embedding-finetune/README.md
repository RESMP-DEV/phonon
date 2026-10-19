# EmbeddingGemma fine-tuning data

This tool creates contrastive `(anchor, positive, negative)` records for the
pinned multimodal embedding model.

## Public seeds

`public_seeds.py` reads pinned local copies of:

- AllNLI train/dev/test triplets
- Quora duplicate-question triplets
- STSB pairs
- sentence-compression pairs
- Simple Wiki pairs
- SQuAD train/validation question-context pairs
- CodeSearchNet Python docstring-code pairs

It deterministically samples each source, keeps only high-similarity STSB
positives, rotates QA/code negatives across differing groups, validates one
shared schema, and records source revisions plus file hashes.

## Private scrubber

`private_scrub.py` combines:

- The deduplicated consensus YouTube rows,
- The whole-leakage-group split manifest,
- Complete EmbeddingGemma 2 text/audio vectors,
- The local Aqua dictionary,
- Teacher transcripts,
- Verified embedded audio bytes.

It emits four private triplet kinds:

- `teacher_consistency`: best teacher hypothesis to reference text.
- `text_audio`: reference text to matching audio, with same-split cross-group
  audio hard negatives. The JSONL stores portable
  `audio://<sha256>.wav` URIs; the audio directory owns the local bytes.
- `term_context`: dictionary term to a containing transcript, with a similar
  transcript that does not contain that term.
- `replacement`: spoken Aqua replacement source to exact written target, with
  another replacement as the negative.

All private JSONL and reports remain under the local research share with mode
`0600`; output and audio directories are mode `0700`, and audio files are
hash-addressed with mode `0600`.

## Current gate

The source YouTube rows still carry `train_allowed=false`. The scrubbed files
are candidates for model-data experiments and quality review, not proof of
permission promotion. Do not train on them outside the registered experiment
boundary until frozen-evaluation contamination checks and an explicit source
permission decision complete.
